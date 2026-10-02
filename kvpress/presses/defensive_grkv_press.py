# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""GRKV regressors applied to DefensiveKV's strict cross-head selection."""

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.defensivekv_press import DefensiveKVPress
from kvpress.utils import get_prerope_query_states


@dataclass
class DefensiveGRKVEarliestPress(DefensiveKVPress):
    """DefensiveKV selection followed by the earliest, head-mean GRKV update.

    DefensiveKV alone decides which positions survive under one global budget of
    ``int((1 - compression_ratio) * cache_length * num_kv_heads)``.  No sink
    position is protected.  The selected prefix centers are then updated with the
    earliest GRKV objective, which averages the GQA query heads served by each KV
    head and always applies its joint key/value regression result.
    """

    grkv_guard: float = 0.1
    grkv_key_lambda: float = 1e-2
    grkv_value_lambda: float = 1e-2
    grkv_num_steps: int = 1
    grkv_tol: float = 1e-9

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.grkv_guard <= 1:
            raise ValueError("grkv_guard must be in [0, 1]")
        if self.grkv_key_lambda < 0:
            raise ValueError("grkv_key_lambda must be non-negative")
        if self.grkv_value_lambda < 0:
            raise ValueError("grkv_value_lambda must be non-negative")
        if self.grkv_num_steps <= 0:
            raise ValueError("grkv_num_steps must be positive")
        if self.grkv_tol < 0:
            raise ValueError("grkv_tol must be non-negative")

    def __str__(self):
        return (
            f"{self.__class__.__name__}={self.compression_ratio}_win={self.window_size}"
            f"_kernel={self.kernel_size}_guard={self.grkv_guard}"
        )

    @staticmethod
    def _global_keep_mask(scores: torch.Tensor, compression_ratio: float) -> torch.Tensor:
        """Select exactly the same flattened cross-head top-k as DefensiveKV."""

        batch_size, num_kv_heads, cache_length = scores.shape
        total_positions = num_kv_heads * cache_length
        n_kept = DefensiveKVPress.get_num_kept(cache_length, num_kv_heads, compression_ratio)
        kept_indices = scores.reshape(batch_size, total_positions).topk(n_kept, dim=-1).indices
        keep_mask = torch.zeros(
            (batch_size, total_positions),
            dtype=torch.bool,
            device=scores.device,
        )
        keep_mask.scatter_(1, kept_indices, True)
        return keep_mask.view(batch_size, num_kv_heads, cache_length)

    @staticmethod
    def _set_masked_indices(module: nn.Module, keep_mask: torch.Tensor) -> None:
        batch_indices, head_indices, sequence_indices = torch.nonzero(~keep_mask, as_tuple=True)
        module.masked_key_indices = (  # type: ignore[assignment]
            batch_indices,
            head_indices,
            sequence_indices,
        )

    @staticmethod
    def _query_states_at_positions(
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        source_positions: torch.Tensor,
        rotary_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected_hidden = hidden_states.index_select(1, source_positions)
        query_states = get_prerope_query_states(module, selected_hidden)
        cos, sin = position_embeddings
        positions = source_positions if rotary_positions is None else rotary_positions
        selected_cos = cos.index_select(1, positions)
        selected_sin = sin.index_select(1, positions)
        return (query_states * selected_cos.unsqueeze(1)) + (rotate_half(query_states) * selected_sin.unsqueeze(1))

    @staticmethod
    def _group_queries(module: nn.Module, query_states: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
        batch_size, num_query_heads, query_count, head_dim = query_states.shape
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("The number of query heads must be divisible by the number of KV heads")
        return query_states.view(
            batch_size,
            num_kv_heads,
            num_query_heads // num_kv_heads,
            query_count,
            head_dim,
        )

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.masked_fill(~visible, torch.finfo(logits.dtype).min)
        weights = F.softmax(masked_logits, dim=-1, dtype=torch.float32)
        weights = weights * visible.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)

    def _full_output_by_kv_head(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        query_positions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        grouped_queries = self._group_queries(module, query_states.to(torch.float32), keys.shape[1])
        logits = torch.matmul(grouped_queries, keys.to(torch.float32).unsqueeze(2).transpose(-1, -2))
        logits = logits / math.sqrt(keys.shape[-1])
        key_positions = torch.arange(keys.shape[2], device=keys.device)
        visible = key_positions.view(1, 1, 1, 1, -1) <= query_positions.view(1, 1, 1, -1, 1)
        weights = self._masked_softmax(logits, visible)
        return torch.matmul(weights, values.to(torch.float32).unsqueeze(2))

    def _build_selected_layout(
        self,
        keep_mask: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Partition selected tokens into fixed and regression slots.

        Selected observation-window tokens remain fixed, matching GRKV's
        treatment of its final query window.  ``grkv_guard`` applies only to
        selected prefix centers.  All tensors are padded across heads; validity
        masks ensure padding contributes neither attention nor regression loss.
        """

        batch_size, num_kv_heads, cache_length = keep_mask.shape
        prefix_length = max(0, cache_length - self.window_size)
        fixed_lists: list[list[torch.Tensor]] = []
        regression_lists: list[list[torch.Tensor]] = []
        max_fixed = 0
        max_regression = 0
        for batch_index in range(batch_size):
            batch_fixed = []
            batch_regression = []
            for head_index in range(num_kv_heads):
                prefix_positions = torch.nonzero(
                    keep_mask[batch_index, head_index, :prefix_length],
                    as_tuple=False,
                ).flatten()
                fixed_budget = int(prefix_positions.numel() * self.grkv_guard)
                if prefix_positions.numel() > 0:
                    ranking = scores[batch_index, head_index, prefix_positions].topk(prefix_positions.numel()).indices
                    ranked_positions = prefix_positions[ranking]
                    guard_positions = ranked_positions[:fixed_budget].sort().values
                    regression_positions = ranked_positions[fixed_budget:].sort().values
                else:
                    guard_positions = prefix_positions
                    regression_positions = prefix_positions
                window_positions = torch.nonzero(
                    keep_mask[batch_index, head_index, prefix_length:],
                    as_tuple=False,
                ).flatten()
                window_positions = window_positions + prefix_length
                fixed_positions = torch.cat([guard_positions, window_positions.sort().values])
                batch_fixed.append(fixed_positions)
                batch_regression.append(regression_positions)
                max_fixed = max(max_fixed, fixed_positions.numel())
                max_regression = max(max_regression, regression_positions.numel())
            fixed_lists.append(batch_fixed)
            regression_lists.append(batch_regression)

        fixed_positions = torch.zeros(
            batch_size,
            num_kv_heads,
            max_fixed,
            dtype=torch.long,
            device=keep_mask.device,
        )
        regression_positions = torch.zeros(
            batch_size,
            num_kv_heads,
            max_regression,
            dtype=torch.long,
            device=keep_mask.device,
        )
        fixed_valid = torch.zeros_like(fixed_positions, dtype=torch.bool)
        regression_valid = torch.zeros_like(regression_positions, dtype=torch.bool)
        for batch_index in range(batch_size):
            for head_index in range(num_kv_heads):
                fixed = fixed_lists[batch_index][head_index]
                regression = regression_lists[batch_index][head_index]
                fixed_positions[batch_index, head_index, : fixed.numel()] = fixed
                regression_positions[batch_index, head_index, : regression.numel()] = regression
                fixed_valid[batch_index, head_index, : fixed.numel()] = True
                regression_valid[batch_index, head_index, : regression.numel()] = True
        return fixed_positions, fixed_valid, regression_positions, regression_valid

    @staticmethod
    def _gather_positions(states: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        head_dim = states.shape[-1]
        gathered = states.gather(2, positions.unsqueeze(-1).expand(*positions.shape, head_dim))
        return gathered * valid.unsqueeze(-1).to(gathered.dtype)

    def _selected_attention_weights(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        query_positions: torch.Tensor,
        fixed_keys: torch.Tensor,
        regression_keys: torch.Tensor,
        fixed_positions: torch.Tensor,
        fixed_valid: torch.Tensor,
        regression_positions: torch.Tensor,
        regression_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grouped_queries = self._group_queries(module, query_states.to(torch.float32), fixed_keys.shape[1])
        kept_keys = torch.cat([fixed_keys, regression_keys], dim=2)
        kept_positions = torch.cat([fixed_positions, regression_positions], dim=2)
        kept_valid = torch.cat([fixed_valid, regression_valid], dim=2)
        logits = torch.matmul(grouped_queries, kept_keys.unsqueeze(2).transpose(-1, -2))
        logits = logits / math.sqrt(kept_keys.shape[-1])
        visible = kept_valid[:, :, None, None, :] & (
            kept_positions[:, :, None, None, :] <= query_positions.view(1, 1, 1, -1, 1)
        )
        weights = self._masked_softmax(logits, visible)
        fixed_count = fixed_keys.shape[2]
        return weights[..., :fixed_count], weights[..., fixed_count:]

    @staticmethod
    def _apply_values(weights: torch.Tensor, values: torch.Tensor, preserve_query_heads: bool) -> torch.Tensor:
        if preserve_query_heads:
            return torch.matmul(weights, values.unsqueeze(2))
        return torch.matmul(weights.mean(dim=2), values)

    @staticmethod
    def _flatten_observations(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten(start_dim=2, end_dim=tensor.ndim - 2)

    @staticmethod
    def _select_lowest_error_update(
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
        next_keys: torch.Tensor,
        next_values: torch.Tensor,
        target_output: torch.Tensor,
        current_output: torch.Tensor,
        value_output: torch.Tensor,
        joint_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduction_dims = tuple(range(2, target_output.ndim))
        errors = torch.stack(
            [
                (current_output - target_output).square().mean(dim=reduction_dims),
                (value_output - target_output).square().mean(dim=reduction_dims),
                (joint_output - target_output).square().mean(dim=reduction_dims),
            ],
            dim=0,
        )
        errors = torch.nan_to_num(errors, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
        best_state = errors.argmin(dim=0)
        use_value_update = (best_state >= 1).unsqueeze(-1).unsqueeze(-1)
        use_key_update = (best_state == 2).unsqueeze(-1).unsqueeze(-1)
        return (
            torch.where(use_key_update, next_keys, current_keys),
            torch.where(use_value_update, next_values, current_values),
        )

    @staticmethod
    def _scatter_regression_states(
        original: torch.Tensor,
        updates: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        result = original.clone()
        for batch_index in range(original.shape[0]):
            for head_index in range(original.shape[1]):
                count = int(valid[batch_index, head_index].sum().item())
                if count:
                    result[batch_index, head_index].index_copy_(
                        0,
                        positions[batch_index, head_index, :count],
                        updates[batch_index, head_index, :count].to(original.dtype),
                    )
        return result.contiguous()

    def _fit_selected_cache(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        keep_mask: torch.Tensor,
        scores: torch.Tensor,
        fit_queries: torch.Tensor,
        fit_positions: torch.Tensor,
        preserve_query_heads: bool,
        select_safe_path: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = self._build_selected_layout(keep_mask, scores)
        fixed_positions, fixed_valid, regression_positions, regression_valid = layout
        if regression_positions.shape[-1] == 0:
            return keys, values

        fixed_keys = self._gather_positions(keys, fixed_positions, fixed_valid).to(torch.float32)
        fixed_values = self._gather_positions(values, fixed_positions, fixed_valid).to(torch.float32)
        regression_keys_initial = self._gather_positions(
            keys,
            regression_positions,
            regression_valid,
        ).to(torch.float32)
        regression_values_initial = self._gather_positions(
            values,
            regression_positions,
            regression_valid,
        ).to(torch.float32)
        full_output = self._full_output_by_kv_head(
            module,
            fit_queries,
            fit_positions,
            keys,
            values,
        )
        if not preserve_query_heads:
            full_output = full_output.mean(dim=2)

        current_keys = regression_keys_initial.clone()
        current_values = regression_values_initial.clone()
        cg_max_iterations = min(16, max(4, fit_queries.shape[-2] // 2))
        cg_tolerance = max(self.grkv_tol, 1e-6 if select_safe_path else 1e-9)

        def attention_splits(key_regression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self._selected_attention_weights(
                module,
                fit_queries,
                fit_positions,
                fixed_keys,
                key_regression,
                fixed_positions,
                fixed_valid,
                regression_positions,
                regression_valid,
            )

        def reconstruction_output(key_regression: torch.Tensor, value_regression: torch.Tensor) -> torch.Tensor:
            fixed_weights, regression_weights = attention_splits(key_regression)
            return self._apply_values(fixed_weights, fixed_values, preserve_query_heads) + self._apply_values(
                regression_weights,
                value_regression,
                preserve_query_heads,
            )

        with torch.inference_mode(False), torch.enable_grad():
            for _ in range(self.grkv_num_steps):
                fixed_weights, regression_weights = attention_splits(current_keys)
                residual_target = full_output - self._apply_values(
                    fixed_weights,
                    fixed_values,
                    preserve_query_heads,
                )
                initial_prediction = self._apply_values(
                    regression_weights,
                    regression_values_initial,
                    preserve_query_heads,
                )
                value_design_weights = regression_weights if preserve_query_heads else regression_weights.mean(dim=2)
                value_design = self._flatten_observations(value_design_weights)
                value_residual = self._flatten_observations(residual_target - initial_prediction)
                observation_count = value_design.shape[-2]
                identity = torch.eye(observation_count, device=keys.device, dtype=torch.float32).view(
                    1,
                    1,
                    observation_count,
                    observation_count,
                )
                gram = torch.matmul(value_design, value_design.transpose(-1, -2))
                dual_solution = torch.linalg.solve(
                    gram + self.grkv_value_lambda * identity,
                    value_residual,
                )
                value_delta = torch.matmul(value_design.transpose(-1, -2), dual_solution)
                next_values = regression_values_initial + value_delta

                def key_objective(key_regression: torch.Tensor) -> torch.Tensor:
                    return reconstruction_output(key_regression, next_values.detach())

                key_base = current_keys.detach().clone()
                key_error = full_output - key_objective(key_base)
                key_shift = key_base - regression_keys_initial
                vector_jacobian_product = torch.func.vjp(key_objective, key_base)[1]

                def transpose_jacobian(vector: torch.Tensor) -> torch.Tensor:
                    return vector_jacobian_product(vector)[0]

                def jacobian(vector: torch.Tensor) -> torch.Tensor:
                    return torch.func.jvp(key_objective, (key_base,), (vector,))[1]

                right_hand_side = key_error + jacobian(key_shift)
                right_hand_side_norm = right_hand_side.reshape(-1).norm().clamp_min(1e-12)

                def conjugate_gradient_operator(vector: torch.Tensor) -> torch.Tensor:
                    return jacobian(transpose_jacobian(vector)) + self.grkv_key_lambda * vector

                dual_key_solution = torch.zeros_like(right_hand_side)
                residual = right_hand_side - conjugate_gradient_operator(dual_key_solution)
                direction = residual.clone()
                residual_norm_squared = torch.dot(residual.reshape(-1), residual.reshape(-1))
                tolerance_squared = (cg_tolerance * right_hand_side_norm) ** 2
                for _ in range(cg_max_iterations):
                    operator_direction = conjugate_gradient_operator(direction)
                    denominator = torch.dot(direction.reshape(-1), operator_direction.reshape(-1)).clamp_min(1e-12)
                    step_size = residual_norm_squared / denominator
                    dual_key_solution = dual_key_solution + step_size * direction
                    residual = residual - step_size * operator_direction
                    new_residual_norm_squared = torch.dot(residual.reshape(-1), residual.reshape(-1))
                    if new_residual_norm_squared <= tolerance_squared:
                        break
                    direction = (
                        residual + (new_residual_norm_squared / residual_norm_squared.clamp_min(1e-12)) * direction
                    )
                    residual_norm_squared = new_residual_norm_squared

                key_delta = transpose_jacobian(dual_key_solution) - key_shift
                next_keys = (key_base + key_delta).detach()
                if select_safe_path:
                    selected_keys, selected_values = self._select_lowest_error_update(
                        current_keys,
                        current_values,
                        next_keys,
                        next_values.detach(),
                        full_output,
                        reconstruction_output(current_keys, current_values).detach(),
                        reconstruction_output(current_keys, next_values).detach(),
                        reconstruction_output(next_keys, next_values).detach(),
                    )
                else:
                    selected_keys, selected_values = next_keys, next_values.detach()

                selected_keys = torch.where(
                    regression_valid.unsqueeze(-1),
                    selected_keys,
                    regression_keys_initial,
                )
                selected_values = torch.where(
                    regression_valid.unsqueeze(-1),
                    selected_values,
                    regression_values_initial,
                )
                max_key_update = (selected_keys - current_keys).abs().max()
                max_value_update = (selected_values - current_values).abs().max()
                current_keys = selected_keys.detach()
                current_values = selected_values.detach()
                if max(max_key_update.item(), max_value_update.item()) <= self.grkv_tol:
                    break

        return (
            self._scatter_regression_states(
                keys,
                current_keys,
                regression_positions,
                regression_valid,
            ),
            self._scatter_regression_states(
                values,
                current_values,
                regression_positions,
                regression_valid,
            ),
        )

    def _fit_anchor(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        keep_mask: torch.Tensor,
        scores: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        preserve_query_heads: bool,
        select_safe_path: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_length = keys.shape[2]
        query_count = min(self.window_size, cache_length)
        query_positions = torch.arange(cache_length - query_count, cache_length, device=keys.device)
        query_states = self._query_states_at_positions(
            module,
            hidden_states,
            position_embeddings,
            query_positions,
        )
        return self._fit_selected_cache(
            module,
            keys,
            values,
            keep_mask,
            scores,
            query_states,
            query_positions,
            preserve_query_heads,
            select_safe_path,
        )

    def _compress_variant(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
        preserve_query_heads: bool,
        select_safe_path: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values
        if module.config._attn_implementation == "eager":
            raise ValueError("DefensiveKV cross-head cache masking does not support eager attention")

        with torch.no_grad():
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            keep_mask = self._global_keep_mask(scores, self.compression_ratio)
        self._set_masked_indices(module, keep_mask)
        return self._fit_anchor(
            module,
            hidden_states,
            keys,
            values,
            keep_mask,
            scores,
            kwargs["position_embeddings"],
            preserve_query_heads,
            select_safe_path,
        )

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compress_variant(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            preserve_query_heads=False,
            select_safe_path=False,
        )


@dataclass
class DefensiveGRKVPurePress(DefensiveGRKVEarliestPress):
    """DefensiveKV selection followed by sum-loss GRKV with safe path choice."""

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compress_variant(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            preserve_query_heads=True,
            select_safe_path=True,
        )


@dataclass
class DefensiveGRKVBehaviorCVPress(DefensiveGRKVPurePress):
    """DefensiveKV + sum-loss GRKV, gated by two-fold Behavior-CV."""

    behavior_probe_windows: int = 9
    behavior_probes_per_window: int = 2
    behavior_relative_margin: float = 0.0
    behavior_cv_diagnostics: dict[str, float | int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        super().__post_init__()
        if self.behavior_probe_windows < 3 or self.behavior_probe_windows % 3 != 0:
            raise ValueError("behavior_probe_windows must be a positive multiple of 3")
        if self.behavior_probes_per_window <= 0:
            raise ValueError("behavior_probes_per_window must be positive")
        if self.behavior_relative_margin < 0:
            raise ValueError("behavior_relative_margin must be non-negative")
        self.reset_behavior_cv_diagnostics()

    def reset_behavior_cv_diagnostics(self) -> None:
        self.behavior_cv_diagnostics = {
            "total_layers": 0,
            "accepted_layers": 0,
            "rejected_layers": 0,
            "short_context_fallback_layers": 0,
            "nonfinite_fallback_layers": 0,
            "fold_1_improved_layers": 0,
            "fold_2_improved_layers": 0,
            "anchor_loss_sum": 0.0,
            "candidate_loss_sum": 0.0,
        }

    def _source_probe_folds(
        self,
        prefix_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        required = self.behavior_probe_windows * self.behavior_probes_per_window
        if prefix_length < required:
            return None
        latest_start = prefix_length - self.behavior_probes_per_window
        starts = (
            torch.linspace(
                0,
                latest_start,
                steps=self.behavior_probe_windows,
                device=device,
                dtype=torch.float64,
            )
            .round()
            .to(torch.long)
        )
        offsets = torch.arange(self.behavior_probes_per_window, device=device)
        windows = starts[:, None] + offsets[None, :]
        if torch.unique(windows).numel() != required:
            return None
        return windows[0::3].reshape(-1), windows[1::3].reshape(-1), windows[2::3].reshape(-1)

    def _virtual_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        source_positions: torch.Tensor,
        virtual_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        virtual_positions = torch.full_like(source_positions, virtual_position)
        queries = self._query_states_at_positions(
            module,
            hidden_states,
            position_embeddings,
            source_positions,
            virtual_positions,
        )
        return queries, virtual_positions

    def _projected_output(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        query_positions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        keep_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if keep_mask is None:
            output = self._full_output_by_kv_head(
                module,
                query_states,
                query_positions,
                keys,
                values,
            )
        else:
            selected_positions, selected_valid, regression_positions, regression_valid = self._build_selected_layout(
                keep_mask,
                torch.zeros_like(keep_mask, dtype=torch.float32),
            )
            # ``grkv_guard=1`` is not assumed here, so concatenate the two layout
            # partitions back into one selected set before computing behavior.
            selected_keys = self._gather_positions(keys, selected_positions, selected_valid).to(torch.float32)
            selected_values = self._gather_positions(values, selected_positions, selected_valid).to(torch.float32)
            regression_keys = self._gather_positions(keys, regression_positions, regression_valid).to(torch.float32)
            regression_values = self._gather_positions(values, regression_positions, regression_valid).to(torch.float32)
            fixed_weights, regression_weights = self._selected_attention_weights(
                module,
                query_states,
                query_positions,
                selected_keys,
                regression_keys,
                selected_positions,
                selected_valid,
                regression_positions,
                regression_valid,
            )
            output = self._apply_values(fixed_weights, selected_values, True) + self._apply_values(
                regression_weights,
                regression_values,
                True,
            )

        batch_size, num_kv_heads, num_groups, query_count, head_dim = output.shape
        concatenated = output.permute(0, 3, 1, 2, 4).reshape(
            batch_size,
            query_count,
            num_kv_heads * num_groups * head_dim,
        )
        bias = module.o_proj.bias
        return F.linear(
            concatenated,
            module.o_proj.weight.to(torch.float32),
            bias.to(torch.float32) if bias is not None else None,
        )

    def _behavior_loss(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        query_positions: torch.Tensor,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        candidate_keys: torch.Tensor,
        candidate_values: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self._projected_output(
            module,
            query_states,
            query_positions,
            full_keys,
            full_values,
            None,
        )
        prediction = self._projected_output(
            module,
            query_states,
            query_positions,
            candidate_keys,
            candidate_values,
            keep_mask,
        )
        target_energy = target.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
        return (prediction - target).square().mean() / target_energy

    def _candidate_passes(
        self,
        anchor_losses: torch.Tensor,
        candidate_losses: torch.Tensor,
    ) -> tuple[bool, torch.Tensor]:
        finite = torch.isfinite(anchor_losses) & torch.isfinite(candidate_losses)
        denominator = anchor_losses.clamp_min(torch.finfo(anchor_losses.dtype).tiny)
        relative_gains = (anchor_losses - candidate_losses) / denominator
        passes = finite & (candidate_losses < anchor_losses) & (relative_gains > self.behavior_relative_margin)
        return bool(passes.all().item()), relative_gains

    def _record_decision(
        self,
        accepted: bool,
        anchor_losses: torch.Tensor,
        candidate_losses: torch.Tensor,
        relative_gains: torch.Tensor,
    ) -> None:
        self.behavior_cv_diagnostics["total_layers"] += 1
        decision = "accepted_layers" if accepted else "rejected_layers"
        self.behavior_cv_diagnostics[decision] += 1
        finite = torch.isfinite(anchor_losses).all() and torch.isfinite(candidate_losses).all()
        if not bool(finite.item() if isinstance(finite, torch.Tensor) else finite):
            self.behavior_cv_diagnostics["nonfinite_fallback_layers"] += 1
        for index in range(2):
            if bool(torch.isfinite(relative_gains[index]).item()) and relative_gains[index].item() > 0:
                self.behavior_cv_diagnostics[f"fold_{index + 1}_improved_layers"] += 1
        self.behavior_cv_diagnostics["anchor_loss_sum"] += float(
            torch.nan_to_num(anchor_losses.detach(), nan=0.0, posinf=0.0, neginf=0.0).sum().cpu()
        )
        self.behavior_cv_diagnostics["candidate_loss_sum"] += float(
            torch.nan_to_num(candidate_losses.detach(), nan=0.0, posinf=0.0, neginf=0.0).sum().cpu()
        )

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values
        if module.config._attn_implementation == "eager":
            raise ValueError("DefensiveKV cross-head cache masking does not support eager attention")

        with torch.no_grad():
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            keep_mask = self._global_keep_mask(scores, self.compression_ratio)
        self._set_masked_indices(module, keep_mask)
        position_embeddings = kwargs["position_embeddings"]
        anchor_keys, anchor_values = self._fit_anchor(
            module,
            hidden_states,
            keys,
            values,
            keep_mask,
            scores,
            position_embeddings,
            preserve_query_heads=True,
            select_safe_path=True,
        )

        cache_length = keys.shape[2]
        prefix_length = cache_length - self.window_size
        folds = self._source_probe_folds(prefix_length, keys.device)
        if folds is None:
            self.behavior_cv_diagnostics["total_layers"] += 1
            self.behavior_cv_diagnostics["rejected_layers"] += 1
            self.behavior_cv_diagnostics["short_context_fallback_layers"] += 1
            return anchor_keys, anchor_values

        fit_sources, validation_1_sources, validation_2_sources = folds
        virtual_position = cache_length - 1
        fit_queries, fit_positions = self._virtual_queries(
            module,
            hidden_states,
            position_embeddings,
            fit_sources,
            virtual_position,
        )
        candidate_keys, candidate_values = self._fit_selected_cache(
            module,
            keys,
            values,
            keep_mask,
            scores,
            fit_queries,
            fit_positions,
            preserve_query_heads=True,
            select_safe_path=True,
        )

        anchor_losses = []
        candidate_losses = []
        for validation_sources in (validation_1_sources, validation_2_sources):
            validation_queries, validation_positions = self._virtual_queries(
                module,
                hidden_states,
                position_embeddings,
                validation_sources,
                virtual_position,
            )
            anchor_losses.append(
                self._behavior_loss(
                    module,
                    validation_queries,
                    validation_positions,
                    keys,
                    values,
                    anchor_keys,
                    anchor_values,
                    keep_mask,
                )
            )
            candidate_losses.append(
                self._behavior_loss(
                    module,
                    validation_queries,
                    validation_positions,
                    keys,
                    values,
                    candidate_keys,
                    candidate_values,
                    keep_mask,
                )
            )
        anchor_loss_tensor = torch.stack(anchor_losses)
        candidate_loss_tensor = torch.stack(candidate_losses)
        accepted, relative_gains = self._candidate_passes(anchor_loss_tensor, candidate_loss_tensor)
        self._record_decision(accepted, anchor_loss_tensor, candidate_loss_tensor, relative_gains)
        if accepted:
            return candidate_keys, candidate_values
        return anchor_keys, anchor_values
