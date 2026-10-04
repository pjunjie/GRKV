# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRKV Default and its historical Critical-AdaKV baseline."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import ClassVar

import torch
from transformers.models.llama.modeling_llama import rotate_half

from grkv.cache import Block
from grkv.config import DefaultConfig
from grkv.io import tensor_sha
from grkv.operators import AttentionMap, Observations
from grkv.queries import query_metadata, source_positions
from grkv.solver import fit_default
from kvpress.presses.base_press import BasePress
from kvpress.presses.criticalkv_press import CriticalAdaKVPress
from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class CaptureSnapKV(SnapKVPress):
    """Keep an unmodified score copy before CriticalAdaKV's in-place selection."""

    def score(self, *args, **kwargs):
        scores = super().score(*args, **kwargs)
        self.raw_scores = scores.clone()
        return scores


@dataclass
class SelectionPress(BasePress):
    compression_ratio: float = 0.9
    window_size: int = 32
    kernel_size: int = 5
    grkv_guard: float = 0.1
    critical_alpha_safeguard: float = 0.2
    critical_epsilon: float = 1e-4
    critical_first_stage_ratio: float = 0.5
    layer_records: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.compression_ratio not in (0.8, 0.9):
            raise ValueError("GRKV Default supports 10% and 20% KV retention")
        if (self.window_size, self.kernel_size) != (32, 5) or self.grkv_guard not in (0.0, 0.1):
            raise ValueError("GRKV Default requires its frozen selection settings")
        if (self.critical_alpha_safeguard, self.critical_epsilon, self.critical_first_stage_ratio) != (0.2, 1e-4, 0.5):
            raise ValueError("Historical Critical-AdaKV selection settings differ")

    def post_init_from_model(self, model):
        self.layer_records = []

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        if keys.shape[0] != 1:
            raise ValueError("this experiment uses the original CriticalAdaKV batch-size-one path")
        scorer = CaptureSnapKV(
            compression_ratio=self.compression_ratio,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
        )
        selector = CriticalAdaKVPress(
            scorer,
            alpha_safeguard=self.critical_alpha_safeguard,
            epsilon=self.critical_epsilon,
            first_stage_ratio=self.critical_first_stage_ratio,
        )
        selector.compress(module, hidden_states, keys, values, attentions, kwargs)
        keep = torch.ones(keys.shape[:3], device=keys.device, dtype=torch.bool)
        keep[module.masked_key_indices] = False
        self._critical_keep = keep
        self._critical_scores = scorer.raw_scores
        return scorer.raw_scores

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


@dataclass
class CriticalAdaKVBaselinePress(SelectionPress):
    """Historical baseline selection without regression."""

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if keys.shape[2] <= self.window_size:
            tokens = keys.shape[2]
            module.masked_key_indices = tuple(torch.empty(0, dtype=torch.long, device=keys.device) for _ in range(3))
            self.layer_records.append(
                dict(
                    layer=int(module.layer_idx),
                    context_tokens=tokens,
                    returned_path="native_short_untouched",
                    compression_applied=False,
                    native_short_path_supported=False,
                    kept=keys.shape[1] * tokens,
                    kept_per_head=[tokens] * keys.shape[1],
                    keep_sha256=tensor_sha(torch.ones(keys.shape[:3], dtype=torch.bool, device=keys.device)),
                    regression_slots=0,
                    fixed=keys.shape[1] * tokens,
                    changed=0,
                    fixed_unchanged=True,
                    unchosen_unchanged=True,
                )
            )
            return keys, values
        with torch.no_grad():
            self.score(module, hidden_states, keys, values, attentions, kwargs)
        keep = self._critical_keep
        fixed, fixed_valid, regression, regression_valid = self._build_selected_layout(keep, self._critical_scores)
        expected = keys.shape[1] * int(keys.shape[2] * (1 - self.compression_ratio))
        if int(keep.sum()) != expected:
            raise RuntimeError("CriticalAdaKV budget changed")
        self.layer_records.append(
            dict(
                layer=int(module.layer_idx),
                returned_path="critical_adakv",
                context_tokens=keys.shape[2],
                kept=int(keep.sum()),
                kept_per_head=keep.sum(-1).flatten().tolist(),
                keep_sha256=tensor_sha(keep),
                fixed=int(fixed_valid.sum()),
                regression_slots=int(regression_valid.sum()),
                changed=0,
                unchosen_unchanged=True,
                fixed_unchanged=True,
            )
        )
        return keys, values


def device_pack(keys, values, keep, layout):
    fp, fv, rp, rv = layout
    positions = torch.cat((fp, rp), -1)
    valid = torch.cat((fv, rv), -1)
    mutable = torch.cat((torch.zeros_like(fv), rv), -1)
    indices = positions[..., None].expand(-1, -1, -1, keys.shape[-1])
    return Block(
        keys.gather(2, indices) * valid[..., None],
        values.gather(2, indices) * valid[..., None],
        positions,
        valid,
        mutable,
    )


@dataclass
class GRKVDefaultPress(SelectionPress):
    """GRKV Default with Llama context-prefill query capture."""

    fit_config: DefaultConfig = field(default_factory=DefaultConfig)
    _default_guard: ClassVar[float] = 0.1
    audit: bool = True
    query_aware: bool = False
    _pending: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.fit_config, DefaultConfig) or self.grkv_guard != self._default_guard:
            raise ValueError("GRKV Default requires the frozen regression configuration and model-specific guard")
        if not self.audit or self.query_aware:
            raise ValueError("GRKV Default requires audited context-only fitting")

    def post_init_from_model(self, model):
        super().post_init_from_model(model)
        self._pending.clear()
        if model.config.model_type != "llama":
            raise ValueError("This query capture requires Llama")

    @contextmanager
    def __call__(self, model):
        hooks = []

        def capture(index, module, inputs, output):
            # The hook retains the sampled raw query vectors. It never runs q_proj again.
            if output.shape[1] <= self.window_size:
                return
            folds = self._source_probe_folds(output.shape[1] - self.window_size, output.device)
            if folds is not None:
                self._pending[index] = (output.index_select(1, folds[0]).detach(), folds[0])

        try:
            for layer in model.model.layers:
                index = layer.self_attn.layer_idx
                hooks.append(
                    layer.self_attn.q_proj.register_forward_hook(lambda m, i, o, index=index: capture(index, m, i, o))
                )
            with super().__call__(model):
                yield
        finally:
            for hook in hooks:
                hook.remove()
            self._pending.clear()

    def _source_probe_folds(self, prefix_length, device):
        positions = source_positions(self.fit_config, prefix_length + self.window_size, device)
        empty = torch.empty(0, device=device, dtype=torch.long)
        return positions, empty, empty

    @torch.no_grad()
    def _compress_selected(self, module, hidden_states, keys, values, attentions, kwargs):
        begin = perf_counter()
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        keep = self._critical_keep
        layout = self._build_selected_layout(keep, scores)
        anchor = device_pack(keys, values, keep, layout)
        selected_at = perf_counter()
        captured = self._pending.pop(module.layer_idx, None)
        record = dict(
            layer=int(module.layer_idx),
            context_tokens=keys.shape[2],
            kept=int(keep.sum()),
            kept_per_head=keep.sum(-1).flatten().tolist(),
            fixed=int((anchor.valid & ~anchor.regression).sum()),
            regression_slots=int(anchor.regression.sum()),
            query_source="actual_context_prefill_q_proj",
            query_aware=False,
            sink_tokens=0,
        )
        if self.audit:
            record["keep_sha256"] = tensor_sha(keep)
        if captured is None or not bool(anchor.regression.any()):
            record["skip"] = "no_history_queries_or_mutable_slots"
            self.layer_records.append(record)
            return keys, values
        raw, sources = captured
        heads, dim = keys.shape[1], keys.shape[-1]
        q = raw.view(1, len(sources), -1, dim).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q = q * cos[:, -1:, None, :].transpose(1, 2) + rotate_half(q) * sin[:, -1:, None, :].transpose(1, 2)
        q = q[0].reshape(heads, -1, len(sources), dim).float()
        positions = torch.full_like(sources, keys.shape[2] - 1)
        obs = Observations(
            q,
            positions,
            torch.empty((len(sources), q.shape[0] * q.shape[1] * dim), device=q.device),
            torch.full(q.shape[:3], -torch.inf, device=q.device),
            torch.zeros_like(q),
            torch.full((len(sources),), 6 / len(sources), device=q.device),
        )
        full_positions = torch.arange(keys.shape[2], device=q.device).expand(heads, -1)
        full_valid = torch.ones_like(full_positions, dtype=torch.bool)
        teacher = AttentionMap(obs, full_positions, full_valid, torch.zeros_like(full_valid))
        obs.target = teacher.state(keys[0].float(), values[0].float())[0]
        del teacher
        target_at = perf_counter()
        candidate, fit_record = fit_default(anchor, obs, module.o_proj.weight.detach(), self.fit_config)
        for head in range(heads):
            valid = candidate.regression[0, head]
            p = candidate.positions[0, head, valid]
            keys[0, head, p] = candidate.key[0, head, valid]
            values[0, head, p] = candidate.value[0, head, valid]
        record.update(
            fit=fit_record,
            source_positions=sources.tolist(),
            virtual_position=keys.shape[2] - 1,
            selection_host_seconds=selected_at - begin,
            target_host_seconds=target_at - selected_at,
            fit_host_seconds=perf_counter() - target_at,
        )
        self.layer_records.append(record)
        return keys, values

    @torch.no_grad()
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        tokens = keys.shape[2]
        if tokens <= self.window_size:
            # SnapKV scoring is undefined here. Preserve the entire protected input,
            # and disclose this boundary fallback rather than claiming a 90% budget.
            self._pending.pop(module.layer_idx, None)
            module.masked_key_indices = tuple(torch.empty(0, dtype=torch.long, device=keys.device) for _ in range(3))
            self.layer_records.append(
                dict(
                    layer=int(module.layer_idx),
                    **query_metadata(self.fit_config, tokens, skip_reason="context_not_larger_than_protection_window"),
                    compression_applied=False,
                    native_short_path_supported=False,
                    kept=keys.shape[1] * tokens,
                    kept_per_head=[tokens] * keys.shape[1],
                    regression_slots=0,
                    fixed=keys.shape[1] * tokens,
                    changed=0,
                    fixed_unchanged=True,
                    unchosen_unchanged=True,
                )
            )
            return keys, values
        output = self._compress_selected(module, hidden_states, keys, values, attentions, kwargs)
        record = self.layer_records[-1]
        fitted = "fit" in record
        metadata = query_metadata(
            self.fit_config,
            tokens,
            used_fit_q=min(tokens, self.fit_config.query_count) if fitted else 0,
            skip_reason=None if fitted else record.get("skip", "no_mutable_slots"),
        )
        if fitted:
            assert record["source_positions"] == metadata["source_positions"]
            assert abs(record["fit"]["total_query_weight"] - 6.0) < 1e-5
        record.update(metadata, compression_applied=True, native_short_path_supported=True)
        return output


@dataclass
class MistralGRKVDefaultPress(GRKVDefaultPress):
    """Mistral adapter for the same GRKV Default regression."""

    grkv_guard: float = 0.0
    _default_guard: ClassVar[float] = 0.0

    def post_init_from_model(self, model):
        if model.config.model_type != "mistral":
            raise ValueError("This Q capture is qualified only for Mistral")
        cfg = model.config
        if (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads) != (32, 32, 8):
            raise ValueError("Unexpected Mistral GQA geometry")
        if cfg.head_dim != 128 or cfg.sliding_window is not None:
            raise ValueError("Unexpected Mistral head dimension or sliding window")
        if cfg._attn_implementation != "flash_attention_2":
            raise ValueError("GRKV Default requires the frozen FlashAttention2 protocol")
        # GRKVDefaultPress's private hook uses q_proj before RoPE. The installed
        # Mistral forward has the same projection/reshape/rotation sequence;
        # bypass only GRKVDefaultPress's Llama-only model-type assertion.
        SelectionPress.post_init_from_model(self, model)
        self._pending.clear()
