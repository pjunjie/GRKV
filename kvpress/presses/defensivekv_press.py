# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""DefensiveKV with adaptive budget allocation across KV heads."""

import math
from dataclasses import dataclass
from decimal import Decimal

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.ops.vw_norm import vw_l1norm
from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


@dataclass
class DefensiveKVPress(BasePress):
    """DefensiveKV's standard cross-head cache selection method.

    The implementation preserves the official method's defensive max
    aggregation, mean floor, output-projection value norm, 90% attention-mass
    first stage, and one shared top-k budget across all KV heads in a layer.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of KV positions to mask across all heads.
    window_size : int, default=32
        Number of final context queries used as the observation window.
    kernel_size : int, default=5
        Width of the average-pooling smoother over prefix positions.
    """

    compression_ratio: float = 0.0
    window_size: int = 32
    kernel_size: int = 5

    def __post_init__(self):
        if not 0 <= self.compression_ratio < 1:
            raise ValueError("compression_ratio must be in [0, 1)")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

    def __str__(self):
        return f"DefensiveKVPress={self.compression_ratio}_win={self.window_size}" f"_kernel={self.kernel_size}"

    @staticmethod
    def get_num_kept(cache_length: int, num_key_value_heads: int, compression_ratio: float) -> int:
        """Return the exact integer global budget using decimal ratio semantics."""

        keep_ratio = Decimal(1) - Decimal(str(compression_ratio))
        return int(Decimal(cache_length * num_key_value_heads) * keep_ratio)

    @staticmethod
    def compute_window_attention(
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        window_size: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        """Compute final-window attention to prefix keys and its reference bias."""

        _, _, k_len, _ = keys.shape
        head_dim = module.head_dim
        num_key_value_groups = module.config.num_attention_heads // module.config.num_key_value_heads

        query_states = get_prerope_query_states(module, hidden_states[:, -window_size:])
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        key_states = repeat_kv(keys, num_key_value_groups)
        attention = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        causal_mask = torch.full_like(attention, float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=k_len - window_size + 1)
        attention = attention + causal_mask
        attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(query_states.dtype)
        prefix_attention = attention[..., :-window_size]

        # Preserve the statistic used by the released DefensiveKV implementation:
        # mass on the final window_size positions of the scored prefix.
        window_bias = prefix_attention[..., -window_size:].sum(dim=-1).mean().item()
        return prefix_attention, window_bias

    @staticmethod
    def _projected_value_norm(values: torch.Tensor, module: nn.Module) -> torch.Tensor:
        batch_size, num_key_value_heads, prefix_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        output_weight = module.o_proj.weight.transpose(0, 1).view(
            module.config.num_attention_heads,
            module.head_dim,
            module.config.hidden_size,
        )
        repeated_values = repeat_kv(values, num_key_value_groups)
        projected_norm = vw_l1norm(repeated_values, output_weight)
        return projected_norm.view(
            batch_size,
            num_key_value_heads,
            num_key_value_groups,
            prefix_len,
        ).mean(dim=2)

    def _apply_value_norm_and_first_stage(
        self,
        values: torch.Tensor,
        module: nn.Module,
        scores: torch.Tensor,
        window_bias: float,
        mean_attention: torch.Tensor,
    ) -> torch.Tensor:
        projected_norm = self._projected_value_norm(values, module)
        # The official cross-head implementation normalizes each token position
        # across KV heads before multiplying the defensive score.
        projected_norm = projected_norm / projected_norm.sum(dim=-2, keepdim=True)
        scores = scores * projected_norm

        normalized_attention = mean_attention / mean_attention.sum(dim=-1, keepdim=True)
        sorted_attention, sorted_indices = torch.sort(normalized_attention, dim=-1, descending=True)
        cumulative_attention = torch.cumsum(sorted_attention, dim=-1)
        threshold = 0.9 - window_bias
        first_crossing = torch.argmax((cumulative_attention >= threshold).to(torch.int32), dim=-1)
        # Match the released implementation exactly: ``values`` already excludes
        # the observation window, and that prefix length is used before the
        # additional window-size subtraction in the stage-1 cap.
        prefix_len = values.shape[-2]
        max_first_stage = int(prefix_len * (1 - self.compression_ratio) - self.window_size)
        first_crossing.clamp_(max=max_first_stage)

        ranks = torch.arange(scores.shape[-1], device=scores.device)
        selected_by_rank = ranks.view(1, 1, -1) < first_crossing.unsqueeze(-1)
        first_stage_mask = torch.zeros_like(selected_by_rank)
        first_stage_mask.scatter_(-1, sorted_indices, selected_by_rank)
        return torch.where(first_stage_mask, scores.amax(), scores)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor:
        batch_size, num_key_value_heads, k_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        if hidden_states.shape[1] <= self.window_size:
            raise ValueError(f"Query length {hidden_states.shape[1]} must exceed window_size {self.window_size}")

        if attentions is None:
            attention, window_bias = self.compute_window_attention(
                module,
                hidden_states,
                keys,
                self.window_size,
                kwargs["position_embeddings"],
            )
        else:
            final_window_attention = attentions[..., -self.window_size :, :]
            window_bias = final_window_attention[..., -self.window_size :].sum(dim=-1).mean().item()
            attention = final_window_attention[..., : -self.window_size]

        prefix_len = k_len - self.window_size
        grouped_attention = attention.view(
            batch_size,
            num_key_value_heads,
            num_key_value_groups,
            self.window_size,
            prefix_len,
        )
        mean_attention = grouped_attention.mean(dim=2).mean(dim=-2)
        attention_mass = grouped_attention.sum(dim=-1, keepdim=True)

        pooled_scores = F.avg_pool1d(
            grouped_attention.view(
                batch_size * num_key_value_heads * num_key_value_groups,
                self.window_size,
                prefix_len,
            ),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).view_as(grouped_attention)
        pooled_scores = pooled_scores / pooled_scores.sum(dim=-1, keepdim=True)
        pooled_scores = pooled_scores * attention_mass

        # Defensive aggregation: keep evidence from any GQA group/query and
        # prevent any prefix score from falling below its head-wise mean.
        scores = pooled_scores.amax(dim=2).amax(dim=-2)
        scores = scores.clamp(min=scores.mean(dim=-1, keepdim=True))
        scores = self._apply_value_norm_and_first_stage(
            values[..., :prefix_len, :],
            module,
            scores,
            window_bias,
            mean_attention,
        )

        return F.pad(scores, (0, self.window_size), value=scores.amax().item())

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
            raise ValueError("DefensiveKV's cross-head cache masking does not support eager attention")

        with torch.no_grad():
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        batch_size, num_key_value_heads, k_len = scores.shape
        total_positions = num_key_value_heads * k_len
        n_kept = self.get_num_kept(k_len, num_key_value_heads, self.compression_ratio)
        kept_indices = scores.reshape(batch_size, total_positions).topk(n_kept, dim=-1).indices
        keep_mask = torch.zeros(
            (batch_size, total_positions),
            dtype=torch.bool,
            device=scores.device,
        )
        keep_mask.scatter_(1, kept_indices, True)
        batch_indices, flat_indices = torch.nonzero(~keep_mask, as_tuple=True)
        module.masked_key_indices = (  # type: ignore[assignment]
            batch_indices,
            flat_indices // k_len,
            flat_indices % k_len,
        )
        return keys, values


# The official release uses this longer class name.  Keep it as a public alias
# so configurations and downstream imports can be transferred without renaming.
EfficientDefensiveKVPress = DefensiveKVPress
