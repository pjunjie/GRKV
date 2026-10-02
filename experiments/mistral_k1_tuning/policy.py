# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Mistral-qualified adapter for the frozen Q32 K1 selection and fit policy."""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

from experiments.critical_adakv_direct_grkv.key_policy import KeyPolicyConfig
from experiments.critical_adakv_direct_grkv.wide_query_policy import WideQueryConfig, WideQueryPress
from experiments.critical_adakv_grkv.press import CriticalAdaGRKVPress


@dataclass(frozen=True)
class S1Config(WideQueryConfig):
    query_sampling: str = "context_stratified"
    query_count: int = 32
    key_relative_cap: float = 0.025

    def __post_init__(self):
        KeyPolicyConfig.__post_init__(self)
        if self.query_sampling != "context_stratified" or self.query_count != 32:
            raise ValueError("S1 requires frozen context-stratified Q32")
        if self.relative_cap != 0.1 or self.key_relative_cap != 0.025:
            raise ValueError("S1 requires frozen K1 cap values")
        if self.objective != "pre" or self.steps != 1 or self.max_iterations != 16:
            raise ValueError("S1 requires the one-round pre-loss solver")
        if self.tolerance != 1e-6 or self.precision != "float32":
            raise ValueError("S1 requires the qualified FP32 solver controls")


@dataclass
class MistralS1Press(WideQueryPress):
    fit_config: S1Config = field(default_factory=S1Config)
    validate_queries: bool = False

    def post_init_from_model(self, model):
        if model.config.model_type != "mistral":
            raise ValueError("This Q capture is qualified only for Mistral")
        cfg = model.config
        if (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads) != (32, 32, 8):
            raise ValueError("Unexpected Mistral GQA geometry")
        if cfg.head_dim != 128 or cfg.sliding_window is not None:
            raise ValueError("Unexpected Mistral head dimension or sliding window")
        if cfg._attn_implementation != "flash_attention_2":
            raise ValueError("S1 requires the frozen FlashAttention2 protocol")
        # DirectPress's private hook uses q_proj before RoPE. The installed
        # Mistral forward has the same projection/reshape/rotation sequence;
        # bypass only DirectPress's Llama-only model-type assertion.
        CriticalAdaGRKVPress.post_init_from_model(self, model)
        self._pending.clear()

    @torch.no_grad()
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        capture_check = None
        if self.validate_queries and module.layer_idx == 0 and module.layer_idx in self._pending:
            raw, sources = self._pending[module.layer_idx]
            expected = F.linear(hidden_states.index_select(1, sources), module.q_proj.weight, module.q_proj.bias)
            raw_error = (raw.float() - expected.float()).abs().max().item()
            heads, dim = module.config.num_attention_heads, module.head_dim
            queries = raw.view(1, len(sources), heads, dim).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            rotated, _ = apply_rotary_pos_emb(queries, queries, cos[:, -1:, :], sin[:, -1:, :])
            virtual, _ = self._virtual_queries(
                module, hidden_states, kwargs["position_embeddings"], sources, keys.shape[2] - 1
            )
            rope_error = (rotated.float() - virtual.float()).abs().max().item()
            capture_check = dict(raw_q_max_abs_error=raw_error, virtual_rope_max_abs_error=rope_error)
        result = super().compress(module, hidden_states, keys, values, attentions, kwargs)
        if capture_check is not None:
            self.layer_records[-1]["mistral_q_check"] = capture_check
        return result


def make_press(ridge, budget):
    if budget not in (10, 20):
        raise ValueError("S1 budget must be 10 or 20 percent")
    return MistralS1Press(
        compression_ratio=0.9 if budget == 10 else 0.8,
        window_size=32,
        kernel_size=5,
        grkv_guard=0.1,
        critical_alpha_safeguard=0.2,
        critical_first_stage_ratio=0.5,
        critical_epsilon=1e-4,
        fit_config=S1Config(ridge=float(ridge)),
    )
