# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit frozen K1 constructors, separate from other GRKV variants."""

from experiments.critical_adakv_direct_grkv.tune50_protocol import make_press as llama_press
from experiments.critical_adakv_direct_grkv.wide_query_policy import WideQueryPress as LlamaK1Press
from experiments.mistral_k1_tuning.policy import MistralS1Press as MistralK1Press

__all__ = ["LlamaK1Press", "MistralK1Press", "make_press"]


def make_press(model, budget, method="candidate"):
    guard = {"llama": 0.1, "mistral": 0.0}[model]
    config = dict(ridge=0.01, key_relative_cap=0.025, relative_cap=0.1, grkv_guard=guard)
    base = llama_press("critical" if method == "baseline" else "K1", config, budget)
    if model == "llama" or method == "baseline":
        return base
    return MistralK1Press(
        compression_ratio=base.compression_ratio,
        window_size=base.window_size,
        kernel_size=base.kernel_size,
        grkv_guard=base.grkv_guard,
        critical_alpha_safeguard=base.critical_alpha_safeguard,
        critical_first_stage_ratio=base.critical_first_stage_ratio,
        critical_epsilon=base.critical_epsilon,
        fit_config=base.fit_config,
    )
