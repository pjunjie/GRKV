# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public constructors for GRKV Default and the historical baseline."""

from grkv.press import CriticalAdaKVBaselinePress, GRKVDefaultPress, MistralGRKVDefaultPress

__all__ = ["GRKVDefaultPress", "MistralGRKVDefaultPress", "make_press"]


def make_press(model, budget, method="default"):
    if model not in ("llama", "mistral"):
        raise ValueError("GRKV Default supports llama and mistral")
    if budget not in (10, 20):
        raise ValueError("KV retention must be 10 or 20 percent")
    if method not in ("default", "baseline"):
        raise ValueError("Method must be default or baseline")
    guard = 0.1 if method == "baseline" or model == "llama" else 0.0
    cls = (
        CriticalAdaKVBaselinePress
        if method == "baseline"
        else (GRKVDefaultPress if model == "llama" else MistralGRKVDefaultPress)
    )
    return cls(compression_ratio=0.9 if budget == 10 else 0.8, grkv_guard=guard)
