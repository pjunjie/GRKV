# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from experiments.critical_adakv_direct_grkv.wide_query_policy import WideQueryConfig, WideQueryPress
from experiments.critical_adakv_direct_grkv.wide_query_worker import NativeBoundaryPress

FAMILIES = {
    "window_q16": ("window_q16_compat", 16),
    "tail64_q64": ("tail64", 64),
    "head_tail_q128": ("head_tail", 128),
    "context_stratified_q32": ("context_stratified", 32),
    "tail_global_q64": ("tail_global", 64),
}

CONFIGS = {
    family + "_" + mode: WideQueryConfig(query_sampling=sampling, query_count=count, key_relative_cap=cap)
    for family, (sampling, count) in FAMILIES.items()
    for mode, cap in (("v", 0.0), ("kv", 0.1))
}


def make_press(method):
    params = dict(compression_ratio=0.9, window_size=32, kernel_size=5, grkv_guard=0.1)
    return (
        NativeBoundaryPress(**params) if method == "critical" else WideQueryPress(**params, fit_config=CONFIGS[method])
    )
