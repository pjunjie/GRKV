# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from math import isfinite

from experiments.critical_adakv_direct_grkv.historical_query_worker import make_press as original
from experiments.critical_adakv_direct_grkv.key_policy import KeyPolicyConfig
from experiments.critical_adakv_direct_grkv.wide_query_policy import WideQueryPress


@dataclass(frozen=True)
class TuningConfig(KeyPolicyConfig):
    query_sampling: str = "context_stratified"
    query_count: int = 32

    def __post_init__(self):
        super().__post_init__()
        assert self.query_sampling == "context_stratified" and self.query_count == 32
        assert self.max_iterations == 16 and self.tolerance == 1e-6 and self.precision == "float32"
        assert isfinite(self.ridge) and self.ridge > 0
        assert 0 < self.key_relative_cap <= self.relative_cap <= 0.2


def make_press(method, config, budget):
    assert budget in (10, 20)
    if method in ("critical", "original"):
        press = original("critical" if method == "critical" else "context_stratified_q32_kv")
        press.compression_ratio = 0.9 if budget == 10 else 0.8
        return press
    values = dict(config)
    guard = values.pop("grkv_guard")
    assert 0 <= guard <= 1
    return WideQueryPress(
        compression_ratio=0.9 if budget == 10 else 0.8,
        window_size=32,
        kernel_size=5,
        grkv_guard=guard,
        fit_config=TuningConfig(**values),
    )


def replay_layer(layer):
    """Exclude only measured wall times; retain all numerical and layout checks."""
    timing = {"selection_host_seconds", "fit_host_seconds", "target_host_seconds"}
    return {k: v for k, v in layer.items() if k not in timing}
