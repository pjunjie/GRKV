# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single regression configuration used by the reproduced results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultConfig:
    objective: str = "pre"
    steps: int = 1
    ridge: float = 0.01
    relative_cap: float = 0.1
    max_iterations: int = 16
    tolerance: float = 1e-6
    precision: str = "float32"
    key_relative_cap: float = 0.025
    query_sampling: str = "context_stratified"
    query_count: int = 32

    def __post_init__(self):
        expected = dict(
            objective="pre",
            steps=1,
            ridge=0.01,
            relative_cap=0.1,
            max_iterations=16,
            tolerance=1e-6,
            precision="float32",
            key_relative_cap=0.025,
            query_sampling="context_stratified",
            query_count=32,
        )
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("GRKV Default requires its frozen regression settings")
