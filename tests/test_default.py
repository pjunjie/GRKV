# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from grkv.api import GRKVDefaultPress, MistralGRKVDefaultPress, make_press
from grkv.cache import Block
from grkv.config import DefaultConfig
from grkv.operators import AttentionMap, Observations
from grkv.solver import fit_default


@pytest.mark.parametrize("cls,guard", [(GRKVDefaultPress, 0.1), (MistralGRKVDefaultPress, 0.0)])
def test_direct_constructors_use_the_reproduced_model_defaults(cls, guard):
    assert cls().grkv_guard == guard
    with pytest.raises(ValueError, match="model-specific guard"):
        cls(grkv_guard=0.0 if guard else 0.1)
    with pytest.raises(ValueError, match="frozen regression configuration"):
        cls(fit_config=SimpleNamespace())


@pytest.mark.parametrize(
    "setting,value",
    [("query_count", 64), ("query_sampling", "tail"), ("key_relative_cap", 0), ("objective", "post"), ("steps", 2)],
)
def test_only_the_reproduced_regression_settings_are_accepted(setting, value):
    with pytest.raises(ValueError, match="frozen regression settings"):
        DefaultConfig(**{setting: value})


def test_public_factory_rejects_unregistered_method_names():
    with pytest.raises(ValueError, match="Method must be default or baseline"):
        make_press("llama", 10, method="unregistered")


def test_regression_preserves_protected_states_layout_and_bf16_update_caps():
    torch.manual_seed(42)
    heads, groups, queries, dim, slots = 2, 4, 32, 8, 40
    keys = torch.randn(heads, slots, dim).to(torch.bfloat16)
    values = torch.randn_like(keys)
    positions = torch.arange(slots).expand(heads, -1)
    valid = torch.ones(heads, slots, dtype=torch.bool)
    mutable = valid.clone()
    mutable[:, -8:] = False
    observations = Observations(
        torch.randn(heads, groups, queries, dim),
        torch.full((queries,), slots - 1),
        torch.randn(queries, heads * groups * dim),
        torch.full((heads, groups, queries), -torch.inf),
        torch.zeros(heads, groups, queries, dim),
        torch.full((queries,), 6 / queries),
    )
    anchor = Block(keys[None], values[None], positions[None], valid[None], mutable[None])
    config = DefaultConfig()
    result, record = fit_default(anchor, observations, None, config)
    assert result.key.dtype == result.value.dtype == torch.bfloat16
    for name in ("positions", "valid", "regression"):
        assert torch.equal(getattr(result, name), getattr(anchor, name))
    for name, cap in (("key", config.key_relative_cap), ("value", config.relative_cap)):
        before, after = getattr(anchor, name), getattr(result, name)
        assert torch.equal(before[0][~mutable], after[0][~mutable])
        correction = torch.linalg.vector_norm(after.float() - before.float(), dim=(-2, -1))
        limit = cap * torch.linalg.vector_norm(before.float() * mutable[None, ..., None], dim=(-2, -1))
        assert bool((correction <= limit).all())
    assert record["final_objective"] <= record["initial_objective"]
    assert record["student_layout"]["fixed_unchanged"]
    assert record["total_query_weight"] == 6.0
    assert AttentionMap(observations, positions, valid, mutable).mutable.equal(mutable)
