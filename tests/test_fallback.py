# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from experiments.critical_adakv_cross_model.mask_fallback import FAILURE, with_fallback


def test_exact_fallback_masks_heads_and_preserves_query_causality():
    module = SimpleNamespace(
        num_key_value_groups=1,
        training=False,
        masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
    )
    query = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])
    key = torch.tensor([[[[100.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    value = torch.tensor([[[[1000.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]]]])

    def impossible(*args, **kwargs):
        raise ValueError(FAILURE)

    output, weights = with_fallback(impossible)(module, query, key, value, None, dropout=0.0, scaling=1.0)
    assert weights[0, 0, :, 0].eq(0).all()  # head-pruned key
    assert weights[0, 0, 0, 3].item() == 0  # first query cannot see future key
    assert torch.isfinite(output).all()
    assert output[0, 0, 0, 0].item() < 10  # the future value 100 is excluded


def test_successful_fake_key_path_is_unchanged():
    sentinel = object()

    def successful(*args, **kwargs):
        return sentinel

    assert with_fallback(successful)(None, None, None, None, None, 0.0) is sentinel
