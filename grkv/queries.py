# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-stratified reference queries for GRKV Default."""

import torch


def stratified(a, b, count, device):
    if not 0 < count <= b - a:
        raise ValueError("Stratified sample must fit its source interval")
    edges = a + torch.arange(count + 1, device=device) * (b - a) // count
    return (edges[:-1] + edges[1:] - 1) // 2


def source_positions(config, context_tokens, device="cpu"):
    if type(context_tokens) is not int or context_tokens < 1:
        raise ValueError("Actual context token count must be positive")
    if context_tokens < config.query_count:
        return torch.arange(context_tokens, device=device)
    return stratified(0, context_tokens, config.query_count, device)


def query_metadata(config, context_tokens, used_fit_q=0, skip_reason=None):
    positions = source_positions(config, context_tokens).tolist()
    effective = min(config.query_count, context_tokens)
    if used_fit_q not in (0, effective):
        raise ValueError("A fit must use exactly the planned effective Q count")
    return dict(
        requested_Q=config.query_count,
        planned_Q=effective,
        effective_Q=effective,
        used_fit_Q=used_fit_q,
        context_tokens=context_tokens,
        source_positions=positions,
        source_range=[positions[0], positions[-1] + 1],
        short_context_fallback=context_tokens < config.query_count,
        virtual_position=context_tokens - 1,
        planned_total_query_weight=6.0,
        planned_per_query_weight=6.0 / effective,
        total_query_weight=6.0 if used_fit_q else 0.0,
        skip_reason=skip_reason,
    )
