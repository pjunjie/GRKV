# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Exact head-wise masking if the usual fake-key search has no solution.

The standard kvpress FlashAttention wrapper first searches for one fake key that
nullifies every query in a GQA group. A long question can have no such common
hyperplane. Only in that case, calculate attention with an explicit per-head
mask for the affected forward call. Successful fake-key calls are unchanged.
"""

from functools import wraps

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward

FAILURE = "Could not find fake keys such that for every query q, exp(<q, k>) = 0"
FALLBACK_CALLS = {"count": 0}


def exact_masked_attention(module, query, key, value, attention_mask, dropout, **kwargs):
    """Return the standard eager attention result with the press's head mask."""
    bsz, num_heads, q_len, _ = query.shape
    kv_heads, key_len = key.shape[1:3]
    if num_heads % kv_heads or key_len < q_len:
        raise ValueError("Invalid grouped-query attention dimensions")
    groups = num_heads // kv_heads
    blocked = torch.zeros((bsz, kv_heads, key_len), dtype=torch.bool, device=query.device)
    batch, head, position = module.masked_key_indices
    blocked[batch, head, position] = True
    blocked = blocked.repeat_interleave(groups, dim=1).unsqueeze(2)
    past_len = key_len - q_len
    key_position = torch.arange(key_len, device=query.device)
    query_position = past_len + torch.arange(q_len, device=query.device)
    causal = key_position[None, :] > query_position[:, None]
    blocked = blocked | causal[None, None, :, :]
    sliding_window = kwargs.get("sliding_window")
    if sliding_window is not None:
        too_old = key_position[None, :] < query_position[:, None] - int(sliding_window) + 1
        blocked = blocked | too_old[None, None, :, :]
    bias = torch.zeros((bsz, num_heads, q_len, key_len), dtype=query.dtype, device=query.device)
    bias.masked_fill_(blocked, -torch.inf)
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            bias = bias + attention_mask
        elif attention_mask.ndim == 2:
            bias.masked_fill_(~attention_mask[:, None, None, :].bool(), -torch.inf)
        else:
            raise ValueError("Unsupported attention mask shape in exact fallback")
    return eager_attention_forward(module, query, key, value, bias, dropout=dropout, **kwargs)


def with_fallback(original):
    """Preserve the original path except when fake-key feasibility fails."""

    @wraps(original)
    def wrapped(module, query, key, value, attention_mask, dropout, **kwargs):
        try:
            return original(module, query, key, value, attention_mask, dropout, **kwargs)
        except ValueError as exc:
            if str(exc) != FAILURE or getattr(module, "masked_key_indices", None) is None:
                raise
            FALLBACK_CALLS["count"] += 1
            return exact_masked_attention(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapped


def install():
    """Wrap the already-patched FlashAttention2 entry once per worker process."""
    name = "flash_attention_2"
    original = ALL_ATTENTION_FUNCTIONS[name]
    if getattr(original, "_critical_adakv_exact_fallback", False):
        return
    wrapped = with_fallback(original)
    wrapped._critical_adakv_exact_fallback = True
    ALL_ATTENTION_FUNCTIONS[name] = wrapped
