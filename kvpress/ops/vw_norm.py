# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Fused value/output-projection norm used by DefensiveKV.

The Triton kernel is adapted from the official DefensiveKV implementation.  A
PyTorch fallback keeps the operator usable on CPU and on unsupported devices.
"""

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on installations without Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def _torch_vw_l1norm(value_states: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    """Compute ``||V_h W_h||_1`` one head at a time to bound peak memory."""

    norms = []
    for head_idx in range(value_states.shape[1]):
        projected = torch.matmul(value_states[:, head_idx], output_weight[head_idx].unsqueeze(0))
        norms.append(projected.abs().sum(dim=-1))
    return torch.stack(norms, dim=1).to(value_states.dtype)


if triton is not None:

    @triton.jit
    def _vw_l1norm_kernel(
        values,
        weights,
        output,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_dim,
        stride_w_head,
        stride_w_dim,
        stride_w_output,
        stride_o_batch,
        stride_o_head,
        stride_o_token,
        num_heads,
        seq_len,
        HEAD_DIM: tl.constexpr,
        OUTPUT_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_OUTPUT: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch_idx = batch_head // num_heads
        head_idx = batch_head % num_heads

        value_ptr = tl.make_block_ptr(
            base=values + batch_idx * stride_v_batch + head_idx * stride_v_head,
            shape=(seq_len, HEAD_DIM),
            strides=(stride_v_token, stride_v_dim),
            offsets=(token_block * BLOCK_TOKENS, 0),
            block_shape=(BLOCK_TOKENS, HEAD_DIM),
            order=(1, 0),
        )
        weight_ptr = tl.make_block_ptr(
            base=weights + head_idx * stride_w_head,
            shape=(HEAD_DIM, OUTPUT_DIM),
            strides=(stride_w_dim, stride_w_output),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_OUTPUT),
            order=(1, 0),
        )

        value_block = tl.load(value_ptr, boundary_check=(0, 1), padding_option="zero")
        norm = tl.zeros([BLOCK_TOKENS], dtype=tl.float32)
        for _ in range(0, OUTPUT_DIM, BLOCK_OUTPUT):
            weight_block = tl.load(weight_ptr, boundary_check=(0, 1), padding_option="zero")
            projected = tl.dot(value_block, weight_block)
            norm += tl.sum(tl.abs(projected), axis=-1)
            weight_ptr = tl.advance(weight_ptr, (0, BLOCK_OUTPUT))

        token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        output_ptr = output + batch_idx * stride_o_batch + head_idx * stride_o_head + token_offsets * stride_o_token
        tl.store(output_ptr, norm, mask=token_offsets < seq_len)


def _triton_config(seq_len: int, head_dim: int) -> tuple[int, int, int, int]:
    capability = torch.cuda.get_device_capability()
    if capability == (8, 0):
        if head_dim <= 64:
            return 128, 64, 3, 4
        if seq_len <= 1024:
            return 128, 32, 3, 4
        return 128, 128, 3, 8
    if capability == (8, 6):
        if head_dim <= 64:
            return 128, 64, 3, 4
        return 128, 32, 2, 4
    return 32, 32, 1, 4


def _can_use_triton(value_states: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and value_states.is_cuda
        and value_states.dtype in (torch.float16, torch.bfloat16)
        and value_states.shape[-1] in {16, 32, 64, 128, 256}
    )


def vw_l1norm(value_states: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    """Return per-token L1 norms of head-specific output projections.

    Parameters use shapes ``[batch, query_heads, tokens, head_dim]`` and
    ``[query_heads, head_dim, hidden_size]``.  The result has shape
    ``[batch, query_heads, tokens]``.
    """

    if value_states.ndim != 4 or output_weight.ndim != 3:
        raise ValueError("value_states and output_weight must be rank-4 and rank-3 tensors")
    if value_states.shape[1] != output_weight.shape[0] or value_states.shape[-1] != output_weight.shape[1]:
        raise ValueError("value_states and output_weight head dimensions do not match")
    if not _can_use_triton(value_states):
        return _torch_vw_l1norm(value_states, output_weight)

    batch_size, num_heads, seq_len, head_dim = value_states.shape
    output = torch.empty(
        (batch_size, num_heads, seq_len),
        device=value_states.device,
        dtype=value_states.dtype,
    )
    block_tokens, block_output, num_stages, num_warps = _triton_config(seq_len, head_dim)
    grid = (triton.cdiv(seq_len, block_tokens), batch_size * num_heads, 1)
    kernel: Any = _vw_l1norm_kernel
    kernel[grid](
        value_states,
        output_weight,
        output,
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        output_weight.stride(0),
        output_weight.stride(1),
        output_weight.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        seq_len,
        HEAD_DIM=head_dim,
        OUTPUT_DIM=output_weight.shape[-1],
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUTPUT=block_output,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return output
