# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Experimental fused key JVP/VJP; not installed into the running reference press."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S"])
def _key_vjp(
    Q,
    V,
    W,
    O,
    U,
    MUTABLE,
    DK,
    R: tl.constexpr,
    S,
    D: tl.constexpr,
    BR: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    h = tl.program_id(0)
    s = tl.program_id(1) * BS + tl.arange(0, BS)
    r = tl.arange(0, BR)
    d = tl.arange(0, BD)
    v = tl.load(V + h * S * D + s[:, None] * D + d[None, :], (s[:, None] < S) & (d[None, :] < D), 0)
    q = tl.load(Q + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    u = tl.load(U + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    out = tl.load(O + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    w = tl.load(W + h * R * S + r[:, None] * S + s[None, :], (r[:, None] < R) & (s[None, :] < S), 0)
    mutable = tl.load(MUTABLE + h * S + s, s < S, 0)
    uv = tl.dot(u, tl.trans(v), input_precision="tf32x3")
    uo = tl.sum(u * out, 1)
    dz = w * (uv - uo[:, None]) * mutable[None, :]
    result = tl.dot(tl.trans(dz), q, input_precision="tf32x3") * (D**-0.5)
    tl.store(DK + h * S * D + s[:, None] * D + d[None, :], result, (s[:, None] < S) & (d[None, :] < D))


@triton.jit(do_not_specialize=["S", "TILES"])
def _key_jvp(
    Q,
    V,
    W,
    O,
    DELTA,
    MUTABLE,
    PARTIAL,
    R: tl.constexpr,
    S,
    D: tl.constexpr,
    TILES,
    BR: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    h, tile = tl.program_id(0), tl.program_id(1)
    s = tile * BS + tl.arange(0, BS)
    r = tl.arange(0, BR)
    d = tl.arange(0, BD)
    q = tl.load(Q + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    delta = tl.load(DELTA + h * S * D + s[:, None] * D + d[None, :], (s[:, None] < S) & (d[None, :] < D), 0)
    v = tl.load(V + h * S * D + s[:, None] * D + d[None, :], (s[:, None] < S) & (d[None, :] < D), 0)
    out = tl.load(O + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    w = tl.load(W + h * R * S + r[:, None] * S + s[None, :], (r[:, None] < R) & (s[None, :] < S), 0)
    mutable = tl.load(MUTABLE + h * S + s, s < S, 0)
    dl = tl.dot(q, tl.trans(delta), input_precision="tf32x3") * (D**-0.5)
    weighted = w * dl * mutable[None, :]
    result = tl.dot(weighted, v, input_precision="tf32x3") - tl.sum(weighted, 1)[:, None] * out
    offset = (h * TILES + tile) * R * D + r[:, None] * D + d[None, :]
    tl.store(PARTIAL + offset, result, (r[:, None] < R) & (d[None, :] < D))


class TritonKeyDerivatives:
    """Cache current linearization weights, process all GQA queries/head together."""

    def __init__(self, op, keys, values):
        self.heads, self.groups, self.queries, self.dim = op.obs.queries.shape
        self.slots = keys.shape[1]
        _, weights, out = op.state(keys, values)
        self.q = op.obs.queries.reshape(self.heads, -1, self.dim).contiguous()
        self.v = values.contiguous()
        self.weights = weights.reshape(self.heads, -1, self.slots).contiguous()
        self.out = out.reshape(self.heads, -1, self.dim).contiguous()
        self.mutable = op.mutable.contiguous()
        if any(x.dtype != torch.float32 or not x.is_cuda for x in (self.q, self.v, self.weights, self.out)):
            raise ValueError("Experimental Triton derivatives require CUDA float32")
        self.rows = self.groups * self.queries
        self.bs = 64
        self.tiles = triton.cdiv(self.slots, self.bs)
        self.kwargs = dict(
            R=self.rows,
            S=self.slots,
            D=self.dim,
            BR=max(16, triton.next_power_of_2(self.rows)),
            BS=self.bs,
            BD=max(32, triton.next_power_of_2(self.dim)),
            num_warps=4,
        )

    def forward(self, delta):
        partial = torch.empty((self.heads, self.tiles, self.rows, self.dim), device=self.q.device, dtype=torch.float32)
        _key_jvp[(self.heads, self.tiles)](
            self.q,
            self.v,
            self.weights,
            self.out,
            delta.contiguous(),
            self.mutable,
            partial,
            TILES=self.tiles,
            **self.kwargs,
        )
        result = partial.sum(1).reshape(self.heads, self.groups, self.queries, self.dim)
        return result.permute(2, 0, 1, 3).reshape(self.queries, -1)

    def adjoint(self, residual):
        u = residual.reshape(self.queries, self.heads, self.groups, self.dim).permute(1, 2, 0, 3).contiguous()
        result = torch.empty_like(self.v)
        _key_vjp[(self.heads, self.tiles)](
            self.q, self.v, self.weights, self.out, u, self.mutable, result, **self.kwargs
        )
        return result
