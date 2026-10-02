# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Tile query rows to fit A6000 shared memory; preserve the original small-Q kernel."""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl

from experiments.critical_adakv_direct_grkv import triton_backend as original_backend
from experiments.critical_adakv_direct_grkv.triton_derivatives import TritonKeyDerivatives as OriginalDerivatives


@triton.jit(do_not_specialize=["S"])
def _wide_vjp(
    Q,
    V,
    W,
    O,
    U,
    MUTABLE,
    PARTIAL,
    R: tl.constexpr,
    S,
    D: tl.constexpr,
    TR: tl.constexpr,
    BR: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    h, st, rt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    s = st * BS + tl.arange(0, BS)
    r = rt * BR + tl.arange(0, BR)
    d = tl.arange(0, BD)
    v = tl.load(V + h * S * D + s[:, None] * D + d[None, :], (s[:, None] < S) & (d[None, :] < D), 0)
    q = tl.load(Q + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    u = tl.load(U + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    out = tl.load(O + h * R * D + r[:, None] * D + d[None, :], (r[:, None] < R) & (d[None, :] < D), 0)
    w = tl.load(W + h * R * S + r[:, None] * S + s[None, :], (r[:, None] < R) & (s[None, :] < S), 0)
    mutable = tl.load(MUTABLE + h * S + s, s < S, 0)
    uv = tl.dot(u, tl.trans(v), input_precision="tf32x3")
    dz = w * (uv - tl.sum(u * out, 1)[:, None]) * mutable[None, :]
    result = tl.dot(tl.trans(dz), q, input_precision="tf32x3") * (D**-0.5)
    offset = ((h * TR + rt) * S + s[:, None]) * D + d[None, :]
    tl.store(PARTIAL + offset, result, (s[:, None] < S) & (d[None, :] < D))


@triton.jit(do_not_specialize=["S", "TILES"])
def _wide_jvp(
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
    h, st, rt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    s = st * BS + tl.arange(0, BS)
    r = rt * BR + tl.arange(0, BR)
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
    offset = (h * TILES + st) * R * D + r[:, None] * D + d[None, :]
    tl.store(PARTIAL + offset, result, (r[:, None] < R) & (d[None, :] < D))


class TiledDerivatives(OriginalDerivatives):
    def __init__(self, op, keys, values):
        super().__init__(op, keys, values)
        self.row_tiles = triton.cdiv(self.rows, 64)
        self.kwargs.update(BR=64, num_stages=1)

    def forward(self, delta):
        partial = torch.empty((self.heads, self.tiles, self.rows, self.dim), device=self.q.device, dtype=torch.float32)
        _wide_jvp[(self.heads, self.tiles, self.row_tiles)](
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
        partial = torch.empty(
            (self.heads, self.row_tiles, self.slots, self.dim), device=self.q.device, dtype=torch.float32
        )
        _wide_vjp[(self.heads, self.tiles, self.row_tiles)](
            self.q, self.v, self.weights, self.out, u, self.mutable, partial, TR=self.row_tiles, **self.kwargs
        )
        return partial.sum(1)


def derivatives_for(op, keys, values):
    # This exact original class (not a reimplementation) retains the Q16 control
    # at group4*Q16=64 rows and its previously measured numerical behavior.
    cls = OriginalDerivatives if op.obs.queries.shape[1] * op.obs.queries.shape[2] <= 64 else TiledDerivatives
    return cls(op, keys, values)


@contextmanager
def key_backend(memoize_state=True):
    previous = original_backend.TritonKeyDerivatives
    setattr(original_backend, "TritonKeyDerivatives", derivatives_for)
    try:
        with original_backend.key_backend(memoize_state=memoize_state):
            yield
    finally:
        setattr(original_backend, "TritonKeyDerivatives", previous)
