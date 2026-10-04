# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two historical derivative kernels required by GRKV Default."""

from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from triton._C.libtriton import llvm

TILES = {"jvp": (32, 64, 4), "vjp": (64, 32, 8)}
_primed = False


def prime_compiler():
    """Make LLVM pointer layout independent of earlier kernel compilation order."""
    global _primed
    if not _primed:
        llvm.init_targets()
        assembly = llvm.translate_to_asm(
            "define void @grkv_prime() { ret void }",
            "nvptx64-nvidia-cuda",
            "sm_86",
            "+ptx80",
            ["nvptx-short-ptr"],
            True,
            False,
        )
        assert "grkv_prime" in assembly
        _primed = True


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


class KernelDerivatives:
    """Fixed JVP or VJP launch geometry; immutable attention-state inputs."""

    def __init__(self, op, keys, values, kind):
        self.heads, self.groups, self.queries, self.dim = op.obs.queries.shape
        if (self.groups, self.queries, self.dim) != (4, 32, 128):
            raise ValueError("Historical kernels require the GRKV Default query geometry")
        self.slots = keys.shape[1]
        _, weights, out = op.state(keys, values)
        self.q = op.obs.queries.reshape(self.heads, -1, self.dim).contiguous()
        self.v = values.contiguous()
        self.weights = weights.reshape(self.heads, -1, self.slots).contiguous()
        self.out = out.reshape(self.heads, -1, self.dim).contiguous()
        self.mutable = op.mutable.contiguous()
        if any(x.dtype != torch.float32 or not x.is_cuda for x in (self.q, self.v, self.weights, self.out)):
            raise ValueError("Historical Triton derivatives require CUDA float32")
        self.rows = self.groups * self.queries
        br, self.bs, warps = TILES[kind]
        self.tiles = triton.cdiv(self.slots, self.bs)
        self.row_tiles = triton.cdiv(self.rows, br)
        self.kwargs = dict(
            R=self.rows, S=self.slots, D=self.dim, BR=br, BS=self.bs, BD=128, num_warps=warps, num_stages=1
        )

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


def make_derivatives(op, keys, values):
    prime_compiler()
    jvp = KernelDerivatives(op, keys, values, "jvp")
    vjp = KernelDerivatives(op, keys, values, "vjp")
    return SimpleNamespace(forward=jvp.forward, adjoint=vjp.adjoint)
