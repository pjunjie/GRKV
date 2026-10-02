# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Isolated diagnostic tile choices; never installed in the frozen runtime."""

from contextlib import contextmanager

import triton

from experiments.critical_adakv_direct_grkv import triton_backend
from experiments.critical_adakv_direct_grkv.triton_derivatives import TritonKeyDerivatives
from experiments.critical_adakv_direct_grkv.wide_query_backend import TiledDerivatives

VARIANTS = {
    "current": (64, 64, 4),
    "r32_s64_w4": (32, 64, 4),
    "r32_s64_w8": (32, 64, 8),
    "r32_s32_w4": (32, 32, 4),
    "r64_s32_w8": (64, 32, 8),
}


def launch_parameters(name, rows, slots):
    if name not in VARIANTS or rows <= 0 or slots <= 0:
        raise ValueError("Unknown tile variant or nonpositive shape")
    br, bs, warps = VARIANTS[name]
    return dict(row_tiles=triton.cdiv(rows, br), slot_tiles=triton.cdiv(slots, bs), BR=br, BS=bs, num_warps=warps)


class DiagnosticDerivatives(TiledDerivatives):
    def __init__(self, op, keys, values, variant):
        super().__init__(op, keys, values)
        config = launch_parameters(variant, self.rows, self.slots)
        self.row_tiles, self.tiles = config.pop("row_tiles"), config.pop("slot_tiles")
        self.bs = config["BS"]
        self.kwargs.update(config)


def make_derivatives(op, keys, values, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown diagnostic variant")
    rows = op.obs.queries.shape[1] * op.obs.queries.shape[2]
    if rows <= 64:
        return TritonKeyDerivatives(op, keys, values)
    if variant == "current":
        return TiledDerivatives(op, keys, values)
    return DiagnosticDerivatives(op, keys, values, variant)


@contextmanager
def diagnostic_backend(variant, capture=None):
    if variant not in VARIANTS:
        raise ValueError("Unknown diagnostic variant")
    previous = triton_backend.TritonKeyDerivatives

    def factory(op, keys, values):
        result = make_derivatives(op, keys, values, variant)
        if capture is not None:
            capture(op, keys, values, result)
        return result

    setattr(triton_backend, "TritonKeyDerivatives", factory)
    try:
        with triton_backend.key_backend(memoize_state=True):
            yield
    finally:
        setattr(triton_backend, "TritonKeyDerivatives", previous)
