# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped historical derivatives and immutable-input attention-state reuse."""

from contextlib import contextmanager
from weakref import WeakKeyDictionary

from grkv.kernels import make_derivatives
from grkv.operators import AttentionMap


@contextmanager
def key_backend(capture=None):
    original = AttentionMap.derivatives
    original_state = AttentionMap.state
    cache: WeakKeyDictionary = WeakKeyDictionary()

    def state(op, keys, values):
        # GRKV Default fitting treats input K/V tensors as immutable. Strong input
        # references prevent id reuse; weak operator keys release each layer's
        # memoized states as soon as that fit ends. No cache survives this scope.
        entries = cache.setdefault(op, {})
        key = (id(keys), id(values))
        if key not in entries:
            entries[key] = (keys, values, original_state(op, keys, values))
        return entries[key][2]

    def derivatives(op, keys, values, kind):
        if kind != "key":
            return original(op, keys, values, kind)
        fast = make_derivatives(op, keys, values)
        if capture is not None:
            capture(op, keys, values, fast)
        return fast.forward, fast.adjoint

    setattr(AttentionMap, "derivatives", derivatives)
    setattr(AttentionMap, "state", state)
    try:
        yield
    finally:
        setattr(AttentionMap, "derivatives", original)
        setattr(AttentionMap, "state", original_state)
        cache.clear()
