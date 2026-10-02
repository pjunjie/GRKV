# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Operator-specific tiles preserve original reductions with verified cubins."""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from experiments.critical_adakv_direct_grkv import triton_backend, wide_query_backend
from experiments.critical_adakv_direct_grkv.triton_derivatives import TritonKeyDerivatives
from experiments.critical_adakv_direct_grkv.wide_query_kernel_variants import DiagnosticDerivatives
from experiments.critical_adakv_direct_grkv.wide_query_optimized_backend import prime_wide_compiler
from experiments.critical_adakv_history_ruler.common import read, sha

VARIANT = "jvp_r32_s64_w4_vjp_r64_s32_w8"
TILES = {"jvp": (32, 64, 4), "vjp": (64, 32, 8)}


class SplitBackend:
    """One validated backend per worker; reuse guards across warm/measured calls."""

    def __init__(self, manifest_path, cache_root):
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = sha(manifest_path)
        self.manifest = read(manifest_path)
        assert self.manifest["status"] == "complete" and self.manifest["variant"] == VARIANT
        assert self.manifest["target"] == dict(backend="cuda", arch=86, warp_size=32)
        assert all(sha(p) == digest for p, digest in self.manifest["source_hashes"].items())
        self.cache_root = Path(cache_root).resolve()
        assert self.cache_root != Path.home().joinpath(".triton/cache").resolve()
        assert Path(os.environ["TRITON_CACHE_DIR"]).resolve() == self.cache_root
        self.expected = {}
        for entry in self.manifest["kernels"]:
            assert entry["cubin"]["sha256"] == sha(entry["cubin"]["path"])
            self.expected[(entry["query"] * 4, entry["kind"])] = entry
        self.verified = {}
        self.launch_records = []

    def identity(self):
        return dict(
            name="wide_query_split_reduction_tiles_primed_cubin_v1",
            manifest=dict(path=str(self.manifest_path), sha256=self.manifest_sha256),
            variant=VARIANT,
            query16_path="Original TritonKeyDerivatives class unchanged",
            numerical_precision="FP32/tf32x3, BF16 cache writeback unchanged",
        )

    def _guard(self, kind, previous):
        def run(*args, **kwargs):
            assert not kwargs.get("warmup", False), "Only actual derivative launches enter this guard"
            br, bs, warps = TILES[kind]
            assert kwargs["D"] == kwargs["BD"] == 128 and kwargs["BR"] == br and kwargs["BS"] == bs
            assert kwargs["num_warps"] == warps and kwargs["num_stages"] == 1
            assert torch.cuda.get_device_capability() == (8, 6)
            assert len(args) == 7 and all(t.is_cuda and t.is_contiguous() and t.data_ptr() % 16 == 0 for t in args)
            assert all(t.dtype == (torch.bool if i == 5 else torch.float32) for i, t in enumerate(args))
            key = (kwargs["R"], kind)
            expected = self.expected[key]
            if key not in self.verified:
                prime_wide_compiler()
                # Warmup compiles/looks up without executing the kernel. Reject
                # any contaminated cache artifact before its first GPU launch.
                compiled = previous(*args, **dict(kwargs, warmup=True))
                digest = hashlib.sha256(compiled.asm["cubin"]).hexdigest()
                assert digest == expected["cubin"]["sha256"], "Actual cubin differs from qualified binary"
                assert compiled.metadata.shared == expected["shared_bytes"]
                self.verified[key] = compiled
                self.launch_records.append(
                    dict(
                        rows=key[0],
                        kind=kind,
                        cubin_sha256=digest,
                        compiled_hash=compiled.hash,
                        checked_before_launch=True,
                    )
                )
            compiled = previous(*args, **kwargs)
            assert compiled is self.verified[key], "Unexpected compiler cache replacement"
            return compiled

        return run

    @contextmanager
    def __call__(self, capture=None):
        assert Path(os.environ["TRITON_CACHE_DIR"]).resolve() == self.cache_root
        original_factory = triton_backend.TritonKeyDerivatives
        patches = []

        def factory(op, keys, values):
            result: Any
            rows = op.obs.queries.shape[1] * op.obs.queries.shape[2]
            if rows <= 64:
                result = TritonKeyDerivatives(op, keys, values)
            else:
                assert op.obs.queries.shape[1] == 4 and op.obs.queries.shape[-1] == 128
                prime_wide_compiler()
                jvp = DiagnosticDerivatives(op, keys, values, "r32_s64_w4")
                vjp = DiagnosticDerivatives(op, keys, values, "r64_s32_w8")
                result = SimpleNamespace(forward=jvp.forward, adjoint=vjp.adjoint)
            if capture is not None:
                capture(op, keys, values, result)
            return result

        try:
            for kind in ("jvp", "vjp"):
                kernel = getattr(wide_query_backend, "_wide_" + kind)
                patches.append((kernel, kernel.run))
                kernel.run = self._guard(kind, kernel.run)
            setattr(triton_backend, "TritonKeyDerivatives", factory)
            with triton_backend.key_backend(memoize_state=True):
                yield
        finally:
            setattr(triton_backend, "TritonKeyDerivatives", original_factory)
            for kernel, previous in reversed(patches):
                kernel.run = previous
