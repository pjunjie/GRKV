# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auditable launch of historical executable sections with redacted debug paths.

This profile deliberately does not claim original full-file cubin SHA equality.
Both the newly compiled executable sections and the published cubin are checked
before CUDA creates handles. The public binary has separate, mandatory hashes.
"""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from grkv import kernel_runtime, kernels
from grkv.elf import executable_identity
from grkv.io import read, sha
from grkv.kernels import TILES, prime_compiler


class HistoricalExecutableBackend:
    def __init__(self, manifest_path, cache_root):
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = sha(manifest_path)
        self.manifest = read(manifest_path)
        if self.manifest["status"] != "complete" or self.manifest["tiles"] != {
            kind: list(tiles) for kind, tiles in TILES.items()
        }:
            raise ValueError("Kernel manifest/launch geometry mismatch")
        if self.manifest["target"] != dict(backend="cuda", arch=86, warp_size=32):
            raise ValueError("Kernel target must be CUDA SM86")
        self.cache_root = Path(cache_root).resolve()
        if Path(os.environ["TRITON_CACHE_DIR"]).resolve() != self.cache_root:
            raise ValueError("Worker cache path mismatch")
        self.expected = {}
        for entry in self.manifest["kernels"]:
            path = self.manifest_path.parent / entry["cubin"]["path"]
            if sha(path) != entry["cubin"]["sha256"]:
                raise ValueError("Published cubin hash mismatch")
            if executable_identity(path.read_bytes()) != entry["executable_sections"]:
                raise ValueError("Published executable-section identity mismatch")
            self.expected[(entry["query_count"] * 4, entry["kind"])] = entry
        if set(self.expected) != {(128, "jvp"), (128, "vjp")}:
            raise ValueError("GRKV Default requires exactly its two historical derivative kernels")
        self.verified = {}
        self.launch_records = []

    def _guard(self, kind, previous):
        def run(*args, **kwargs):
            if kwargs.get("warmup", False):
                raise ValueError("Launch guard accepts actual launches only")
            br, bs, warps = TILES[kind]
            assert kwargs["D"] == kwargs["BD"] == 128 and kwargs["BR"] == br and kwargs["BS"] == bs
            assert kwargs["num_warps"] == warps and kwargs["num_stages"] == 1
            assert torch.cuda.get_device_capability() == (8, 6)
            assert len(args) == 7 and all(t.is_cuda and t.is_contiguous() and t.data_ptr() % 16 == 0 for t in args)
            assert all(t.dtype == (torch.bool if i == 5 else torch.float32) for i, t in enumerate(args))
            key = (kwargs["R"], kind)
            expected = self.expected[key]
            if key not in self.verified:
                prime_compiler()
                compiled = previous(*args, **dict(kwargs, warmup=True))
                if compiled.module is not None:
                    raise ValueError("CUDA handles created before public cubin verification")
                source_identity = executable_identity(compiled.asm["cubin"])
                if source_identity != expected["executable_sections"]:
                    raise ValueError("Fresh compilation differs from historical nondebug sections")
                if compiled.metadata.shared != expected["shared_bytes"]:
                    raise ValueError("Kernel shared-memory mismatch")
                if compiled.metadata.target.arch != 86:
                    raise ValueError("Compiled kernel architecture mismatch")
                binary = (self.manifest_path.parent / expected["cubin"]["path"]).read_bytes()
                digest = hashlib.sha256(binary).hexdigest()
                if digest != expected["cubin"]["sha256"]:
                    raise ValueError("Cubin changed before launch")
                compiled.asm["cubin"] = binary
                compiled.kernel = binary
                self.verified[key] = compiled
                self.launch_records.append(
                    dict(
                        rows=key[0],
                        kind=kind,
                        cubin_sha256=digest,
                        historical_cubin_sha256=expected["historical_cubin_sha256"],
                        original_cubin_bytes_identical=False,
                        executable_sections_identical=True,
                        checked_before_launch=True,
                        shared_bytes=compiled.metadata.shared,
                        arch=86,
                    )
                )
            compiled = previous(*args, **kwargs)
            if compiled is not self.verified[key]:
                raise ValueError("Unexpected compiler cache replacement")
            return compiled

        return run

    @contextmanager
    def __call__(self, capture=None):
        if Path(os.environ["TRITON_CACHE_DIR"]).resolve() != self.cache_root:
            raise ValueError("Worker cache path changed")
        patches = []
        try:
            for kind in ("jvp", "vjp"):
                kernel = getattr(kernels, "_wide_" + kind)
                patches.append((kernel, kernel.run))
                kernel.run = self._guard(kind, kernel.run)
            with kernel_runtime.key_backend(capture=capture):
                yield
        finally:
            for kernel, previous in reversed(patches):
                kernel.run = previous
