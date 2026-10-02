# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from triton._C.libtriton import llvm

_primed = False


def prime_wide_compiler():
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
