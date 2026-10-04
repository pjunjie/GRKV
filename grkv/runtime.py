# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify imported runtime files, metadata and real hardware."""

import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
import triton

from grkv.io import read, sha
from grkv.settings import ROOT


def runtime():
    expected = {"torch": "2.6.0+cu124", "triton": "3.2.0", "transformers": "5.2.0", "flash-attn": "2.8.3.post1"}
    versions = {name: version(name) for name in expected}
    for name, value in expected.items():
        actual = versions[name] if name == "torch" else versions[name].split("+")[0]
        if actual != value:
            raise ValueError(f"Installed {name} version differs from reference")
    if triton.__version__ != "3.2.0" or platform.python_version() != "3.12.7" or torch.version.cuda != "12.4":
        raise ValueError("Imported Triton, Python or CUDA runtime differs")
    from triton._C import libtriton

    site = Path(triton.__file__).resolve().parent
    if not Path(libtriton.__file__).resolve().is_relative_to(site):
        raise ValueError("libtriton/compiler package mismatch")
    inventory = read(ROOT / "environment/triton_package.json")
    for name, digest in inventory.items():
        if sha(site / name) != digest:
            raise ValueError(f"Imported Triton package file differs: triton/{name}")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8, 6):
        raise ValueError("Use one isolated CUDA SM86 device per worker")
    if torch.cuda.get_device_name(0) != "NVIDIA RTX A6000":
        raise ValueError("Historical profile requires NVIDIA RTX A6000")
    drivers = set(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()
    )
    if drivers != {"555.52.04"}:
        raise ValueError("Historical profile requires NVIDIA driver 555.52.04")
    return dict(
        software=versions,
        python=platform.python_version(),
        torch_cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        capability=[8, 6],
        driver="555.52.04",
        triton_compiler_sha256=sha(site / "backends/nvidia/compiler.py"),
        libtriton_sha256=sha(libtriton.__file__),
        imported_triton_package_verified=True,
    )
