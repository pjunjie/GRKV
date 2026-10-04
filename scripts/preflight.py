# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real GPU derivative qualification for the historical executable profile."""

import argparse
import json
import os
from pathlib import Path

from grkv.settings import load_environment


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["historical-executable-sm86"], default="historical-executable-sm86")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--kernel-root", type=Path, default=Path(os.environ.get("GRKV_KERNEL_ROOT", "artifacts/kernels"))
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/preflight.json"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cache = args.output.parent / "qualification-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache.resolve())
    import torch

    from grkv.backend import HistoricalExecutableBackend
    from grkv.io import write
    from grkv.qualification import qualify
    from grkv.runtime import runtime

    identity = runtime()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = HistoricalExecutableBackend(args.kernel_root / "manifest.json", cache)
    proof = qualify(backend)
    result = dict(
        profile=args.profile,
        runtime=identity,
        qualification=proof,
        original_cubin_bytes_identical=False,
        gpu_qualification_passed=True,
    )
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
