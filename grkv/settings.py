# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local paths and optional environment configuration; no stored credentials."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_environment():
    local = ROOT / ".env.local"
    if local.exists():
        for line in local.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.removeprefix("export ").split("=", 1)
                value = value.strip().strip("\"'")
                if value == "xxxxxx":
                    raise ValueError(f"Replace the placeholder for {key} in .env.local")
                os.environ.setdefault(key.strip(), value)


def asset_root():
    load_environment()
    return Path(os.environ.get("GRKV_ASSET_ROOT", ROOT / "assets"))
