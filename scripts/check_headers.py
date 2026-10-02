# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

root = Path(__file__).resolve().parents[1]
bad = [
    str(p.relative_to(root))
    for name in ("grkv", "kvpress", "experiments", "evaluation", "scripts", "tests")
    for p in (root / name).rglob("*.py")
    if "SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    not in p.read_text()
    or "SPDX-License-Identifier: Apache-2.0" not in p.read_text()
]
if bad:
    raise SystemExit("Missing required SPDX header: " + ", ".join(bad))
print("All Python SPDX headers passed")
