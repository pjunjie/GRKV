# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact sample-set coverage and task-macro scores."""

import math
from collections import defaultdict


def row_key(item, row):
    return item["benchmark"], item["budget"], item["task"], row["original_row_id"]


def expected_rows(units):
    keys = [row_key(w["item"], r) for w in units for r in w["item"]["rows"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate expected sample identity")
    return set(keys)


def summarize(records, units):
    expected = expected_rows(units)
    seen = set()
    tasks = defaultdict(list)
    count = 0
    for record in records:
        item = record["item"]
        for row in record["value"]["rows"]:
            key = row_key(item, row)
            if key in seen:
                raise ValueError("Duplicate answer identity")
            seen.add(key)
            value = float(row["score"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite per-answer score")
            maximum = 1 if item["benchmark"] == "ruler16k" else 100
            if not 0 <= value <= maximum:
                raise ValueError("Per-answer score outside the benchmark scale")
            tasks[key[:3]].append(value * (100 if item["benchmark"] == "ruler16k" else 1))
            count += 1
    if seen != expected:
        raise ValueError(f"Coverage mismatch: missing={len(expected-seen)}, unexpected={len(seen-expected)}")
    task_scores = {f"{b}_{budget}/{t}": sum(v) / len(v) for (b, budget, t), v in sorted(tasks.items())}
    cells = defaultdict(list)
    for (b, budget, _), values in sorted(tasks.items()):
        cells[f"{b}_{budget}"].append(sum(values) / len(values))
    return dict(
        coverage_passed=True,
        answers=count,
        cells={k: sum(v) / len(v) for k, v in cells.items()},
        task_scores=task_scores,
        task_counts={f"{b}_{budget}/{t}": len(v) for (b, budget, t), v in sorted(tasks.items())},
    )
