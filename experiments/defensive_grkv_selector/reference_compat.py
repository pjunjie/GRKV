# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from experiments.defensive_grkv_selector.fit_selector import TASK_FAMILIES, _as_string_list

NONCLASSIFICATION_TASKS = frozenset(TASK_FAMILIES) - {"trec"}


def classes_for_scoring(task: str, value: Any) -> list[str]:
    if task not in TASK_FAMILIES:
        raise RuntimeError(f"unknown frozen LongBench task: {task}")
    if value is None and task in NONCLASSIFICATION_TASKS:
        return []
    result = _as_string_list(value, "all_classes")
    if task == "trec" and not result:
        raise RuntimeError("TREC requires its nonempty source class list")
    return result
