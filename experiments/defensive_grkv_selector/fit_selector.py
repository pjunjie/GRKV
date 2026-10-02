# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from evaluation.benchmarks.longbench.calculate_metrics import dataset2metric

TASK_FAMILIES = {
    "narrativeqa": "single_document_qa",
    "qasper": "single_document_qa",
    "multifieldqa_en": "single_document_qa",
    "hotpotqa": "multi_document_qa",
    "2wikimqa": "multi_document_qa",
    "musique": "multi_document_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "few_shot_learning",
    "triviaqa": "few_shot_learning",
    "samsum": "few_shot_learning",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code_completion",
    "repobench-p": "code_completion",
}


def _as_string_list(value, field_name: str) -> list[str]:
    """Validate one source Arrow list without relying on pandas CSV repr."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"source {field_name} must be a list of strings")
    return list(value)


def score_prediction(task: str, prediction: str, answers: list, all_classes: list) -> float:
    """Return the exact per-row LongBench score on the 0--100 scale."""

    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip().split("\n")[0]
    score = 0.0
    for answer in answers:
        score = max(
            score,
            float(dataset2metric[task](prediction.lstrip(), answer, all_classes=all_classes)),
        )
    return 100.0 * score
