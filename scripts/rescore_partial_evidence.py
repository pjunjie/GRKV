# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently score a downloaded subset against its explicit missing-unit ledger."""

import argparse
import json
from pathlib import Path

from experiments.critical_adakv_history_ruler.common import read, write
from grkv.run import unit_key
from grkv.score import rescore
from grkv.settings import ROOT


def verify(archive, ledger):
    coverage = read(ledger)
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    expected = {unit_key(work) for work in units}
    missing = set(coverage["missing_units"])
    if (
        coverage["full_coverage_passed"]
        or not coverage["subset_coverage_passed"]
        or not missing
        or not missing < expected
        or len(missing) != coverage["missing_context_budget_units"]
    ):
        raise ValueError("Require an explicitly incomplete frozen sample ledger")
    selected = [work for work in units if unit_key(work) not in missing]
    score = rescore(archive, selected)
    if (
        not score["coverage_passed"]
        or score["saved_score_mismatches"]
        or score["answers"] != coverage["answers"]
        or len(selected) != coverage["context_budget_units"]
        or score["cells"] != coverage["observed_subset_cells"]
        or score["task_scores"] != coverage["observed_task_scores"]
        or score["task_counts"] != coverage["observed_task_counts"]
    ):
        raise ValueError("Downloaded subset coverage or independently computed scores differ")
    passed = score.pop("coverage_passed")
    return dict(
        scope="Downloaded observed subset only; this is rescoring, not new inference",
        full_coverage_passed=False,
        subset_coverage_passed=passed,
        full_benchmark_result=False,
        missing_answers=coverage["missing_answers"],
        missing_context_budget_units=len(missing),
        **score,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--coverage-ledger", type=Path, default=Path("results/fresh_partial/llama/verification.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input, args.coverage_ledger)
    write(args.output, result)
    print(f"Independently rescored {result['answers']} subset answers; {result['missing_answers']} remain missing.")


if __name__ == "__main__":
    main()
