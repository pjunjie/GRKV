# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rescore and compare existing outputs without claiming missing answers were generated."""

import argparse
import json
from pathlib import Path

from experiments.critical_adakv_history_ruler.common import read, sha, write
from grkv.qualification import compare
from grkv.run import code_identity, unit_key
from grkv.score import read_records, rescore
from grkv.settings import ROOT


def verify(run, reference, model, kernel_root):
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    expected = {unit_key(work): work for work in units}
    files = {p.stem: p for p in (run / "contexts").glob("*.json")}
    if not files or not files.keys() <= expected.keys():
        raise ValueError("Require existing outputs from the frozen full sample index")
    registration = read(run / "registration.json")
    registration_sha = sha(run / "registration.json")
    if (
        registration["source"] != "fresh_public_checkout"
        or registration["stage"] != "full"
        or registration["config"]["model"] != model
        or registration["config"]["method"] != "candidate"
        or registration["code_hashes"] != code_identity()
        or registration["reference_predictions_required"]
    ):
        raise ValueError("Partial verification requires unchanged fresh candidate code/registration")
    manifest_path = kernel_root / "manifest.json"
    if sha(manifest_path) != registration["kernel_manifest_sha256"]:
        raise ValueError("Kernel manifest differs from registration")
    kernels = {
        (entry["query"] * 4, entry["kind"]): entry["cubin"]["sha256"] for entry in read(manifest_path)["kernels"]
    }
    observed = [work for work in units if unit_key(work) in files]
    scored = rescore(run, observed)
    seen = set()
    differences = []
    prediction_mismatches = score_mismatches = fit_mismatches = fallback_mismatches = 0
    fallback_calls = 0
    for old in read_records(reference):
        key = old["unit"]
        if key not in expected or key in seen or old["item"] != expected[key]["item"]:
            raise ValueError("Full reference sample identity or coverage differs")
        seen.add(key)
        if key not in files:
            continue
        record = read(files[key])
        if (
            record["status"] != "complete"
            or record["unit"] != key
            or record["item"] != expected[key]["item"]
            or record["registration_sha256"] != registration_sha
            or record["prediction_source"] != "fresh_public_checkout"
            or any(row["prediction_source"] != "fresh_public_checkout" for row in record["value"]["rows"])
        ):
            raise ValueError("Existing output provenance differs")
        checks = record["actual_binary_checks"]
        if not checks or any(
            not check["checked_before_launch"]
            or check["arch"] != 86
            or not check["executable_sections_identical"]
            or check["original_cubin_bytes_identical"]
            or kernels[(check["rows"], check["kind"])] != check["cubin_sha256"]
            for check in checks
        ):
            raise ValueError("Existing output kernel launch proof differs")
        comparison = compare(old["value"], record["value"])
        prediction_mismatches += sum(not row["prediction_exact"] for row in comparison["rows"])
        score_mismatches += sum(not row["score_exact"] for row in comparison["rows"])
        fit_mismatches += len(comparison["fit_mismatch_layers"])
        fallback = record["value"]["exact_mask_fallback_calls"]
        fallback_calls += fallback
        fallback_differs = fallback != old["value"].get("exact_mask_fallback_calls", 0)
        fallback_mismatches += fallback_differs
        if not comparison["exact"] or fallback_differs:
            differences.append(dict(unit=key, comparison=comparison, fallback_differs=fallback_differs))
    if seen != expected.keys():
        raise ValueError("Full reference has missing frozen units")
    missing = sorted(expected.keys() - files.keys())
    expected_answers = sum(len(work["item"]["rows"]) for work in units)
    result = dict(
        model=model,
        scope="Observed existing fresh outputs only; missing units are neither filled nor counted",
        registration_sha256=registration_sha,
        expected_context_budget_units=len(units),
        context_budget_units=len(files),
        expected_answers=expected_answers,
        answers=scored["answers"],
        missing_context_budget_units=len(missing),
        missing_answers=expected_answers - scored["answers"],
        missing_units=missing,
        full_coverage_passed=not missing,
        subset_coverage_passed=scored["coverage_passed"],
        independent_rescoring=True,
        saved_score_mismatches=scored["saved_score_mismatches"],
        observed_subset_cells=scored["cells"],
        observed_task_scores=scored["task_scores"],
        observed_task_counts=scored["task_counts"],
        full_target_scores_reproduced=False,
        strict_observed_output_reproduction=not differences and not scored["saved_score_mismatches"],
        prediction_mismatches=prediction_mismatches,
        score_mismatches=score_mismatches,
        fit_mismatch_layers=fit_mismatches,
        fallback_mismatch_units=fallback_mismatches,
        observed_exact_mask_fallback_calls=fallback_calls,
        differences=differences,
        original_cubin_bytes_identical=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", choices=["llama", "mistral"], required=True)
    parser.add_argument("--kernel-root", type=Path, default=Path("artifacts/kernels"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.reference, args.model, args.kernel_root)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("missing_units", "differences")}, indent=2))
    if not result["strict_observed_output_reproduction"]:
        raise SystemExit("Observed output comparison failed; no full completion is inferred")


if __name__ == "__main__":
    main()
