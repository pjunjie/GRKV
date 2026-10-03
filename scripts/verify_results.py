# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare newly generated predictions, independently rescored targets and layers."""

import argparse
import json
from pathlib import Path

from experiments.critical_adakv_history_ruler.common import read, sha, write
from grkv.qualification import compare
from grkv.run import unit_key
from grkv.score import read_records, rescore
from grkv.settings import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["llama", "mistral"])
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--kernel-root", type=Path, default=Path("artifacts/kernels"))
    args = parser.parse_args()
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    if args.stage == "smoke":
        selected = set(read(ROOT / "manifests/smoke.json")["units"])
        units = [w for w in units if unit_key(w) in selected]
    wanted = {unit_key(w) for w in units}
    registration = read(args.run / "registration.json")
    registration_sha = sha(args.run / "registration.json")
    if registration["config"]["model"] != args.model or registration["stage"] != args.stage:
        raise ValueError("Verification model/stage differs from the run")
    manifest_path = args.kernel_root / "manifest.json"
    manifest = read(manifest_path)
    if sha(manifest_path) != registration["kernel_manifest_sha256"]:
        raise ValueError("Kernel manifest differs from run registration")
    expected = {(x["query"] * 4, x["kind"]): x["cubin"]["sha256"] for x in manifest["kernels"]}
    if registration["source"] != "fresh_public_checkout" or registration["config"]["method"] != "candidate":
        raise ValueError("Verification requires a fresh candidate registration")
    summary = rescore(args.run, units)
    differences = []
    text = scores = fits = fallback = 0
    fresh_fallback = historical_fallback = 0
    seen = set()
    for old in read_records(args.reference):
        key = old["unit"]
        if key not in wanted:
            continue
        if key in seen:
            raise ValueError("Duplicate reference unit")
        seen.add(key)
        record = read(args.run / "contexts" / (key + ".json"))
        if record["registration_sha256"] != registration_sha or record["prediction_source"] != "fresh_public_checkout":
            raise ValueError("Candidate source/registration differs")
        new = record["value"]
        if not record["actual_binary_checks"] or not all(
            x["checked_before_launch"] and x["arch"] == 86 and x["executable_sections_identical"]
            for x in record["actual_binary_checks"]
        ):
            raise ValueError("Missing real kernel qualification/launch proof")
        if any(expected[(x["rows"], x["kind"])] != x["cubin_sha256"] for x in record["actual_binary_checks"]):
            raise ValueError("Unexpected actual cubin hash")
        result = compare(old["value"], new)
        text += sum(not row["prediction_exact"] for row in result["rows"])
        scores += sum(not row["score_exact"] for row in result["rows"])
        fits += len(result["fit_mismatch_layers"])
        a = old["value"].get("exact_mask_fallback_calls", 0)
        b = new["exact_mask_fallback_calls"]
        historical_fallback += a
        fresh_fallback += b
        fallback += a != b
        if not result["exact"] or a != b:
            differences.append(dict(unit=key, comparison=result, historical_fallback_calls=a, fresh_fallback_calls=b))
    if seen != wanted:
        raise ValueError("Reference coverage mismatch")
    targets = read(ROOT / "results/reference/scores.json")
    candidate = targets[args.model + "_candidate"]
    baseline = targets[args.model + "_critical"]
    target_pass = args.stage == "full" and all(abs(summary["cells"][k] - v) <= 1e-8 for k, v in candidate.items())
    gains = {k: summary["cells"][k] - baseline[k] for k in candidate} if args.stage == "full" else None
    result = dict(
        **summary,
        target_scores_reproduced=target_pass,
        strict_output_reproduction=not differences and not summary["saved_score_mismatches"],
        original_cubin_bytes_identical=False,
        prediction_mismatches=text,
        score_mismatches=scores,
        fit_mismatch_layers=fits,
        fallback_mismatch_units=fallback,
        fresh_exact_mask_fallback_calls=fresh_fallback,
        reference_exact_mask_fallback_calls=historical_fallback,
        gains_against_historical_critical=gains,
        reference_sha256=sha(args.reference),
        registration_sha256=registration_sha,
        differences=differences,
    )
    write(args.run / "verification.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["differences", "task_scores", "task_counts"]}, indent=2
        )
    )
    if args.strict and (not result["strict_output_reproduction"] or (args.stage == "full" and not target_pass)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
