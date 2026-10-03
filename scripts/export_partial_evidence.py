# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export genuinely generated partial evidence with an explicit missing-sample ledger."""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

from experiments.critical_adakv_history_ruler.common import read, sha, write
from grkv.run import code_identity, unit_key
from grkv.score import rescore
from grkv.settings import ROOT


def export(run, verification, source_commit, destination):
    registration = read(run / "registration.json")
    report = read(verification)
    if (
        report["full_coverage_passed"]
        or not report["subset_coverage_passed"]
        or not report["independent_rescoring"]
        or not report["strict_observed_output_reproduction"]
        or report["full_target_scores_reproduced"]
        or report["saved_score_mismatches"]
        or report["differences"]
        or report["original_cubin_bytes_identical"]
        or report["registration_sha256"] != sha(run / "registration.json")
        or registration["code_hashes"] != code_identity()
        or not re.fullmatch(r"[a-f0-9]{40}", source_commit)
    ):
        raise ValueError("Require verified genuine partial generation; no full result can be inferred")
    for name, digest in registration["code_hashes"].items():
        content = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Generation commit numerical source differs from registration")
    files = sorted((run / "contexts").glob("*.json"))
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    present = {file.stem for file in files}
    observed = [work for work in units if unit_key(work) in present]
    if len(files) != report["context_budget_units"] or len(observed) != len(files):
        raise ValueError("Partial source coverage changed since verification")
    model = report["model"]
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"fresh_{model}_candidate_partial.jsonl.gz"
    temporary = archive.with_suffix(archive.suffix + ".partial")
    stream_hash = hashlib.sha256()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            for file in files:
                record = read(file)
                if record["prediction_source"] != "fresh_public_checkout":
                    raise ValueError("Partial source is not current fresh generation")
                record["source_output_sha256"] = sha(file)
                encoded = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
                zipped.write(encoded)
                stream_hash.update(encoded)
    rescored = rescore(temporary, observed)
    if (
        rescored["saved_score_mismatches"]
        or rescored["answers"] != report["answers"]
        or rescored["task_counts"] != report["observed_task_counts"]
        or rescored["task_scores"] != report["observed_task_scores"]
    ):
        raise ValueError("Independent exported-subset rescoring differs")
    temporary.replace(archive)
    artifact = dict(
        name=archive.name,
        bytes=archive.stat().st_size,
        sha256=sha(archive),
        decompressed_stream_sha256=stream_hash.hexdigest(),
        url=f"https://github.com/pjunjie/GRKV/releases/download/v0.2.0/{archive.name}",
        prediction_source="fresh_public_checkout",
        generation_checkout_commit=source_commit,
        registration_sha256=report["registration_sha256"],
        context_budget_units=report["context_budget_units"],
        answers=report["answers"],
        missing_context_budget_units=report["missing_context_budget_units"],
        missing_answers=report["missing_answers"],
        full_coverage_passed=False,
        full_benchmark_result=False,
        scope="Only current generated answers; no missing answers are filled from references",
    )
    target = ROOT / "results/fresh_partial" / model
    write(target / "artifact.json", artifact)
    subset_passed = rescored.pop("coverage_passed")
    write(
        target / "exported_archive_rescore.json",
        dict(
            scope="Observed subset only", full_coverage_passed=False, subset_coverage_passed=subset_passed, **rescored
        ),
    )
    print(json.dumps(artifact, indent=2))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--destination", type=Path, default=Path("artifacts/fresh"))
    args = parser.parse_args()
    export(args.run, args.verification, args.source_commit, args.destination)


if __name__ == "__main__":
    main()
