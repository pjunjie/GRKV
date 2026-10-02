# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export completed new inference without logs, caches or machine paths."""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from experiments.critical_adakv_history_ruler.common import read, sha, write
from grkv.run import code_identity, unit_key
from grkv.score import rescore
from grkv.settings import ROOT


def export(run, model, source_commit, destination):
    registration = read(run / "registration.json")
    registration_sha = sha(run / "registration.json")
    verification = read(run / "verification.json")
    if (
        registration["source"] != "fresh_public_checkout"
        or registration["stage"] != "full"
        or registration["config"]["model"] != model
        or registration["config"]["method"] != "candidate"
        or registration["code_hashes"] != code_identity()
        or registration["reference_predictions_required"]
    ):
        raise ValueError("Export requires unchanged full candidate generation code and registration")
    if not re.fullmatch(r"[a-f0-9]{40}", source_commit):
        raise ValueError("Supply the full generation checkout commit")
    for name, digest in registration["code_hashes"].items():
        content = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Generation commit source differs from the registered numerical source")
    if (
        verification["registration_sha256"] != registration_sha
        or not verification["independent_rescoring"]
        or not verification["coverage_passed"]
        or verification["answers"] != 20500
        or not verification["strict_output_reproduction"]
        or not verification["target_scores_reproduced"]
        or not verification["all_four_point_gains_positive"]
        or verification["saved_score_mismatches"]
        or verification["original_cubin_bytes_identical"]
    ):
        raise ValueError("Export requires complete independent rescoring and strict target/output verification")
    jobs = [read(p) for p in sorted((run / "jobs").glob("*.json"))]
    if not jobs or any(p["status"] != "complete" for p in jobs):
        raise ValueError("All inference workers must have completed successfully")
    if sum(p["completed_units"] for p in jobs) != 19630 or sum(p["answers"] for p in jobs) != 20500:
        raise ValueError("Worker coverage differs from the frozen full protocol")
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    expected = {unit_key(work): work for work in units}
    files = sorted((run / "contexts").glob("*.json"))
    if {p.stem for p in files} != set(expected):
        raise ValueError("Output context coverage differs from the full frozen sample index")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"fresh_{model}_candidate.jsonl.gz"
    partial = archive.with_suffix(archive.suffix + ".partial")
    stream_hash = hashlib.sha256()
    seconds = peak_allocated = peak_reserved = 0.0
    fallback_calls = fallback_units = answers = 0
    kernel_checks: dict[tuple[int, str], dict] = {}
    with partial.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            for file in files:
                record = read(file)
                work = expected[file.stem]
                if (
                    record["unit"] != file.stem
                    or record["item"] != work["item"]
                    or record["status"] != "complete"
                    or record["registration_sha256"] != registration_sha
                    or record["prediction_source"] != "fresh_public_checkout"
                    or not record["actual_binary_checks"]
                    or any(r["prediction_source"] != "fresh_public_checkout" for r in record["value"]["rows"])
                ):
                    raise ValueError("Context provenance or source identity differs")
                record["source_output_sha256"] = sha(file)
                for check in record["actual_binary_checks"]:
                    key = (check["rows"], check["kind"])
                    if key in kernel_checks and kernel_checks[key] != check:
                        raise ValueError("Kernel qualification differs between contexts")
                    kernel_checks[key] = check
                encoded = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
                stream_hash.update(encoded)
                zipped.write(encoded)
                value = record["value"]
                seconds += record["seconds"]
                peak_allocated = max(peak_allocated, value["peak_allocated_mib"])
                peak_reserved = max(peak_reserved, value["peak_reserved_mib"])
                fallback_calls += value["exact_mask_fallback_calls"]
                fallback_units += value["exact_mask_fallback_calls"] > 0
                answers += len(value["rows"])
    if answers != 20500 or fallback_calls != verification["fresh_exact_mask_fallback_calls"]:
        raise ValueError("Export answer/fallback totals differ from strict verification")
    exported_scores = rescore(partial, units)
    if (
        not exported_scores["coverage_passed"]
        or exported_scores["saved_score_mismatches"]
        or exported_scores["cells"] != verification["cells"]
        or exported_scores["task_scores"] != verification["task_scores"]
    ):
        raise ValueError("Independent rescoring of the exported archive differs from the completed run")
    partial.replace(archive)
    starts = [datetime.fromisoformat(p["started_at"]) for p in jobs]
    finishes = [datetime.fromisoformat(p["finished_at"]) for p in jobs]
    worker_seconds = [(b - a).total_seconds() for a, b in zip(starts, finishes)]
    resources = dict(
        model=model,
        generation_checkout_commit=source_commit,
        registration_sha256=registration_sha,
        context_budget_units=len(files),
        answers=answers,
        gpu_workers=len(jobs),
        worker_elapsed_seconds=worker_seconds,
        wall_seconds=(max(finishes) - min(starts)).total_seconds(),
        gpu_worker_hours=sum(worker_seconds) / 3600,
        sum_generation_seconds=seconds,
        peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,
        exact_mask_fallback_calls=fallback_calls,
        exact_mask_fallback_units=fallback_units,
        formal_performance_benchmark=False,
        timing_scope=(
            "Worker elapsed time includes model loading and qualification; per-unit timing excludes preprocessing"
        ),
        peak_scope="Maximum of measured per-unit CUDA allocator peaks; excludes non-PyTorch device allocations",
        runtimes=[p["runtime"] for p in jobs],
        derivative_qualifications=[p["qualification"] for p in jobs],
        qualified_and_inference_loaded_kernel_variants=[kernel_checks[k] for k in sorted(kernel_checks)],
        kernel_record_scope="Cumulative first-load checks include derivative qualification; not a CUDA call count",
        original_cubin_bytes_identical=False,
    )
    report = ROOT / "results/fresh" / model
    write(report / "registration.json", registration)
    if sha(report / "registration.json") != registration_sha:
        raise ValueError("Copied registration serialization differs")
    write(report / "verification.json", verification)
    write(report / "resources.json", resources)
    write(report / "exported_archive_rescore.json", exported_scores)
    asset = dict(
        name=archive.name,
        bytes=archive.stat().st_size,
        sha256=sha(archive),
        decompressed_stream_sha256=stream_hash.hexdigest(),
        url=f"https://github.com/pjunjie/GRKV/releases/download/v0.1.0-k1/{archive.name}",
        prediction_source="fresh_public_checkout",
        generation_checkout_commit=source_commit,
        registration_sha256=registration_sha,
        context_budget_units=len(files),
        answers=answers,
    )
    write(report / "artifact.json", asset)
    print(json.dumps(asset, indent=2))
    return asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=["llama", "mistral"], required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--destination", type=Path, default=Path("artifacts/fresh"))
    args = parser.parse_args()
    export(args.run, args.model, args.source_commit, args.destination)


if __name__ == "__main__":
    main()
