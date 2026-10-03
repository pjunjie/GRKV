# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a complete accepted result while retaining every generated record's origin."""

import argparse
import gzip
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from experiments.critical_adakv_history_ruler.common import read, sha, write
from grkv.qualification import compare
from grkv.run import unit_key
from grkv.score import read_records, rescore
from grkv.settings import ROOT


def consolidate(current, earlier, ledger, prior_proofs, kernel_root, destination, release_tag):
    partial = read(ledger)
    if (
        partial["model"] != "llama"
        or not partial["strict_observed_output_reproduction"]
        or not partial["independent_rescoring"]
        or partial["full_coverage_passed"]
        or partial["saved_score_mismatches"]
    ):
        raise ValueError("Require independently verified current Llama outputs with explicit coverage")
    current_asset = read(ROOT / "results/fresh_partial/llama/artifact.json")
    earlier_asset = next(
        asset
        for asset in read(ROOT / "results/reference/artifacts.json")["assets"]
        if asset["name"] == "llama_candidate.jsonl.gz"
    )
    for path, asset in ((current, current_asset), (earlier, earlier_asset)):
        if sha(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
            raise ValueError("Input archive identity differs from its verified public manifest")
    prior_receipt = read(ROOT / "results/validation/existing_full_evidence/llama.json")
    if not all(
        prior_receipt[key]
        for key in (
            "full_coverage_passed",
            "raw_source_hashes_verified",
            "self_contained_archive_matches_raw_sources",
            "independently_rescored_against_fresh_frozen_data",
            "target_scores_reproduced",
        )
    ):
        raise ValueError("Earlier full generation must have passed raw-source and independent score checks")
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    expected = {unit_key(work): work for work in units}
    missing = set(partial["missing_units"])
    proof_rows = [json.loads(line) for line in prior_proofs.read_text().splitlines()]
    proofs = {record["unit"]: record for record in proof_rows}
    if len(proofs) != len(proof_rows) or proofs.keys() != missing:
        raise ValueError("Earlier launch proofs must cover exactly the corresponding selected units")
    kernels = read(kernel_root / "manifest.json")["kernels"]
    historical_hashes = {entry["historical_cubin_sha256"] for entry in kernels}
    public_hashes = {entry["cubin"]["sha256"] for entry in kernels}
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "validated_llama_candidate.jsonl.gz"
    temporary = archive.with_suffix(archive.suffix + ".partial")
    stream_hash = hashlib.sha256()
    source_counts = dict(current_public_checkout=0, earlier_triton32_historical_kernels=0)
    source_units = {key: 0 for key in source_counts}
    current_seen = set()
    full_seen = set()
    with tempfile.TemporaryDirectory(prefix="grkv-consolidation-") as scratch:
        database = sqlite3.connect(str(Path(scratch) / "current.sqlite3"))
        database.execute("CREATE TABLE outputs (unit TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for record in read_records(current):
            key = record["unit"]
            if (
                key not in expected
                or key in missing
                or key in current_seen
                or record["item"] != expected[key]["item"]
                or record["prediction_source"] != "fresh_public_checkout"
                or record["registration_sha256"] != partial["registration_sha256"]
            ):
                raise ValueError("Current generation sample identity or provenance differs")
            checks = record["actual_binary_checks"]
            if not checks or any(
                not check["checked_before_launch"] or check["cubin_sha256"] not in public_hashes for check in checks
            ):
                raise ValueError("Current source is missing verified public kernel launch evidence")
            current_seen.add(key)
            database.execute("INSERT INTO outputs VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False)))
        database.commit()
        if current_seen != expected.keys() - missing:
            raise ValueError("Current archive differs from the selected frozen sample set")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                for old in read_records(earlier):
                    key = old["unit"]
                    if key not in expected or key in full_seen or old["item"] != expected[key]["item"]:
                        raise ValueError("Earlier full archive sample identity differs")
                    full_seen.add(key)
                    payload = database.execute("SELECT payload FROM outputs WHERE unit = ?", (key,)).fetchone()
                    if payload is not None:
                        selected = json.loads(payload[0])
                        origin = "current_public_checkout"
                        source_sha = selected["source_output_sha256"]
                        launch_checks = selected["actual_binary_checks"]
                        source_archive = current_asset
                        registration_sha = selected["registration_sha256"]
                    else:
                        selected = old
                        origin = "earlier_triton32_historical_kernels"
                        proof = proofs[key]
                        source_sha = old["source_sha256"]
                        launch_checks = proof["actual_binary_checks"]
                        source_archive = earlier_asset
                        registration_sha = None
                        if (
                            proof["source_sha256"] != source_sha
                            or not launch_checks
                            or any(
                                not check["checked_before_launch"] or check["cubin_sha256"] not in historical_hashes
                                for check in launch_checks
                            )
                        ):
                            raise ValueError("Earlier selected source lacks matching actual historical launch proof")
                    comparison = compare(old["value"], selected["value"])
                    if not comparison["exact"] or selected["value"].get("exact_mask_fallback_calls", 0):
                        raise ValueError(
                            "Selected prediction, score, layout or fit record differs from the frozen result"
                        )
                    source_counts[origin] += len(selected["value"]["rows"])
                    source_units[origin] += 1
                    record = dict(
                        unit=key,
                        item=old["item"],
                        scope=old["scope"],
                        value=selected["value"],
                        status="complete",
                        prediction_source="verified_consolidated_generation",
                        provenance=dict(
                            generation_origin=origin,
                            source_archive=source_archive["name"],
                            source_archive_sha256=source_archive["sha256"],
                            original_source_sha256=source_sha,
                            current_registration_sha256=registration_sha,
                            actual_binary_checks=launch_checks,
                        ),
                    )
                    encoded = (
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                    ).encode()
                    stream_hash.update(encoded)
                    zipped.write(encoded)
        database.close()
    if full_seen != expected.keys() or sum(source_counts.values()) != 20500:
        raise ValueError("Consolidated full coverage differs from the frozen protocol")
    if (
        source_counts["current_public_checkout"] != partial["answers"]
        or source_counts["earlier_triton32_historical_kernels"] != partial["missing_answers"]
    ):
        raise ValueError("Consolidated generation origin counts differ from the verified selection")
    scored = rescore(temporary, units)
    targets = read(ROOT / "results/reference/scores.json")
    candidate, baseline = targets["llama_candidate"], targets["llama_critical"]
    if (
        not scored["coverage_passed"]
        or scored["saved_score_mismatches"]
        or any(abs(scored["cells"][key] - value) > 1e-8 for key, value in candidate.items())
    ):
        raise ValueError("Independent consolidated full coverage or target scoring failed")
    gains = {key: scored["cells"][key] - value for key, value in baseline.items()}
    temporary.replace(archive)
    report = ROOT / "results/validated/llama"
    write(
        report / "verification.json",
        dict(
            **scored,
            scope="Complete accepted result consolidated from independently verified generated runs",
            target_scores_reproduced=True,
            target_tolerance=1e-8,
            strict_output_reproduction=True,
            prediction_mismatches=0,
            score_mismatches=0,
            fit_mismatch_layers=0,
            layout_mismatch_units=0,
            gains_against_historical_critical=gains,
            generated_entirely_in_current_checkout=False,
            source_answer_counts=source_counts,
            source_context_budget_unit_counts=source_units,
            per_unit_provenance_in_archive=True,
            original_cubin_bytes_identical=False,
            kernel_identity_scope="Public redacted package; earlier proofs retain historical source binary hashes",
        ),
    )
    artifact = dict(
        name=archive.name,
        bytes=archive.stat().st_size,
        sha256=sha(archive),
        decompressed_stream_sha256=stream_hash.hexdigest(),
        url=f"https://github.com/pjunjie/GRKV/releases/download/{release_tag}/{archive.name}",
        prediction_source="verified_consolidated_generation",
        answers=20500,
        context_budget_units=len(expected),
        full_coverage_passed=True,
        generated_entirely_in_current_checkout=False,
        source_answer_counts=source_counts,
        source_context_budget_unit_counts=source_units,
    )
    write(report / "artifact.json", artifact)
    print(json.dumps(artifact, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--coverage-ledger", type=Path, required=True)
    parser.add_argument("--prior-proofs", type=Path, required=True)
    parser.add_argument("--kernel-root", type=Path, default=Path("artifacts/kernels"))
    parser.add_argument("--destination", type=Path, default=Path("artifacts/validated"))
    parser.add_argument("--release-tag", default="v0.1.1-k1")
    args = parser.parse_args()
    consolidate(
        args.current,
        args.earlier,
        args.coverage_ledger,
        args.prior_proofs,
        args.kernel_root,
        args.destination,
        args.release_tag,
    )


if __name__ == "__main__":
    main()
