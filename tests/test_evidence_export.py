# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from importlib import import_module

import pytest

from experiments.critical_adakv_history_ruler.common import sha, write

consolidate_llama_results = import_module("scripts.consolidate_llama_results")
export_fresh_evidence = import_module("scripts.export_fresh_evidence")


@pytest.mark.parametrize("gains", [None, {"longbench_10": -1, "ruler16k_10": 0}])
def test_export_checks_worker_completion_without_requiring_baseline_improvement(tmp_path, monkeypatch, gains):
    run = tmp_path / "run"
    write(
        run / "registration.json",
        dict(
            source="fresh_public_checkout",
            stage="full",
            config=dict(model="llama", method="candidate"),
            code_hashes={},
            reference_predictions_required=False,
        ),
    )
    verification = dict(
        registration_sha256=sha(run / "registration.json"),
        independent_rescoring=True,
        coverage_passed=True,
        answers=20500,
        strict_output_reproduction=True,
        target_scores_reproduced=True,
        saved_score_mismatches=0,
        original_cubin_bytes_identical=False,
    )
    if gains is not None:
        verification["gains_against_historical_critical"] = gains
    write(run / "verification.json", verification)
    monkeypatch.setattr(export_fresh_evidence, "code_identity", lambda: {})
    with pytest.raises(ValueError, match="All inference workers must have completed successfully"):
        export_fresh_evidence.export(run, "llama", "a" * 40, tmp_path / "export")


@pytest.mark.parametrize("scores_match", [True, False])
def test_consolidation_uses_target_reproduction_and_preserves_negative_gains(tmp_path, monkeypatch, scores_match):
    module = consolidate_llama_results
    monkeypatch.setattr(module, "ROOT", tmp_path)
    item = dict(budget=10, benchmark="longbench", task="example", context_sha256="context")
    work = dict(item=item)
    key = module.unit_key(work)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/units.jsonl").write_text(json.dumps(work) + "\n")
    current, earlier = tmp_path / "current.jsonl.gz", tmp_path / "earlier.jsonl.gz"
    source_sha, historical_sha, public_sha = "a" * 64, "b" * 64, "c" * 64
    checks = [dict(checked_before_launch=True, cubin_sha256=historical_sha)]
    value = dict(
        context_ids_sha256="context",
        question_ids_sha256=["question"] * 20500,
        dataset_rows_sha256="dataset",
        layers=[dict(layer=i, kept=1) for i in range(32)],
        rows=[dict(original_row_id=i, prediction="answer", score=10) for i in range(20500)],
    )
    with gzip.open(current, "wt"):
        pass
    with gzip.open(earlier, "wt") as stream:
        stream.write(json.dumps(dict(unit=key, item=item, scope="full", value=value, source_sha256=source_sha)) + "\n")
    assets = {p.name: dict(name=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in (current, earlier)}
    write(tmp_path / "results/fresh_partial/llama/artifact.json", assets[current.name])
    write(
        tmp_path / "results/reference/artifacts.json",
        dict(assets=[dict(assets[earlier.name], name="llama_candidate.jsonl.gz")]),
    )
    write(
        tmp_path / "results/validation/existing_full_evidence/llama.json",
        dict(
            full_coverage_passed=True,
            raw_source_hashes_verified=True,
            self_contained_archive_matches_raw_sources=True,
            independently_rescored_against_fresh_frozen_data=True,
            target_scores_reproduced=True,
        ),
    )
    ledger = tmp_path / "ledger.json"
    write(
        ledger,
        dict(
            model="llama",
            strict_observed_output_reproduction=True,
            independent_rescoring=True,
            full_coverage_passed=False,
            saved_score_mismatches=0,
            missing_units=[key],
            answers=0,
            missing_answers=20500,
            registration_sha256="registration",
        ),
    )
    proofs = tmp_path / "proofs.jsonl"
    proofs.write_text(json.dumps(dict(unit=key, source_sha256=source_sha, actual_binary_checks=checks)) + "\n")
    kernels = tmp_path / "kernels"
    write(
        kernels / "manifest.json",
        dict(kernels=[dict(historical_cubin_sha256=historical_sha, cubin=dict(sha256=public_sha))]),
    )
    cells = {key: 10 for key in ("longbench_10", "longbench_20", "ruler16k_10", "ruler16k_20")}
    targets = {key: 10 if scores_match else 9 for key in cells}
    baseline = {key: 11 for key in cells}
    write(tmp_path / "results/reference/scores.json", dict(llama_candidate=targets, llama_critical=baseline))
    monkeypatch.setattr(
        module,
        "rescore",
        lambda *args: dict(coverage_passed=True, saved_score_mismatches=0, answers=20500, cells=cells),
    )
    destination = tmp_path / "accepted"
    arguments = (current, earlier, ledger, proofs, kernels, destination, "test-release")
    if not scores_match:
        with pytest.raises(ValueError, match="Independent consolidated full coverage or target scoring failed"):
            module.consolidate(*arguments)
        assert not (destination / "validated_llama_candidate.jsonl.gz").exists()
        return
    module.consolidate(*arguments)
    verification = json.loads((tmp_path / "results/validated/llama/verification.json").read_text())
    assert verification["target_scores_reproduced"] is True
    assert verification["gains_against_historical_critical"] == {key: -1 for key in cells}
    assert verification["source_answer_counts"] == dict(
        current_public_checkout=0, earlier_triton32_historical_kernels=20500
    )
    assert (destination / "validated_llama_candidate.jsonl.gz").exists()
