# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from grkv.api import make_press
from grkv.coverage import summarize
from grkv.qualification import compare


def unit(task, rows, benchmark="longbench"):
    return {"item": dict(benchmark=benchmark, budget=10, task=task, rows=[dict(original_row_id=i) for i in rows])}


def test_macro_is_equal_task_weight_and_ruler_scale_is_applied_once():
    units = [unit("a", [0]), unit("b", [0, 1, 2]), unit("c", [0], "ruler16k")]
    values = [100, 0, 0.5]
    records = [
        dict(item=w["item"], value=dict(rows=[dict(**r, score=v) for r in w["item"]["rows"]]))
        for w, v in zip(units, values)
    ]
    result = summarize(records, units)
    assert result["cells"] == {"longbench_10": 50, "ruler16k_10": 50}


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "negative", "out_of_scale"])
def test_incomplete_duplicate_and_nonfinite_results_are_rejected(defect):
    units = [unit("a", [0, 1])]
    record = dict(
        item=units[0]["item"], value=dict(rows=[dict(original_row_id=0, score=0), dict(original_row_id=1, score=1)])
    )
    if defect == "missing":
        record["value"]["rows"].pop()
    elif defect == "duplicate":
        record["value"]["rows"][1]["original_row_id"] = 0
    elif defect == "nan":
        record["value"]["rows"][0]["score"] = float("nan")
    else:
        record["value"]["rows"][0]["score"] = -1 if defect == "negative" else 101
    with pytest.raises(ValueError):
        summarize([record], units)


@pytest.mark.parametrize("model", ["llama", "mistral"])
@pytest.mark.parametrize("budget", [10, 20])
def test_frozen_press_and_short_context_preserve_every_slot(model, budget):
    press = make_press(model, budget)
    assert press.grkv_guard == (0.1 if model == "llama" else 0)
    assert press.fit_config.key_relative_cap == 0.025
    assert press.fit_config.query_count == 32
    key = torch.randn(1, 8, 32, 128)
    value = torch.randn_like(key)
    module = SimpleNamespace(layer_idx=0)
    out = press.compress(module, None, key, value, None, {})
    assert out[0] is key and out[1] is value
    assert press.layer_records[-1]["kept"] == 256
    assert press.layer_records[-1]["compression_applied"] is False


def test_budget_uses_historical_floating_point_truncation():
    press = make_press("llama", 10)
    assert int(50 * (1 - press.compression_ratio)) == 4


def test_same_scores_do_not_prove_text_or_layer_identity():
    value: dict[str, Any] = dict(
        context_ids_sha256="c",
        question_ids_sha256=["q"],
        dataset_rows_sha256="d",
        layers=[dict(layer=i, kept=1) for i in range(32)],
        rows=[dict(original_row_id=0, prediction="yes", score=1)],
    )
    modified = deepcopy(value)
    modified["rows"][0]["prediction"] = "YES"
    assert compare(value, modified)["reasons"] == ["prediction"]
    modified = deepcopy(value)
    modified["layers"].pop()
    assert "layer_cardinality" in compare(value, modified)["reasons"]
    modified = deepcopy(value)
    modified["layers"][0]["fit"] = {"loss": 1}
    result = compare(value, modified)
    assert result["exact"] is False
    assert result["fit_mismatch_layers"] == [0]
    assert "fit_record" in result["reasons"]


def test_frozen_input_hash_rejects_changed_question():
    from experiments.critical_adakv_direct_grkv.run_wide_query_compatibility import group_digest
    from experiments.critical_adakv_direct_grkv.tune50_worker import load_group
    from experiments.critical_adakv_history_ruler.common import text_sha

    group = [dict(context="context", question="original", _id="id")]
    item = dict(
        benchmark="longbench",
        task="task",
        context_sha256=text_sha("context"),
        dataset_rows_sha256=group_digest(group),
        rows=[dict(original_row_id=0, context_sha256=text_sha("context"), example_id="id", ab="A")],
    )
    assert load_group(item, {("longbench", "task"): group}) == group
    group[0]["question"] = "changed"
    with pytest.raises(AssertionError):
        load_group(item, {("longbench", "task"): group})


def test_tampered_kernel_is_rejected_before_cuda_launch(tmp_path, monkeypatch):
    from experiments.critical_adakv_direct_grkv.wide_query_split_backend import VARIANT
    from experiments.critical_adakv_history_ruler.common import write
    from grkv.backend import HistoricalExecutableBackend

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    (tmp_path / "kernel.cubin").write_bytes(b"changed")
    write(
        tmp_path / "manifest.json",
        dict(
            status="complete",
            variant=VARIANT,
            target=dict(backend="cuda", arch=86, warp_size=32),
            kernels=[dict(query=32, kind="jvp", cubin=dict(path="kernel.cubin", sha256="0" * 64))],
        ),
    )
    with pytest.raises(ValueError, match="Published cubin hash mismatch"):
        HistoricalExecutableBackend(tmp_path / "manifest.json", tmp_path)


def test_resume_rejects_changed_asset_identity(tmp_path, monkeypatch):
    import json

    from experiments.critical_adakv_history_ruler.common import sha, write
    from grkv import run

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "code_identity", lambda: {})
    monkeypatch.setattr(run, "asset_identity", lambda model: {"model": "new"})
    for name in [
        "manifests/units.jsonl",
        "manifests/models.json",
        "manifests/datasets.json",
        "uv.lock",
        "kernels/manifest.json",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    config = dict(model="llama", budgets=[10, 20])
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps(config))
    identity = dict(
        config=config,
        stage="full",
        code_hashes={},
        units_sha256=sha(tmp_path / "manifests/units.jsonl"),
        models_manifest_sha256=sha(tmp_path / "manifests/models.json"),
        datasets_manifest_sha256=sha(tmp_path / "manifests/datasets.json"),
        lock_sha256=sha(tmp_path / "uv.lock"),
        kernel_manifest_sha256=sha(tmp_path / "kernels/manifest.json"),
        kernel_profile="historical-executable-sm86",
        asset_hashes={"model": "old"},
    )
    output = tmp_path / "output"
    write(output / "registration.json", dict(identity=identity))
    monkeypatch.setattr(
        "sys.argv",
        [
            "grkv.run",
            "--config",
            str(configuration),
            "--stage",
            "full",
            "--resume",
            "--output",
            str(output),
            "--kernel-root",
            str(tmp_path / "kernels"),
        ],
    )
    with pytest.raises(ValueError, match="Resume requires identical"):
        run.main()
