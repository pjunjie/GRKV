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
