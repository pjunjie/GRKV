# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from grkv.io import replay_layer, tensor_sha


def qualify(backend):
    """Run real JVP/VJP through unchanged pre-launch guards."""
    from grkv import kernels

    h, r, s, d = 8, 128, 256, 128
    q = torch.randn(h, r, d, device="cuda", dtype=torch.float32)
    v = torch.randn(h, s, d, device="cuda", dtype=torch.float32)
    w = torch.randn(h, r, s, device="cuda", dtype=torch.float32).softmax(-1).contiguous()
    o = torch.bmm(w, v).contiguous()
    delta = torch.randn_like(v)
    u = torch.randn_like(q)
    mutable = torch.ones(h, s, device="cuda", dtype=torch.bool)
    partial_j = torch.empty(h, 4, r, d, device="cuda", dtype=torch.float32)
    partial_v = torch.empty(h, 2, s, d, device="cuda", dtype=torch.float32)
    with backend():
        kernels._wide_jvp[(h, 4, 4)](
            q, v, w, o, delta, mutable, partial_j, R=r, S=s, D=d, TILES=4, BR=32, BS=64, BD=d, num_warps=4, num_stages=1
        )
        kernels._wide_vjp[(h, 8, 2)](
            q, v, w, o, u, mutable, partial_v, R=r, S=s, D=d, TR=2, BR=64, BS=32, BD=d, num_warps=8, num_stages=1
        )
    torch.cuda.synchronize()
    if not bool(torch.isfinite(partial_j).all() and torch.isfinite(partial_v).all()):
        raise ValueError("Nonfinite synthetic derivative outputs")
    if {(x["rows"], x["kind"]) for x in backend.launch_records} != {(128, "jvp"), (128, "vjp")}:
        raise ValueError("Both actual kernels were not checked")
    return dict(
        actual_binary_checks=backend.launch_records,
        jvp_sha256=tensor_sha(partial_j),
        vjp_sha256=tensor_sha(partial_v),
        jvp_norm=float(partial_j.norm()),
        vjp_norm=float(partial_v.norm()),
    )


def compare(old, new):
    """Require cardinality and row identities before comparing values."""
    reasons = []
    for name in ("context_ids_sha256", "question_ids_sha256", "dataset_rows_sha256"):
        if old[name] != new[name]:
            reasons.append(name)
    if len(old["layers"]) != len(new["layers"]) or len(old["layers"]) != 32:
        reasons.append("layer_cardinality")
    layout_fields = (
        "layer",
        "context_tokens",
        "keep_sha256",
        "kept",
        "kept_per_head",
        "source_positions",
        "virtual_position",
        "student_layout",
        "fixed",
        "regression_slots",
    )
    layout_mismatches, fit_mismatches, rows = [], [], []
    for a, b in zip(old["layers"], new["layers"]):
        changes = [name for name in layout_fields if a.get(name) != b.get(name)]
        if changes:
            layout_mismatches.append(dict(layer=a["layer"], fields=changes))
        if a.get("fit", {}).get("config") != b.get("fit", {}).get("config"):
            reasons.append("fit_config")
        for name in ("student_layout", "total_query_weight", "query_count"):
            if a.get("fit", {}).get(name) != b.get("fit", {}).get(name):
                reasons.append("fit_" + name)
        if replay_layer(a) != replay_layer(b):
            fit_mismatches.append(a["layer"])
    if layout_mismatches:
        reasons.append("layout")
    if fit_mismatches:
        reasons.append("fit_record")
    if len(old["rows"]) != len(new["rows"]):
        reasons.append("row_cardinality")
    for a, b in zip(old["rows"], new["rows"]):
        if a["original_row_id"] != b["original_row_id"]:
            reasons.append("row_identity")
        rows.append(
            dict(
                original_row_id=a["original_row_id"],
                prediction_exact=a["prediction"] == b["prediction"],
                score_exact=a["score"] == b["score"],
                historical_score=a["score"],
                score=b["score"],
            )
        )
    if not all(x["prediction_exact"] for x in rows):
        reasons.append("prediction")
    if not all(x["score_exact"] for x in rows):
        reasons.append("score")
    return dict(
        exact=not reasons,
        reasons=sorted(set(reasons)),
        layout_mismatches=layout_mismatches,
        fit_mismatch_layers=fit_mismatches,
        rows=rows,
    )
