# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from experiments.critical_adakv_direct_grkv.run_wide_query_compatibility import group_digest
from experiments.critical_adakv_history_ruler.common import row_identity, text_sha
from grkv import data as evaluation


def load_group(item, datasets):
    bench = item["benchmark"]
    assert bench in ("longbench", "ruler4k", "ruler16k")
    subset = item["task"] if bench == "longbench" else ("4096" if bench == "ruler4k" else "16384")
    key = (bench, subset)
    if key not in datasets:
        datasets[key] = evaluation._load_offline_test_split(
            "Xnhyacinth/LongBench" if bench == "longbench" else "simonjegou/ruler", subset
        )
    group = [dict(datasets[key][r["original_row_id"]]) for r in item["rows"]]
    for row, identity in zip(group, item["rows"]):
        assert text_sha(row["context"]) == item["context_sha256"] == identity["context_sha256"]
        if bench == "longbench":
            assert row["_id"] == identity["example_id"] and identity["ab"] in ("A", "B")
        else:
            assert row_identity(row, identity["original_row_id"]) == identity
    assert group_digest(group) == item["dataset_rows_sha256"]
    return group
