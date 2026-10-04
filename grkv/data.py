# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load pinned, separately downloaded public parquet data."""

from functools import lru_cache

from datasets import Dataset

from grkv.io import group_digest, row_identity, text_sha
from grkv.settings import asset_root


@lru_cache(maxsize=17)
def _load_offline_test_split(repo_id, data_dir):
    name = {"Xnhyacinth/LongBench": "longbench", "simonjegou/ruler": "ruler"}[repo_id]
    path = asset_root() / "datasets" / name / data_dir / "test-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError("Missing pinned dataset; run scripts/fetch_assets.py --datasets all")
    return Dataset.from_parquet(str(path), cache_dir=str(asset_root() / "arrow"))


def load_group(item, datasets):
    bench = item["benchmark"]
    assert bench in ("longbench", "ruler16k")
    subset = item["task"] if bench == "longbench" else "16384"
    key = (bench, subset)
    if key not in datasets:
        datasets[key] = _load_offline_test_split(
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
