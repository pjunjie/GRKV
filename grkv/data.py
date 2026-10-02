# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load pinned, separately downloaded public parquet data."""

from functools import lru_cache

from datasets import Dataset

from grkv.settings import asset_root


@lru_cache(maxsize=17)
def _load_offline_test_split(repo_id, data_dir):
    name = {"Xnhyacinth/LongBench": "longbench", "simonjegou/ruler": "ruler"}[repo_id]
    path = asset_root() / "datasets" / name / data_dir / "test-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError("Missing pinned dataset; run scripts/fetch_assets.py --datasets all")
    return Dataset.from_parquet(str(path), cache_dir=str(asset_root() / "arrow"))
