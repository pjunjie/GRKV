# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download pinned public resources; validate every file and frozen input group."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from huggingface_hub import hf_hub_download, snapshot_download

from experiments.critical_adakv_history_ruler.common import read, sha, timestamp, write
from grkv.settings import ROOT, asset_root


def fetch_models(names):
    manifest = read(ROOT / "manifests/models.json")
    for name in names:
        record = manifest[name]
        directory = asset_root() / "models" / name
        snapshot_download(
            record["repo_id"],
            revision=record["revision"],
            local_dir=str(directory),
            cache_dir=str(asset_root() / "hub"),
            allow_patterns=[entry["name"] for entry in record["files"]],
            max_workers=4,
        )
        for entry in record["files"]:
            path = directory / entry["name"]
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError(f"Model file mismatch: {name}/{entry['name']}")
        print(f"Verified model {name}: {record['revision']}", flush=True)


def fetch_datasets():
    records = read(ROOT / "manifests/datasets.json")
    for record in records:
        directory = asset_root() / "datasets" / record["name"]
        for filename in record["files"]:
            path = Path(
                hf_hub_download(
                    record["repo_id"],
                    filename,
                    repo_type="dataset",
                    revision=record["revision"],
                    local_dir=str(directory),
                    cache_dir=str(asset_root() / "hub"),
                )
            )
            expected = record.get("file_sha256", {}).get(filename)
            if expected and sha(path) != expected:
                raise ValueError(f"Dataset file mismatch: {record['name']}/{filename}")
        print(f"Fetched dataset {record['name']}: {record['revision']}", flush=True)
    from experiments.critical_adakv_direct_grkv.tune50_worker import load_group

    cache = {}
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    seen = set()
    for work in units:
        item = work["item"]
        identity = (item["benchmark"], item["task"], item["context_sha256"])
        if identity not in seen:
            load_group(item, cache)
            seen.add(identity)
    print(f"Verified {len(seen)} frozen context groups and all 10,250 source rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["all", "llama", "mistral", "none"], default="none")
    parser.add_argument("--datasets", choices=["all", "none"], default="none")
    args = parser.parse_args()
    begin = perf_counter()
    if args.models != "none":
        fetch_models(["llama", "mistral"] if args.models == "all" else [args.models])
    if args.datasets == "all":
        fetch_datasets()
    write(
        asset_root() / "download_receipt.json",
        dict(
            completed_at=timestamp(),
            seconds=perf_counter() - begin,
            models=args.models,
            datasets=args.datasets,
        ),
    )


if __name__ == "__main__":
    main()
