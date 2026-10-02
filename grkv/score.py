# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently rescore predictions against freshly downloaded answers."""

import argparse
import gzip
import json
from pathlib import Path

from experiments.critical_adakv_direct_grkv.tune50_worker import load_group
from experiments.critical_adakv_direct_grkv.wide_query_worker_base import score
from experiments.critical_adakv_history_ruler.common import read, write
from grkv.coverage import summarize
from grkv.settings import ROOT


def read_records(path):
    path = Path(path)
    if path.is_dir():
        for file in sorted((path / "contexts").glob("*.json")):
            record = read(file)
            if record["status"] != "complete":
                raise ValueError("Incomplete output context")
            yield record
    else:
        with gzip.open(path, "rt", encoding="utf8") as stream:
            for line in stream:
                yield json.loads(line)


def rescore(path, units):
    cache: dict = {}
    mismatches = []

    def records():
        for record in read_records(path):
            item, value = record["item"], record["value"]
            group = load_group(item, cache)
            if len(value["rows"]) != len(group):
                raise ValueError("Reference answer cardinality differs")
            for row, expected, source in zip(value["rows"], item["rows"], group):
                if row["original_row_id"] != expected["original_row_id"]:
                    raise ValueError("Answer order/identity mismatch")
                computed = score(item, source, row["prediction"])
                if "score" in row and computed != row["score"]:
                    mismatches.append(dict(unit=record["unit"], row_id=row["original_row_id"]))
                row["score"] = computed
            yield record

    result = summarize(records(), units)
    result.update(independent_rescoring=True, saved_score_mismatches=mismatches)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    if args.stage == "smoke":
        from grkv.run import unit_key

        selected = set(read(ROOT / "manifests/smoke.json")["units"])
        units = [w for w in units if unit_key(w) in selected]
    write(args.output, rescore(args.input, units))


if __name__ == "__main__":
    main()
