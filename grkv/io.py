# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def row_identity(row, index):
    value = {k: str(row[k]) for k in ("task", "context", "question", "answer_prefix")}
    value.update(answer=[str(x) for x in row["answer"]], max_new_tokens=int(row["max_new_tokens"]))
    return dict(
        original_row_id=index,
        task=value["task"],
        context_sha256=text_sha(value["context"]),
        row_sha256=text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True)),
        max_new_tokens=value["max_new_tokens"],
    )


def sample_score(task, prediction, answers):
    prediction = re.sub(r"[\x00-\x1f]", "", prediction.strip()).strip().lower()
    matches = [float(str(answer).lower() in prediction) for answer in answers]
    if not matches:
        raise ValueError("RULER reference list is empty")
    return max(matches) if task.split("_")[0] == "qa" else sum(matches) / len(matches)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def group_digest(group):
    return hashlib.sha256(json.dumps(group, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def replay_layer(layer):
    """Exclude only measured wall times; retain all numerical and layout checks."""
    timing = {"selection_host_seconds", "fit_host_seconds", "target_host_seconds"}
    return {k: v for k, v in layer.items() if k not in timing}
