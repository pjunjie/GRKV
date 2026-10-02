# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from experiments.critical_adakv_grkv.press import tensor_sha
from experiments.critical_adakv_history_ruler.common import sample_score
from experiments.defensive_grkv_selector.fit_selector import score_prediction
from experiments.defensive_grkv_selector.reference_compat import classes_for_scoring


def score(item, row, prediction):
    if item["benchmark"] == "longbench":
        return score_prediction(
            item["task"], prediction, row["answers"], classes_for_scoring(item["task"], row["all_classes"])
        )
    return sample_score(item["task"], prediction, row["answer"])


def verify_prefill_layout(model, cache, layers):
    """Bind actual installed decode masks, including explicit all-kept short inputs."""
    assert len(cache.layers) == len(layers) == len(model.model.layers)
    for module, cached, record in zip(model.model.layers, cache.layers, layers):
        keep = torch.ones(cached.keys.shape[:3], device=cached.keys.device, dtype=torch.bool)
        keep[module.self_attn.masked_key_indices] = False
        digest = tensor_sha(keep)
        if "keep_sha256" not in record:
            assert record["context_tokens"] <= 32 and record["compression_applied"] is False
            assert bool(keep.all())
            record["keep_sha256"] = digest
        assert record["keep_sha256"] == digest
        assert record["kept_per_head"] == keep.sum(-1).flatten().tolist()
        assert record["kept"] == int(keep.sum())
        record["installed_decode_mask_verified"] = True
