# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen prefill, shared-context generation and installed-mask verification."""

from contextlib import nullcontext
from time import perf_counter

import torch
from transformers import DynamicCache

from grkv.io import group_digest, tensor_sha
from grkv.scoring import score


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


@torch.inference_mode()
def generate(pipeline, press, backend, item, group, tensors):
    ids = tensors["context_ids"].to(pipeline.model.device)
    cache = DynamicCache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = perf_counter()
    with backend() if backend is not None else nullcontext():
        with press(pipeline.model):
            pipeline.model.model(ids, past_key_values=cache)
    torch.cuda.synchronize()
    prefill_seconds = perf_counter() - begin
    records = press.layer_records
    verify_prefill_layout(pipeline.model, cache, records)
    cache_bytes = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
    rows = []
    for row, identity, question in zip(group, item["rows"], tensors["questions_ids"]):
        prediction = pipeline.generate_answer(
            question.to(pipeline.model.device), cache, ids.shape[1], int(row["max_new_tokens"])
        )
        rows.append(
            dict(
                **identity,
                prediction=prediction,
                score=score(item, row, prediction),
                prediction_source="fresh_independent_matrix",
            )
        )
        pipeline._remove_answer_from_cache(cache, [ids.shape[1]] * len(records))
    result = dict(
        rows=rows,
        layers=records,
        context_ids_sha256=tensor_sha(tensors["context_ids"]),
        question_ids_sha256=[tensor_sha(q) for q in tensors["questions_ids"]],
        dataset_rows_sha256=group_digest(group),
        weights_frozen=True,
        query_aware=False,
        exploratory_prefill_seconds=prefill_seconds,
        formal_cost_measurement=False,
        physical_cache_bytes=cache_bytes,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1048576,
        live_allocated_mib=torch.cuda.memory_allocated() / 1048576,
    )
    del cache
    return result
