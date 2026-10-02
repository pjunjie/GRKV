# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from time import perf_counter

import torch
from transformers import DynamicCache

from experiments.critical_adakv_direct_grkv.run_wide_query_compatibility import group_digest
from experiments.critical_adakv_direct_grkv.wide_query_worker_base import score, verify_prefill_layout
from experiments.critical_adakv_grkv.press import tensor_sha


def check_sinks(press):
    # Frozen K1 has no IndependentSinkPress: the historical context manager is a no-op.
    return nullcontext()


@torch.inference_mode()
def generate(pipeline, press, backend, item, group, tensors):
    ids = tensors["context_ids"].to(pipeline.model.device)
    cache = DynamicCache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = perf_counter()
    with backend() if backend is not None else nullcontext():
        with check_sinks(press), press(pipeline.model):
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
