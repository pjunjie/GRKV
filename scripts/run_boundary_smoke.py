# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU checks for protected short inputs and shared-context cache restoration."""

import argparse
import os
from pathlib import Path

from grkv.settings import asset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["llama", "mistral"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from transformers import DynamicCache, pipeline

    import kvpress  # noqa: F401
    from experiments.critical_adakv_cross_model.mask_fallback import install
    from experiments.critical_adakv_history_ruler.common import write
    from grkv.api import make_press
    from grkv.runtime import runtime

    identity = runtime()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    pipe = pipeline(
        "kv-press-text-generation",
        model=str(asset_root() / "models" / args.model),
        model_kwargs=dict(attn_implementation="flash_attention_2", dtype="auto"),
        trust_remote_code=True,
        device="cuda:0",
    )
    pipe.model.eval().requires_grad_(False)
    if args.model == "mistral":
        install()
    tensors = pipe.preprocess(
        "The meeting starts at noon.\n",
        questions=["When does the meeting start?", "Repeat the time."],
        answer_prefix="",
        max_context_length=32,
    )
    ids = tensors["context_ids"].to("cuda")
    if ids.shape[1] > 32:
        raise ValueError("Boundary context is not short")
    results = []
    with torch.inference_mode():
        for budget in [10, 20]:
            cache = DynamicCache()
            press = make_press(args.model, budget)
            with press(pipe.model):
                pipe.model.model(ids, past_key_values=cache)
            if not all(not r["compression_applied"] and r["kept"] == 8 * ids.shape[1] for r in press.layer_records):
                raise ValueError("Short input budget branch changed")
            before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
            predictions = []
            # A/B/A checks intervening question independence on one context cache.
            for question in [tensors["questions_ids"][0], tensors["questions_ids"][1], tensors["questions_ids"][0]]:
                predictions.append(pipe.generate_answer(question.to("cuda"), cache, ids.shape[1], 12))
                pipe._remove_answer_from_cache(cache, [ids.shape[1]] * 32)
                if not all(
                    torch.equal(k, layer.keys) and torch.equal(v, layer.values)
                    for (k, v), layer in zip(before, cache.layers)
                ):
                    raise ValueError("Question generation mutated the protected context cache")
            if predictions[0] != predictions[2]:
                raise ValueError("Intervening question changed the repeated answer")
            results.append(
                dict(
                    budget=budget,
                    context_tokens=ids.shape[1],
                    predictions=predictions,
                    cache_restoration_exact=True,
                    repeated_answer_exact=True,
                )
            )
    write(
        args.output,
        dict(
            model=args.model,
            runtime=identity,
            short_context_passed=True,
            shared_context_isolation_passed=True,
            checks=results,
        ),
    )
    print(f"{args.model}: short context and A/B/A shared cache checks passed at both budgets")


if __name__ == "__main__":
    main()
