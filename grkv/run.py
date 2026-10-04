# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh, identity-bound candidate/baseline generation without reference predictions."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from time import perf_counter

import yaml

from grkv.io import read, sha, timestamp, write
from grkv.settings import ROOT, asset_root, load_environment


def unit_key(work):
    item = work["item"]
    return f"{item['budget']}__{item['benchmark']}__{item['task']}__{item['context_sha256']}"


def code_identity():
    return {
        str(p.relative_to(ROOT)): sha(p)
        for directory in ("grkv", "kvpress", "evaluation")
        for p in sorted((ROOT / directory).rglob("*.py"))
    }


def asset_identity(model):
    record = read(ROOT / "manifests/models.json")[model]
    result = {}
    for entry in record["files"]:
        path = asset_root() / "models" / model / entry["name"]
        digest = sha(path)
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Frozen model file mismatch: {model}/{entry['name']}")
        result[f"models/{model}/{entry['name']}"] = digest
    for dataset in read(ROOT / "manifests/datasets.json"):
        for name, digest in dataset["file_sha256"].items():
            if sha(asset_root() / "datasets" / dataset["name"] / name) != digest:
                raise ValueError("Frozen dataset file mismatch")
            result[f"datasets/{dataset['name']}/{name}"] = digest
    return result


def configuration(press):
    return {
        f.name: asdict(getattr(press, f.name)) if is_dataclass(getattr(press, f.name)) else getattr(press, f.name)
        for f in fields(press)
        if f.init
    }


def worker(root, job_path):
    import torch
    from transformers import pipeline

    import kvpress  # noqa: F401 - registers the frozen pipeline and attention patch
    from grkv.api import make_press
    from grkv.backend import HistoricalExecutableBackend
    from grkv.data import load_group
    from grkv.generation import generate
    from grkv.mask_fallback import FALLBACK_CALLS, install
    from grkv.qualification import qualify
    from grkv.runtime import runtime

    registration = read(root / "registration.json")
    job = read(job_path)
    identity = sha(root / "registration.json")
    if job["registration_sha256"] != identity or registration["code_hashes"] != code_identity():
        raise ValueError("Worker code/registration identity differs")
    state = dict(status="running", started_at=timestamp(), completed_units=0, answers=0)
    state_path = root / "jobs" / (job_path.stem + ".json")
    write(state_path, state)
    try:
        state["runtime"] = runtime()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(42)
        backend = None
        if registration["config"]["method"] == "default":
            backend = HistoricalExecutableBackend(os.environ["GRKV_KERNEL_MANIFEST"], os.environ["TRITON_CACHE_DIR"])
            state["qualification"] = qualify(backend)
        config = registration["config"]
        name = config["model"]
        pipe = pipeline(
            "kv-press-text-generation",
            model=str(asset_root() / "models" / name),
            model_kwargs=dict(attn_implementation="flash_attention_2", dtype="auto"),
            trust_remote_code=True,
            device="cuda:0",
        )
        pipe.model.eval().requires_grad_(False)
        if pipe.model.dtype != torch.bfloat16 or pipe.model.config._attn_implementation != "flash_attention_2":
            raise ValueError("Model dtype or attention differs")
        torch.cuda.reset_peak_memory_stats()
        if name == "mistral":
            if pipe.model.config.model_type != "mistral":
                raise ValueError("Mistral-only fallback model mismatch")
            install()
        cache: dict = {}
        for work in job["units"]:
            if (root / "STOP").exists():
                raise RuntimeError("STOP requested")
            item = work["item"]
            key = unit_key(work)
            state["current_unit"] = key
            write(state_path, state)
            output = root / "contexts" / (key + ".json")
            if output.exists():
                record = read(output)
                if record["registration_sha256"] != identity or record["item"] != item:
                    raise ValueError("Resume context identity differs")
                if record["status"] != "complete" or record["prediction_source"] != "fresh_public_checkout":
                    raise ValueError("Resume context is incomplete or from another inference source")
            else:
                group = load_group(item, cache)
                limit = min(pipe.tokenizer.model_max_length, int(1e10))
                if config["method"] == "baseline" and name == "mistral":
                    limit = 32768
                tensors = pipe.preprocess(
                    group[0]["context"],
                    questions=[r["question"] for r in group],
                    answer_prefix=group[0]["answer_prefix"],
                    max_context_length=limit,
                )
                before = FALLBACK_CALLS["count"]
                press = make_press(name, item["budget"], config["method"])
                if configuration(press) != dict(
                    config["press"], compression_ratio=0.9 if item["budget"] == 10 else 0.8
                ):
                    raise ValueError("Frozen dataclass configuration differs")
                begin = perf_counter()
                value = generate(pipe, press, backend, item, group, tensors)
                value["exact_mask_fallback_calls"] = FALLBACK_CALLS["count"] - before
                for row in value["rows"]:
                    row["prediction_source"] = "fresh_public_checkout"
                record = dict(
                    status="complete",
                    unit=key,
                    item=item,
                    scope=work["scope"],
                    value=value,
                    registration_sha256=identity,
                    prediction_source="fresh_public_checkout",
                    completed_at=timestamp(),
                    seconds=perf_counter() - begin,
                    actual_binary_checks=list(backend.launch_records) if backend else [],
                )
                write(output, record)
            state["completed_units"] += 1
            state["answers"] += len(record["value"]["rows"])
            state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            state["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            state["updated_at"] = timestamp()
            write(state_path, state)
            print(f"{job_path.stem}: {state['completed_units']}/{len(job['units'])} units", flush=True)
            torch.cuda.empty_cache()
        state.update(status="complete", finished_at=timestamp())
        write(state_path, state)
    except BaseException as error:
        import traceback

        state.update(status="failed", error=type(error).__name__, traceback=traceback.format_exc())
        write(state_path, state)
        raise


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--gpus", default=os.environ.get("GRKV_GPU_IDS", "0"))
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fresh", action="store_true")
    modes.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--kernel-root", type=Path, default=Path(os.environ.get("GRKV_KERNEL_ROOT", "artifacts/kernels"))
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.worker)
        return
    if not args.config:
        parser.error("--config is required")
    config = yaml.safe_load(args.config.read_text())
    units = [json.loads(line) for line in (ROOT / "manifests/units.jsonl").read_text().splitlines()]
    if args.stage == "smoke":
        subset = set(read(ROOT / "manifests/smoke.json")["units"])
        units = [w for w in units if unit_key(w) in subset]
    units = [w for w in units if w["item"]["budget"] in config["budgets"]]
    root = args.output or Path(os.environ.get("GRKV_OUTPUT_ROOT", "outputs")) / config["model"] / args.stage
    root = root.resolve()
    kernel_manifest = (args.kernel_root / "manifest.json").resolve()
    identity = dict(
        config=config,
        stage=args.stage,
        code_hashes=code_identity(),
        units_sha256=sha(ROOT / "manifests/units.jsonl"),
        models_manifest_sha256=sha(ROOT / "manifests/models.json"),
        datasets_manifest_sha256=sha(ROOT / "manifests/datasets.json"),
        lock_sha256=sha(ROOT / "uv.lock"),
        kernel_manifest_sha256=sha(kernel_manifest),
        kernel_profile="historical-executable-sm86",
        asset_hashes=asset_identity(config["model"]),
    )
    registration = root / "registration.json"
    if args.resume:
        if not registration.exists() or read(registration)["identity"] != identity:
            raise ValueError("Resume requires identical code, config, data, model, kernels and environment lock")
    else:
        if root.exists():
            raise FileExistsError("Output already exists; use --resume or choose a new --output directory")
        if args.stage == "full" and not args.fresh:
            parser.error("Full generation requires --fresh or --resume")
        root.mkdir(parents=True)
        write(
            registration,
            dict(
                **identity,
                identity=identity,
                created_at=timestamp(),
                source="fresh_public_checkout",
                reference_predictions_required=False,
            ),
        )
    gpus = [g.strip() for g in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("Specify unique visible GPU IDs")
    loads = [0] * len(gpus)
    assignments: list[list[dict]] = [[] for _ in gpus]
    for work in sorted(units, key=lambda w: (-len(w["item"]["rows"]), unit_key(w))):
        index = min(range(len(gpus)), key=lambda i: (loads[i], i))
        assignments[index].append(work)
        loads[index] += len(work["item"]["rows"]) + 2
    workers = []
    for index, (gpu, works) in enumerate(zip(gpus, assignments)):
        job = root / "specs" / f"worker{index}.json"
        write(job, dict(registration_sha256=sha(registration), units=works))
        cache = root / "cache" / f"worker{index}"
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=gpu,
            TRITON_CACHE_DIR=str(cache),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            GRKV_KERNEL_MANIFEST=str(kernel_manifest),
        )
        log = open(root / f"worker{index}.log", "a")
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "grkv.run", "--worker", str(job), "--output", str(root)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        workers.append((process, log))
    failed = False
    for process, log in workers:
        failed = process.wait() != 0 or failed
        log.close()
    if failed:
        raise RuntimeError("Worker failed; inspect local job records and logs")
    from grkv.score import rescore

    summary = rescore(root, units)
    write(root / "scores.json", summary)
    print(json.dumps(summary["cells"], indent=2))


if __name__ == "__main__":
    main()
