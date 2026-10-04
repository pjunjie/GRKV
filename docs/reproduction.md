# Reproducing GRKV Default

Use the native reference profile in `environment/reference.json`: Linux x86_64,
Python 3.12.7, CUDA toolkit 12.4.131, NVIDIA RTX A6000 (SM86), driver 555.52.04.
The lock fixes Python packages, including the exact public Triton 3.2.0 wheel.
The historical GPU profile rejects another architecture, driver or imported
Triton package. BF16 model/KV and FP32 fitting are required. Other GPU profiles
are not validated. The digest-pinned Dockerfile has not been executed on this
host; the native installation is the tested path.

Install uv 0.9.24 from its public release and make the reference CUDA toolkit
available locally. Set `CUDA_HOME` to that installation before setup.

```bash
git clone https://github.com/pjunjie/GRKV.git
cd GRKV
# Replace xxxxxx with your local CUDA 12.4.131 toolkit directory.
export CUDA_HOME=xxxxxx
bash scripts/setup_reference.sh
cp .env.example .env.local
```

The setup script installs the locked environment and builds FlashAttention
2.8.3.post1 from the pinned public source. Its official prebuilt wheel requires
a newer glibc than the reference Ubuntu 20.04 host. Initial source compilation
took about 25 minutes with 16 build jobs. A subsequent clean checkout can reuse
that verified uv build cache; this is distinct from reusing prediction results.
Use an empty `UV_CACHE_DIR` to repeat initial downloading and compilation.

`.env.local` is read by the runner and fetcher. Default paths are relative to
the checkout; set another local `GRKV_ASSET_ROOT` if needed. Accept upstream
model access conditions and authenticate with Hugging Face locally. Never
commit credentials; `xxxxxx` is an example placeholder that must be replaced
when enabled. No model weights, benchmark bodies or private caches are shipped.

```bash
uv run --no-sync python scripts/fetch_assets.py --models all --datasets all --kernels historical
uv run --no-sync python scripts/preflight.py --profile historical-executable-sm86 --gpu 0
```

Model file sizes and SHA-256 values, raw parquet hashes and every frozen input
group are checked. Kernel download size/SHA, its manifest and each extracted
cubin are checked. Real derivative qualification verifies imported runtime
files, compiled nondebug ELF sections and the public binary before CUDA loads
it. The public kernels have redacted debug paths and separate SHA values;
original complete-file cubin equality is false. This distinction remains part
of all validation reports. See `historical_kernels.md`.

Generation does not download, open or require a reference prediction archive.
The smoke selection contains both benchmarks, both budgets, shared-question
groups and a known Mistral fallback case. GPU short-context/cache checks are
separate commands.

```bash
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage smoke --gpus 0,1,2,3
uv run --no-sync python -m grkv.run --config configs/llama.yaml --stage smoke --gpus 0,1,2,3
uv run --no-sync python scripts/run_boundary_smoke.py --model mistral --gpu 0 --output outputs/boundary/mistral.json
uv run --no-sync python scripts/run_boundary_smoke.py --model llama --gpu 0 --output outputs/boundary/llama.json
```

Choose free GPU IDs. Each worker sees one device. The supplied validation uses
four workers per model; the runner accepts a different unique list. Single-card
full scheduling has not been validated. Actual Mistral full-run GPU hours and memory peaks are in `results/fresh/mistral/resources.json`.
Llama's consolidated evidence does not establish single-run timing. Earlier smoke-based estimates remain
labeled estimates in `environment/reference.json`; they are not timings of a
completed current Llama full run or minimum hardware guarantees.

```bash
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage full --fresh --gpus 0,1,2,3
uv run --no-sync python -m grkv.run --config configs/llama.yaml --stage full --fresh --gpus 0,1,2,3
```

Default outputs are `outputs/<model>/<stage>`. A fresh run rejects an existing
output directory. Use `--output` to choose a distinct destination. Full runs
must explicitly use `--fresh` or `--resume`. Resume reuses only complete answers
from the same run identity: config, numerical source hashes, lock, input/model
hashes and kernel manifest. It rejects changed assets or another method's
answers. To stop after the current unit, create `STOP` inside the run directory;
remove that file before resuming.

```bash
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage full --resume --gpus 0,1,2,3
```

Each output unit contains newly generated text, row identities, numeric/layer
records, launch proof and a registration hash. Worker logs/cache remain local;
publish reviewed evidence exports rather than entire output folders. The
final summarizer independently recomputes scores from predictions and fresh
dataset answers, rejects missing/duplicate/nonfinite samples, and applies
task-equal macros with the RULER percent conversion exactly once.

Reference assets are optional verification inputs, downloaded separately:

```bash
uv run --no-sync python scripts/fetch_assets.py --references all
uv run --no-sync python -m grkv.score --input artifacts/reference/mistral_critical.jsonl.gz --output outputs/rescore/mistral_critical.json
uv run --no-sync python -m grkv.score --input artifacts/reference/llama_critical.jsonl.gz --output outputs/rescore/llama_critical.json
uv run --no-sync python -m grkv.score --input artifacts/reference/mistral_candidate.jsonl.gz --output outputs/rescore/mistral_candidate.json
uv run --no-sync python -m grkv.score --input artifacts/reference/llama_candidate.jsonl.gz --output outputs/rescore/llama_candidate.json
uv run --no-sync python scripts/verify_results.py --run outputs/mistral/full --model mistral --reference artifacts/reference/mistral_candidate.jsonl.gz --strict
uv run --no-sync python scripts/verify_results.py --run outputs/llama/full --model llama --reference artifacts/reference/llama_candidate.jsonl.gz --strict
```

For smoke verification add `--stage smoke` and use the smoke run directory.
The verifier reports coverage, independently rescored scores, target tolerance
1e-8, prediction/score/layout/fit differences, fallback differences and positive
gains as separate fields. It never turns reference-only rescoring into fresh
generation evidence. `--strict` requires zero output differences, no saved-score
mismatch and, for full runs, target scores. Original cubin byte identity remains
a separate false field, rather than being hidden inside this output test.

`configs/baselines/` provides historical Critical-AdaKV constructors and its
model-specific protocol. Give new baseline runs distinct `--output` directories.
The published baseline scores come from historical Critical-AdaKV inference
and have been independently rescored with the pinned data and public scorer.
The earlier Mistral GRKV control is not a Critical-AdaKV baseline.

Download the accepted complete evidence for both models and independently
rescore it with the ordinary full scorer:

```bash
uv run --no-sync python scripts/fetch_assets.py --validated-evidence all
uv run --no-sync python -m grkv.score --input artifacts/validated/fresh_mistral_candidate.jsonl.gz --output outputs/rescore/accepted_mistral.json
uv run --no-sync python -m grkv.score --input artifacts/validated/validated_llama_candidate.jsonl.gz --output outputs/rescore/accepted_llama.json
```

Both archives have full frozen coverage: 20,500 answers in 19,630 context/budget
units each. The download manifest lists exact asset URLs, sizes and hashes;
all eight evidence and kernel archives are available together in the unified
[GRKV release](https://github.com/pjunjie/GRKV/releases/tag/v0.3.0). Its
`SHA256SUMS` covers every archive. The seven prediction archives retain their
original contents, sizes and hashes. The smaller kernel archive contains only
the two required, unchanged public cubins and an updated manifest, with new
archive and manifest hashes. See the [asset guide](release.md).

Downloading this evidence does not perform new inference on your machine.
Llama's full accepted result consolidates current generated records with the
corresponding verified earlier Triton 3.2.0 / historical-kernel outputs. Each
record retains its origin, source archive and checkpoint hashes, and actual
pre-launch binary checks. It is not labeled as entirely current generation.
The assembly script independently rescores the complete selection, verifies
the frozen sample set and rejects any strict output differences.

To reproduce that assembly from its published source-run evidence:

```bash
uv run --no-sync python scripts/fetch_assets.py --references all --fresh-evidence all --kernels historical
uv run --no-sync python scripts/consolidate_llama_results.py --current artifacts/fresh/fresh_llama_candidate_partial.jsonl.gz --earlier artifacts/reference/llama_candidate.jsonl.gz --coverage-ledger results/fresh_partial/llama/verification.json --prior-proofs results/validated/llama/selected_prior_launch_proofs.jsonl
```

Source-run evidence retains its original coverage and provenance.
`scripts/export_fresh_evidence.py` exports only completed, strictly verified full
candidate runs. `scripts/verify_partial_results.py` and
`scripts/export_partial_evidence.py` verify and export genuinely generated
incomplete runs with explicit coverage and missing-unit records. Both exporters
independently rescore their archives and exclude worker logs, caches and machine
paths. The source-run archives previously passed remote size/SHA verification;
their audit receipts retain the original checkout, URLs and validation scope.
Those URLs describe earlier publications; use the current download manifests
and unified release for downloads.

Run `make style` and `make test` for the targeted CPU checks. CI installs without
FlashAttention or GPU qualification, imports the public APIs, validates CPU
protocol/fallback behavior and scans source/history. GPU evidence is recorded
separately. Manifest download links point to public Release assets fetched and checked from a new remote clone.
On a host with an unavailable unrelated GPU, the optional
`scripts/isolated_gpu_tools/nvidia-smi` wrapper scopes only the existing
driver-version query to each worker's `CUDA_VISIBLE_DEVICES` selection. Add
`$PWD/scripts/isolated_gpu_tools` to `PATH` and select healthy devices using
`--gpus`. The wrapper invokes the real system `nvidia-smi -i` command and
preserves the required driver check; it does not supply a replacement value.
All other queries retain their original behavior.
