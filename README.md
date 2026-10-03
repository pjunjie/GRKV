# GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs (EMNLP 2026)

[![Paper](https://img.shields.io/badge/arXiv-2605.31105-red)](https://arxiv.org/abs/2605.31105)
[![CPU validation](https://github.com/pjunjie/GRKV/actions/workflows/ci.yml/badge.svg)](https://github.com/pjunjie/GRKV/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

This repository contains the implementation of **GRKV**, a training-free method for KV cache compression introduced in our [paper](https://arxiv.org/abs/2605.31105). It is built on [NVIDIA's kvpress](https://github.com/NVIDIA/kvpress) and provides the code, default configurations, fixed environment and evaluation tools needed to reproduce GRKV on **Mistral-7B-Instruct-v0.3** and **Llama-3.1-8B-Instruct**.

The results below come from the verified reproduction of the current GRKV defaults. Full prediction evidence, sample indices and checksums are available in the [evaluation manifest](results/validated/artifacts.json) and [GitHub Releases](https://github.com/pjunjie/GRKV/releases).

## Overview

Evicting KV cache entries saves memory but loses information from the removed tokens. GRKV uses global regression to update the retained keys and values so that attention with the compressed cache better matches attention with the full cache. The model weights remain frozen, and compression requires no additional training.

The default implementation follows three steps:

1. **Select retained entries** using Critical-AdaKV at the requested cache budget.
2. **Fit regularized updates** using queries sampled from the input context, distributing information across retained entries.
3. **Generate with the compressed cache**, using the same model and decoding protocol as the baseline.

The public API uses these defaults directly:

```python
from grkv.api import make_press

press = make_press("mistral", budget=10)  # Retain 10% of the KV cache.
# Use "llama" for Llama-3.1-8B-Instruct, or budget=20 to retain 20%.
```

| Default setting | Value |
|---|---|
| KV retention budgets | 10% and 20% |
| Regression queries | 32, sampled across the input context |
| Ridge regularization | 0.01 |
| Relative update caps | Keys: 0.025; values: 0.1 |
| Regression precision | FP32; model and KV cache: BF16 |
| Safety guard | Mistral: 0.0; Llama: 0.1 |

Complete defaults are provided in [configs/mistral.yaml](configs/mistral.yaml) and [configs/llama.yaml](configs/llama.yaml). See the [evaluation protocol](docs/protocol.md) for budget rounding, protected tokens, attention behavior and scoring details.

## Installation

The validated native environment is:

| Component | Version / hardware |
|---|---|
| OS / Python | Ubuntu 20.04.6, Python 3.12.7 |
| GPU / driver | NVIDIA RTX A6000, SM86 / 555.52.04 |
| CUDA toolkit | 12.4.131 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.2.0 |
| FlashAttention | 2.8.3.post1 |
| Triton | 3.2.0, pinned public PyTorch wheel |
| Package manager | uv 0.9.24 |

Install [uv 0.9.24](https://github.com/astral-sh/uv/releases/tag/0.9.24) and make the CUDA toolkit available locally, then run:

```bash
git clone https://github.com/pjunjie/GRKV.git
cd GRKV

# Replace xxxxxx with your local CUDA 12.4.131 toolkit directory.
export CUDA_HOME=xxxxxx
bash scripts/setup_reference.sh

# Keep local paths and credentials in this untracked file.
cp .env.example .env.local
```

The setup installs the locked dependencies and builds FlashAttention from its pinned public source. The source build is required on the reference Ubuntu 20.04 environment because the public prebuilt wheel requires a newer glibc. The full software and compiler record is in [environment/reference.json](environment/reference.json).

### Preparing Models and Datasets

Models and benchmark data are obtained from their upstream sources; they are not bundled in this repository.

| Resource | Fixed revision |
|---|---|
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71) | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| [Formatted LongBench](https://huggingface.co/datasets/Xnhyacinth/LongBench/tree/2e9ade51ebf45d98942056c0716234f9d5d257d5) | `2e9ade51ebf45d98942056c0716234f9d5d257d5` |
| [RULER, configuration 16384](https://huggingface.co/datasets/simonjegou/ruler/tree/24adceac8a0e6532936e8d721cd9e9084d2e4686) | `24adceac8a0e6532936e8d721cd9e9084d2e4686` |

Accept the upstream Llama access requirements and configure your Hugging Face token locally if needed. `.env.local` sets the asset, output and kernel directories; the defaults are relative to this checkout. Any enabled `xxxxxx` credential placeholder must be replaced with your own local value.

```bash
uv run --no-sync python scripts/fetch_assets.py --models all --datasets all --kernels historical
```

The fetcher verifies model file sizes and SHA-256 hashes, dataset parquet hashes, and every frozen evaluation input. Strict GPU reproduction also uses the supplied executable kernels, whose download and extracted files are checksum-verified. The runtime checks the actual kernel before loading it. See [reproduction instructions](docs/reproduction.md) and [kernel provenance](docs/historical_kernels.md).

## Quick Evaluation

Start with the fixed smoke evaluation to check installation, generation and compression for both models:

```bash
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage smoke --fresh --gpus 0,1,2,3
uv run --no-sync python -m grkv.run --config configs/llama.yaml --stage smoke --fresh --gpus 0,1,2,3
```

Choose four available GPU IDs on your host. Each model's smoke evaluation covers both benchmarks and both cache budgets: **60 context/budget units and 88 answers**, including shared-context questions and a Mistral exact-mask fallback case. Outputs and independently recomputed scores are saved under `outputs/<model>/smoke/`. Smoke scores describe this subset and should not be compared with the full benchmark scores below.

The runner accepts other GPU lists; the validated runs use four workers per model. Single-card full evaluation has not yet been validated. Additional short-context and shared-cache checks are described in the [reproduction guide](docs/reproduction.md).

## Comprehensive Evaluation: LongBench + RULER

Run the full evaluation with the default GRKV configurations:

```bash
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage full --fresh --gpus 0,1,2,3
uv run --no-sync python -m grkv.run --config configs/llama.yaml --stage full --fresh --gpus 0,1,2,3
```

Each model is evaluated at **10% and 20% KV retention** on **16 LongBench tasks** and **13 RULER tasks at 16k context length**. The complete evaluation contains **19,630 context/budget units and 20,500 answers per model**. Predictions, run identities and scores are saved under `outputs/<model>/full/`. Generation does not require reference prediction downloads.

Use `--resume` instead of `--fresh` to continue an interrupted run:

```bash
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage full --resume --gpus 0,1,2,3
```

Resume requires identical configuration, numerical source, model/data hashes, kernels and dependency lock. For a separate run, choose a new `--output` directory. Historical Critical-AdaKV baseline configurations are provided under [configs/baselines/](configs/baselines/); use distinct output directories when generating new baseline results.

### Reproduced Results

Scores are on a **0–100 scale**, averaged equally across tasks. Higher is better. The percentages denote retained KV cache, and gains are absolute score differences.

**Mistral-7B-Instruct-v0.3**

| Method | LongBench / 10% | LongBench / 20% | RULER16k / 10% | RULER16k / 20% |
|---|---:|---:|---:|---:|
| Historical Critical-AdaKV | 35.0089 | 38.8074 | 26.8210 | 48.2644 |
| **GRKV** | **37.2587** | **40.0311** | **27.0987** | **51.8290** |
| Absolute gain | +2.2498 | +1.2238 | +0.2777 | +3.5646 |

**Llama-3.1-8B-Instruct**

| Method | LongBench / 10% | LongBench / 20% | RULER16k / 10% | RULER16k / 20% |
|---|---:|---:|---:|---:|
| Historical Critical-AdaKV | 37.6718 | 43.1407 | 44.8662 | 66.3128 |
| **GRKV** | **38.7237** | **43.3293** | **45.4649** | **66.6577** |
| Absolute gain | +1.0519 | +0.1885 | +0.5987 | +0.3449 |

Both complete GRKV result sets passed independent rescoring, matching all eight full-precision targets within `1e-8`. Per-answer text, scores, regression records and cache layouts have zero strict differences against the frozen reference. Mistral uses the complete clean-checkout run; Llama consolidates corresponding verified outputs from the current run and an earlier experiment under the fixed Triton 3.2.0 and kernel protocol. Every Llama record retains its generation origin and source hashes. See the [full acceptance report](results/validated/summary.json).

The Critical-AdaKV rows are independently rescored **historical baselines**, not newly generated baseline runs. Older experimental control results are documented separately and are not substituted for this baseline.

These comparisons are descriptive. Historical full results had been exposed, and the remaining 50% was held out only from the parameter search. Llama's LongBench/20% historical paired confidence interval crosses zero. Its original criterion requiring at least 0.5 points of gain in every cell remains unmet; the positive gains shown here use a separate criterion. The [protocol](docs/protocol.md) and [historical statistics](results/reference/historical_statistics.json) retain these definitions.

### Analyzing Results

To independently rescore the published complete evidence, download the frozen data and accepted prediction archives:

```bash
uv run --no-sync python scripts/fetch_assets.py --datasets all --validated-evidence all
uv run --no-sync python -m grkv.score --input artifacts/validated/fresh_mistral_candidate.jsonl.gz --output outputs/rescore/mistral.json
uv run --no-sync python -m grkv.score --input artifacts/validated/validated_llama_candidate.jsonl.gz --output outputs/rescore/llama.json
```

This scoring path needs no GPU. In a separate CPU-only checkout, install with `uv sync --frozen --extra eval` instead of the CUDA setup. Scores are recomputed from prediction text and the pinned dataset answers; missing, duplicate or nonfinite answers fail validation. Downloaded evidence retains its recorded generation source and does not perform inference on your machine.

To strictly compare your own full runs with the reference outputs:

```bash
uv run --no-sync python scripts/fetch_assets.py --references all
uv run --no-sync python scripts/verify_results.py --run outputs/mistral/full --model mistral --reference artifacts/reference/mistral_candidate.jsonl.gz --strict
uv run --no-sync python scripts/verify_results.py --run outputs/llama/full --model llama --reference artifacts/reference/llama_candidate.jsonl.gz --strict
```

For smoke verification, add `--stage smoke` and use the corresponding smoke output directory. Full-precision scores, per-task results and generation provenance are available under [results/](results/).

## Efficiency and Validation

The complete Mistral run measured the following resources on RTX A6000 GPUs:

| Measurement | Observed value |
|---|---:|
| Successful GPU workers | 4 |
| Wall time | 14.118 hours |
| Total GPU worker time | 55.891 hours |
| Peak allocated / reserved CUDA memory | 35.585 / 41.406 GiB |

Worker time includes model loading and qualification. Memory values are PyTorch allocator measurements and exclude other device allocations. These are run measurements, not a formal efficiency comparison or minimum hardware requirements. No single-run timing is inferred from Llama's consolidated evidence. See [the resource record](results/fresh/mistral/resources.json).

Native installation, two-model GPU smoke, GPU derivative checks, strict output checks and independent rescoring passed. The remote CPU checkout also passed style/type/SPDX checks and 17 targeted tests; [remote acceptance](results/validated/remote_checkout/summary.json) records its scope. Run `make style` and `make test` for these CPU development checks.

Mistral uses FlashAttention2 normally and falls back to explicit causal/per-head masking only when fake-key feasibility fails. The complete run independently measured **22 fallback calls in 22 units**, matching the reference. Llama does not enable this fallback.

The public kernel assets redact private debug paths while preserving every nondebug ELF section, including GPU instruction bytes. Numerical qualification and full output comparisons are reported separately. The complete original binary files are not byte-identical: `original_cubin_bytes_identical=false`. Instructions, manifests and separate public hashes are in [the kernel documentation](docs/historical_kernels.md).

## TODO

- [x] Publish the default GRKV implementation, configurations and locked environment.
- [x] Verify complete two-model results and provide independently rescorable evidence.
- [x] Validate public downloads and installation from a remote checkout.
- [ ] Complete Docker container build and GPU acceptance. The [Dockerfile](environment/Dockerfile) is provided; container validation remains incomplete.
- [ ] Validate single-card full evaluation and additional GPU architectures.

## Contact

For questions about GRKV or reproduction, please [open an issue](https://github.com/pjunjie/GRKV/issues) in this repository. Contributions require DCO sign-off; automated submissions use the `🤖🤖🤖` marker described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Paper

If you use GRKV in your research, please cite our paper:

```bibtex
@article{peng2026grkv,
  title={GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs},
  author={Peng, Junjie and Wu, You and Wu, Haoyi and Han, Jialong and Xie, Xiaohua and Tu, Kewei and Lai, Jianhuang},
  journal={arXiv preprint arXiv:2605.31105},
  year={2026}
}
```

**Paper:** [arXiv:2605.31105](https://arxiv.org/abs/2605.31105), accepted to EMNLP 2026 Main.

## Acknowledgments

We thank the authors of [kvpress](https://github.com/NVIDIA/kvpress), Critical-AdaKV, [Triton](https://github.com/triton-lang/triton), [FlashAttention](https://github.com/Dao-AILab/flash-attention), LongBench and RULER. This repository retains the upstream Apache-2.0 license and copyright notices. Model, dataset and dependency licenses remain with their respective upstream projects; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
