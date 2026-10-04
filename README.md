# GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs (EMNLP 2026)

[![Paper](https://img.shields.io/badge/arXiv-2605.31105-red)](https://arxiv.org/abs/2605.31105)
[![CPU validation](https://github.com/pjunjie/GRKV/actions/workflows/ci.yml/badge.svg)](https://github.com/pjunjie/GRKV/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

This repository contains the implementation of **GRKV**, a training-free method for KV cache compression introduced in our [paper](https://arxiv.org/abs/2605.31105). It is built on [NVIDIA's kvpress](https://github.com/NVIDIA/kvpress) and provides the code, default configurations, fixed environment and evaluation tools needed to reproduce GRKV on **Llama-3.1-8B-Instruct** and **Mistral-7B-Instruct-v0.3**.

Full prediction evidence, sample indices and checksums are available in the [evaluation manifest](results/validated/artifacts.json) and [the unified GRKV release](https://github.com/pjunjie/GRKV/releases/tag/v0.2.0).

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

GRKV achieves higher task-equal mean scores than the historical Critical-AdaKV baseline in all eight reported model, benchmark and retention-budget settings. The [evaluation protocol](docs/protocol.md) documents the frozen sample set, parameter-search split and scoring procedure; [paired confidence intervals](results/reference/historical_statistics.json) provide the statistical context for the measured gains.

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

These figures describe the completed Mistral inference run. Worker time includes model loading and qualification; memory values cover PyTorch allocator measurements and exclude other device allocations. See [the Mistral resource record](results/fresh/mistral/resources.json).

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

For questions about GRKV or reproduction, please [open an issue](https://github.com/pjunjie/GRKV/issues) in this repository. Contributions require DCO sign-off; see [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution guidelines.

## Paper

If you use GRKV in your research, please cite our paper:

```bibtex
@misc{peng2026grkvglobalregressiontrainingfree,
      title={{GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs}},
      author={Junjie Peng and You Wu and Haoyi Wu and Jialong Han and Xiaohua Xie and Kewei Tu and Jianhuang Lai},
      year={2026},
      eprint={2605.31105},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.31105},
}
```

**Paper:** [arXiv:2605.31105](https://arxiv.org/abs/2605.31105), accepted to EMNLP 2026 Main.

## Acknowledgments

We thank the authors of [kvpress](https://github.com/NVIDIA/kvpress), Critical-AdaKV, [Triton](https://github.com/triton-lang/triton), [FlashAttention](https://github.com/Dao-AILab/flash-attention), LongBench and RULER. This repository retains the upstream Apache-2.0 license and copyright notices. Model, dataset and dependency licenses remain with their respective upstream projects; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## KV Cache Merging Methods

Selected related work on KV cache merging, covering token merging, merging across layers, and multimodal inference. The table follows the year, venue, title, paper and code format of [Transformer-based Long Document Modeling](https://github.com/pjunjie/Transformer-based-Long-Document-Modeling), with a method column for easier lookup. Years refer to the listed publication; entries marked arXiv link to preprints.

| Year | Venue | Method | Title | Paper | Code |
|---|---|---|---|---|---|
| 2024 | ICML | CaM | CaM: Cache Merging for Memory-efficient LLMs Inference | [Paper](https://proceedings.mlr.press/v235/zhang24n.html) | [Code](https://github.com/zyxxmu/cam) |
| 2024 | ICML | DMC (continued pretraining) | Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference | [Paper](https://proceedings.mlr.press/v235/nawrot24a.html) | — |
| 2024 | NeurIPS | MiniCache (cross-layer) | MiniCache: KV Cache Compression in Depth Dimension for Large Language Models | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html) | — |
| 2024 | Findings of EMNLP | LOOK-M (multimodal) | LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference | [Paper](https://aclanthology.org/2024.findings-emnlp.235/) | [Code](https://github.com/SUSTechBruce/LOOK-M) |
| 2024 | arXiv | KVMerger | Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks | [Paper](https://arxiv.org/abs/2407.08454) | — |
| 2024 | arXiv | EMS | EMS: Adaptive Evict-then-Merge Strategy for Head-wise KV Cache Compression Based on Global-Local Importance | [Paper](https://arxiv.org/abs/2412.08521) | — |
| 2025 | ICLR | D2O | D2O: Dynamic Discriminative Operations for Efficient Long-Context Inference of Large Language Models | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html) | [Code](https://github.com/AIoT-MLSys-Lab/d2o) |
| 2025 | [ICASSP](https://jhc.sjtu.edu.cn/~bjiang/) | WeightedKV | WeightedKV: Attention Scores Weighted Key-Value Cache Merging for Large Language Models | [Paper](https://arxiv.org/abs/2503.01330) | — |
| 2025 | NeurIPS | AsymKV | Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs | [Paper](https://papers.nips.cc/paper_files/paper/2025/hash/750b0f9fccafad88e0da366315e03d1a-Abstract-Conference.html) | [Code](https://github.com/the-scale-lab/Asymkv) |
| 2025 | arXiv | ZSMerge (formerly ZeroMerge) | ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs | [Paper](https://arxiv.org/abs/2503.10714) | [Code](https://github.com/SusCom-Lab/ZSMerge) |
| 2026 | AAAI | KeepKV | KeepKV: Achieving Periodic Lossless KV Cache Compression for Efficient LLM Inference | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40611) | [Code](https://github.com/kkvcache/KeepKV) |
| 2026 | TMLR | LightKV (multimodal) | Make Your LVLM KV Cache More Lightweight | [Paper](https://openreview.net/forum?id=n77IeySrQl) | [Code](https://github.com/howtoosee/LightKV) |
| 2026 | [EMNLP](https://ai4gc.org/blog/flowmm-emnlp-2026) | FlowMM (multimodal) | FlowMM: Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference | [Paper](https://arxiv.org/abs/2511.05534) | — |
| 2026 | arXiv | KVSlimmer | KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging | [Paper](https://arxiv.org/abs/2603.00907) | [Code](https://github.com/lianjunl13-sudo/KVSlimmer) |
| 2026 | arXiv | SelKV | SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation | [Paper](https://arxiv.org/abs/2607.16213) | — |

Code links point to author-maintained implementations. — means no accessible author-maintained implementation was verified in the linked sources. AsymKV here refers to the merging method in *Homogeneous Keys, Heterogeneous Values*. GRKV's reproduced measurements are reported in [Reproduced Results](#reproduced-results).
