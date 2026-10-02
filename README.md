# GRKV: frozen Q32 K1 reproduction

Release preparation is in progress. Independent reference rescoring, Triton
package matching, CPU checks and real derivative qualification have passed.
Both-model fresh smoke and full inference are still pending; the reference
scores below must not be mistaken for new inference results.

K1 fits global regression over the historical Critical-AdaKV selected cache
positions. Use `grkv.api.make_press("mistral", budget)` or
`grkv.api.make_press("llama", budget)` for the frozen candidates. Mistral uses
guard 0.0 and Llama uses guard 0.1. Full parameters are in `configs/`.

| Benchmark / KV retained | Mistral historical Critical-AdaKV | Mistral K1 reference | Gain | Llama historical Critical-AdaKV | Llama K1 reference | Gain |
|---|---:|---:|---:|---:|---:|---:|
| LongBench / 10% | 35.0089 | 37.2587 | +2.2498 | 37.6718 | 38.7237 | +1.0519 |
| LongBench / 20% | 38.8074 | 40.0311 | +1.2238 | 43.1407 | 43.3293 | +0.1885 |
| RULER16k / 10% | 26.8210 | 27.0987 | +0.2777 | 44.8662 | 45.4649 | +0.5987 |
| RULER16k / 20% | 48.2644 | 51.8290 | +3.5646 | 66.3128 | 66.6577 | +0.3449 |

Scores are task-equal macros on 0–100; gains are absolute points. The older
Mistral K1 control is a separate comparison, not the historical Critical-AdaKV
baseline. See [the protocol](docs/protocol.md) for exact scoring, templates,
budgets, sample counts, baseline identities and statistical disclosures.
Historical `all_four_pass=false` uses the original >=0.5-point rule; the new
positive-gain gate is separately named. The comparisons are descriptive,
and the Llama LongBench/20% historical paired interval crosses zero.

| External resource | Fixed revision |
|---|---|
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71) | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| [Formatted LongBench](https://huggingface.co/datasets/Xnhyacinth/LongBench/tree/2e9ade51ebf45d98942056c0716234f9d5d257d5) | `2e9ade51ebf45d98942056c0716234f9d5d257d5` |
| [RULER, configuration 16384](https://huggingface.co/datasets/simonjegou/ruler/tree/24adceac8a0e6532936e8d721cd9e9084d2e4686) | `24adceac8a0e6532936e8d721cd9e9084d2e4686` |

Accept the upstream Llama access requirements and configure your Hugging Face
credential locally when needed. Models and benchmark bodies are downloaded
from these sources and are not included in Git or kernel artifacts. The fetcher
checks model file size/SHA-256, dataset parquet SHA-256 and all frozen row,
example, context and full input-group hashes. Original LongBench and RULER
sources and upstream licenses are linked in [third-party notices](THIRD_PARTY_NOTICES.md).

The tested native package profile is Python 3.12.7, PyTorch 2.6.0+cu124,
Transformers 5.2.0, FlashAttention 2.8.3.post1 built with CUDA 12.4.131,
Triton 3.2.0 from the exact public PyTorch wheel, BF16 model/cache and FP32 fits.
Hardware is NVIDIA RTX A6000, SM86, with driver 555.52.04. Compiler tiles and
all 315 historical Triton package hashes are verified. Other hardware and
single-card full scheduling have not yet been validated. Smoke peak memory
and full GPU hours will be recorded from fresh runs rather than estimated from
reference prediction files.

Install uv 0.9.24 from its [public release](https://github.com/astral-sh/uv/releases/tag/0.9.24),
make CUDA 12.4.131 available and set `CUDA_HOME` to that toolkit locally. The
public prebuilt FlashAttention wheel requires a newer glibc than the historical
Ubuntu 20.04 host, so the reference setup builds the pinned public source.

```bash
git clone https://github.com/pjunjie/GRKV.git
cd GRKV
bash scripts/setup_reference.sh
cp .env.example .env.local
# The runner reads .env.local; keep real credentials and local paths untracked.
uv run --no-sync python scripts/fetch_assets.py --models all --datasets all
uv run --no-sync python -m grkv.run --help
```

The runnable local kernel preview and its separate public hashes are under
validation. It redacts private debug paths and preserves every nondebug ELF
section; original complete-file SHA equality is explicitly false. Real Q32
JVP/VJP outputs match the historical qualification tensor hashes. See
[the historical kernel documentation](docs/historical_kernels.md). Kernel
fetching and final smoke/full commands will be documented after their actual
validation. Generation requires no reference prediction archive.

Mistral preserves successful FlashAttention2 calls and only uses explicit
causal/per-head masking when fake-key feasibility fails. The historical
reference contains 22 such calls in 22 units; fresh runs measure this statistic.
Llama never installs that fallback. Installation, CPU and GPU validation
statuses are reported separately. The digest-pinned Dockerfile is provided,
but container execution remains unvalidated because this host does not allow
Docker access and has no rootless subordinate UID/GID range.

Run `make style` and `make test` for SPDX, style/type checks and the targeted CPU
protocol/fallback tests. Keep upstream Apache-2.0 notices. Contributions require
DCO sign-off; automated commit messages and issue/PR titles/comments end with
`🤖🤖🤖` as specified in `CONTRIBUTING.md`.
