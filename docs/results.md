# Validation status and historical evidence

Native full-inference acceptance passed using explicitly separate sources.
Mistral newly generated all 19,630 context/budget units and 20,500 answers from
clean checkout `246d124b5c1de0ed2a6cc14714261b5f1f75604d`. Complete row coverage,
independent rescoring, all four full-precision targets within 1e-8 and four
positive point gains passed, with zero text, per-answer score, fit-layer, layout
and exact-mask fallback differences against the final reference.

Llama's earlier genuine Triton 3.2.0 recovery experiment already generated all
19,630 units and 20,500 answers. The audit rechecked every raw checkpoint SHA,
frozen sample identity, actual historical pre-launch binary proof, exported
record and original K1 comparison. Prediction, per-answer score and fit-layer
differences are zero. These earlier full outputs were independently rescored
against freshly downloaded frozen data and reproduce all four complete-precision
targets within 1e-8. They remain earlier reproduction evidence, not current
clean-checkout generation. The earlier Mistral complete evidence was also
traced and independently rescored. See `results/validation/existing_full_evidence/`.

The current Llama attempt contains 16,841 units and 17,711 genuinely new answers.
Every observed answer was independently rescored and strictly compared: zero
text, score, fit-layer, layout and fallback differences. Its 2,789 missing units
and answers are explicitly listed in `results/fresh_partial/llama/verification.json`.
No prediction is filled from references. This subset does not reproduce a full
benchmark score or establish current full coverage. Host GPU/driver faults
prevented reliable new CUDA initialization; no additional GPU jobs were submitted
after the scope change. Existing Mistral workers were allowed to finish naturally.

| Model | Current new answers | Missing answers | Current coverage | Accepted full-score source |
|---|---:|---:|---|---|
| Mistral | 20,500 | 0 | Full | Current clean-checkout inference |
| Llama | 17,711 | 2,789 | Observed subset | Earlier full Triton 3.2.0 reproduction |

The two current-generation archives have separate names and manifests; the
Llama archive is explicitly suffixed `_partial`. Exported predictions were
independently rescored again. Downloading any archive is not new local inference.
The current generated total is 38,211, not 41,000. Historical Critical-AdaKV
columns are separately rescored historical baselines, not regenerated baselines.
The baseline constructors passed CPU checks; a new baseline GPU-entry smoke
and a new remote-checkout GPU smoke were not executed. Earlier clean-checkout
GPU smoke and numerical qualification below passed before the host fault.

The current Mistral full run used 4 successful workers,
55.891 worker GPU hours and 14.118 wall hours.
Its measured peak allocated/reserved CUDA memory was 35.585/41.406 GiB. These are allocator measurements,
excluding non-PyTorch device allocations. Worker time includes model loading
and qualification; per-unit generation time excludes preprocessing. These
are run resource measurements, not a formal performance benchmark or minimum
hardware guarantee. The current Llama attempt is incomplete; no completed
full-run timing is inferred from it. See `results/fresh/mistral/resources.json`.

All four historical prediction assets have been independently rescored against
freshly obtained, revision-pinned data: 82,000 answers, complete task/row
coverage and zero saved-score mismatches. All 16 candidate/baseline macro
scores agree with the original full-precision targets within 1e-8. See
`results/reference/scores.json`, `task_scores.csv`, `coverage.json` and
`provenance.json`. Per-unit source content hashes are included in the
self-contained reference records; private source paths are removed.

The clean-checkout smoke generated 88 answers in 60 context/budget units per
model. Both models have zero prediction, per-answer score, fit-layer and
fallback differences. Mistral measured two exact-mask fallback calls in that
subset; Llama measured zero. These subset macro scores are not full targets.
Both models passed actual GPU short-context preservation and A/B/A shared-cache
restoration checks at both budgets. The separate real derivative qualification
matched both historical Q32 JVP/VJP tensor hashes. Machine-readable reports are
under `results/validation/`.

The source/type/SPDX checks and 17 targeted tests passed. An independent CPU
environment without FlashAttention or visible GPUs also passed those 17 tests
and imported the public constructors, runner and scorer. This supports the
scope of CPU CI; it is not a substitute for GPU/full-run evidence.

Historical binary identity is disclosed separately. The public cubins redact
private strings only in nonallocated debug sections. Every nondebug ELF
section matches the historical artifact, but original complete-file SHA
equality is false. The pre-launch backend verifies compiled sections, public
binary SHA, shared memory and architecture before first loading a kernel;
subsequent calls must reuse that same verified compiled object. No report
claims the public cubins retain the original full-file SHA.

Native installation is validated. Container build and GPU execution remain
uncompleted limitations: this validation process could not access the Docker
daemon. Native installation and GPU execution are the validated path. Single-card full scheduling and other
architectures are not validated. Kernel/reference Release downloads are
pending publication; their fetch/extraction/checksum logic passed a local HTTP
rehearsal, which is explicitly distinguished from a remote Release download.

The historical Critical-AdaKV baseline columns are independently rescored
historical results, not new baseline generation. The older Mistral K1 control
uses another compiler/protocol and is not substituted for that baseline.
Mistral's final reference is assembled from 10,191 answers from prior stages
of the final candidate run and 10,309 continuation answers with verified source
content hashes. Its historical fallback total is 22 calls in 22 units. The current full run
independently measured 22 calls in 22 units with zero per-unit differences.

The original Llama `all_four_pass=false` means all four gains must be at least
0.5 points. The release's positive-gain check is a separate field. Historical
full results had previously been exposed; the remaining 50% was held out only
from that tuning round. Comparisons remain descriptive, and the historical
Llama LongBench/20% paired interval crosses zero. Exact output reproduction,
positive point gains and statistical significance are distinct statements.
