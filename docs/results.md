# Validation status and historical evidence

Full fresh generation is running from a clean checkout of candidate commit
`246d124b5c1de0ed2a6cc14714261b5f1f75604d`. Each model must produce 19,630
context/budget units and 20,500 answers. The eight full-run target checks are
pending. Historical reference scores are not reported as new inference scores.

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
content hashes. Its historical fallback total is 22 calls in 22 units; the full
fresh total will be measured and compared rather than filled from this number.

The original Llama `all_four_pass=false` means all four gains must be at least
0.5 points. The release's positive-gain check is a separate field. Historical
full results had previously been exposed; the remaining 50% was held out only
from that tuning round. Comparisons remain descriptive, and the historical
Llama LongBench/20% paired interval crosses zero. Exact output reproduction,
positive point gains and statistical significance are distinct statements.
