# Validation status and historical evidence

Native full-inference acceptance is complete: each model covers all 19,630
context/budget units and 20,500 answers. Both full result sets were independently
rescored from prediction text and freshly obtained frozen data. All eight
complete-precision targets match within 1e-8, and all eight point gains over the
historical Critical-AdaKV baselines are positive. Text, per-answer scores,
fit-layer records and layout have zero strict differences.

| Model | Accepted answers | Context/budget units | Accepted generation source |
|---|---:|---:|---|
| Mistral | 20,500 | 19,630 | Current complete clean-checkout inference |
| Llama | 20,500 | 19,630 | Verified consolidation of current and earlier Triton 3.2.0 / historical-kernel inference |

Mistral generated the complete set from clean checkout
`246d124b5c1de0ed2a6cc14714261b5f1f75604d`. Its exact-mask fallback total is
22 calls in 22 units, independently measured and identical to the final
reference. The consolidated Llama set selects current generated outputs where
available and the corresponding genuine earlier experiment outputs otherwise.
Every selected earlier raw checkpoint, frozen sample and actual historical
pre-launch cubin proof was checked. Every accepted record was compared against
the frozen result, followed by independent full rescoring with strict coverage.
The numerical source, configuration, model/data revisions, historical Triton
package and kernel instruction identity remain frozen. The measured scores and
algorithm parameters are unchanged.

Each Llama record retains its generation origin, source archive SHA, original
checkpoint SHA and actual pre-launch binary checks. Consolidation is not a new
inference run; it does not label earlier outputs as generated in the current
checkout. The primary full-result manifest and verification are under
`results/validated/`; the assembly entry point is
`scripts/consolidate_llama_results.py`. The source-run archives and earlier
audit receipts remain available with their original identities. The existing
Git tags and history are preserved. All eight evidence and kernel archives
are now published together in the [unified release](release.md), with unchanged
contents and SHA-256 hashes. Operational download manifests use that release;
earlier audit receipts retain their actual historical URLs and checkout scope.

Historical Critical-AdaKV columns are separately rescored historical baselines.
Baseline constructors passed CPU checks; a new baseline GPU-entry smoke and a
new remote-checkout GPU smoke were not executed. Earlier clean-checkout GPU
smoke and numerical qualification passed before the host fault. No additional
GPU jobs were submitted after that scope change; existing Mistral workers
finished naturally.

The current Mistral full run used 4 successful workers,
55.891 worker GPU hours and 14.118 wall hours.
Its measured peak allocated/reserved CUDA memory was 35.585/41.406 GiB. These are allocator measurements,
excluding non-PyTorch device allocations. Worker time includes model loading
and qualification; per-unit generation time excludes preprocessing. These
figures describe the completed Mistral inference run. See
`results/fresh/mistral/resources.json`.

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
and imported the public constructors, runner and scorer. CPU CI checks
installation and protocol behavior; the separate GPU receipts record smoke,
numerical and full-result validation.

Historical binary identity is disclosed separately. The public cubins redact
private strings only in nonallocated debug sections. Every nondebug ELF
section matches the historical artifact, but original complete-file SHA
equality is false. The pre-launch backend verifies compiled sections, public
binary SHA, shared memory and architecture before first loading a kernel;
subsequent calls must reuse that same verified compiled object. The reports record
`original_cubin_bytes_identical=false` and separately verify executable-section
identity.

Native installation is validated. Container build and GPU execution remain
uncompleted limitations: this validation process could not access the Docker
daemon. Native installation and GPU execution are the validated path. Single-card full scheduling and other
architectures are not validated. The earlier remote checkout of the initial
publication installed the locked CPU environment without
FlashAttention, passed 17 tests and style/type/SPDX checks, downloaded all seven
evidence/kernel archives and SHA256SUMS from the public GitHub Release, and
verified every size/SHA plus actual downloaded member privacy. It independently
rescored all 82,000 historical answers and all 38,211 source-run answers. Those
receipts retain their original source-run scope. No new remote GPU
smoke was launched. The verified uv cache and previously freshly downloaded
model/data assets were reused and are disclosed in
`results/validation/remote_checkout/summary.json`.

The historical Critical-AdaKV baseline columns are independently rescored
historical results, not new baseline generation. The older Mistral K1 control
uses another compiler/protocol and is not substituted for that baseline.
Mistral's final reference is assembled from 10,191 answers from prior stages
of the final candidate run and 10,309 continuation answers with verified source
content hashes. Its historical fallback total is 22 calls in 22 units. The current full run
independently measured 22 calls in 22 units with zero per-unit differences.

Reproduction validation checks complete sample coverage, independent rescoring
and agreement with the frozen outputs and scores. GRKV has higher task-equal
mean scores than the historical Critical-AdaKV baseline across all eight
reported settings. The archived 50/50 development/remaining labels describe
the parameter-search split within the already evaluated full sample set.
[The evaluation protocol](protocol.md) records that split and the paired
confidence intervals, including the Llama LongBench/20% mean gain of +0.1885
points and its paired 95% interval of [-0.2648, 0.6340] points.

The earlier remote checkout of the consolidated Llama publication installed
the locked CPU environment,
passed style/type/SPDX checks on 84 Python files and all 17 targeted
tests, and downloaded the consolidated Llama archive and checksum list
from the public Release. Independent full rescoring and strict
per-question comparison of all 20,500 downloaded answers passed.
Generation origins and pre-launch binary proofs were checked for every
record. The unchanged initial-publication assets retain their verified size/SHA
identities and earlier public-download audit. No new GPU task was
launched. See `results/validated/remote_checkout/summary.json`.
