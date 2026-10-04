# GRKV evaluation protocol

GRKV applies global regression to the exact historical Critical-AdaKV selected
positions. It captures Q from the actual context prefill, chooses 32 stratified
positions, rotates them at the last context position, and fits one FP32 round
with pre-projection loss. The ridge is 0.01, the relative K cap is 0.025, the V
cap is 0.1, CG uses 16 iterations and tolerance 1e-6, and selected cache updates
are written back as BF16. Total query weight is 6.0. The protected window is 32,
SnapKV kernel size is 5, and there are no additional sink tokens. The safety
guard is 0.0 for Mistral and 0.1 for Llama. Complete default parameters are in
the YAML files, and `grkv.api` supplies the model-specific GRKV constructors.

KV retention 10% and 20% corresponds to `compression_ratio` 0.9 and 0.8.
The implementation preserves `int(tokens * (1 - compression_ratio))`, including
floating point truncation. Inputs of at most 32 context tokens retain every
slot; their effective KV retention is 100%.

Both models run in eval mode with frozen BF16 weights, FlashAttention2, seed 42,
and PyTorch matmul TF32 disabled. Triton explicitly uses `tf32x3`; PyTorch's
switch does not control that setting. Normal attention calls use FlashAttention2.
Only Mistral installs the exact-mask fallback, triggered by the specific
fake-key feasibility error. It applies per-head pruning and the causal mask
for that call; unrelated exceptions propagate. The reference Mistral candidate
used 22 fallback calls in 22 context/budget units. Fresh runs count calls rather
than assuming that value. Llama does not install this fallback.

Generation uses the vendored `KVPressTextGenerationPipeline` and the historical
`generate` function: greedy decoding, original answer prefixes, task-specific
`max_new_tokens`, termination rules, cache cropping after each answer, and the
original chat-template preprocessing. Multiple questions in one context share
the compressed context cache. Candidate context truncation is
`min(tokenizer.model_max_length, int(1e10))`. Mistral candidates do not inherit
the older K1 control's explicit 32,768-token limit.

Each model has 19,630 context/budget units and 20,500 answers across two budgets.
Per budget, LongBench contains 16 tasks and 3,750 answers: `multifieldqa_en` has
150, `lcc` and `repobench-p` have 500 each, and all other tasks have 200.
RULER16k contains 13 tasks with 500 answers each. Exact task names, original row
indices, example/context hashes and development/remaining labels are in
`manifests/units.jsonl`; dataset bodies are downloaded from the pinned upstream
revisions. The RULER `16384` dataset is the fixed published data prepared using
the Llama tokenizer, also used unchanged for Mistral.

LongBench's historical per-answer scorer already produces 0–100 scores. RULER
scores are 0–1 and are multiplied by 100 once. First average within each task,
then average tasks equally. Unequal task sizes do not weight the macro score.
Duplicate, unexpected, missing or nonfinite answers fail validation. Scoring
reloads the pinned public answers and computes scores from prediction text.

Historical Critical-AdaKV is a separately identified baseline. The earlier
Mistral K1 control used recompiled Triton 3.7.1 kernels and an explicit context
limit; it is not the Critical-AdaKV column. Historical baseline references are
not labeled newly generated baselines. Baseline inference has separate configs
and outputs; a fresh baseline result never overwrites the historical column.

Reproduction validation checks complete sample coverage, independent rescoring,
and agreement with the frozen outputs and full-precision scores. Gains against
the historical baseline are reported as measurements.

Full benchmark scores use the complete frozen evaluation set. The archived
development/remaining labels record the 50/50 parameter-search split within
an already evaluated full sample set. Paired context-level 95% confidence
intervals are preserved in
[the statistical record](../results/reference/historical_statistics.json).
For Llama on LongBench at 20% KV retention, the measured mean gain is +0.1885
score points, with a recorded paired 95% interval of [-0.2648, 0.6340] points.
Mean gains and their intervals are reported together to describe the measured
results.
