# Historical kernels and the privacy boundary

The historical backend targets CUDA SM86 with warp size 32. JVP uses
`(BR=32, BS=64, num_warps=4)`; VJP uses `(64, 32, 8)`; both use one stage.
LLVM priming, reduction order, explicit `tf32x3`, FP32 fitting and BF16 writes
are retained. Every first kernel load checks architecture, shared memory, signature,
alignment and public cubin SHA before CUDA creates handles; subsequent calls
must reuse that verified compiled object.

The original 18 cubins contain private build paths in nonallocated debug
sections. The public kernel package
redacts those strings with equal-length anonymous strings and verifies every
nondebug section is byte-identical, including GPU instructions, constants,
symbol tables and allocation metadata. This changes the complete file SHA.
Original and public hashes are separate manifest fields. No claim is made that
redacted binaries have the original complete-file hashes.

For Q32 the original JVP SHA is
`df2ff825ab436b57318c50bb266593d3625aac93b870f60011a90d57bb8acae1` and original
VJP SHA is `fc7d86adf5b41b7903c70c31a360c1960dd1b643af47a41255300179c1fc8a04`.
The public Q32 JVP SHA is
`a95bc3cf598ae2b83b82d78b53a67e5e9b184e6ef3433984c0d93f91774ba5c1`;
the public Q32 VJP SHA is
`09e970025e87d60198db2e4d9545c64cb7e9035d6ef69de389c34e76a524a890`.
All 18 original/public identities are recorded in the downloaded manifest,
with package size/SHA and its Release link in [kernels.json](../manifests/kernels.json).
It compiles the unchanged numerical kernel in Triton 3.2.0, requires the compiled
nondebug sections to equal the historical sections, then installs the redacted
cubin before the first launch. It does not execute an unchecked newly compiled
binary or bypass a failed executable-section comparison.

The real synthetic JVP and VJP qualification has passed, and both resulting
tensor hashes exactly match the historical qualification. This checks real GPU
execution, but does not replace smoke or full generation validation. Original
complete-file binary equality and full prediction equality remain separate
checks. `original_cubin_bytes_identical=false` is retained in the reports.
The real GPU numerical receipt is [derivative_qualification.json](../results/validation/derivative_qualification.json);
per-question full and observed-subset comparisons are reported separately in
[results.md](results.md).

Triton 3.2.0 comes from the pinned PyTorch wheel. All 315 package file hashes
match the historical package, including `compiler.py` and `libtriton.so`.
The PyPI manylinux2014 wheel has different compiled libraries and is not silently
substituted. Runtime checks verify actual imported modules and every package
file. Third-party runtime packages are obtained from their public sources;
private live compiler caches and cache JSON paths are not distributed.
