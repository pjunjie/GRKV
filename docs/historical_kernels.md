# Historical kernels and the privacy boundary

The historical backend targets CUDA SM86 with warp size 32. JVP uses
`(BR=32, BS=64, num_warps=4)`; VJP uses `(64, 32, 8)`; both use one stage.
LLVM priming, reduction order, explicit `tf32x3`, FP32 fitting and BF16 writes
are retained. Every real launch checks architecture, shared memory, signature,
alignment and public cubin SHA before CUDA creates handles.

The original 18 cubins contain private build paths in nonallocated debug
sections. They cannot be published unchanged. The local release preview
redacts those strings with equal-length anonymous strings and verifies every
nondebug section is byte-identical, including GPU instructions, constants,
symbol tables and allocation metadata. This changes the complete file SHA.
Original and public hashes are separate manifest fields. No claim is made that
redacted binaries have the original complete-file hashes.

For Q32 the original JVP SHA is
`df2ff825ab436b57318c50bb266593d3625aac93b870f60011a90d57bb8acae1` and original
VJP SHA is `fc7d86adf5b41b7903c70c31a360c1960dd1b643af47a41255300179c1fc8a04`.
The preview uses separate verified public SHA values recorded in its manifest.
It compiles the unchanged numerical kernel in Triton 3.2.0, requires the compiled
nondebug sections to equal the historical sections, then installs the redacted
cubin before the first launch. It does not execute an unchecked newly compiled
binary or bypass a failed executable-section comparison.

The real synthetic JVP and VJP qualification has passed, and both resulting
tensor hashes exactly match the historical qualification. This checks real GPU
execution, but does not replace smoke or full generation validation. Original
complete-file binary equality and full prediction equality remain separate
checks. A release using this profile must disclose the binary identity change.

Triton 3.2.0 comes from the pinned PyTorch wheel. All 315 package file hashes
match the historical package, including `compiler.py` and `libtriton.so`.
The PyPI manylinux2014 wheel has different compiled libraries and is not silently
substituted. Runtime checks verify actual imported modules and every package
file. Third-party runtime packages are obtained from their public sources;
private live compiler caches and cache JSON paths are not distributed.
