# Third-party notices

This implementation vendors a required subset of [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
and retains its Apache-2.0 license and NVIDIA copyright headers. The source
commit, source file hashes and export transformations are recorded in
`manifests/source_provenance.json`. Numerical experiment adapters retain the
same SPDX notices as the source project. The GRKV Default dependency subset
includes the exact Critical-AdaKV selector, cache representation, solver,
pipeline, attention patch and LongBench metrics used by the frozen protocol.

External dependencies retain their own licenses. [Triton](https://github.com/triton-lang/triton)
is installed from the pinned public PyTorch wheel; [FlashAttention](https://github.com/Dao-AILab/flash-attention)
is built from the pinned public PyPI source distribution using CUDA 12.4.
Neither third-party package is copied from the author's installation into Git.
Kernel assets are derived from the included Triton source; their manifests
record the historical executable provenance and debug-path redaction.

Models and benchmark bodies are not distributed here. Obtain the exact
revisions listed in `manifests/models.json` and `manifests/datasets.json` from
Hugging Face under their upstream access requirements and licenses. See the
Mistral and Llama model cards, the formatted LongBench and RULER dataset cards,
[the original LongBench](https://huggingface.co/datasets/zai-org/LongBench), and
[the RULER generator](https://github.com/NVIDIA/RULER). This repository's code
license does not relicense those resources. Generated prediction references
and numerical records are labeled separately from upstream dataset bodies.
