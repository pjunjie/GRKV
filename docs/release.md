# Unified GRKV release

The [GRKV v0.3.0 release](https://github.com/pjunjie/GRKV/releases/tag/v0.3.0)
brings the complete implementation, fixed environment, evaluation protocol and
all eight evidence/kernel archives into one publication. It also includes the
current README and the evaluation protocol without experimental gain or TTFT
acceptance gates. The measured scores and default numerical behavior are preserved.
Both the current main branch and release source archives contain GRKV Default
and its required dependencies. Earlier Git tags retain their original source snapshots.

| Archive | Content |
|---|---|
| `grkv-default-sm86-kernels.tar.gz` | Historical SM86 executable kernels, sanitized debug strings and kernel manifest |
| `fresh_mistral_candidate.jsonl.gz` | Accepted complete Mistral inference: 20,500 answers |
| `validated_llama_candidate.jsonl.gz` | Accepted complete Llama evidence: 20,500 answers, verified current and earlier generation origins retained |
| `fresh_llama_candidate_partial.jsonl.gz` | Original current Llama source-run evidence, retained for provenance |
| `mistral_candidate.jsonl.gz` | Historical Mistral GRKV reference |
| `llama_candidate.jsonl.gz` | Historical Llama GRKV reference and verified earlier consolidation source |
| `mistral_critical.jsonl.gz` | Historical Mistral Critical-AdaKV baseline reference |
| `llama_critical.jsonl.gz` | Historical Llama Critical-AdaKV baseline reference |

`SHA256SUMS` covers the seven unchanged prediction archives and the GRKV Default
kernel archive. The kernel archive contains only the required JVP and VJP
cubins and their manifest; both public cubin byte sequences and SHA-256 hashes
are unchanged. Removing unused kernels and renaming archive members changes
the archive and manifest hashes, recorded in the active download manifest.
The download commands in [reproduction.md](reproduction.md) use updated,
size- and checksum-verified manifest URLs. GitHub also supplies the source
archives for the release tag. Public models and datasets remain upstream
downloads pinned by revision and file checksums.

Downloading evidence does not generate predictions. Both accepted full results
were independently rescored and strictly compared with the frozen outputs;
Llama records retain their actual generation origins. Original cubin full-file
byte equality remains false, while executable-section identity, numerical
qualification and per-question comparisons are reported separately. See
[validation results](results.md). Docker build and GPU container acceptance
remain uncompleted limitations.

Earlier Git tags and commit history remain available. Earlier publication audits
are summarized with their original content hashes and source-snapshot links;
complete original records remain in Git history. Use the unified
release and current download manifests for active downloads.
