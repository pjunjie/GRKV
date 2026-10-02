# Native environment is the tested reference. Container execution is pending validation.
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04@sha256:8d577fd078ae56c37493af4454a5b700c72a7f1aeb9ff3d0adc0459fc47752a4
ENV DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential git ripgrep && rm -rf /var/lib/apt/lists/*
# Fixed public uv installer; verify its version before creating the environment.
RUN curl -LsSf https://astral.sh/uv/0.9.24/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
RUN uv --version && uv python install 3.12.7
WORKDIR /opt/grkv
COPY . .
RUN uv sync --frozen --extra eval && FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 NVCC_THREADS=2 uv sync --frozen --extra eval --extra flash-attn
CMD ["uv", "run", "--no-sync", "python", "-m", "grkv.run", "--help"]
