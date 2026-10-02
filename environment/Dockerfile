# Reference container; see environment/reference.json for its measured validation status.
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04@sha256:8d577fd078ae56c37493af4454a5b700c72a7f1aeb9ff3d0adc0459fc47752a4
ENV DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential git ripgrep && rm -rf /var/lib/apt/lists/*
# Install tools outside the root home so an unprivileged runtime user can execute them.
ENV UV_INSTALL_DIR=/usr/local/bin UV_PYTHON_INSTALL_DIR=/opt/python
RUN curl -LsSf https://astral.sh/uv/0.9.24/install.sh | sh
RUN uv --version && uv python install 3.12.7
WORKDIR /opt/grkv
COPY . .
RUN uv sync --frozen --extra eval && FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 NVCC_THREADS=2 uv sync --frozen --extra eval --extra flash-attn
RUN mkdir -p assets/models assets/datasets assets/arrow artifacts/kernels outputs
CMD ["/opt/grkv/.venv/bin/python", "-m", "grkv.run", "--help"]
