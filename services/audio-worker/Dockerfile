# Audio worker: NVIDIA RE-USE enhancement, Qwen3-ASR, ForcedAligner.
# Everything is a prebuilt wheel: torch 2.10.0+cu130 bundles its CUDA runtime, and the
# mamba_ssm selective-scan kernel comes from the project's release wheel for exactly that
# torch/CUDA/Python combination. Nothing is compiled during the build.
FROM ghcr.io/astral-sh/uv:0.12.2 AS uv
FROM python:3.12.12-slim-bookworm

ARG TORCH_VERSION=2.10.0
ARG MAMBA_WHEEL=https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/worker \
    PATH=/opt/worker/bin:${PATH} \
    HF_HOME=/cache/huggingface \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    VOICEAPI_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 2000 --gid 0 --create-home --shell /bin/bash worker \
    && mkdir -p /data /cache && chown -R 2000:0 /data /cache && chmod -R g+rwX /data /cache

COPY --from=uv /uv /usr/local/bin/uv
RUN uv venv /opt/worker --python /usr/local/bin/python3.12 \
    && uv pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
    && uv pip install --no-deps "${MAMBA_WHEEL}"

WORKDIR /opt/src
COPY pyproject.toml ./
COPY src ./src
RUN uv pip install --no-cache . \
    && python -c "import torch, selective_scan_cuda, transformers; print(torch.__version__, transformers.__version__)"

USER worker
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s --start-period=900s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8001/health || exit 1
CMD ["voiceapi-worker"]
