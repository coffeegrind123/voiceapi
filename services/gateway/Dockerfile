FROM ghcr.io/astral-sh/uv:0.12.2 AS uv
FROM python:3.12.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/gateway \
    PATH=/opt/gateway/bin:${PATH} \
    VOICEAPI_DATA_DIR=/data

# Debian's ffmpeg ships libmp3lame and libopus for the mp3/opus response formats.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 2000 --gid 0 --create-home --shell /bin/bash gateway \
    && mkdir -p /data && chown -R 2000:0 /data && chmod -R g+rwX /data

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/src
COPY pyproject.toml ./
COPY src ./src
RUN uv venv /opt/gateway --python /usr/local/bin/python3.12 && uv pip install .

USER gateway
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8080/health || exit 1
CMD ["voiceapi-gateway"]
