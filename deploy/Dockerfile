# syntax=docker/dockerfile:1
#
# API image only — the three model servers are separate llama.cpp processes
# (see deploy/llama/*.sh and deploy/docker-compose.example.yml).
#
#   docker build -f deploy/Dockerfile -t memorysum-vl:latest .
#
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMORYSUM_DB_PATH=/data/memorysum.db

WORKDIR /app

# Install the package first from the pieces the build backend needs, so a code
# change does not invalidate the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 10001 memory \
 && mkdir -p /data && chown memory:memory /data

USER memory
VOLUME ["/data"]
EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/healthz').read()"

CMD ["uvicorn", "memorysum_vl.main:app", \
     "--host", "0.0.0.0", "--port", "8003", "--no-access-log"]
