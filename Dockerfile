# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Compile psycopg2 against system libpq (avoids the -binary wheel in prod).
# All heavy pip work is isolated here so the runtime image stays lean.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Strip psycopg2-binary, install compiled psycopg2 instead.
# Pin CPU-only torch so sentence-transformers does not pull multi-GB CUDA wheels.
RUN pip install --upgrade pip \
    && grep -v 'psycopg2-binary' requirements.txt > requirements_prod.txt \
    && pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements_prod.txt \
        psycopg2==2.9.9 \
        torch==2.3.1+cpu

# Pre-warm the tiktoken cl100k_base encoding cache during build.
# TIKTOKEN_CACHE_DIR is set explicitly to a known path so the COPY in the
# runtime stage has a predictable source. Without this, tiktoken writes to a
# platform-dependent location that may not be /root/.tiktoken.
RUN mkdir -p /build/.tiktoken_cache \
    && PYTHONPATH=/install/lib/python3.11/site-packages \
       TIKTOKEN_CACHE_DIR=/build/.tiktoken_cache \
       python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken cache warmed')" \
    && ls -la /build/.tiktoken_cache


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="analyst-copilot" \
      org.opencontainers.image.description="LLM-powered data analyst copilot" \
      org.opencontainers.image.source="https://github.com/your-org/analyst_copilot"

# libpq5: runtime shared library required by compiled psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder
COPY --from=builder /install/lib /usr/local/lib
COPY --from=builder /install/bin /usr/local/bin

# Copy pre-warmed tiktoken cache from the explicit build-time cache directory.
# TIKTOKEN_CACHE_DIR was set to /build/.tiktoken_cache during the builder stage
# so this COPY has a predictable, verified source path.
COPY --from=builder /build/.tiktoken_cache /tiktoken_cache

WORKDIR /app

# Non-root user — principle of least privilege
RUN useradd --uid 1001 --create-home --shell /bin/sh appuser

# Application source (only what the server needs at runtime)
COPY analyst_copilot/ ./analyst_copilot/
COPY app.py main.py .slowapi.env ./

# Persistent data directories — override with volume mounts in production.
# model_cache MUST be listed here: Docker named volumes are created with root
# ownership; pre-creating the dir as appuser before the volume mounts over it
# gives the non-root process write access. Without this line, sentence-transformers
# gets PermissionError: [Errno 13] on first download of BAAI/bge-large-en-v1.5.
RUN mkdir -p data/faiss_index data/files data/lt_memory data/model_cache \
    && chown -R appuser:appuser /app \
    # Install pre-warmed tiktoken cache for appuser.
    # /tiktoken_cache was copied from the builder's explicit TIKTOKEN_CACHE_DIR.
    && mkdir -p /home/appuser/.tiktoken \
    && cp -r /tiktoken_cache/. /home/appuser/.tiktoken/ \
    && chown -R appuser:appuser /home/appuser/.tiktoken \
    && rm -rf /tiktoken_cache

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # BAAI/bge-large-en-v1.5 sentence-transformers cache
    SENTENCE_TRANSFORMERS_HOME=/app/data/model_cache \
    # Adds analyst_copilot/ inner dir to sys.path so flat imports resolve
    # (same effect as the sys.path injection in main.py — belt-and-suspenders)
    PYTHONPATH=/app/analyst_copilot \
    # Pre-warmed tiktoken encoding cache (cl100k_base downloaded at build time).
    # Prevents runtime download attempt to openaipublic.blob.core.windows.net.
    TIKTOKEN_CACHE_DIR=/home/appuser/.tiktoken

EXPOSE 8000

# start-period covers the ~40s BAAI/bge-large-en-v1.5 warmup
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=5); \
         sys.exit(0 if r.status == 200 else 1)"

# Single worker default — scale horizontally with multiple containers.
# For multi-worker single-container: --workers $(nproc) (each worker owns its own
# _app_state global; FAISS index and embedding model are loaded per-worker).
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
