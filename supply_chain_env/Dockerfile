# ── Supply Chain Disruption Management — OpenEnv Environment ──────────────
# Dockerfile for Hugging Face Spaces (Docker SDK)
# Build : docker build -t supply-chain-env .
# Run   : docker run -p 8000:8000 supply-chain-env
# HF    : tagged with "openenv" — deploys as HF Space
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="OpenEnv Hackathon"
LABEL description="Supply Chain Disruption Management — OpenEnv RL Environment"
LABEL org.opencontainers.image.title="supply-chain-env"
LABEL org.opencontainers.image.version="1.0.0"

# HF Spaces requires port 8000
EXPOSE 8000

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py          /app/models.py
COPY graders.py         /app/graders.py
COPY client.py          /app/client.py
COPY inference.py       /app/inference.py
COPY openenv.yaml       /app/openenv.yaml
COPY server/            /app/server/
COPY pyproject.toml     /app/pyproject.toml

# Create output directories
RUN mkdir -p /app/outputs/logs /app/outputs/evals \
 && chown -R appuser:appuser /app

USER appuser

# Health check — HF Spaces pings / to verify deployment
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/'); r.raise_for_status()" \
    || exit 1

# Start FastAPI server on 0.0.0.0:8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
