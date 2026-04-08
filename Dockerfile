FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml pyproject.toml
COPY supply_chain_env/ supply_chain_env/
COPY models/ models/
COPY setup.sh setup.sh

# Install dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn

# Create .env with defaults (will be overridden by HF Spaces secrets)
RUN cat > .env << 'EOF'
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN=${HF_TOKEN}
ENV_BASE_URL=http://localhost:7860
EOF

# Expose port for HF Spaces (uses 7860)
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "supply_chain_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
