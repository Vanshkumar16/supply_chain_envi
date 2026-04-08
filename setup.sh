#!/usr/bin/env bash
# Quick setup script for supply_chain_env

set -e

echo "================================================"
echo "Supply Chain Environment Setup"
echo "OpenEnv Framework + LLM Inference"
echo "================================================"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT"

# 1. Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
if [ -f "supply_chain_env/pyproject.toml" ]; then
    pip install -e .
    echo "✓ Dependencies installed"
else
    echo "✗ pyproject.toml not found"
    exit 1
fi

# 2. Verify OpenEnv installation
echo ""
echo "[2/4] Verifying OpenEnv framework..."
python -c "import openenv; print(f'✓ OpenEnv {openenv.__version__} ready')" || {
    echo "✗ OpenEnv installation failed"
    exit 1
}

# 3. Verify core packages
echo ""
echo "[3/4] Verifying core packages..."
python -c "
import fastapi
import pydantic
import numpy
import openai
print('✓ FastAPI, Pydantic, NumPy, OpenAI ready')
"

# 4. Show next steps
echo ""
echo "[4/4] Setup Summary"
echo "================================================"
echo "✓ OpenEnv framework configured"
echo "✓ FastAPI server ready"
echo "✓ Dependencies locked in pyproject.toml"
echo ""
echo "Next steps:"
echo ""
echo "1. Start environment server:"
echo "   cd supply_chain_env && python -m server.app"
echo ""
echo "2. Run inference (in another terminal):"
echo "   export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct"
echo "   export HF_TOKEN=your_token"
echo "   python supply_chain_env/inference.py"
echo ""
echo "3. (Optional) Use local model:"
echo "   export USE_LOCAL_MODEL=true"
echo "   export LOCAL_MODEL_ID=meta-llama/Llama-2-7b-chat-hf"
echo "   python models/setup_model.py --model-id <model-id>"
echo ""
echo "================================================"
