# Models Directory

This directory manages AI models and framework setup for the supply chain disruption management environment.

## What is OpenEnv?

**OpenEnv** is an environment framework (not a model) from Meta PyTorch. It standardizes how to build and interact with isolated execution environments for reinforcement learning. Your `supply_chain_env` already implements OpenEnv's architecture with:
- `reset()` - Initialize episode
- `step()` - Transition to next state
- `grade()` - Score the episode

See: https://github.com/meta-pytorch/OpenEnv

## Installation

### Option 1: Install from source (recommended)
```bash
cd /path/to/supply_chain_env
pip install -e .  # Installs with all dependencies
```

### Option 2: Install specific extras
```bash
# Core only (OpenEnv + FastAPI server)
pip install -e .

# With model support
pip install -e ".[models]"

# Everything
pip install -e ".[all]"
```

## LLM Model Integration

For running inference with language models:

```bash
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=your_hf_token
python ../supply_chain_env/inference.py
```

### Local Model Support
To use a local Hugging Face model instead of API:

```bash
export USE_LOCAL_MODEL=true
export LOCAL_MODEL_ID=meta-llama/Llama-2-7b-chat-hf
python setup_model.py --model-id meta-llama/Llama-2-7b-chat-hf
python ../supply_chain_env/inference.py
```

## Structure

- `model_loader.py` - Model downloading and caching
- `setup_model.py` - Setup script to prepare models
- `.cache/` - Local model cache (gitignored)
- `.env.example` - Environment template

## Dependencies

All dependencies defined in `../supply_chain_env/pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| openenv-core | >=0.2.0 | OpenEnv framework |
| fastapi | >=0.110.0 | HTTP API server |
| transformers | >=4.36.0 | HF models |
| torch | >=2.1.0 | ML inference |
| openai | >=1.20.0 | LLM API client |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_LOCAL_MODEL` | `false` | Use local model instead of API |
| `LOCAL_MODEL_ID` | (none) | Which local model to load |
| `MODEL_NAME` | `meta-llama/Llama-3.3-70B-Instruct` | API model ID |
| `HF_TOKEN` | (none) | Hugging Face API token |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `ENV_BASE_URL` | `http://localhost:8000` | Environment server |

## Testing

```bash
# Verify OpenEnv installation
python -c "import openenv; print(openenv.__version__)"

# Test environment server
python -m supply_chain_env.server

# Run inference
python supply_chain_env/inference.py
```

