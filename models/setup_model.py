#!/usr/bin/env python3
"""
Setup script to prepare OpenEnv model for supply chain environment.
Usage: python setup_model.py [--model-id <model_id>] [--hf-token <token>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent


def install_dependencies():
    """Install required packages for model loading."""
    deps = [
        "huggingface_hub",
        "transformers",
        "torch",
    ]
    print("[*] Installing model dependencies...")
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  ✓ {dep} already installed")
        except ImportError:
            print(f"  → Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
            print(f"  ✓ {dep} installed")


def download_model(model_id: str, hf_token: str = None):
    """Download model from Hugging Face."""
    from model_loader import get_model_path

    print(f"\n[*] Setting up model: {model_id}")
    print(f"  Cache dir: {MODELS_DIR / '.cache'}")

    try:
        path = get_model_path(model_id, hf_token)
        print(f"  ✓ Model ready at: {path}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup OpenEnv model for supply chain environment"
    )
    parser.add_argument(
        "--model-id",
        default="openenv/supply-chain-model",
        help="Hugging Face model ID (default: openenv/supply-chain-model)",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip installing dependencies",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OpenEnv Model Setup for Supply Chain Disruption Management")
    print("=" * 60)

    if not args.skip_deps:
        install_dependencies()

    success = download_model(args.model_id, args.hf_token)

    print("\n" + "=" * 60)
    if success:
        print("✓ Setup complete! Model is ready to use.")
        print("\nTo use the local model in inference.py, set:")
        print("  export USE_LOCAL_MODEL=true")
        print(f"  export LOCAL_MODEL_ID={args.model_id}")
    else:
        print("✗ Setup failed. Check your Hugging Face token and model ID.")
        sys.exit(1)


if __name__ == "__main__":
    main()
