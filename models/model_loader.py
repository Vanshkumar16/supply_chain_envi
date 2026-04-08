"""
Model Loader — Load OpenEnv and other Hugging Face models
=========================================================
Handles model downloading, caching, and loading with proper error handling.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Model cache directory
MODELS_DIR = Path(__file__).parent
CACHE_DIR = MODELS_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


class ModelLoader:
    """Load models from Hugging Face with caching and error handling."""

    def __init__(self, model_id: str, hf_token: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            model_id: Hugging Face model ID (e.g., "username/openenv")
            hf_token: Hugging Face API token
        """
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.cache_path = CACHE_DIR / model_id.replace("/", "--")

    def is_cached(self) -> bool:
        """Check if model is already cached locally."""
        return self.cache_path.exists() and (self.cache_path / "config.json").exists()

    def download_from_huggingface(self) -> None:
        """Download model from Hugging Face using huggingface_hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install it with: pip install huggingface_hub"
            )

        logger.info(f"Downloading model {self.model_id} from Hugging Face...")
        try:
            snapshot_download(
                self.model_id,
                cache_dir=str(CACHE_DIR),
                token=self.hf_token,
                local_dir=str(self.cache_path),
            )
            logger.info(f"Model cached at {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to download model {self.model_id}: {e}")
            raise

    def load(self):
        """Load model from cache or download if needed."""
        if not self.is_cached():
            self.download_from_huggingface()
        return self.cache_path


def get_model_path(model_id: str, hf_token: Optional[str] = None) -> Path:
    """Get or download a model, return its local path."""
    loader = ModelLoader(model_id, hf_token)
    return loader.load()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model_loader.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    path = get_model_path(model_id)
    print(f"Model path: {path}")
