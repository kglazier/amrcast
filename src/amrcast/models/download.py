"""Download pre-trained models from GitHub Releases."""

import logging
import tarfile
from io import BytesIO
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = (
    "https://github.com/kglazier/amrcast/releases/download/v0.1.0"
)

MODEL_ARCHIVES = {
    "ecoli": "ecoli_models.tar.gz",
    "salmonella": "salmonella_models.tar.gz",
    "klebsiella": "klebsiella_models.tar.gz",
}

# Where each species' models are expected
MODEL_DIRS = {
    "ecoli": Path("data/narms/models"),
    "salmonella": Path("data/salmonella/models"),
    "klebsiella": Path("data/klebsiella/models"),
}


def models_exist(species: str) -> bool:
    """Check if models are already downloaded for a species."""
    model_dir = MODEL_DIRS.get(species)
    if model_dir is None:
        return False
    return (model_dir / "feature_columns.json").exists()


def download_models(species: str) -> Path:
    """Download pre-trained models for a species from GitHub Releases.

    Returns the model directory path.
    """
    archive_name = MODEL_ARCHIVES.get(species)
    if archive_name is None:
        raise ValueError(f"Unknown species: {species}. Available: {list(MODEL_ARCHIVES)}")

    model_dir = MODEL_DIRS[species]

    if models_exist(species):
        logger.info(f"Models already exist: {model_dir}")
        return model_dir

    url = f"{GITHUB_RELEASE_URL}/{archive_name}"
    logger.info(f"Downloading {species} models from {url}...")

    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    # Extract to parent directory (archive contains models/ folder)
    extract_dir = model_dir.parent
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz") as tar:
        tar.extractall(path=extract_dir)

    logger.info(f"Models extracted to {model_dir}")
    return model_dir


def ensure_models(species: str) -> Path:
    """Ensure models exist, downloading if needed. Returns model dir."""
    if models_exist(species):
        return MODEL_DIRS[species]
    return download_models(species)
