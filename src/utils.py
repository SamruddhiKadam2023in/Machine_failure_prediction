"""Small shared helpers."""

import hashlib
from pathlib import Path

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for directory in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    """Checksum used to record exactly which dataset file was used."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
