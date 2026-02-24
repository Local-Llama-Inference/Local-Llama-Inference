"""Extract binary bundle on first use."""

import tarfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

INSTALL_DIR = Path.home() / ".local" / "share" / "local-llama-inference"
BUNDLE_PATTERN = "local-llama-inference-bins-linux-x86_64*.tar.gz"


def get_bundle_path() -> Optional[Path]:
    """
    Find the tar.gz binary bundle.

    Searches in:
    1. LLAMA_BUNDLE environment variable
    2. Package data directory
    3. binaries/ subdirectory in package

    Returns:
        Path to bundle or None if not found
    """
    import os

    # Check environment variable
    if bundle_path := os.getenv("LLAMA_BUNDLE"):
        path = Path(bundle_path)
        if path.exists():
            return path

    # Check package data (assumes bundle is distributed with package)
    # For now, return None - bundle distribution is external
    return None


def extract_bundle(bundle_path: Path) -> Path:
    """
    Extract tar.gz bundle to install directory.

    Creates install directory if needed. Skips extraction if already done.

    Args:
        bundle_path: Path to tar.gz file

    Returns:
        Path to extracted directory

    Raises:
        FileNotFoundError: If bundle not found
        tarfile.TarError: If extraction fails
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    # Create install directory
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    marker_file = INSTALL_DIR / ".extracted"
    if marker_file.exists():
        logger.debug(f"Bundle already extracted to {INSTALL_DIR}")
        return INSTALL_DIR

    logger.info(f"Extracting bundle to {INSTALL_DIR}")

    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(INSTALL_DIR)
    except tarfile.TarError as e:
        logger.error(f"Failed to extract bundle: {e}")
        raise

    # Create marker file
    marker_file.touch()
    logger.info("Bundle extraction complete")

    return INSTALL_DIR


def ensure_extracted() -> Path:
    """
    Ensure binaries are extracted.

    Called on module import to set up binaries.

    Returns:
        Path to extracted binaries directory

    Raises:
        FileNotFoundError: If bundle not found
    """
    # For now, we assume binaries are found via finder.py
    # In a full implementation, this would extract from bundle
    if bundle_path := get_bundle_path():
        return extract_bundle(bundle_path)

    logger.debug("Using system binaries (no bundle extraction needed)")
    return INSTALL_DIR
