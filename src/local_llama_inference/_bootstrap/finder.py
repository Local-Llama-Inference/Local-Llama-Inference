"""Locate binaries and shared libraries."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from ..exceptions import BinaryNotFound, LibraryNotFound


def find_binary(name: str) -> str:
    """
    Find an executable binary.

    Searches in:
    1. LLAMA_BIN_DIR environment variable
    2. HF-downloaded bundle (BinaryInstaller)
    3. System PATH

    Args:
        name: Binary name (e.g., 'llama-server')

    Returns:
        Path to binary

    Raises:
        BinaryNotFound: If binary not found
    """
    # Check environment variable
    if bin_dir := os.getenv("LLAMA_BIN_DIR"):
        path = Path(bin_dir) / name
        if path.exists() and path.is_file():
            return str(path.absolute())

    # Check HF-downloaded bundle via BinaryInstaller
    try:
        from .installer import BinaryInstaller
        installer = BinaryInstaller()
        paths = installer.get_binary_paths()
        if paths.get("llama_bin"):
            path = paths["llama_bin"] / name
            if path.exists():
                return str(path.absolute())
    except Exception:
        pass

    # Check system PATH
    result = subprocess.run(
        ["which", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()

    raise BinaryNotFound(f"Binary '{name}' not found in PATH or bundle")


def find_library(name: str) -> str:
    """
    Find a shared library (.so file).

    Searches in:
    1. HF-downloaded bundle (BinaryInstaller)
    2. System library paths
    3. ~/.local/lib (NCCL installation)

    Args:
        name: Library name (e.g., 'libllama.so' or 'libnccl.so.2')

    Returns:
        Path to library

    Raises:
        LibraryNotFound: If library not found
    """
    candidates = []

    # Check HF-downloaded bundle via BinaryInstaller
    try:
        from .installer import BinaryInstaller
        installer = BinaryInstaller()
        paths = installer.get_binary_paths()

        if paths.get("llama_lib"):
            candidates.append(paths["llama_lib"] / name)
            for f in paths["llama_lib"].glob(f"{name}*"):
                candidates.append(f)

        if paths.get("nccl_lib"):
            candidates.append(paths["nccl_lib"] / name)
            for f in paths["nccl_lib"].glob(f"{name}*"):
                candidates.append(f)
    except Exception:
        pass

    # Check ~/.local/lib (NCCL)
    local_lib = Path.home() / ".local" / "lib"
    if local_lib.exists():
        candidates.append(local_lib / name)
        for f in local_lib.glob(f"{name}*"):
            candidates.append(f)

    # Check system paths
    for path in [Path("/usr/lib"), Path("/usr/local/lib"), Path("/usr/lib/x86_64-linux-gnu")]:
        if path.exists():
            candidates.append(path / name)
            for f in path.glob(f"{name}*"):
                candidates.append(f)

    # Find first existing
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path.absolute())

    raise LibraryNotFound(f"Library '{name}' not found in system paths or bundle")


def get_server_binary() -> str:
    """Get path to llama-server executable."""
    return find_binary("llama-server")


def get_llama_library() -> str:
    """Get path to libllama.so."""
    try:
        return find_library("libllama.so")
    except LibraryNotFound:
        return find_library("libllama.so.0")


def get_nccl_library() -> str:
    """Get path to libnccl.so.2."""
    try:
        return find_library("libnccl.so.2")
    except LibraryNotFound:
        return find_library("libnccl.so")
