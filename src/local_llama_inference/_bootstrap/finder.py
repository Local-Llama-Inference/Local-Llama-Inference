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
    2. System PATH
    3. Extracted bundle directory
    4. llama.cpp local build directory

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

    # Check extracted bundle
    bundle_dir = Path.home() / ".local" / "share" / "local-llama-inference" / "bin"
    if bundle_dir.exists():
        path = bundle_dir / name
        if path.exists():
            return str(path.absolute())

    # Check system PATH
    result = subprocess.run(
        ["which", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()

    # Check llama.cpp local build
    local_build = (
        Path("/media/waqasm86/External1/Project-Nvidia-Office")
        / "Project-LlamaInference"
        / "llama.cpp"
        / "build"
        / "bin"
        / name
    )
    if local_build.exists():
        return str(local_build.absolute())

    raise BinaryNotFound(f"Binary '{name}' not found in PATH or bundle")


def find_library(name: str) -> str:
    """
    Find a shared library (.so file).

    Searches in:
    1. LLAMA_LIB_DIR or extracted bundle
    2. System library paths
    3. ~/.local/lib (NCCL installation)
    4. llama.cpp local build

    Args:
        name: Library name (e.g., 'libllama.so' or 'libnccl.so.2')

    Returns:
        Path to library

    Raises:
        LibraryNotFound: If library not found
    """
    candidates = []

    # Check extracted bundle
    bundle_lib = Path.home() / ".local" / "share" / "local-llama-inference" / "lib"
    if bundle_lib.exists():
        candidates.append(bundle_lib / name)
        # Check for version variants (e.g., libllama.so.0)
        for f in bundle_lib.glob(f"{name}*"):
            candidates.append(f)

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

    # Check llama.cpp local build
    local_lib_dir = (
        Path("/media/waqasm86/External1/Project-Nvidia-Office")
        / "Project-LlamaInference"
        / "llama.cpp"
        / "build"
        / "lib"
    )
    if local_lib_dir.exists():
        candidates.append(local_lib_dir / name)
        for f in local_lib_dir.glob(f"{name}*"):
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
