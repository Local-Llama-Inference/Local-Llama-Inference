"""
local-llama-inference: Python SDK for llama.cpp + NVIDIA NCCL GPU inference.

This package provides a complete Python integration for running large language models
with GPU acceleration using llama.cpp and NVIDIA's NCCL library for multi-GPU support.

Quick Start:
    >>> from local_llama_inference import LlamaServer, LlamaClient
    >>> server = LlamaServer(model="model.gguf")
    >>> server.start()
    >>> client = LlamaClient()
    >>> response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
    >>> print(response)

For more information, visit:
    https://github.com/Local-Llama-Inference/Local-Llama-Inference
"""

from ._version import __version__, __author__, __license__
from .config import ServerConfig, SamplingConfig, ModelConfig
from .gpu import GPUInfo, detect_gpus, check_cuda_version, suggest_tensor_split
from .server import LlamaServer
from .client import LlamaClient
from .exceptions import (
    LlamaError,
    ServerError,
    ClientError,
    GPUError,
    NCCLError,
    BinaryNotFound,
    LibraryNotFound,
)
from ._bootstrap.installer import BinaryInstaller, ensure_binaries_installed

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "ServerConfig",
    "SamplingConfig",
    "ModelConfig",
    "GPUInfo",
    "detect_gpus",
    "check_cuda_version",
    "suggest_tensor_split",
    "LlamaServer",
    "LlamaClient",
    "LlamaError",
    "ServerError",
    "ClientError",
    "GPUError",
    "NCCLError",
    "BinaryNotFound",
    "LibraryNotFound",
    "BinaryInstaller",
    "ensure_binaries_installed",
]

# Auto-download binaries on first import (can be disabled with env var)
# Set LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1 to skip auto-download
# (useful for offline systems, CI, or air-gapped deployments)
import os
import sys

if not os.getenv("LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL"):
    try:
        # Ensure binaries are installed before SDK can be used
        paths = ensure_binaries_installed()

        # Verify that binaries were actually found/downloaded
        if not paths or not any(paths.values()):
            print("\n⚠️  WARNING: Binary paths could not be resolved!")
            print("   The SDK requires CUDA binaries to function.")
            print("   Please ensure binaries are installed:")
            print("   https://huggingface.co/datasets/waqasm86/Local-Llama-Inference")
            print("   Or manually run: python -c \"from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary()\"")
            print()
    except Exception as e:
        # Binary download failed - provide detailed error info
        print("\n❌ ERROR: Failed to download CUDA binaries automatically!")
        print(f"   Error: {str(e)}")
        print("\n   The local-llama-inference SDK requires CUDA binaries from Hugging Face CDN.")
        print("   Please download manually from:")
        print("   https://huggingface.co/datasets/waqasm86/Local-Llama-Inference")
        print("\n   Or manually install by running:")
        print("   python -c \"from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary(force=True)\"")
        print("\n   To skip this download (not recommended):")
        print("   export LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1")
        print()
        # Don't fail import, but user will see clear error message
        import warnings
        warnings.warn(
            f"Binary auto-download failed: {str(e)}",
            RuntimeWarning,
            stacklevel=2
        )
else:
    # Auto-install is disabled
    print("⚠️  LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL is set")
    print("   Binary auto-download is disabled.")
    print("   Ensure binaries are already installed at:")
    print("   ~/.local/share/local-llama-inference/extracted/")
    print()
