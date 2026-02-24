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

if not os.getenv("LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL"):
    try:
        ensure_binaries_installed()
    except Exception as e:
        # Don't fail import if binary download fails - user can still call it manually
        import warnings
        warnings.warn(
            f"Could not automatically download binaries on import: {str(e)}\n"
            f"You can manually download by running: llama-inference install\n"
            f"Or set LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1 to skip auto-download",
            RuntimeWarning,
            stacklevel=2
        )
else:
    # Auto-install is disabled
    import warnings
    warnings.warn(
        "LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL is set - binary auto-download is disabled",
        RuntimeWarning,
        stacklevel=2
    )
