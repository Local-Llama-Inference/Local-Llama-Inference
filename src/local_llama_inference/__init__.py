"""local-llama-inference: Python SDK for llama.cpp + NVIDIA NCCL GPU inference."""

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
]
