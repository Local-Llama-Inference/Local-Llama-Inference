"""Exception classes for local-llama-inference SDK."""


class LlamaError(Exception):
    """Base exception for all local-llama-inference errors."""

    pass


class ServerError(LlamaError):
    """Server-related errors (start/stop/communication)."""

    pass


class ServerNotRunning(ServerError):
    """Server is not running when expected."""

    pass


class ServerAlreadyRunning(ServerError):
    """Server is already running."""

    pass


class ServerStartupTimeout(ServerError):
    """Server did not become ready within timeout."""

    pass


class ClientError(LlamaError):
    """HTTP client errors."""

    pass


class APIError(ClientError):
    """HTTP API returned an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"API Error ({status_code}): {message}")


class GPUError(LlamaError):
    """GPU detection or configuration errors."""

    pass


class GPUNotFound(GPUError):
    """No NVIDIA GPUs found."""

    pass


class CUDAError(GPUError):
    """CUDA runtime error."""

    pass


class NCCLError(LlamaError):
    """NCCL operation error."""

    pass


class BinaryNotFound(LlamaError):
    """Required binary (llama-server) not found."""

    pass


class LibraryNotFound(LlamaError):
    """Required shared library (.so file) not found."""

    pass


class ConfigError(LlamaError):
    """Invalid configuration."""

    pass
