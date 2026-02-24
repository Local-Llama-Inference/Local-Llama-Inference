"""Native bindings for libllama.so and libnccl.so via ctypes."""

try:
    from .nccl_binding import NCCLBinding
except ImportError:
    NCCLBinding = None

try:
    from .llama_binding import LlamaBinding
except ImportError:
    LlamaBinding = None

__all__ = ["NCCLBinding", "LlamaBinding"]
