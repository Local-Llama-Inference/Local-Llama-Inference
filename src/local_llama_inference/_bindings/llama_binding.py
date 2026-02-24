"""ctypes wrapper for llama.cpp (libllama.so) - Direct C API access for advanced users."""

import ctypes
from typing import Optional

from .._bootstrap.finder import get_llama_library
from ..exceptions import LibraryNotFound


class LlamaBinding:
    """ctypes wrapper for libllama.so - Direct llama.h C API access.

    This is for advanced users who want direct access to the llama.cpp C API.
    Most users should use LlamaServer + LlamaClient instead.
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize llama.cpp binding.

        Args:
            lib_path: Path to libllama.so (auto-detected if None)

        Raises:
            LibraryNotFound: If library not found
        """
        if lib_path is None:
            lib_path = get_llama_library()

        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise LibraryNotFound(f"Failed to load llama library: {e}") from e

        self._setup_signatures()

    def _setup_signatures(self):
        """Setup basic ctypes function signatures."""
        # Backend
        self._lib.llama_backend_init.argtypes = []
        self._lib.llama_backend_init.restype = None

        self._lib.llama_backend_free.argtypes = []
        self._lib.llama_backend_free.restype = None

        # Version check
        if hasattr(self._lib, "llama_get_version"):
            self._lib.llama_get_version.argtypes = []
            self._lib.llama_get_version.restype = ctypes.c_char_p

    def backend_init(self) -> None:
        """Initialize llama.cpp backend."""
        self._lib.llama_backend_init()

    def backend_free(self) -> None:
        """Free llama.cpp backend resources."""
        self._lib.llama_backend_free()

    def supports_gpu_offload(self) -> bool:
        """Check if GPU offloading is supported."""
        if hasattr(self._lib, "llama_supports_gpu_offload"):
            result = self._lib.llama_supports_gpu_offload()
            return bool(result)
        return False

    def max_devices(self) -> int:
        """Get maximum number of devices supported."""
        if hasattr(self._lib, "llama_max_devices"):
            return self._lib.llama_max_devices()
        return 1

    def __repr__(self) -> str:
        return "LlamaBinding(libllama.so)"
