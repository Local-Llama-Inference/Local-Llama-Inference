"""GPU detection and configuration utilities."""

import subprocess
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .exceptions import GPUError, GPUNotFound, CUDAError


@dataclass
class GPUInfo:
    """Information about an NVIDIA GPU."""

    index: int
    name: str
    uuid: str
    compute_capability: Tuple[int, int]  # (major, minor)
    total_memory_mb: int
    free_memory_mb: int

    def supports_flash_attn(self) -> bool:
        """Flash Attention requires sm_70+."""
        major, minor = self.compute_capability
        return (major, minor) >= (7, 0)

    def is_supported(self, min_compute_capability: Tuple[int, int] = (5, 0)) -> bool:
        """Check if GPU meets minimum compute capability."""
        return self.compute_capability >= min_compute_capability


def detect_gpus() -> List[GPUInfo]:
    """
    Detect available NVIDIA GPUs using nvidia-smi.

    Returns:
        List of GPUInfo objects

    Raises:
        GPUNotFound: If no NVIDIA GPUs found or nvidia-smi not available
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise GPUNotFound(f"Failed to run nvidia-smi: {e}") from e

    if result.returncode != 0:
        raise GPUNotFound(f"nvidia-smi error: {result.stderr}")

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        try:
            index = int(parts[0])
            name = parts[1]
            uuid = parts[2]
            compute_cap = parts[3]  # format: "5.0" or "sm_50"

            # Parse compute capability
            if "." in compute_cap:
                major, minor = compute_cap.split(".")
            else:
                # Handle "sm_50" format
                match = re.search(r"(\d)(\d)", compute_cap)
                if match:
                    major, minor = match.groups()
                else:
                    continue

            compute_capability = (int(major), int(minor))

            total_memory = int(parts[4])
            free_memory = int(parts[5])

            gpu = GPUInfo(
                index=index,
                name=name,
                uuid=uuid,
                compute_capability=compute_capability,
                total_memory_mb=total_memory,
                free_memory_mb=free_memory,
            )
            gpus.append(gpu)

        except (ValueError, IndexError):
            continue

    if not gpus:
        raise GPUNotFound("No NVIDIA GPUs detected by nvidia-smi")

    return sorted(gpus, key=lambda g: g.index)


def check_cuda_version() -> Tuple[int, int]:
    """
    Get CUDA version from nvidia-smi.

    Returns:
        Tuple of (major_version, minor_version)

    Raises:
        CUDAError: If CUDA version cannot be determined
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise CUDAError(f"Failed to run nvidia-smi: {e}") from e

    # Look for "CUDA Version: X.X" in output
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    raise CUDAError("Could not determine CUDA version from nvidia-smi")


def suggest_tensor_split(gpus: List[GPUInfo]) -> List[float]:
    """
    Suggest tensor_split proportions based on GPU memory.

    For multiple GPUs, distributes the model proportionally to each GPU's
    available memory.

    Args:
        gpus: List of detected GPUs

    Returns:
        List of proportions (will sum to number of GPUs)

    Example:
        >>> gpus = [GPUInfo(..., total_memory_mb=8000), GPUInfo(..., total_memory_mb=16000)]
        >>> suggest_tensor_split(gpus)
        [1.0, 2.0]  # Proportions, normalized to GPU memory
    """
    if not gpus:
        return []

    if len(gpus) == 1:
        return [1.0]

    # Use total memory as proportion
    total_memory = sum(g.total_memory_mb for g in gpus)
    proportions = [g.total_memory_mb / total_memory * len(gpus) for g in gpus]

    return proportions


def validate_tensor_split(tensor_split: List[float], n_gpus: int) -> bool:
    """
    Validate tensor_split configuration.

    Args:
        tensor_split: List of proportions
        n_gpus: Number of GPUs

    Returns:
        True if valid
    """
    if not tensor_split:
        return True

    if len(tensor_split) != n_gpus:
        return False

    if any(x < 0 for x in tensor_split):
        return False

    # At least one GPU should have non-zero proportion
    if sum(tensor_split) == 0:
        return False

    return True


def get_nvml_device_count() -> int:
    """
    Get GPU count using NVIDIA Management Library.

    Returns:
        Number of GPUs

    Raises:
        GPUError: If NVIDIA GPU System Management Interface not available
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown

        nvmlInit()
        count = nvmlDeviceGetCount()
        nvmlShutdown()
        return count
    except (ImportError, Exception) as e:
        raise GPUError(f"NVML not available: {e}") from e
