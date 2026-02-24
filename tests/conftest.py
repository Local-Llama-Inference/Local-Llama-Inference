"""pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information."""
    from local_llama_inference import GPUInfo

    return GPUInfo(
        index=0,
        name="NVIDIA GeForce 940M",
        uuid="GPU-123456",
        compute_capability=(5, 0),
        total_memory_mb=1024,
        free_memory_mb=512,
    )


@pytest.fixture
def server_config():
    """Mock server configuration."""
    from local_llama_inference import ServerConfig

    return ServerConfig(
        model_path="/test/model.gguf",
        host="127.0.0.1",
        port=8080,
        n_gpu_layers=33,
        ctx_size=2048,
    )
