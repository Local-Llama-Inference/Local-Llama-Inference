"""Tests for configuration module."""

import pytest
from local_llama_inference import ServerConfig


def test_server_config_defaults():
    """Test default configuration values."""
    config = ServerConfig(model_path="/test/model.gguf")

    assert config.model_path == "/test/model.gguf"
    assert config.host == "127.0.0.1"
    assert config.port == 8080
    assert config.n_gpu_layers == 99
    assert config.ctx_size == 4096
    assert config.batch_size == 512
    assert config.main_gpu == 0


def test_server_config_to_args():
    """Test converting config to CLI arguments."""
    config = ServerConfig(
        model_path="/test/model.gguf",
        host="0.0.0.0",
        port=9000,
        n_gpu_layers=33,
        ctx_size=2048,
    )

    args = config.to_args()

    assert "-m" in args
    assert "/test/model.gguf" in args
    assert "--host" in args
    assert "0.0.0.0" in args
    assert "--port" in args
    assert "9000" in args
    assert "-ngl" in args
    assert "33" in args


def test_server_config_tensor_split():
    """Test tensor split configuration."""
    config = ServerConfig(
        model_path="/test/model.gguf",
        tensor_split=[1.0, 2.0],
        split_mode="layer",
    )

    args = config.to_args()
    assert "--tensor-split" in args
    assert "1.0,2.0" in args


def test_server_config_to_url():
    """Test URL generation."""
    config = ServerConfig(
        model_path="/test/model.gguf",
        host="192.168.1.100",
        port=9000,
    )

    assert config.to_url() == "http://192.168.1.100:9000"


def test_server_config_with_api_key():
    """Test configuration with API key."""
    config = ServerConfig(
        model_path="/test/model.gguf",
        api_key="test-key-123",
    )

    assert config.api_key == "test-key-123"
    args = config.to_args()
    assert "--api-key" in args
    assert "test-key-123" in args
