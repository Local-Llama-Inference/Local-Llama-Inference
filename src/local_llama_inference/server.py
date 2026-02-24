"""Server process management for llama-server."""

import subprocess
import time
import os
import logging
from pathlib import Path
from typing import Optional

from .config import ServerConfig
from .exceptions import (
    ServerError,
    ServerNotRunning,
    ServerAlreadyRunning,
    ServerStartupTimeout,
    BinaryNotFound,
)

logger = logging.getLogger(__name__)


class LlamaServer:
    """Manages llama-server subprocess."""

    def __init__(self, config: Optional[ServerConfig] = None, binary_path: Optional[str] = None, **kwargs):
        """
        Initialize LlamaServer.

        Args:
            config: ServerConfig instance (alternative to kwargs)
            binary_path: Path to llama-server binary (auto-detected if not provided)
            **kwargs: ServerConfig parameters as keyword arguments
                - model_path (required if config not provided)
                - host (default: "127.0.0.1")
                - port (default: 8080)
                - n_gpu_layers (default: 99)
                - n_threads (default: -1)
                - ctx_size (default: 4096)
                - batch_size (default: 512)
                - And all other ServerConfig parameters

        Examples:
            # Using ServerConfig object
            config = ServerConfig(model_path="model.gguf", n_gpu_layers=33)
            server = LlamaServer(config=config)

            # Using keyword arguments (convenience)
            server = LlamaServer(model_path="model.gguf", n_gpu_layers=33)

        Raises:
            ValueError: If neither config nor model_path provided
        """
        if config is not None:
            # Use provided config
            self.config = config
        elif kwargs:
            # Create config from kwargs
            if "model_path" not in kwargs:
                raise ValueError("Either 'config' or 'model_path' parameter must be provided")
            self.config = ServerConfig(**kwargs)
        else:
            raise ValueError("Either 'config' or 'model_path' parameter must be provided")

        self._process: Optional[subprocess.Popen] = None
        self._binary = binary_path or self._find_binary()

    def _find_binary(self) -> str:
        """
        Find llama-server binary.

        Searches in:
        1. LLAMA_BIN_DIR environment variable
        2. System PATH
        3. Local llama.cpp build directory
        """
        # Check environment variable
        if env_dir := os.getenv("LLAMA_BIN_DIR"):
            binary_path = Path(env_dir) / "llama-server"
            if binary_path.exists():
                return str(binary_path)

        # Check PATH
        result = subprocess.run(
            ["which", "llama-server"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # Check local llama.cpp build
        local_build = Path(
            "/media/waqasm86/External1/Project-Nvidia-Office/Project-LlamaInference/llama.cpp/build/bin/llama-server"
        )
        if local_build.exists():
            return str(local_build)

        raise BinaryNotFound("llama-server binary not found")

    def start(self, wait_ready: bool = True, timeout: float = 60.0) -> None:
        """
        Start llama-server subprocess.

        Args:
            wait_ready: If True, wait for server to be ready via /health endpoint
            timeout: Timeout in seconds for startup

        Raises:
            ServerAlreadyRunning: If server is already running
            ServerStartupTimeout: If server doesn't start within timeout
        """
        if self.is_running():
            raise ServerAlreadyRunning("Server is already running")

        logger.info(f"Starting llama-server from {self._binary}")
        logger.debug(f"Server config: {self.config}")

        args = self.config.to_args()
        cmd = [self._binary] + args

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not self.config.verbose else None,
                stderr=subprocess.PIPE if not self.config.verbose else None,
                text=True,
            )
        except FileNotFoundError as e:
            raise BinaryNotFound(f"Failed to start llama-server: {e}") from e
        except Exception as e:
            raise ServerError(f"Failed to start server: {e}") from e

        logger.info(f"Server process started (PID: {self._process.pid})")

        if wait_ready:
            self.wait_ready(timeout=timeout)

    def stop(self) -> None:
        """
        Gracefully terminate llama-server.

        Sends SIGTERM and waits for process to exit.
        """
        if not self.is_running():
            logger.debug("Server is not running")
            return

        logger.info(f"Stopping server (PID: {self._process.pid})")
        self._process.terminate()

        try:
            self._process.wait(timeout=5.0)
            logger.info("Server stopped")
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully, killing...")
            self._process.kill()
            self._process.wait()
            logger.info("Server killed")

        self._process = None

    def restart(self) -> None:
        """Restart the server with same configuration."""
        self.stop()
        time.sleep(1)
        self.start()

    def is_running(self) -> bool:
        """Check if server process is running."""
        if self._process is None:
            return False

        # Check if process is still alive
        poll_result = self._process.poll()
        return poll_result is None

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """
        Wait for server to become ready.

        Polls the /health endpoint until it returns 200 or timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if server became ready

        Raises:
            ServerStartupTimeout: If timeout exceeded
        """
        import httpx

        url = self.config.to_url()
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = httpx.get(f"{url}/health", timeout=2.0)
                if response.status_code == 200:
                    logger.info("Server is ready")
                    return True
            except (httpx.RequestError, httpx.TimeoutException):
                # Server not ready yet
                pass

            time.sleep(0.5)

        raise ServerStartupTimeout(
            f"Server did not become ready within {timeout} seconds"
        )

    def get_client(self):
        """
        Get a configured HTTP client for this server.

        Returns:
            LlamaClient instance

        Raises:
            ServerNotRunning: If server is not running
        """
        if not self.is_running():
            raise ServerNotRunning("Server is not running")

        from .client import LlamaClient

        return LlamaClient(
            base_url=self.config.to_url(),
            api_key=self.config.api_key,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops server."""
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return f"LlamaServer(model={self.config.model_path}, status={status})"
