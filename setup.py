#!/usr/bin/env python3
"""
Setup configuration for local-llama-inference Python SDK.

This package provides a comprehensive Python SDK for GPU-accelerated LLM inference
using llama.cpp and NVIDIA NCCL for multi-GPU support.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from _version.py
version_file = Path(__file__).parent / "src" / "local_llama_inference" / "_version.py"
version = "0.1.0"
if version_file.exists():
    content = version_file.read_text()
    for line in content.split("\n"):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
            break

setup(
    name="local-llama-inference",
    version=version,
    author="waqasm86",
    author_email="waqasm86@example.com",
    description="GPU-accelerated LLM inference SDK with NVIDIA NCCL multi-GPU support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Local-Llama-Inference/Local-Llama-Inference",
    project_urls={
        "Bug Tracker": "https://github.com/Local-Llama-Inference/Local-Llama-Inference/issues",
        "Documentation": "https://github.com/Local-Llama-Inference/Local-Llama-Inference/blob/main/README.md",
        "Source Code": "https://github.com/Local-Llama-Inference/Local-Llama-Inference",
        "Hugging Face": "https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",           # Async HTTP client for REST API
        "huggingface-hub>=0.16.0", # For downloading binaries from HF
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
        "all": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "llm",
        "inference",
        "gpu",
        "cuda",
        "nccl",
        "llama",
        "gguf",
        "transformer",
        "multi-gpu",
        "tensor-parallel",
    ],
    entry_points={
        "console_scripts": [
            "llama-inference=local_llama_inference.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "local_llama_inference": [
            "py.typed",
        ],
    },
    zip_safe=False,
)
