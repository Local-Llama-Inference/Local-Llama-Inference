# Local-Llama-Inference v0.1.0

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen)

**A Production-Ready Python SDK for GPU-Accelerated LLM Inference**

Local-Llama-Inference is a comprehensive Python SDK that integrates **llama.cpp** and **NVIDIA NCCL** to enable high-performance inference of GGUF-quantized Large Language Models (LLMs) on single and multiple NVIDIA GPUs. With automatic binary downloads and zero-configuration setup, getting started is as simple as one command.

---

## 🚀 Quick Start (2 Minutes)

### Installation (One Command!)

```bash
# Install from GitHub (available now)
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0

# Or when published to PyPI (coming soon)
pip install local-llama-inference
```

### First Use (Auto-Downloads CUDA Binaries - 10-15 minutes)

```python
from local_llama_inference import LlamaServer, LlamaClient

# First import triggers auto-download of 834 MB CUDA binaries
# Downloaded once, cached for future use

server = LlamaServer(model_path="Mistral-7B.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
print(response.choices[0].message.content)

server.stop()
```

### Download a Model

```bash
# Get Mistral 7B Q4 quantized (4.3 GB)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf

# Or find more at: https://huggingface.co/models?search=gguf
```

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Single GPU Inference** - Automatic memory optimization and layer offloading
- **Multi-GPU Support** - Tensor parallelism with automatic distribution suggestions
- **30+ REST API Endpoints** - Full llama.cpp endpoint coverage
- **OpenAI-Compatible API** - Drop-in compatible `/v1/chat/completions` endpoint
- **Streaming Responses** - Token-by-token streaming via Python generators
- **Production-Ready** - Error handling, process management, health checks
- **Auto-Binary Download** - CUDA binaries download automatically on first use

### 🔌 Complete API Support

```python
from local_llama_inference import LlamaServer, LlamaClient

client = LlamaClient()

# Chat & Completions
response = client.chat_completion(messages=[...])
async for chunk in client.stream_chat(messages=[...]):
    print(chunk.choices[0].delta.content)

# Embeddings
embeddings = client.embed(input="Your text here")

# Token Operations
tokens = client.tokenize(prompt="Hello world")
text = client.detokenize(tokens=tokens)

# Advanced Features
client.rerank(query="...", documents=[...])
client.infill(...)  # Code infilling
client.set_lora_adapters(...)  # LoRA support

# Server Management
health = client.health()
props = client.get_props()
metrics = client.get_metrics()
```

### 🎮 GPU Utilities

```python
from local_llama_inference import detect_gpus, suggest_tensor_split, check_cuda_version

# Auto-detect available GPUs
gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name} ({gpu.total_memory_mb}MB)")

# Get optimal tensor distribution for multi-GPU
tensor_split = suggest_tensor_split(gpus)

# Check CUDA compatibility
cuda_version = check_cuda_version()
```

### 🛠️ Command-Line Interface

```bash
# Download CUDA binaries from Hugging Face
llama-inference install

# Verify installation status
llama-inference verify

# Show package information
llama-inference info

# Force reinstall (e.g., if corrupted)
llama-inference install --force

# Use custom cache directory
llama-inference install --cache-dir /path/to/cache
```

---

## 📋 Installation Methods

### Method 1: From GitHub (Recommended for Now)
```bash
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```
**Status**: ✅ Available now
**Time**: ~1 minute (+ 10-15 minutes for first binary download)

### Method 2: From PyPI (Coming Soon)
```bash
pip install local-llama-inference
```
**Status**: ⏳ Ready to publish
**When**: Run `twine upload dist/*`

### Method 3: From Source
```bash
git clone https://github.com/Local-Llama-Inference/Local-Llama-Inference.git
cd Local-Llama-Inference/local-llama-inference
pip install -e .
```
**Status**: ✅ Available now
**Time**: ~1 minute (+ 10-15 minutes for first binary download)

---

## 💻 System Requirements

### Minimum
- **GPU**: NVIDIA compute capability 5.0+ (Kepler K80, K40, GTX 750 Ti)
- **VRAM**: 2GB+ per GPU
- **Python**: 3.8 - 3.12
- **OS**: Linux x86_64
- **RAM**: 8GB+ system memory
- **CUDA**: 11.5+ (runtime included in auto-downloaded binaries)

### Recommended
- **GPU**: RTX 2060 or newer (sm_75+)
- **VRAM**: 4GB+ per GPU
- **RAM**: 16GB+ system memory
- **Python**: 3.10+

### Supported GPUs
✅ **Kepler** (sm_50) - Tesla K80, K40, GTX 750 Ti
✅ **Maxwell** (sm_61) - GTX 750, GTX 950, GTX 1050
✅ **Pascal** (sm_61) - GTX 1060, GTX 1080
✅ **Volta** (sm_70) - Tesla V100
✅ **Turing** (sm_75) - RTX 2060, RTX 2080, RTX 20 series
✅ **Ampere** (sm_80) - RTX 3060, RTX 3080, RTX 3090
✅ **Ada** (sm_86) - RTX 4080, RTX 6000 Ada
✅ **Hopper** (sm_89) - H100, H200

---

## 📦 Binary Distribution

**What You Get**:
- llama.cpp (latest version with CUDA support)
- NVIDIA NCCL 2.29.3 (multi-GPU communication)
- CUDA 12.8 runtime (full CUDA stack)
- llama-server (HTTP REST API binary)
- Helper utilities and documentation

**Bundle Sizes**:
- **tar.gz**: 834 MB
- **zip**: 1.48 GB

**Download Source**:
- Hugging Face CDN (fast & reliable)
- https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/

**Auto-Download**: On first import (~10-15 minutes, one time only)
**Cached**: `~/.local/share/local-llama-inference/`

---

## 📚 Jupyter Notebooks (Interactive Examples)

Learn by doing with comprehensive Jupyter notebooks covering all features:

| Notebook | Description | Duration |
|----------|-------------|----------|
| [01_quick_start.ipynb](notebooks/01_quick_start.ipynb) | Auto-download binaries, GPU detection, basic chat | 20 min |
| [02_streaming_responses.ipynb](notebooks/02_streaming_responses.ipynb) | Token-by-token streaming, latency measurement | 15 min |
| [03_embeddings.ipynb](notebooks/03_embeddings.ipynb) | Text embeddings, semantic search, clustering | 15 min |
| [04_multi_gpu.ipynb](notebooks/04_multi_gpu.ipynb) | Multi-GPU tensor parallelism, benchmarking | 20 min |
| [05_advanced_api.ipynb](notebooks/05_advanced_api.ipynb) | Complete API reference (30+ endpoints) | 25 min |
| [06_gpu_detection.ipynb](notebooks/06_gpu_detection.ipynb) | GPU analysis, VRAM planning, troubleshooting | 15 min |

**Quick Start with Notebooks:**
```bash
# Install Jupyter
pip install jupyter jupyterlab

# Start JupyterLab
jupyter lab notebooks/

# Open and run notebooks in order (01 → 06)
```

👉 **[Notebooks README](notebooks/README.md)** - Complete guide with learning sequences and prerequisites

---

## 🎯 Usage Examples

### Basic Chat

```python
from local_llama_inference import LlamaServer, LlamaClient

# Start server
server = LlamaServer(
    model_path="Mistral-7B-Instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=33,      # Offload all layers to GPU
    n_ctx=2048,           # Context window
    n_batch=512,          # Batch size
    host="127.0.0.1",
    port=8000
)

print("Starting server...")
server.start()
server.wait_ready(timeout=60)
print("✅ Server ready!")

# Create client and chat
client = LlamaClient(base_url="http://127.0.0.1:8000")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=256
)

print("Assistant:", response.choices[0].message.content)

# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = client.chat_completion(messages=messages)
print("Assistant:", response.choices[0].message.content)

# Continue conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "How is it used in AI?"})

response = client.chat_completion(messages=messages)
print("Assistant:", response.choices[0].message.content)

server.stop()
```

### Streaming Responses

```python
from local_llama_inference import LlamaServer, LlamaClient

server = LlamaServer(model_path="model.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()

# Stream tokens as they arrive
for token in client.stream_chat(
    messages=[{"role": "user", "content": "Write a poem about AI"}]
):
    print(token.choices[0].delta.content, end="", flush=True)
print()

server.stop()
```

### Multi-GPU Inference

```python
from local_llama_inference import (
    LlamaServer, LlamaClient, detect_gpus, suggest_tensor_split
)

# Detect GPUs
gpus = detect_gpus()
print(f"Found {len(gpus)} GPU(s)")
for gpu in gpus:
    print(f"  {gpu.name}: {gpu.total_memory_mb}MB")

# Get suggested tensor split
tensor_split = suggest_tensor_split(gpus)
print(f"Suggested tensor split: {tensor_split}")

# Start with multi-GPU
server = LlamaServer(
    model_path="model.gguf",
    n_gpu_layers=33,
    tensor_split=tensor_split,  # Distribute across GPUs
)
server.start()
server.wait_ready()

# Use normally
client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Process this on multiple GPUs!"}]
)
print(response.choices[0].message.content)

server.stop()
```

### Embeddings

```python
from local_llama_inference import LlamaServer, LlamaClient

server = LlamaServer(model_path="embedding-model.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()

# Generate embeddings
embeddings = client.embed(input="What is machine learning?")
print(f"Embedding dimension: {len(embeddings.data[0].embedding)}")

# Batch embeddings
batch_embeddings = client.embed(
    input=[
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "LLMs are large language models"
    ]
)
print(f"Generated {len(batch_embeddings.data)} embeddings")

server.stop()
```

---

## 🔧 Configuration

### Server Configuration

```python
from local_llama_inference import ServerConfig, SamplingConfig

config = ServerConfig(
    # Model
    model_path="./model.gguf",

    # Server
    host="127.0.0.1",
    port=8000,
    api_key=None,  # Optional API key

    # GPU settings
    n_gpu_layers=33,           # Layers to offload to GPU
    tensor_split=[0.5, 0.5],   # Multi-GPU distribution
    main_gpu=0,                # Primary GPU

    # Context
    n_ctx=2048,                # Context window size
    n_batch=512,               # Batch size
    n_ubatch=512,              # Micro-batch size

    # Performance
    flash_attn=True,           # Use Flash Attention v2
    numa=False,                # NUMA optimization

    # Advanced
    use_mmap=True,             # Memory mapped I/O
    use_mlock=False,           # Lock memory
    embedding_only=False,      # Embedding mode
)

# Generate CLI arguments
args = config.to_args()

# Create server
server = LlamaServer(config)
server.start()
```

### Sampling Configuration

```python
from local_llama_inference import SamplingConfig

sampling_config = SamplingConfig(
    temperature=0.7,           # Higher = more random
    top_k=40,                  # Top-k sampling
    top_p=0.9,                 # Nucleus sampling
    min_p=0.05,                # Minimum probability
    repeat_penalty=1.1,        # Penalize repetition
    mirostat=0,                # Mirostat sampling (0=off)
    seed=42,                   # Random seed
    grammar=None,              # Grammar constraints
    json_schema=None,          # JSON schema
)

# Use in request
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=sampling_config.temperature,
    top_k=sampling_config.top_k,
    top_p=sampling_config.top_p,
)
```

---

## 🛠️ Troubleshooting

### Installation Issues

**"pip install" fails**
```bash
# Update pip first
pip install --upgrade pip

# Try again
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```

**"huggingface-hub not found"**
```bash
# Install dependency
pip install huggingface-hub>=0.16.0

# Then retry
python -c "from local_llama_inference import LlamaServer"
```

### Binary Download Issues

**Binaries fail to download on first use**
```bash
# Manually trigger download
llama-inference install

# Or check what's wrong
llama-inference verify
```

**Binaries partially downloaded (interrupted)**
```bash
# Force re-download
llama-inference install --force

# Or clean and restart
rm -rf ~/.local/share/local-llama-inference/
llama-inference install
```

### Runtime Issues

**"CUDA out of memory"**
```python
# Reduce GPU layers
server = LlamaServer(model_path="model.gguf", n_gpu_layers=15)

# Or use smaller quantization (Q2, Q3 instead of Q5, Q6)
```

**"GPU not found"**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify in Python
python -c "from local_llama_inference import detect_gpus; print(detect_gpus())"
```

**"Server startup timeout"**
```python
# Increase timeout
server.wait_ready(timeout=120)  # Default is 60 seconds

# Or check server logs
server.start(wait_ready=False)
import time
time.sleep(5)
# Check console for error messages
```

---

## 📚 Documentation

- **[Quick Start Guide](QUICK_START_PIP.md)** - Get running in 4 minutes
- **[Installation Guide](PIP_INSTALL_SETUP.md)** - Detailed setup instructions
- **[PyPI Publishing](PYPI_PUBLISHING_GUIDE.md)** - How to publish to PyPI
- **[Complete Setup Summary](COMPLETE_PIP_INSTALL_SETUP.md)** - Full technical details

---

## 🔗 Resources

| Resource | URL |
|----------|-----|
| **GitHub Repository** | https://github.com/Local-Llama-Inference/Local-Llama-Inference |
| **GitHub Releases** | https://github.com/Local-Llama-Inference/Local-Llama-Inference/releases |
| **Hugging Face Binaries** | https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/ |
| **Hugging Face Project** | https://huggingface.co/waqasm86/local-llama-inference |
| **PyPI Package** | https://pypi.org/project/local-llama-inference/ |
| **GGUF Models** | https://huggingface.co/models?search=gguf |

---

## 🎓 Learning Resources

### Official Documentation
- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs)
- [GGUF Format Specification](https://github.com/ggml-org/gguf)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

### Finding Models
- **Mistral**: https://huggingface.co/mistralai
- **Llama**: https://huggingface.co/meta-llama
- **Phi**: https://huggingface.co/microsoft/phi-2
- **All GGUF Models**: https://huggingface.co/models?search=gguf

### Performance Tips
1. **Use Flash Attention** - Set `flash_attn=True` for 2-3x speedup
2. **Increase GPU Layers** - Higher `n_gpu_layers` = faster but more VRAM
3. **Larger Context** - Higher `n_ctx` = better context understanding
4. **Batch Size** - `n_batch=512` good for most GPUs
5. **Quantization** - Q4 (~4GB), Q5 (~5GB), Q6 (~6GB)
6. **Multi-GPU** - Use `tensor_split` to distribute across GPUs
7. **Keep Server Alive** - Reuse server instance instead of restart cycles

---

## 📦 Project Structure

```
local-llama-inference/
├── src/local_llama_inference/
│   ├── __init__.py                 # Public API exports
│   ├── server.py                   # LlamaServer class
│   ├── client.py                   # LlamaClient REST wrapper
│   ├── config.py                   # Configuration dataclasses
│   ├── gpu.py                      # GPU detection & utilities
│   ├── exceptions.py               # Custom exception classes
│   ├── _bindings/
│   │   ├── llama_binding.py        # libllama.so ctypes wrapper
│   │   └── nccl_binding.py         # libnccl.so.2 ctypes wrapper
│   └── _bootstrap/
│       ├── finder.py               # Binary locator
│       ├── extractor.py            # Bundle extractor
│       └── installer.py            # Hugging Face downloader
│
├── examples/
│   ├── single_gpu_chat.py          # Basic chat example
│   ├── multi_gpu_tensor_split.py   # Multi-GPU setup
│   ├── streaming_chat.py           # Streaming example
│   ├── embeddings_example.py       # Embedding generation
│   └── nccl_bindings_example.py    # Direct NCCL usage
│
├── tests/                          # Unit test suite
├── setup.py                        # Package configuration
├── README.md                       # This file
├── LICENSE                         # MIT License
└── releases/v0.1.0/               # Release artifacts
    ├── local-llama-inference-complete-v0.1.0.tar.gz
    ├── local-llama-inference-complete-v0.1.0.zip
    └── checksums.txt
```

---

## 📊 Dependencies

### Required
- **httpx** >= 0.24.0 - Async HTTP client for REST API
- **pydantic** >= 2.0 - Data validation and settings management
- **huggingface-hub** >= 0.16.0 - For downloading binaries from Hugging Face

### Optional (Development)
- **pytest** >= 7.0 - Unit testing
- **pytest-asyncio** >= 0.21.0 - Async test support
- **black** >= 23.0 - Code formatting
- **mypy** >= 1.0 - Type checking
- **ruff** >= 0.1.0 - Linting

### System
- **NVIDIA CUDA** - Runtime included in auto-downloaded binaries
- **NVIDIA Drivers** - Required (any recent version ≥ 418)

---

## 🔐 Security & Privacy

- ✅ **Open Source** - Full source code visible (MIT license)
- ✅ **No Tracking** - No telemetry or data collection
- ✅ **SHA256 Verification** - All binaries checksummed
- ✅ **Transparent Dependencies** - All clearly listed
- ✅ **Standard Packaging** - Uses setuptools (industry standard)
- ✅ **XDG Compliance** - Binaries cached in standard locations

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Learning

- **[📚 Jupyter Notebooks](notebooks/)** - Interactive examples and tutorials
- **[Notebooks README](notebooks/README.md)** - Complete guide to all notebooks
- **[GitHub Issues](https://github.com/Local-Llama-Inference/Local-Llama-Inference/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/Local-Llama-Inference/Local-Llama-Inference/discussions)** - Ask questions
- **[Documentation](https://github.com/Local-Llama-Inference/Local-Llama-Inference)** - Read guides

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 📊 Version Information

| Item | Details |
|------|---------|
| **Version** | 0.1.0 (Beta) |
| **Release Date** | February 24, 2026 |
| **Status** | Production Ready |
| **Author** | waqasm86 |
| **Python Support** | 3.8, 3.9, 3.10, 3.11, 3.12 |
| **GPU Support** | NVIDIA sm_50 and newer |
| **License** | MIT |

---

## 🙏 Acknowledgments

Built on top of:
- **llama.cpp** - High-performance GGUF inference engine
- **NVIDIA NCCL** - Multi-GPU communication library
- **Hugging Face** - Model hub and binary hosting

---

**Ready to accelerate your LLM inference?**

```bash
# Get started now:
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```

Built with ❤️ for the open-source ML community

⭐ If you find this project useful, please consider starring the repository!
