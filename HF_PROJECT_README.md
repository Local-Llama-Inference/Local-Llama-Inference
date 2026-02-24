# Local-Llama-Inference

**A Production-Ready Python SDK for GPU-Accelerated LLM Inference**

A comprehensive Python SDK integrating **llama.cpp** and **NVIDIA NCCL** to enable high-performance inference of GGUF-quantized Large Language Models on single and multiple NVIDIA GPUs.

## Quick Start

### Installation (One Command!)

```bash
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```

When published to PyPI:
```bash
pip install local-llama-inference
```

### First Use (Auto-Downloads CUDA Binaries)

```python
from local_llama_inference import LlamaServer, LlamaClient

# First import: Auto-downloads 834 MB from Hugging Face (10-15 min)
# Future uses: Instant (cached)

server = LlamaServer(model_path="Mistral-7B-Q4.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
print(response.choices[0].message.content)
server.stop()
```

## Features

✨ **Core Capabilities**
- Single & Multi-GPU inference with automatic tensor parallelism
- 30+ REST API endpoints (full llama.cpp coverage)
- OpenAI-compatible `/v1/chat/completions` endpoint
- Streaming responses with Python generators
- Embeddings & reranking support
- Production-ready error handling

🔌 **API Support**
```python
# Chat & Completions
client.chat_completion()          # Chat (non-streaming)
client.stream_chat()              # Chat (streaming)
client.complete()                 # Text completion
client.stream_complete()          # Streaming completion

# Embeddings & Reranking
client.embed()                    # Generate embeddings
client.rerank()                   # Cross-encoder reranking

# Token Utilities
client.tokenize()                 # Text → tokens
client.detokenize()               # Tokens → text
client.apply_template()           # Chat template

# Advanced
client.infill()                   # Code infilling
client.set_lora_adapters()        # LoRA support
```

🎮 **GPU Utilities**
```python
from local_llama_inference import detect_gpus, suggest_tensor_split

gpus = detect_gpus()
tensor_split = suggest_tensor_split(gpus)
```

## Installation Methods

### 1️⃣ **From GitHub** (Available Now)
```bash
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```

### 2️⃣ **From PyPI** (When Published)
```bash
pip install local-llama-inference
```

### 3️⃣ **From Source**
```bash
git clone https://github.com/Local-Llama-Inference/Local-Llama-Inference.git
cd Local-Llama-Inference/local-llama-inference
pip install -e .
```

## CLI Tools

Manage CUDA binaries and installation:

```bash
# Download CUDA binaries from Hugging Face
llama-inference install

# Verify installation status
llama-inference verify

# Show package information
llama-inference info

# Force reinstall binaries
llama-inference install --force

# Use custom cache location
llama-inference install --cache-dir /path/to/cache
```

## System Requirements

### Minimum
- **GPU**: NVIDIA compute capability 5.0+ (Kepler K80, K40)
- **VRAM**: 2GB+ per GPU
- **Python**: 3.8+
- **OS**: Linux x86_64
- **RAM**: 8GB+

### Recommended
- **GPU**: RTX 2060 or newer
- **VRAM**: 4GB+ per GPU
- **RAM**: 16GB+
- **CUDA**: Any recent version (runtime included)

### Supported Architectures
✅ Kepler (sm_50) - Tesla K80, K40
✅ Maxwell (sm_61) - GTX 750, GTX 950
✅ Pascal (sm_61) - GTX 1060, GTX 1080
✅ Volta (sm_70) - Tesla V100
✅ Turing (sm_75) - RTX 2060, RTX 2080
✅ Ampere (sm_80) - RTX 3060, RTX 3090
✅ Ada (sm_86) - RTX 4080, RTX 6000
✅ Hopper (sm_89) - H100, H200

## Binary Distribution

**Size**: 834 MB (tar.gz) + 1.48 GB (zip)

**Includes**:
- llama.cpp (latest version with CUDA support)
- NVIDIA NCCL 2.29.3 (multi-GPU communication)
- CUDA 12.8 runtime (complete CUDA stack)
- llama-server (HTTP REST API binary)

**Hosted on**: Hugging Face CDN
https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/

**Auto-Downloads**: On first import, ~10-15 minutes

## Examples

### Basic Chat
```python
from local_llama_inference import LlamaServer, LlamaClient

server = LlamaServer(model_path="model.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "What is Python?"}]
)
print(response.choices[0].message.content)

server.stop()
```

### Streaming Responses
```python
for token in client.stream_chat(
    messages=[{"role": "user", "content": "Write a poem"}]
):
    print(token.choices[0].delta.content, end="", flush=True)
```

### Multi-GPU Inference
```python
from local_llama_inference import detect_gpus, suggest_tensor_split

gpus = detect_gpus()
tensor_split = suggest_tensor_split(gpus)

server = LlamaServer(
    model_path="model.gguf",
    n_gpu_layers=33,
    tensor_split=tensor_split
)
server.start()
```

## Documentation

- **[GitHub Repository](https://github.com/Local-Llama-Inference/Local-Llama-Inference)** - Full source code
- **[README.md](https://github.com/Local-Llama-Inference/Local-Llama-Inference/blob/main/README.md)** - Complete documentation
- **[Quick Start Guide](https://github.com/Local-Llama-Inference/Local-Llama-Inference/blob/main/QUICK_START_PIP.md)** - 4-minute setup
- **[Implementation Details](https://github.com/Local-Llama-Inference/Local-Llama-Inference/blob/main/PIP_INSTALL_SETUP.md)** - Technical details

## Resources

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/Local-Llama-Inference/Local-Llama-Inference |
| **GitHub Releases** | https://github.com/Local-Llama-Inference/Local-Llama-Inference/releases |
| **Hugging Face Binaries** | https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/ |
| **Hugging Face Project** | https://huggingface.co/waqasm86/local-llama-inference |
| **PyPI Package** | https://pypi.org/project/local-llama-inference/ (when published) |

## License

MIT License - See [LICENSE](https://github.com/Local-Llama-Inference/Local-Llama-Inference/blob/main/LICENSE) for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- **[GitHub Issues](https://github.com/Local-Llama-Inference/Local-Llama-Inference/issues)** - Bug reports
- **[GitHub Discussions](https://github.com/Local-Llama-Inference/Local-Llama-Inference/discussions)** - Questions & ideas
- **[Documentation](https://github.com/Local-Llama-Inference/Local-Llama-Inference)** - Full guides

## Getting GGUF Models

Find GGUF quantized models on Hugging Face:
https://huggingface.co/models?search=gguf

Popular options:
- **Mistral 7B**: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- **Llama 2 7B**: TheBloke/Llama-2-7B-Chat-GGUF
- **Phi 2**: TheBloke/phi-2-GGUF
- **Orca Mini**: TheBloke/orca_mini-3B-GGUF

## Project Information

- **Version**: 0.1.0 (Beta)
- **Status**: Production Ready
- **Release Date**: February 24, 2026
- **Author**: waqasm86
- **Python Support**: 3.8 - 3.12
- **GPU Support**: NVIDIA sm_50 and newer

---

**Ready to use GPU-accelerated LLM inference!**

```bash
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```

Built with ❤️ for the open-source ML community
