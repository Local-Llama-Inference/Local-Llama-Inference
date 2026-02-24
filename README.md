# Local-Llama-Inference

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A Production-Ready Python SDK for GPU-Accelerated LLM Inference**

Local-Llama-Inference is a comprehensive Python SDK that integrates **llama.cpp** and **NVIDIA NCCL** to enable high-performance inference of GGUF-quantized Large Language Models (LLMs) on single and multiple NVIDIA GPUs.

---

## 🎯 Features

### 🚀 Core Capabilities
- **Single GPU Inference** - Automatic memory optimization and layer offloading
- **Multi-GPU Support** - Tensor parallelism with automatic split suggestions
- **30+ REST API Endpoints** - Full llama.cpp endpoint coverage
- **OpenAI-Compatible API** - Drop-in compatible `/v1/chat/completions` endpoint
- **Streaming Responses** - Token-by-token streaming via Python generators
- **Production-Ready** - Error handling, process management, health checks

### 🔌 API Support
```python
# Chat & Completions
client.chat()              # Chat completion (non-streaming)
client.stream_chat()       # Chat with token streaming
client.complete()          # Text completion
client.stream_complete()   # Streaming completion

# Embeddings & Reranking
client.embed()            # Generate embeddings
client.rerank()           # Cross-encoder reranking

# Token Utilities
client.tokenize()         # Text to tokens
client.detokenize()       # Tokens to text
client.apply_template()   # Apply chat template

# Advanced Features
client.infill()           # Code infilling
client.set_lora_adapters() # LoRA support
client.save_slot()        # Slot management
client.restore_slot()     # Restore saved state

# Server Management
client.health()           # Health check
client.get_props()        # Get server properties
client.get_metrics()      # Performance metrics
```

### 🎮 GPU Utilities
```python
from local_llama_inference import detect_gpus, suggest_tensor_split, check_cuda_version

# Detect available GPUs
gpus = detect_gpus()

# Get automatic tensor split for multi-GPU
tensor_split = suggest_tensor_split(gpus)

# Check CUDA version
cuda_version = check_cuda_version()
```

### 📊 NCCL Collective Operations
```python
from local_llama_inference._bindings.nccl_binding import NCCLBinding

# Direct access to NCCL primitives
nccl = NCCLBinding('/path/to/libnccl.so.2')
nccl.all_reduce(sendbuff, recvbuff, ncclFloat32, ncclSum, comm)
nccl.broadcast(buffer, ncclFloat32, root, comm)
nccl.all_gather(sendbuff, recvbuff, ncclFloat32, comm)
```

---

## 📋 System Requirements

### Minimum
- **GPU**: NVIDIA compute capability 5.0+ (sm_50)
  - Tesla K80, K40 | GeForce GTX 750 Ti
- **VRAM**: 2GB+ per GPU
- **Python**: 3.8+
- **OS**: Linux x86_64
- **RAM**: 8GB+ system memory

### Recommended
- **GPU**: RTX 2060 or newer (sm_75+)
- **VRAM**: 4GB+ per GPU
- **RAM**: 16GB+ system memory
- **CUDA**: Any version 11.5+ (runtime included)

### Supported GPUs
✅ Kepler (sm_50) - Tesla K80, K40, GTX 750 Ti
✅ Maxwell (sm_61) - GTX 750, GTX 950, GTX 1050
✅ Pascal (sm_61) - GTX 1060, GTX 1080
✅ Volta (sm_70) - Tesla V100
✅ Turing (sm_75) - RTX 2060, RTX 2080
✅ Ampere (sm_80) - RTX 3060, RTX 3090
✅ Ada (sm_86) - RTX 4080, RTX 6000
✅ Hopper (sm_89) - H100, H200

---

## ⚡ Quick Start (5 Minutes)

### 1. Installation

#### Option A: From Release Package (Recommended)
```bash
# Download from GitHub Releases
# https://github.com/Local-Llama-Inference/Local-Llama-Inference/releases/tag/v0.1.0

tar -xzf local-llama-inference-complete-v0.1.0.tar.gz
cd local-llama-inference-v0.1.0
pip install -e ./python
```

#### Option B: From Source (Developer)
```bash
git clone https://github.com/Local-Llama-Inference/Local-Llama-Inference.git
cd Local-Llama-Inference/local-llama-inference
pip install -e .
```

### 2. Verify Installation
```bash
python -c "from local_llama_inference import LlamaServer, detect_gpus; print('✅ SDK Ready'); print(detect_gpus())"
```

### 3. Download a Model
```bash
# Download Mistral 7B Q4 (quantized, ~4GB)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf

# Or find more models at: https://huggingface.co/models?search=gguf
```

### 4. Run Your First Inference
```python
from local_llama_inference import LlamaServer, LlamaClient

# Start the server
server = LlamaServer(
    model_path="./Mistral-7B-Instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=33,  # Offload all layers to GPU
    n_ctx=2048,       # Context window
    host="127.0.0.1",
    port=8000
)

print("Starting server...")
server.start()
server.wait_ready(timeout=60)
print("✅ Server ready!")

# Create client and send request
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

# Cleanup
server.stop()
```

---

## 📚 Getting Started Tutorials

### Basic Chat Example
```python
from local_llama_inference import LlamaServer, LlamaClient

# Configure and start server
server = LlamaServer(
    model_path="model.gguf",
    n_gpu_layers=33,      # Use GPU
    main_gpu=0,           # Primary GPU
    n_ctx=2048,           # Context size
    n_batch=512,          # Batch size
)
server.start()
server.wait_ready()

# Simple chat
client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = client.chat_completion(messages=messages)
print("Assistant:", response.choices[0].message.content)

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

# Stream tokens in real-time
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

# Get suggested tensor split
tensor_split = suggest_tensor_split(gpus)
print(f"Suggested tensor split: {tensor_split}")

# Start with multi-GPU
server = LlamaServer(
    model_path="model.gguf",
    n_gpu_layers=33,
    tensor_split=tensor_split,  # Distribute layers across GPUs
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

# Single embedding
embedding = client.embed(input="What is machine learning?")
print(f"Embedding dimension: {len(embedding.data[0].embedding)}")

# Batch embeddings
embeddings = client.embed(
    input=[
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "LLMs are large language models"
    ]
)
print(f"Generated {len(embeddings.data)} embeddings")

server.stop()
```

### Advanced: NCCL Operations
```python
from local_llama_inference._bindings.nccl_binding import NCCLBinding, NCCLDataType, NCCLRedOp
import numpy as np

# Load NCCL
nccl = NCCLBinding('/path/to/libnccl.so.2')

# AllReduce operation
sendbuff = np.array([1.0, 2.0, 3.0], dtype=np.float32)
recvbuff = np.zeros_like(sendbuff)

# This would require NCCL communicator setup
# nccl.all_reduce(sendbuff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#                 recvbuff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#                 len(sendbuff), NCCLDataType.FLOAT32, NCCLRedOp.SUM, comm)
```

---

## 🔧 Configuration

### Server Configuration
```python
from local_llama_inference import ServerConfig, SamplingConfig

# Create configuration
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
```

### Sampling Configuration
```python
from local_llama_inference import SamplingConfig

sampling_config = SamplingConfig(
    temperature=0.7,           # Higher = more random
    top_k=40,                  # Nucleus sampling
    top_p=0.9,                 # Cumulative probability
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

## 📖 API Reference

### `LlamaServer` - Process Management
```python
server = LlamaServer(config, binary_path=None)

# Methods
server.start(wait_ready=False, timeout=60)     # Start server
server.stop()                                   # Stop server
server.restart()                                # Restart server
server.is_running()                             # Check status
server.wait_ready(timeout=60)                   # Wait for /health
```

### `LlamaClient` - HTTP REST Client
```python
client = LlamaClient(base_url="http://127.0.0.1:8000", api_key=None)

# Chat & Completions
client.chat_completion(messages, model=None, **kwargs)
client.stream_chat(messages, model=None, **kwargs)
client.complete(prompt, model=None, **kwargs)
client.stream_complete(prompt, model=None, **kwargs)

# Embeddings
client.embed(input, model=None)
client.rerank(model, query, documents)

# Tokens
client.tokenize(prompt, add_special=True)
client.detokenize(tokens)
client.apply_template(messages, add_generation_prompt=True)

# Server Info
client.health()                    # GET /health
client.get_props()                 # GET /props
client.set_props(props)            # POST /props
client.get_metrics()               # GET /metrics
client.get_models()                # GET /models
client.get_slots()                 # GET /slots
```

### `detect_gpus()` - GPU Detection
```python
gpus = detect_gpus()
# Returns: List[GPUInfo]
# Each GPUInfo has: index, name, uuid, compute_capability, total_memory_mb, free_memory_mb

for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name}")
    print(f"  Compute Capability: {gpu.compute_capability}")
    print(f"  VRAM: {gpu.total_memory_mb} MB ({gpu.free_memory_mb} MB free)")
    print(f"  Supports Flash Attention: {gpu.supports_flash_attn()}")
```

### `suggest_tensor_split()` - Auto Multi-GPU
```python
tensor_split = suggest_tensor_split(gpus)
# Automatically calculates optimal layer distribution
# Returns: List[float] summing to 1.0
```

---

## 🛠️ Troubleshooting

### "CUDA out of memory"
```python
# Solution 1: Reduce GPU layers
server = LlamaServer(model_path="model.gguf", n_gpu_layers=15)

# Solution 2: Use smaller quantization
# Download Q2 or Q3 instead of Q5/Q6

# Solution 3: Reduce batch size
server = LlamaServer(model_path="model.gguf", n_batch=256)
```

### "GPU not found"
```bash
# Check NVIDIA driver
nvidia-smi

# Verify NVIDIA driver is installed
# https://www.nvidia.com/Download/driverDetails.aspx

# Check compute capability
python -c "from local_llama_inference import detect_gpus; print(detect_gpus())"
```

### "libcudart.so.12 not found"
```bash
# The complete package includes CUDA runtime

# Or install NVIDIA drivers:
sudo apt update
sudo apt install nvidia-driver-XXX  # Replace XXX with version
sudo reboot
```

### "Server startup timeout"
```python
# Increase timeout
server.wait_ready(timeout=120)  # Default is 60 seconds

# Or check server logs for errors
server.start(wait_ready=False)
time.sleep(5)
# Check console for error messages
```

### Slow Inference
```python
# Increase GPU offloading
n_gpu_layers=33  # Offload all layers

# Check GPU utilization
nvidia-smi -l 1  # Refresh every second

# Use larger models with better quantization (Q5, Q6 instead of Q2)
# Reduce context size if not needed
```

---

## 🔗 Key Files & Directories

```
local-llama-inference/
├── src/local_llama_inference/        # Python SDK source
│   ├── __init__.py                   # Public API
│   ├── server.py                     # LlamaServer class
│   ├── client.py                     # LlamaClient REST wrapper
│   ├── config.py                     # Configuration classes
│   ├── gpu.py                        # GPU utilities
│   ├── exceptions.py                 # Custom exceptions
│   ├── _bindings/
│   │   ├── llama_binding.py          # libllama.so ctypes wrapper
│   │   └── nccl_binding.py           # libnccl.so.2 ctypes wrapper
│   └── _bootstrap/
│       ├── finder.py                 # Binary locator
│       └── extractor.py              # Bundle extractor
├── examples/                          # Tutorial scripts
│   ├── single_gpu_chat.py
│   ├── multi_gpu_tensor_split.py
│   ├── streaming_chat.py
│   ├── embeddings_example.py
│   └── nccl_bindings_example.py
├── tests/                             # Unit tests
├── pyproject.toml                    # Package metadata
├── README.md                         # This file
├── LICENSE                           # MIT License
└── releases/v0.1.0/                 # Release artifacts
    ├── local-llama-inference-complete-v0.1.0.tar.gz
    ├── local-llama-inference-sdk-v0.1.0.tar.gz
    └── CHECKSUMS.txt
```

---

## 📦 Dependencies

### Required
- **httpx** >= 0.24.0 - Async HTTP client for REST API
- **pydantic** >= 2.0 - Data validation and settings management

### Optional (Development)
- **pytest** >= 7.0 - Unit testing
- **pytest-asyncio** >= 0.21.0 - Async test support

### System
- **NVIDIA CUDA** - Any version 11.5+ (runtime included in package)
- **NVIDIA Drivers** - Required, any recent version

---

## 🚀 Performance Tips

1. **Use Flash Attention** - Set `flash_attn=True` for 2-3x speedup
2. **Increase Context** - Larger `n_ctx` = slower but better context
3. **Batch Size** - `n_batch=512` good for most cases
4. **GPU Layers** - Higher `n_gpu_layers` = faster but more VRAM
5. **Quantization** - Q4 = 4GB, Q5 = 5GB, Q6 = 6GB typical sizes
6. **Multi-GPU** - Use `tensor_split` to distribute across GPUs
7. **Keep Alive** - Reuse server instance instead of restart/start cycles

---

## 🔐 Security

- **API Keys** - Optional API key support via `ServerConfig.api_key`
- **Local Only** - Bind to `127.0.0.1` for local development
- **Production** - Consider authentication/TLS for production deployments
- **Model Files** - Keep GGUF files private, don't share URLs publicly

---

## 📄 License

MIT License - See `LICENSE` file for details

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Resources

- **GitHub Issues**: [Report bugs or request features](https://github.com/Local-Llama-Inference/Local-Llama-Inference/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/Local-Llama-Inference/Local-Llama-Inference/discussions)
- **Releases**: [Download packages](https://github.com/Local-Llama-Inference/Local-Llama-Inference/releases)

### Related Projects
- **llama.cpp** - Core inference engine: https://github.com/ggml-org/llama.cpp
- **NCCL** - GPU collective communication: https://github.com/NVIDIA/nccl
- **Hugging Face GGUF Models** - https://huggingface.co/models?search=gguf

---

## 📊 Project Status

- **Version**: 0.1.0 (Beta)
- **Status**: Production Ready
- **Last Updated**: February 24, 2026
- **Python Support**: 3.8 - 3.12
- **GPU Support**: NVIDIA sm_50 and newer

---

## 🎓 Learning Resources

### Official Documentation
- See `00-START-HERE.md` in release package
- See `RELEASE_NOTES_v0.1.0.md` for detailed feature list
- Check `examples/` directory for code samples

### External Resources
- **llama.cpp Documentation**: https://github.com/ggml-org/llama.cpp/tree/master/docs
- **GGUF Format**: https://github.com/ggml-org/gguf
- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/

---

**Built with ❤️ for the open-source ML community**

⭐ If you find this project useful, please consider starring the repository!
