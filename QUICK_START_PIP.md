# Quick Start Guide: pip install local-llama-inference

**The simplest way to get GPU-accelerated LLM inference running in 4 steps!**

---

## ⚡ 4-Minute Setup

### Step 1: Install Package (1 minute)
```bash
pip install local-llama-inference
```

### Step 2: Download Binaries (10-15 minutes, first time only)
```bash
# This happens automatically on first use, OR:
llama-inference install
```

### Step 3: Download a Model
```bash
# Get Mistral 7B Q4 quantized (4.3 GB)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf

# Or use any GGUF model from: https://huggingface.co/models?search=gguf
```

### Step 4: Run Inference
```python
from local_llama_inference import LlamaServer, LlamaClient

# Start server
server = LlamaServer(
    model_path="./Mistral-7B-Instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=33,  # Use GPU
)
server.start()
server.wait_ready()

# Create client
client = LlamaClient()

# Chat with the model
response = client.chat_completion(
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
print(response.choices[0].message.content)

# Clean up
server.stop()
```

**Done!** 🎉 You're running GPU-accelerated LLM inference!

---

## 📊 What Happens Behind the Scenes

```
pip install local-llama-inference
        ↓
  Downloads 29 KB wheel from PyPI
        ↓
  Installs: httpx, pydantic, huggingface-hub
        ↓
from local_llama_inference import LlamaServer
        ↓
  Package checks: Do binaries exist?
        ↓
  NO? → Downloads 834 MB from Hugging Face CDN (1-2 Mbps)
        ↓
  Extracts to ~/.local/share/local-llama-inference/
        ↓
  ✅ Ready to use! (and cached for next time)
```

---

## 🛠️ Useful Commands

```bash
# Check installation
llama-inference verify

# Show package info
llama-inference info

# Force reinstall binaries
llama-inference install --force

# Use custom cache directory
llama-inference install --cache-dir /path/to/cache
```

---

## 🐍 Python Examples

### Example 1: Simple Chat
```python
from local_llama_inference import LlamaServer, LlamaClient

server = LlamaServer(model_path="model.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

server.stop()
```

### Example 2: Streaming Responses
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

server.stop()
```

### Example 3: Multi-GPU Inference
```python
from local_llama_inference import (
    LlamaServer, LlamaClient, detect_gpus, suggest_tensor_split
)

# Auto-detect GPUs
gpus = detect_gpus()
print(f"Found {len(gpus)} GPUs")

# Get optimal distribution
tensor_split = suggest_tensor_split(gpus)

# Run on multiple GPUs
server = LlamaServer(
    model_path="model.gguf",
    n_gpu_layers=33,
    tensor_split=tensor_split,
)
server.start()
server.wait_ready()

client = LlamaClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello from multi-GPU!"}]
)
print(response.choices[0].message.content)

server.stop()
```

### Example 4: Embeddings
```python
from local_llama_inference import LlamaServer, LlamaClient

server = LlamaServer(model_path="embedding_model.gguf", n_gpu_layers=33)
server.start()
server.wait_ready()

client = LlamaClient()

# Generate embeddings
result = client.embed(input="What is machine learning?")
embedding = result.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")

server.stop()
```

---

## ❓ Common Questions

### Q: How long does binary download take?
**A:** 10-15 minutes on a typical 1-2 Mbps connection. Happens only once! After that, binaries are cached.

### Q: How much disk space do I need?
**A:** ~1 GB free for download and extraction. Binaries take ~834 MB after extraction.

### Q: Which GPUs are supported?
**A:** NVIDIA GPUs with compute capability 5.0+ (sm_50 and newer)
- Kepler (K80, K40)
- Maxwell (GTX 750, GTX 950)
- Pascal (GTX 1060, GTX 1080)
- Volta (V100)
- Turing (RTX 2060, RTX 2080)
- Ampere (RTX 3060, RTX 3090)
- Ada (RTX 4080, RTX 6000)
- Hopper (H100, H200)

### Q: Do I need CUDA installed?
**A:** No! The CUDA runtime is included in the auto-downloaded binaries. Just need NVIDIA drivers.

### Q: Can I use a custom cache location?
**A:** Yes! Use `llama-inference install --cache-dir /custom/path`

### Q: How do I update to a new version?
**A:** Just run `pip install --upgrade local-llama-inference` and re-download binaries if needed.

---

## 🔗 Resources

- **GitHub**: https://github.com/Local-Llama-Inference/Local-Llama-Inference
- **PyPI**: https://pypi.org/project/local-llama-inference/
- **Hugging Face**: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/
- **GGUF Models**: https://huggingface.co/models?search=gguf

---

## 🚨 Troubleshooting

### Issue: "pip install" fails
```bash
# Update pip first
pip install --upgrade pip

# Then try again
pip install local-llama-inference
```

### Issue: Binaries fail to download
```bash
# Check internet connection
ping huggingface.co

# Manually download with CLI
llama-inference install

# Check logs
llama-inference verify
```

### Issue: "CUDA out of memory"
```python
# Reduce GPU layers
server = LlamaServer(model_path="model.gguf", n_gpu_layers=15)

# Or use smaller quantization (Q2, Q3 instead of Q5, Q6)
```

### Issue: GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Verify in Python
python -c "from local_llama_inference import detect_gpus; print(detect_gpus())"
```

---

## ✨ What You Get

✅ GPU-accelerated LLM inference
✅ Single & multi-GPU support
✅ 30+ REST API endpoints
✅ OpenAI-compatible chat API
✅ Streaming responses
✅ Embeddings & reranking
✅ Production-ready error handling
✅ Process management
✅ Health monitoring

---

## 🎓 Next Steps

1. **Install**: `pip install local-llama-inference`
2. **Verify**: `llama-inference verify`
3. **Get a Model**: Download from [Hugging Face](https://huggingface.co/models?search=gguf)
4. **Run Inference**: Use the examples above
5. **Read Docs**: Check [README.md](README.md) for detailed documentation

---

**That's it! You're ready to go!** 🚀

For detailed documentation, see [README.md](README.md).
For setup details, see [PIP_INSTALL_SETUP.md](PIP_INSTALL_SETUP.md).
