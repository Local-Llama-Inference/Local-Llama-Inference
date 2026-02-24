# Local-Llama-Inference Jupyter Notebooks

Complete set of example notebooks demonstrating local-llama-inference v0.1.0 features.

## 📚 Notebooks Overview

### 1️⃣ [01_quick_start.ipynb](01_quick_start.ipynb)
**Quick Start with Auto-Download** (Recommended First)

Learn the basics:
- ✅ Auto-download CUDA binaries from Hugging Face (first import)
- ✅ GPU detection and verification
- ✅ Download GGUF models
- ✅ Start LLM server
- ✅ Basic chat and multi-turn conversation
- ✅ Server health checks

**Time**: ~20 minutes (includes binary download on first run)
**Prerequisites**: NVIDIA GPU (sm_50+), Python 3.8+

---

### 2️⃣ [02_streaming_responses.ipynb](02_streaming_responses.ipynb)
**Token-by-Token Streaming**

Real-time text generation:
- 🔄 Stream chat responses
- 🔄 Stream text completion
- ⏱️ Measure latency (first token, tokens/sec)
- 🎯 Multiple sequential requests
- 💬 Interactive streaming conversation

**Time**: ~15 minutes
**Use Cases**: Chatbots, interactive applications, real-time feedback

---

### 3️⃣ [03_embeddings.ipynb](03_embeddings.ipynb)
**Text Embeddings & Semantic Search**

Vector representations and similarity:
- 🧮 Generate embeddings for single/multiple texts
- 🔍 Semantic similarity search
- 📍 Document clustering
- 📊 Similarity matrix calculation
- 🏆 Reranking search results
- 📈 Token counting

**Time**: ~15 minutes
**Use Cases**: RAG systems, semantic search, document clustering

---

### 4️⃣ [04_multi_gpu.ipynb](04_multi_gpu.ipynb)
**Multi-GPU Tensor Parallelism**

Distributed inference across GPUs:
- 🎮 Detect available GPUs
- 💡 Get tensor split recommendations
- 📊 GPU utilization monitoring
- ⚡ Multi-GPU benchmark
- 📈 Performance scaling
- 🔧 Configuration optimization

**Time**: ~20 minutes
**Prerequisites**: Single or multiple NVIDIA GPUs

---

### 5️⃣ [05_advanced_api.ipynb](05_advanced_api.ipynb)
**Complete API Reference**

All 30+ endpoints and advanced features:
- 💬 Chat, completion, streaming variants
- 🧮 Embeddings, tokenization, detokenization
- 🎯 Advanced features (infill, reranking, LoRA)
- 📊 Server status and metrics
- ⚙️ Sampling parameters (temperature, top-p, top-k)
- 🔄 Batch operations
- 🛡️ Error handling

**Time**: ~25 minutes
**Reference**: Complete API documentation

---

### 6️⃣ [06_gpu_detection.ipynb](06_gpu_detection.ipynb)
**GPU Detection & Configuration**

Comprehensive GPU analysis:
- 🖥️ System information
- 🎮 GPU detection and architecture
- 📊 VRAM analysis and model capacity
- 🎯 Tensor split recommendations
- 📋 Layer distribution calculator
- 📈 Real-time GPU status
- 🔧 Configuration recommendations
- 🛠️ Troubleshooting guide

**Time**: ~15 minutes
**Use Cases**: Optimization, troubleshooting, capacity planning

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Install local-llama-inference
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0

# Install optional dependencies
pip install jupyter jupyterlab scikit-learn scipy
```

### Run Notebooks

**Option 1: JupyterLab (Recommended)**
```bash
# Start JupyterLab
jupyter lab

# Navigate to notebooks folder and open 01_quick_start.ipynb
```

**Option 2: Jupyter Notebook**
```bash
# Start Jupyter Notebook
jupyter notebook

# Open http://localhost:8888
```

**Option 3: Command Line**
```bash
# Run specific notebook
jupyter nbconvert --to notebook --execute 01_quick_start.ipynb
```

---

## 📖 Recommended Sequence

**For Beginners:**
1. `06_gpu_detection.ipynb` - Understand your GPU
2. `01_quick_start.ipynb` - Learn basics
3. `02_streaming_responses.ipynb` - Real-time generation
4. `03_embeddings.ipynb` - Try embeddings

**For Advanced Users:**
1. `01_quick_start.ipynb` - Setup verification
2. `04_multi_gpu.ipynb` - Optimize for your hardware
3. `05_advanced_api.ipynb` - Explore all endpoints
4. `06_gpu_detection.ipynb` - Fine-tune configuration

**For Specific Use Cases:**

- **Chatbot/Conversational AI**:
  - `01_quick_start.ipynb` → `02_streaming_responses.ipynb`

- **Semantic Search/RAG**:
  - `03_embeddings.ipynb` → `05_advanced_api.ipynb`

- **High-Performance Inference**:
  - `04_multi_gpu.ipynb` → `06_gpu_detection.ipynb`

- **Full API Exploration**:
  - `05_advanced_api.ipynb` (comprehensive reference)

---

## 🎮 GPU Compatibility

These notebooks work on:
- ✅ **NVIDIA GPUs**: Compute Capability 5.0+ (Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper)
- ✅ **Single GPU**: All notebooks work with 1 GPU
- ✅ **Multi-GPU**: Notebook 04 demonstrates distributed inference
- ✅ **VRAM Requirements**: 2GB+ per GPU (4GB+ recommended)

**Supported Architectures:**
- sm_50 (Kepler K80, K40)
- sm_61 (Maxwell GTX 750, GTX 950)
- sm_61 (Pascal GTX 1060, GTX 1080)
- sm_70 (Volta V100)
- sm_75 (Turing RTX 2060, RTX 2080)
- sm_80 (Ampere RTX 3060, RTX 3090)
- sm_86 (Ada RTX 4080, RTX 6000)
- sm_89 (Hopper H100, H200)

---

## 📊 What Gets Downloaded

**On First Run (Notebook 01):**
- 834 MB CUDA binaries from Hugging Face
- Extracted to: `~/.local/share/local-llama-inference/`
- Cached for instant future use

**Optional Model Downloads:**
- Phi-2 Q4: ~1.4 GB
- Mistral 7B Q4: ~4.3 GB
- Llama 2 7B Q4: ~3.8 GB
- Other GGUF models: varies

---

## 🔧 Key Packages

Notebooks use these libraries (automatically installed):
- **local-llama-inference**: Main SDK
- **huggingface-hub**: Download models
- **httpx**: HTTP client
- **pydantic**: Data validation

Optional for advanced notebooks:
- **scikit-learn**: Clustering
- **scipy**: Similarity metrics
- **numpy**: Numerical operations

---

## 💡 Tips & Tricks

### Performance Optimization
1. **Check GPU first**: Run `06_gpu_detection.ipynb`
2. **Batch operations**: Process multiple requests together
3. **Stream responses**: Use streaming for faster first-token latency
4. **Cache embeddings**: Reuse embeddings for similar queries
5. **Monitor VRAM**: Run `nvidia-smi` during inference

### Model Selection
- **Small models**: Phi-2 (1.4 GB) - Fast, good quality
- **Medium models**: Mistral 7B (4.3 GB) - Balanced
- **Large models**: Llama 2 13B+ - Better quality, slower
- **Find models**: https://huggingface.co/models?search=gguf

### Common Issues

**"No GPUs detected"**
```bash
nvidia-smi  # Verify GPU is available
```

**"Out of Memory"**
- Reduce `n_gpu_layers` in LlamaServer
- Use smaller model
- Reduce `n_ctx` (context size)

**"Slow inference"**
- Increase `n_gpu_layers` (offload more to GPU)
- Check GPU utilization: `nvidia-smi dmon`
- Use larger batch size if applicable

---

## 📚 Resources

- **GitHub**: https://github.com/Local-Llama-Inference/Local-Llama-Inference
- **Models**: https://huggingface.co/models?search=gguf
- **Binaries**: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/
- **Documentation**: Check README.md in repository

---

## 🎓 Learning Outcomes

After running these notebooks, you'll understand:

✅ Auto-download mechanism from Hugging Face
✅ GPU detection and capability analysis
✅ Starting and managing LLM servers
✅ Chat completions and streaming responses
✅ Text embeddings and semantic search
✅ Multi-GPU tensor parallelism
✅ All 30+ API endpoints
✅ Configuration optimization
✅ Performance monitoring and troubleshooting

---

## 🤝 Contributing

Found issues? Suggestions for notebooks?
- **GitHub Issues**: Report bugs
- **GitHub Discussions**: Ask questions
- **Pull Requests**: Improve notebooks

---

## 📝 License

These notebooks are part of local-llama-inference v0.1.0
MIT License - See LICENSE file in main repository

---

**Ready to get started?** Open **01_quick_start.ipynb** in JupyterLab! 🚀

```bash
jupyter lab 01_quick_start.ipynb
```
