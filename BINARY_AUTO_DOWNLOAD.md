# CUDA Binary Auto-Download - local-llama-inference v0.1.0+

## Overview

When you install local-llama-inference via pip, the SDK code is installed immediately, but the CUDA binaries (~834MB) are downloaded **automatically on first import**.

This document explains:
1. What happens when you `pip install`
2. What happens when you `import local_llama_inference`
3. What to expect during binary download
4. How to troubleshoot if download fails

---

## The Installation Process

### Step 1: Pip Install (Fast - 10-30 seconds)

```bash
pip install git+https://github.com/Local-Llama-Inference/Local-Llama-Inference.git@v0.1.0
```

This installs:
- ✅ Python SDK code (~50KB)
- ✅ Configuration files
- ✅ Documentation
- ❌ NOT the CUDA binaries (downloaded later)

**Output**:
```
Successfully installed local-llama-inference-0.1.0
```

### Step 2: First Import (Slow - 5-15 minutes first time)

```python
import local_llama_inference
```

On first import, the SDK automatically:
1. Checks if CUDA binaries are installed
2. If not, downloads them from Hugging Face CDN
3. Verifies checksum for integrity
4. Extracts binaries to `~/.local/share/local-llama-inference/`

**Expected Output**:
```
======================================================================
FIRST-TIME SETUP: Installing local-llama-inference binaries
======================================================================

This will download ~834MB of CUDA binaries from Hugging Face CDN.
Repository: waqasm86/Local-Llama-Inference
Download location: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference
Installation path: ~/.local/share/local-llama-inference/

This may take 5-15 minutes depending on your internet speed...
Please do NOT interrupt this process.

⏳ Starting download (hf_hub_download may show progress)...
[download progress...]

✅ Download complete!
   Location: /home/user/.cache/huggingface/hub/...

🔒 Verifying integrity (SHA256 checksum)...
✅ Integrity verified - file is not corrupted

📦 Extracting binaries (this may take 1-2 minutes)...
   Source: /home/user/.cache/.../local-llama-inference-complete-v0.1.0.tar.gz
   Target: /home/user/.local/share/local-llama-inference/extracted/

✅ Extraction complete!
   Contents: llama-dist/ nccl-dist/

✅ CUDA binaries ready!
   Binaries: /home/user/.local/share/local-llama-inference/extracted/llama-dist/bin/
   Libraries: /home/user/.local/share/local-llama-inference/extracted/llama-dist/lib/
             /home/user/.local/share/local-llama-inference/extracted/nccl-dist/lib/

======================================================================
✅ CUDA BINARIES READY - SDK is now fully functional
======================================================================
```

### Step 3: Subsequent Imports (Fast - < 1 second)

```python
import local_llama_inference  # Already installed, no download needed
```

After the first import, binaries are cached locally. Subsequent imports are instant.

---

## What Gets Downloaded

### File Details
- **Filename**: `local-llama-inference-complete-v0.1.0.tar.gz`
- **Size**: ~834 MB
- **Format**: Compressed tar archive
- **Source**: Hugging Face CDN: `waqasm86/Local-Llama-Inference`

### Contents
```
extracted/
├── llama-dist/
│   ├── bin/
│   │   └── llama-server          (5.6 MB - main executable)
│   └── lib/
│       ├── libllama.so           (main inference library)
│       ├── libggml-cpu.so        (CPU backend)
│       ├── libggml-cuda.so       (CUDA GPU backend - 45MB)
│       └── [other shared libs]
└── nccl-dist/
    ├── lib/
    │   ├── libnccl.so.2          (NVIDIA Collective Communications)
    │   └── libnccl-net.so        (NCCL networking)
    └── include/
        └── [NCCL headers]
```

---

## Where Binaries Are Stored

### Installation Location
```
~/.local/share/local-llama-inference/
├── .installed                          (marker file, version info)
├── extracted/                          (extracted binaries)
│   ├── llama-dist/
│   └── nccl-dist/
└── [cache files from hf_hub_download]
```

### Environment Variables
```bash
# Binary directory
~/.local/share/local-llama-inference/extracted/llama-dist/bin/

# Library directories
~/.local/share/local-llama-inference/extracted/llama-dist/lib/
~/.local/share/local-llama-inference/extracted/nccl-dist/lib/

# These are automatically configured on server startup via LD_LIBRARY_PATH
```

### Disk Space Required
- Downloaded archive: ~834 MB
- Extracted binaries: ~1.2 GB
- Cache files: ~100 MB
- **Total**: ~2.1 GB (one-time, after first import)

---

## Troubleshooting

### Issue 1: Download Never Starts

**Symptom**: You import the module and nothing happens for a few minutes

**Solution**:
```bash
# Check if binaries are already installed
ls ~/.local/share/local-llama-inference/extracted/

# If directory doesn't exist or is empty, manually trigger download:
python -c "from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary()"
```

---

### Issue 2: Download Fails (Network Error)

**Symptom**: Error message about connection or timeout

**Solution**:
```bash
# Retry with force=True to re-download
python -c "from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary(force=True)"

# Or check your internet connection:
ping huggingface.co

# If you're behind a proxy, configure it:
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

---

### Issue 3: Checksum Verification Failed

**Symptom**: "SHA256 checksum mismatch" error

**Solution**:
```bash
# The downloaded file is corrupted, force re-download:
python -c "from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary(force=True)"

# Or manually delete and retry:
rm -rf ~/.local/share/local-llama-inference/
python -c "from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary()"
```

---

### Issue 4: Extraction Failed

**Symptom**: "Extraction failed" or "Tar member would escape" error

**Solution**:
```bash
# Clean and retry
rm -rf ~/.local/share/local-llama-inference/
python -c "from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary()"
```

---

### Issue 5: "Binary paths could not be resolved"

**Symptom**: Import succeeds but you get a warning about unresolved paths

**Solution**:
```bash
# Check what was extracted
ls -la ~/.local/share/local-llama-inference/extracted/

# Expected output:
# drwxr-xr-x  llama-dist
# drwxr-xr-x  nccl-dist

# If directories are missing, re-extract:
python -c "from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary(force=True)"
```

---

## Advanced Usage

### Skip Auto-Download (Not Recommended)

For offline or air-gapped environments:

```bash
export LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1
python your_script.py  # Won't attempt to download
```

**Note**: You MUST ensure binaries are pre-installed at `~/.local/share/local-llama-inference/extracted/`

### Manual Binary Installation

If you want to download binaries separately:

```python
from local_llama_inference import BinaryInstaller

installer = BinaryInstaller()

# Check if installed
if not installer.is_installed():
    print("Downloading binaries...")
    installer.download_binary()  # Downloads from HF CDN
else:
    print("Binaries already installed")

# Get installed paths
paths = installer.get_binary_paths()
print(f"Binary: {paths['llama_bin']}")
print(f"Libraries: {paths['llama_lib']}, {paths['nccl_lib']}")
```

### Force Re-Download

```python
from local_llama_inference import BinaryInstaller

installer = BinaryInstaller()
installer.download_binary(force=True)  # Re-download even if already installed
```

### Custom Installation Path

```python
from local_llama_inference import BinaryInstaller

# Install to custom location
custom_path = "/opt/local-llama-inference"
installer = BinaryInstaller(cache_dir=custom_path)
installer.download_binary()

# Binaries will be at: /opt/local-llama-inference/extracted/
```

---

## Network & Bandwidth

### Download Speed Estimates

| Connection | Time | Status |
|-----------|------|--------|
| Fiber (300+ Mbps) | 3-5 minutes | ✅ Fast |
| Broadband (100 Mbps) | 7-12 minutes | ✅ Normal |
| DSL (30 Mbps) | 20-30 minutes | ⚠️ Slow |
| Mobile 4G (10 Mbps) | 60+ minutes | ❌ Very Slow |
| Mobile 3G (1 Mbps) | Not recommended | ❌ Too Slow |

### Bandwidth Required

- **Download**: ~834 MB (one-time)
- **Verification**: Negligible (SHA256 checksum)
- **Cache**: ~100 MB (temporary during extraction)

### Offline Installation

For air-gapped environments:

```bash
# On system with internet:
1. Download from: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/tree/main/v0.1.0
2. Get: local-llama-inference-complete-v0.1.0.tar.gz

# Transfer to offline system and extract:
mkdir -p ~/.local/share/local-llama-inference/extracted
tar -xzf local-llama-inference-complete-v0.1.0.tar.gz -C ~/.local/share/local-llama-inference/extracted
touch ~/.local/share/local-llama-inference/.installed
```

---

## How It Works Under the Hood

### Auto-Download Flow

```
pip install
    ↓
    └→ Installs Python SDK code only
    
User: import local_llama_inference
    ↓
    └→ __init__.py calls ensure_binaries_installed()
        ↓
        ├→ Check: Is ~/.local/share/local-llama-inference/.installed present?
        │   ├→ YES: Binaries already here, use them
        │   └→ NO: Continue to download
        ↓
        └→ BinaryInstaller.download_binary()
            ├→ Get platform bundle info (linux x86_64)
            ├→ Download from HF: waqasm86/Local-Llama-Inference/v0.1.0/...
            ├→ Verify SHA256 checksum
            ├→ Extract to ~/.local/share/local-llama-inference/extracted/
            ├→ Create marker file: ~/.local/share/local-llama-inference/.installed
            └→ Return paths to binaries
    
SDK ready: LlamaServer, LlamaClient, etc. work normally
```

### Why This Approach?

1. **Small pip package**: SDK code is only ~50KB, installs in seconds
2. **On-demand binaries**: Large files (834MB) downloaded only when needed
3. **Cache**: Downloaded once, reused forever
4. **Offline option**: Can skip download with environment variable
5. **Transparency**: Clear messages about what's being downloaded

---

## FAQ

**Q: Why is the file so large (834MB)?**  
A: It contains compiled llama.cpp for multiple architectures, NVIDIA NCCL, and all shared libraries. Most of the size is the CUDA GPU backend.

**Q: Can I use the SDK without downloading binaries?**  
A: No, you must have the CUDA binaries. You can skip auto-download with `LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1` but then you must manually provide the binaries.

**Q: What if my internet cuts out during download?**  
A: The download will fail and you'll get an error. Just run the import again—it will retry from where it left off (HF uses resumable downloads).

**Q: Can I move the extracted binaries?**  
A: No, the SDK expects them at `~/.local/share/local-llama-inference/extracted/`. If you move them, the SDK won't find them.

**Q: Do I need to be admin to install?**  
A: No, the binaries are installed to `~/.local/` which is in your home directory. No admin privileges needed.

**Q: Can I use different binaries?**  
A: Yes, set the `LLAMA_BIN_DIR` environment variable to point to your custom llama-server binary before importing.

---

## Summary

1. **pip install**: Quick, installs Python code only
2. **First import**: Automatic CUDA binary download (5-15 min, 834MB)
3. **Subsequent imports**: Instant, uses cached binaries
4. **Storage**: ~2.1GB total in `~/.local/share/`
5. **No internet needed after first import**

The entire process is transparent and automatic. You'll see clear messages showing:
- What's being downloaded
- Download progress
- Time remaining estimate
- Completion status

If something goes wrong, clear error messages guide you to solutions.

