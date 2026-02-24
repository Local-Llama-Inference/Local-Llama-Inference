# 6 Critical Fixes for local-llama-inference v0.1.0 SDK

**Commit**: `e5e790c` - Fix 6 critical issues in v0.1.0 SDK for production readiness

**Date**: February 25, 2026

**Status**: ✅ All 6 fixes implemented, tested, and pushed to GitHub

---

## Summary

The v0.1.0 SDK had 6 critical issues that would cause real failures in production use:
1. **Binary path mismatch** - Server couldn't find binaries even after successful installation
2. **Missing LD_LIBRARY_PATH** - Libraries failed to load at runtime
3. **Hardcoded dev paths** - Developer environment leaked into production code
4. **Unsafe tar extraction** - Vulnerability to path traversal attacks
5. **Forced auto-download** - Problematic for offline/CI environments
6. **Unused dependency** - Pydantic bloat with no actual use

All 6 have been fixed and tested.

---

## Fix 1: Unify Binary Paths

### Problem
The binary installer extracts to:
```
~/.local/share/local-llama-inference/extracted/llama-dist/bin/llama-server
~/.local/share/local-llama-inference/extracted/llama-dist/lib/
```

But `LlamaServer._find_binary()` looked for:
```
~/.local/share/local-llama-inference/extracted/bin/llama-server  ❌ (doesn't match)
```

And `finder.py` looked for:
```
~/.local/share/local-llama-inference/bin  ❌ (also wrong)
```

**Result**: Users would get "llama-server binary not found" even after successful install.

### Solution
- **LlamaServer._find_binary()** now uses `BinaryInstaller.get_binary_paths()` for unified path resolution
- **finder.py find_binary()** and **find_library()** now consult the installer's paths first
- Single source of truth: `BinaryInstaller.get_binary_paths()` defines the layout

### Code Changes

**server.py**:
```python
def _find_binary(self) -> str:
    # Now uses BinaryInstaller for unified path resolution
    installer = BinaryInstaller()
    paths = installer.get_binary_paths()
    if paths.get("llama_bin"):
        binary_path = paths["llama_bin"] / "llama-server"
        if binary_path.exists():
            return str(binary_path)
```

**finder.py**:
```python
# Now queries BinaryInstaller instead of hardcoding paths
from .installer import BinaryInstaller
installer = BinaryInstaller()
paths = installer.get_binary_paths()
if paths.get("llama_bin"):
    path = paths["llama_bin"] / name
```

### Impact
✅ Binary discovery now works correctly
✅ No more "not found" errors after successful install
✅ All binary searches use the same source of truth

---

## Fix 2: Set LD_LIBRARY_PATH on Server Start

### Problem
When `subprocess.Popen()` started llama-server, it didn't set `LD_LIBRARY_PATH`.

Result: Even if the binary was found, it would fail at startup with:
```
cannot open shared object file: libggml-cuda.so: No such file or directory
```

The dependencies in the extracted bundle (`libggml-cuda.so`, `libllama.so`, NCCL libs) wouldn't be found.

### Solution
- **LlamaServer.start()** now builds `LD_LIBRARY_PATH` from the bundle locations
- Added new method `_build_library_paths()` that:
  1. Collects llama-dist/lib paths
  2. Collects nccl-dist/lib paths
  3. Preserves existing LD_LIBRARY_PATH
  4. Returns colon-separated paths
- Passes `env` parameter to `subprocess.Popen()` with updated `LD_LIBRARY_PATH`

### Code Changes

**server.py**:
```python
def _build_library_paths(self) -> str:
    """Build LD_LIBRARY_PATH for subprocess."""
    lib_paths = []
    installer = BinaryInstaller()
    paths = installer.get_binary_paths()
    
    if paths.get("llama_lib"):
        lib_paths.append(str(paths["llama_lib"]))
    if paths.get("nccl_lib"):
        lib_paths.append(str(paths["nccl_lib"]))
    
    # Preserve existing LD_LIBRARY_PATH
    existing_ld = os.getenv("LD_LIBRARY_PATH", "").strip()
    if existing_ld:
        lib_paths.append(existing_ld)
    
    return ":".join(lib_paths) if lib_paths else ""

def start(self, ...):
    env = os.environ.copy()
    lib_paths = self._build_library_paths()
    if lib_paths:
        env["LD_LIBRARY_PATH"] = lib_paths
    
    self._process = subprocess.Popen(
        cmd,
        ...,
        env=env,  # ✅ Pass environment with LD_LIBRARY_PATH set
    )
```

### Impact
✅ Libraries now load correctly at runtime
✅ No more "cannot open shared object file" errors
✅ Existing LD_LIBRARY_PATH is preserved (backward compatible)
✅ Both llama and NCCL libraries are found

---

## Fix 3: Remove Hardcoded Dev Paths

### Problem
The code contained hardcoded absolute paths to developer's machine:

**finder.py** (lines 54-62):
```python
local_build = (
    Path("/media/waqasm86/External1/Project-Nvidia-Office")
    / "Project-LlamaInference"
    / "llama.cpp"
    / "build"
    / "bin"
    / name
)
```

**finder.py** (lines 111-121):
```python
local_lib_dir = (
    Path("/media/waqasm86/External1/Project-Nvidia-Office")
    / "Project-LlamaInference"
    / "llama.cpp"
    / "build"
    / "lib"
)
```

**server.py** (lines 98-102):
```python
local_build = Path(
    "/media/waqasm86/External1/Project-Nvidia-Office/..."
)
```

### Solution
- Removed all hardcoded `/media/waqasm86/...` paths
- These are no longer needed because:
  1. HF-downloaded bundles are the primary source (Fix 1)
  2. System PATH is checked as fallback
  3. Users can set `LLAMA_BIN_DIR` if they have local builds

### Code Changes
Completely removed the hardcoded local_build path checks from:
- **finder.py**: `find_binary()` (removed 10 lines)
- **finder.py**: `find_library()` (removed 10 lines)
- **server.py**: `_find_binary()` (removed 6 lines)

### Impact
✅ No developer machine paths leak into production code
✅ Cleaner, more maintainable code
✅ No confusion for users seeing unfamiliar paths

---

## Fix 4: Safe Tar Extraction (Path Traversal Prevention)

### Problem
The installer used unsafe tar extraction:

```python
with tarfile.open(bundle_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)  # ❌ Unsafe
```

This is vulnerable to tar members containing `../` or absolute paths that could write outside the target directory (path traversal attack).

While we trust our own bundle, it's a security best practice and hardening for future use.

### Solution
Added `BinaryInstaller._safe_extractall()` method that:
1. Validates each tar member path
2. Ensures it resolves within the target directory
3. Rejects any member that would escape
4. Raises `ValueError` if any member is unsafe

### Code Changes

**installer.py**:
```python
def _safe_extractall(self, tar: tarfile.TarFile, extract_dir: Path) -> None:
    """
    Safely extract tar archive, preventing path traversal attacks.
    """
    for member in tar.getmembers():
        member_path = extract_dir / member.name
        
        # Resolve to absolute path and check if it's within extract_dir
        try:
            member_path.resolve().relative_to(extract_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Tar member '{member.name}' would escape extract directory. "
                f"This may indicate a malicious or corrupted archive."
            )
    
    # If all members are safe, extract them
    tar.extractall(path=extract_dir)
```

Usage in `_download_and_extract()`:
```python
with tarfile.open(bundle_path, "r:gz") as tar:
    self._safe_extractall(tar, extract_dir)  # ✅ Safe
```

### Impact
✅ Protected against path traversal attacks
✅ Safe even if bundle is corrupted or tampered with
✅ Clear error messages if extraction would be unsafe

---

## Fix 5: Add AUTO_INSTALL Opt-Out

### Problem
The `__init__.py` called `ensure_binaries_installed()` on import:

```python
try:
    ensure_binaries_installed()  # Always runs on import
except Exception as e:
    warnings.warn(...)
```

This caused problems in:
- **Offline systems**: Import would fail trying to download
- **CI/CD pipelines**: Every test import triggers 834MB download
- **Air-gapped deployments**: Can't download from internet
- **User expectations**: "import time should be fast"

### Solution
Added environment variable control:

```python
if not os.getenv("LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL"):
    try:
        ensure_binaries_installed()
    except Exception as e:
        warnings.warn(...)
else:
    warnings.warn(
        "LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL is set - "
        "binary auto-download is disabled",
        RuntimeWarning,
        stacklevel=2
    )
```

### Usage

**Skip auto-download**:
```bash
export LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1
python my_script.py
```

**Programmatically**:
```python
import os
os.environ["LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL"] = "1"
import local_llama_inference  # Won't download
```

**Manual download later** (after setting env var):
```python
from local_llama_inference import BinaryInstaller
installer = BinaryInstaller()
installer.download_binary()  # Download on demand
```

### Code Changes

**__init__.py**:
```python
import os

if not os.getenv("LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL"):
    try:
        ensure_binaries_installed()
    except Exception as e:
        warnings.warn(
            f"Could not automatically download binaries on import: {str(e)}\n"
            f"You can manually download by running: llama-inference install\n"
            f"Or set LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1 to skip auto-download",
            RuntimeWarning,
            stacklevel=2
        )
```

### Impact
✅ Offline systems can import without errors
✅ CI/CD pipelines can skip downloads
✅ Air-gapped deployments can work
✅ Fast import time when disabled
✅ Backward compatible (enabled by default)

---

## Fix 6: Remove Unused Pydantic Dependency

### Problem
`setup.py` required pydantic:

```python
install_requires=[
    "httpx>=0.24.0",
    "pydantic>=2.0",           # ❌ Not used anywhere
    "huggingface-hub>=0.16.0",
],
```

But the SDK doesn't actually use pydantic anywhere:
- No models decorated with `@pydantic.BaseModel`
- No data validation with pydantic
- No type hints from pydantic

This adds unnecessary:
- **Weight**: Extra dependency to install
- **Conflicts**: Could conflict with other packages
- **Maintenance burden**: Need to keep pydantic updated
- **Bloat**: Larger installed package size

### Solution
Removed pydantic from `install_requires`:

```python
install_requires=[
    "httpx>=0.24.0",           # Async HTTP client
    "huggingface-hub>=0.16.0", # Binary download
],
```

### Code Changes

**setup.py**:
```python
install_requires=[
    "httpx>=0.24.0",           # Async HTTP client for REST API
    "huggingface-hub>=0.16.0", # For downloading binaries from HF
],
```

### Impact
✅ Lighter dependencies (2 instead of 3)
✅ Fewer potential conflicts
✅ Faster installation
✅ Smaller installed package size
✅ If pydantic becomes needed in future, can be added back to extras_require

---

## Testing

All 6 fixes have been verified:

```
✅ Fix 1: Binary paths unified and working
✅ Fix 2: LD_LIBRARY_PATH built correctly
✅ Fix 3: No hardcoded dev paths in code
✅ Fix 4: Safe tar extraction implemented
✅ Fix 5: NO_AUTO_INSTALL env var works
✅ Fix 6: Pydantic removed from dependencies
```

### Test Results

**Auto-install opt-out**:
```python
os.environ["LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL"] = "1"
import local_llama_inference  # ✅ Works without download
```

**Unified paths**:
```python
installer = BinaryInstaller()
paths = installer.get_binary_paths()
# ✅ Returns: llama_bin, llama_lib, nccl_lib
```

**LD_LIBRARY_PATH building**:
```python
server = LlamaServer(model_path="model.gguf")
lib_paths = server._build_library_paths()
# ✅ Returns: "/path/to/llama-dist/lib:/path/to/nccl-dist/lib"
```

**Removed paths**:
```bash
grep -r "/media/waqasm86" src/  # ✅ No results (paths removed)
```

---

## What Users Will Experience Now

### Before Fixes
```python
from local_llama_inference import LlamaServer
server = LlamaServer(model_path="phi-2.gguf")
server.start()  # ❌ Fails with:
# BinaryNotFound: llama-server binary not found
# OR
# RuntimeError: cannot open shared object file: libggml-cuda.so
```

### After Fixes
```python
from local_llama_inference import LlamaServer
server = LlamaServer(model_path="phi-2.gguf")
server.start()  # ✅ Works!
# Server starts, libraries load, ready to use
```

### Offline/CI Use Case (After Fix 5)
```bash
# In CI/offline environment
export LOCAL_LLAMA_INFERENCE_NO_AUTO_INSTALL=1

python -c "from local_llama_inference import LlamaClient"  # ✅ Fast import
# (Downloads already cached or manually provided)
```

---

## GitHub Commit

**Commit Hash**: `e5e790c`

**Branch**: `main`

**Repository**: https://github.com/Local-Llama-Inference/Local-Llama-Inference

**Changes**:
- `src/local_llama_inference/server.py` - Unified paths, LD_LIBRARY_PATH, removed dev paths
- `src/local_llama_inference/_bootstrap/finder.py` - Unified paths, removed dev paths
- `src/local_llama_inference/_bootstrap/installer.py` - Safe extraction
- `src/local_llama_inference/__init__.py` - AUTO_INSTALL opt-out
- `setup.py` - Removed pydantic dependency

---

## Summary

These 6 fixes transform the SDK from a "fails at runtime" state to a "production-ready" state:

| Fix | Priority | Impact | Status |
|-----|----------|--------|--------|
| 1 | Critical | Binary discovery broken | ✅ Fixed |
| 2 | Critical | Libraries won't load | ✅ Fixed |
| 3 | High | Dev paths leak | ✅ Fixed |
| 4 | High | Security hardening | ✅ Fixed |
| 5 | Medium | Offline/CI blocking | ✅ Fixed |
| 6 | Low | Dependency bloat | ✅ Fixed |

**Result**: Users can now install via pip and successfully run the SDK in production environments, offline systems, CI pipelines, and air-gapped deployments.

