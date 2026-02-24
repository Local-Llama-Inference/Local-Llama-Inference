# PyPI Publishing Guide for local-llama-inference

This guide explains how to publish the `local-llama-inference` package to PyPI so users can install it with `pip install local-llama-inference`.

## Prerequisites

Before publishing, ensure you have:

1. **PyPI Account**: Create one at https://pypi.org/account/register/
2. **Twine**: `pip install twine`
3. **Build Tools**: `pip install build`
4. **Hugging Face Binaries**: All binaries must be uploaded to Hugging Face first

## Step 1: Verify Hugging Face Binaries Are Available

Before publishing to PyPI, ensure all binaries are available on Hugging Face:

```bash
# Check the Hugging Face dataset
# https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/
```

Expected files in `v0.1.0/`:
- `local-llama-inference-complete-v0.1.0.tar.gz` (834 MB)
- `local-llama-inference-complete-v0.1.0.tar.gz.sha256`
- `local-llama-inference-complete-v0.1.0.zip` (1.4 GB)
- `local-llama-inference-complete-v0.1.0.zip.sha256`
- `local-llama-inference-sdk-v0.1.0.tar.gz` (45 KB)
- `local-llama-inference-sdk-v0.1.0.tar.gz.sha256`
- Plus documentation files

## Step 2: Build Distribution Packages

Navigate to the `local-llama-inference` directory and build:

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/Project-LlamaInference/Local-Llama-Inference/local-llama-inference

# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# This creates:
# - dist/local-llama-inference-0.1.0.tar.gz (source)
# - dist/local-llama-inference-0.1.0-py3-none-any.whl (wheel)
```

Verify the build:

```bash
ls -lh dist/
```

## Step 3: Verify Package Contents

Before uploading, verify the package is correct:

```bash
# Check wheel contents
unzip -l dist/local-llama-inference-0.1.0-py3-none-any.whl | head -20

# Check source distribution contents
tar -tzf dist/local-llama-inference-0.1.0.tar.gz | head -20
```

Expected files should include:
- `local_llama_inference/` (Python modules)
- `local_llama_inference/cli.py` (CLI module)
- `local_llama_inference/_bootstrap/installer.py` (Binary installer)
- `setup.py`
- `README.md`
- `LICENSE`

## Step 4: Create PyPI API Token

1. Visit: https://pypi.org/manage/account/token/
2. Create a new token with name "local-llama-inference"
3. Select "Entire account" (or just "local-llama-inference" project if it exists)
4. Copy the token (you'll only see it once)

The token format: `pypi-AgEIcHlwaS5vcmc...`

## Step 5: Configure Twine

Create or update `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

**Permissions**: `chmod 600 ~/.pypirc`

## Step 6 (Optional): Test on TestPyPI First

It's recommended to test on TestPyPI first:

```bash
# Create TestPyPI account: https://test.pypi.org/account/register/
# Get token from: https://test.pypi.org/manage/account/token/

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ local-llama-inference

# Test the package
python -c "import local_llama_inference; print(local_llama_inference.__version__)"

# Test CLI
llama-inference info
```

## Step 7: Publish to PyPI

Once verified, publish to the official PyPI:

```bash
# Upload to PyPI
twine upload dist/*

# This will prompt for your PyPI credentials (use __token__ as username)
```

Example output:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading local-llama-inference-0.1.0.tar.gz
Uploading local-llama-inference-0.1.0-py3-none-any.whl
```

## Step 8: Verify PyPI Publication

Visit: https://pypi.org/project/local-llama-inference/

You should see:
- Package name, version, and description
- Download statistics
- Project links
- Release history

## Step 9: Test Installation from PyPI

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate

# Install from PyPI (may take a minute to appear)
pip install local-llama-inference

# Verify installation
python -c "from local_llama_inference import LlamaServer; print('✅ Installation successful')"

# Test CLI
llama-inference install
llama-inference verify
llama-inference info
```

## Step 10: Update GitHub Release

Create a release on GitHub:

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/Project-LlamaInference/Local-Llama-Inference/local-llama-inference

gh release create v0.1.0 \
  --title "local-llama-inference v0.1.0" \
  --body "GPU-accelerated LLM inference with NVIDIA NCCL support.

Install with: \`pip install local-llama-inference\`

See https://pypi.org/project/local-llama-inference/ for details."
```

## Publishing New Versions

For future releases:

1. Update version in `src/local_llama_inference/_version.py`
2. Update `setup.py` if needed
3. Update `CHANGELOG.md` with release notes
4. Build: `python -m build --clean`
5. Upload: `twine upload dist/*`
6. Create GitHub release: `gh release create v<version> ...`

## Troubleshooting

### "File already exists" error

PyPI doesn't allow overwriting existing releases. To fix:

```bash
# Update version number and rebuild
# Or delete the old dist/ and rebuild

rm -rf dist/
python -m build
```

### "Invalid distribution" error

Check package integrity:

```bash
twine check dist/*
```

Fix any reported issues and rebuild.

### Package not appearing on PyPI

PyPI caches updates. Wait a few minutes and refresh:

```bash
# Direct URL to check
https://pypi.org/project/local-llama-inference/
```

### Installation fails with "huggingface-hub not found"

The package depends on huggingface-hub for binary downloads. This should install automatically:

```bash
pip install --upgrade local-llama-inference
```

## Automatic Binary Download on First Use

When users install `local-llama-inference`, the package:

1. **First import**: Detects if binaries are installed
2. **If missing**: Prompts to run `llama-inference install`
3. **Download**: Fetches from Hugging Face (https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/)
4. **Cache**: Stores in `~/.local/share/local-llama-inference/`
5. **Ready**: Package fully functional after binaries download

## Documentation Links

- **GitHub**: https://github.com/Local-Llama-Inference/Local-Llama-Inference
- **PyPI**: https://pypi.org/project/local-llama-inference/
- **Hugging Face**: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/
- **Issues**: https://github.com/Local-Llama-Inference/Local-Llama-Inference/issues

## Support

For questions or issues:
1. Check existing GitHub issues
2. Create a new issue on GitHub
3. Contact: waqasm86@example.com
