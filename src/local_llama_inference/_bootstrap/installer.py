"""
Automatic binary installer for local-llama-inference.

This module handles downloading and extracting pre-built binaries from Hugging Face
on first use, providing a seamless installation experience for end users.
"""

import os
import sys
import tarfile
import hashlib
import tempfile
import platform
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


class BinaryInstaller:
    """Manages downloading and installing pre-built binaries from Hugging Face."""

    # Hugging Face dataset configuration
    HF_REPO_ID = "waqasm86/Local-Llama-Inference"
    HF_REPO_TYPE = "dataset"
    VERSION = "0.1.0"

    # Binary bundle information
    BUNDLES = {
        "linux": {
            "filename": "local-llama-inference-complete-v0.1.0.tar.gz",
            "sha256": "b9b1a813e44f38c249e4d312ee88be94849a907da4f22fe9995c3d29d845c0b9",
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the binary installer.

        Args:
            cache_dir: Custom cache directory for binaries.
                      Defaults to ~/.local/share/local-llama-inference/
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".local" / "share" / "local-llama-inference"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_installed(self) -> bool:
        """Check if binaries are already installed."""
        marker_file = self.cache_dir / ".installed"
        return marker_file.exists()

    def get_platform_bundle(self) -> Optional[dict]:
        """Get the appropriate binary bundle for current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "linux" and machine in ("x86_64", "amd64"):
            return self.BUNDLES.get("linux")

        return None

    def download_binary(self, force: bool = False) -> bool:
        """
        Download and extract binary bundle from Hugging Face.

        Downloads the complete CUDA binaries bundle (~834MB) which includes:
        - llama.cpp compiled for NVIDIA CUDA GPUs
        - NVIDIA NCCL library for multi-GPU support
        - All required shared libraries

        Args:
            force: Force download even if already installed

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            print("✅ CUDA binaries already installed")
            return True

        bundle_info = self.get_platform_bundle()
        if bundle_info is None:
            print(f"❌ Binary bundle not available for your platform: "
                  f"{platform.system()} {platform.machine()}")
            print(f"   Supported: Linux x86_64")
            print(f"   Current: {platform.system()} {platform.machine()}")
            return False

        if hf_hub_download is None:
            print("❌ FATAL: huggingface-hub package required for binary download")
            print("   Install with: pip install huggingface-hub")
            return False

        try:
            print("\n" + "="*70)
            print("DOWNLOADING CUDA BINARIES")
            print("="*70)
            success = self._download_and_extract(bundle_info, force)
            if success:
                print("\n" + "="*70)
                print("✅ CUDA BINARIES READY - SDK is now fully functional")
                print("="*70 + "\n")
            return success
        except Exception as e:
            print(f"\n❌ Fatal error downloading binaries: {str(e)}")
            print(f"   Please check your internet connection and try again.")
            return False

    def _download_and_extract(self, bundle_info: dict, force: bool = False) -> bool:
        """Download and extract binary bundle."""
        filename = bundle_info["filename"]
        remote_path = f"v{self.VERSION}/{filename}"
        file_size_gb = 0.834  # Approximate size in GB

        print(f"\n📥 Downloading {filename} (~{file_size_gb}GB) from Hugging Face CDN...")
        print(f"   Repository: {self.HF_REPO_ID}")
        print(f"   Remote path: v{self.VERSION}/")
        print(f"   Destination: {self.cache_dir}/")
        print(f"\n   This may take 5-15 minutes depending on your internet speed...")
        print(f"   Please do NOT interrupt this process.")

        try:
            # Download the bundle with progress
            print(f"\n⏳ Starting download (hf_hub_download may show progress)...")
            bundle_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=remote_path,
                repo_type=self.HF_REPO_TYPE,
                cache_dir=str(self.cache_dir),
                force_download=force,
            )

            print(f"\n✅ Download complete!")
            print(f"   Location: {bundle_path}")

            # Verify checksum
            expected_sha256 = bundle_info.get("sha256")
            if expected_sha256:
                print(f"\n🔒 Verifying integrity (SHA256 checksum)...")
                if not self.verify_checksum(Path(bundle_path), expected_sha256):
                    raise ValueError(f"SHA256 checksum mismatch for {filename}")
                print(f"✅ Integrity verified - file is not corrupted")

            print(f"\n📦 Extracting binaries (this may take 1-2 minutes)...")
            print(f"   Source: {bundle_path}")
            print(f"   Target: {self.cache_dir}/extracted/")

            # Extract the bundle
            extract_dir = self.cache_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(bundle_path, "r:gz") as tar:
                self._safe_extractall(tar, extract_dir)

            print(f"✅ Extraction complete!")
            print(f"   Extracted to: {extract_dir}")
            print(f"   Contents: llama-dist/ nccl-dist/")

            # Create marker file
            marker_file = self.cache_dir / ".installed"
            marker_file.write_text(self.VERSION)

            print(f"\n✅ CUDA binaries ready!")
            print(f"   Binaries: {extract_dir}/llama-dist/bin/")
            print(f"   Libraries: {extract_dir}/llama-dist/lib/ + {extract_dir}/nccl-dist/lib/")
            return True

        except Exception as e:
            print(f"\n❌ Download/extraction failed!")
            print(f"   Error: {str(e)}")
            print(f"\n   Try again with:")
            print(f"   python -c \"from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary(force=True)\"")
            return False

    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify SHA256 checksum of downloaded file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        calculated = sha256_hash.hexdigest()
        return calculated.lower() == expected_sha256.lower()

    def _safe_extractall(self, tar: tarfile.TarFile, extract_dir: Path) -> None:
        """
        Safely extract tar archive, preventing path traversal attacks.

        Validates that all tar members extract within the target directory.

        Args:
            tar: TarFile object
            extract_dir: Target extraction directory

        Raises:
            ValueError: If any member would escape the target directory
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

    def get_binary_paths(self) -> dict:
        """Get paths to installed binaries."""
        if not self.is_installed():
            print("⚠️  Binaries not installed. Attempting to download...")
            if not self.download_binary():
                return {}

        llama_dist = self.cache_dir / "extracted" / "llama-dist"
        nccl_dist = self.cache_dir / "extracted" / "nccl-dist"

        return {
            "llama_bin": llama_dist / "bin" if llama_dist.exists() else None,
            "llama_lib": llama_dist / "lib" if llama_dist.exists() else None,
            "nccl_lib": nccl_dist / "lib" if nccl_dist.exists() else None,
            "nccl_include": nccl_dist / "include" if nccl_dist.exists() else None,
        }


def ensure_binaries_installed() -> dict:
    """
    Ensure binaries are installed and return their paths.

    This is called automatically on first import of the package.
    Downloads ~834MB CUDA binaries from Hugging Face on first use.

    Returns:
        Dictionary with paths to binary directories

    Raises:
        Exception: If download fails (not caught - let caller handle)
    """
    installer = BinaryInstaller()

    if not installer.is_installed():
        print("\n" + "="*70)
        print("🚀 FIRST-TIME SETUP: Installing local-llama-inference binaries")
        print("="*70)
        print("\nThis will download ~834MB of CUDA binaries from Hugging Face CDN.")
        print("Download location: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference")
        print("Installation path: ~/.local/share/local-llama-inference/")
        print("\nDownload may take 5-15 minutes depending on internet speed...")
        print("-"*70 + "\n")

        try:
            success = installer.download_binary()
            if not success:
                raise RuntimeError(
                    "Binary download failed. Check your internet connection and try again."
                )
            print("\n" + "="*70)
            print("✅ Binaries installed successfully!")
            print("="*70 + "\n")
        except Exception as e:
            print("\n" + "="*70)
            print("❌ Binary download failed!")
            print("="*70)
            print(f"\nError: {str(e)}")
            print("\nPlease try one of the following:")
            print("1. Check your internet connection and try again")
            print("2. Manually download from:")
            print(f"   https://huggingface.co/datasets/{installer.HF_REPO_ID}")
            print("3. Run: python -c \"from local_llama_inference import BinaryInstaller; BinaryInstaller().download_binary(force=True)\"")
            print()
            raise

    paths = installer.get_binary_paths()

    if not paths or not any(paths.values()):
        raise RuntimeError(
            "Binaries installed but paths could not be resolved. "
            "Please check ~/.local/share/local-llama-inference/extracted/ exists."
        )

    return paths


if __name__ == "__main__":
    """Allow running this script directly for testing."""
    installer = BinaryInstaller()

    if installer.download_binary(force=("--force" in sys.argv)):
        paths = installer.get_binary_paths()
        print("\n✅ Binaries installed successfully!")
        print(f"\nBinary paths:")
        for key, value in paths.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Binary installation failed!")
        sys.exit(1)
