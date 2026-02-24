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

        Args:
            force: Force download even if already installed

        Returns:
            True if successful, False otherwise
        """
        if self.is_installed() and not force:
            return True

        bundle_info = self.get_platform_bundle()
        if bundle_info is None:
            print(f"⚠️  Binary bundle not available for your platform: "
                  f"{platform.system()} {platform.machine()}")
            return False

        if hf_hub_download is None:
            print("❌ huggingface-hub package required for binary download")
            print("   Install with: pip install huggingface-hub")
            return False

        try:
            return self._download_and_extract(bundle_info, force)
        except Exception as e:
            print(f"❌ Error downloading binaries: {str(e)}")
            return False

    def _download_and_extract(self, bundle_info: dict, force: bool = False) -> bool:
        """Download and extract binary bundle."""
        filename = bundle_info["filename"]
        remote_path = f"v{self.VERSION}/{filename}"

        print(f"📥 Downloading {filename} from Hugging Face...")
        print(f"   This may take a few minutes...")

        try:
            # Download the bundle
            bundle_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=remote_path,
                repo_type=self.HF_REPO_TYPE,
                cache_dir=str(self.cache_dir),
                force_download=force,
            )

            print(f"✅ Downloaded to: {bundle_path}")

            # Verify checksum
            expected_sha256 = bundle_info.get("sha256")
            if expected_sha256:
                print(f"🔒 Verifying checksum...")
                if not self.verify_checksum(Path(bundle_path), expected_sha256):
                    raise ValueError(f"SHA256 checksum mismatch for {filename}")
                print(f"✅ Checksum verified")

            print(f"📦 Extracting binaries...")

            # Extract the bundle
            extract_dir = self.cache_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(bundle_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)

            print(f"✅ Extracted to: {extract_dir}")

            # Create marker file
            marker_file = self.cache_dir / ".installed"
            marker_file.write_text(self.VERSION)

            print(f"✅ Binary installation complete!")
            return True

        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            return False

    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify SHA256 checksum of downloaded file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        calculated = sha256_hash.hexdigest()
        return calculated.lower() == expected_sha256.lower()

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

    Returns:
        Dictionary with paths to binary directories
    """
    installer = BinaryInstaller()

    if not installer.is_installed():
        print("🚀 First-time setup: Installing local-llama-inference binaries...")
        if not installer.download_binary():
            print("⚠️  Could not download binaries automatically.")
            print("   Please download manually from:")
            print(f"   https://huggingface.co/datasets/{installer.HF_REPO_ID}")
            return {}

    return installer.get_binary_paths()


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
