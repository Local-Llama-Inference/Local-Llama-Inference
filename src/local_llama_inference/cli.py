"""
Command-line interface for local-llama-inference.

Provides utilities for managing the SDK, downloading binaries, and running inference.
"""

import argparse
import sys
from pathlib import Path

from local_llama_inference._bootstrap.installer import BinaryInstaller
from local_llama_inference._version import __version__


def cmd_install(args):
    """Install or update binary dependencies."""
    print(f"🔧 local-llama-inference {__version__}")
    print(f"   Installing/updating binaries...\n")

    installer = BinaryInstaller(cache_dir=args.cache_dir if args.cache_dir else None)

    if installer.download_binary(force=args.force):
        paths = installer.get_binary_paths()
        print("\n✅ Installation successful!")
        print(f"\n📁 Binary paths:")
        for key, value in paths.items():
            if value:
                print(f"   {key}: {value}")
    else:
        print("\n❌ Installation failed!")
        sys.exit(1)


def cmd_verify(args):
    """Verify that binaries are installed and accessible."""
    print(f"🔍 Verifying local-llama-inference {__version__}...\n")

    installer = BinaryInstaller(cache_dir=args.cache_dir if args.cache_dir else None)

    if installer.is_installed():
        print("✅ Binaries are installed")

        paths = installer.get_binary_paths()
        print(f"\n📁 Binary locations:")
        for key, value in paths.items():
            if value and value.exists():
                print(f"   ✅ {key}: {value}")
            elif value:
                print(f"   ⚠️  {key}: {value} (not found)")
            else:
                print(f"   ❌ {key}: Not configured")

        # Check if binaries are accessible
        llama_bin = paths.get("llama_bin")
        if llama_bin and llama_bin.exists():
            llama_cli = llama_bin / "llama-cli"
            if llama_cli.exists():
                print(f"\n✅ llama-cli is available")
            else:
                print(f"\n⚠️  llama-cli not found")
    else:
        print("❌ Binaries are not installed")
        print("\nRun: llama-inference install")
        sys.exit(1)


def cmd_info(args):
    """Show package information."""
    print(f"📦 local-llama-inference {__version__}")
    print(f"\n📍 GitHub: https://github.com/Local-Llama-Inference/Local-Llama-Inference")
    print(f"📍 Hugging Face: https://huggingface.co/datasets/waqasm86/Local-Llama-Inference/")
    print(f"📍 PyPI: https://pypi.org/project/local-llama-inference/")

    print(f"\n✨ Features:")
    print(f"   • GPU-accelerated inference with llama.cpp")
    print(f"   • Multi-GPU support via NVIDIA NCCL")
    print(f"   • OpenAI-compatible REST API (30+ endpoints)")
    print(f"   • Chat, completions, embeddings, reranking")
    print(f"   • Async support with Python asyncio")

    print(f"\n🚀 Quick Start:")
    print(f"   # Install binaries (first time)")
    print(f"   $ llama-inference install")
    print(f"\n   # Use in Python")
    print(f"   $ python")
    print(f"   >>> from local_llama_inference import LlamaServer")
    print(f"   >>> server = LlamaServer(model='model.gguf')")
    print(f"   >>> server.start()")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="llama-inference",
        description="GPU-accelerated LLM inference with NVIDIA NCCL support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install binaries on first use
  llama-inference install

  # Verify installation
  llama-inference verify

  # Show package information
  llama-inference info

  # Force reinstall binaries
  llama-inference install --force

For more information, visit:
  https://github.com/Local-Llama-Inference/Local-Llama-Inference
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"local-llama-inference {__version__}",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for binaries (default: ~/.local/share/local-llama-inference)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Install command
    install_parser = subparsers.add_parser("install", help="Install binary dependencies")
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall even if already installed",
    )
    install_parser.set_defaults(func=cmd_install)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify installation")
    verify_parser.set_defaults(func=cmd_verify)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show package information")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
