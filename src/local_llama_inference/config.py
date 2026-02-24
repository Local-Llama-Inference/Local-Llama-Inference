"""Configuration classes for llama-server and inference."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class SamplingConfig:
    """Sampling parameters for text generation."""

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    mirostat: int = 0  # 0=off, 1=v1, 2=v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    seed: int = -1  # -1 = random
    grammar: str = ""
    json_schema: dict = field(default_factory=dict)


@dataclass
class ServerConfig:
    """Configuration for llama-server process."""

    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = 99  # -ngl: 99 = all layers
    main_gpu: int = 0  # -mg: which GPU for main compute
    tensor_split: List[float] = field(default_factory=list)  # -ts: proportions per GPU
    split_mode: str = "layer"  # -sm: none/layer/row
    ctx_size: int = 4096  # -c: context window
    n_parallel: int = 1  # -np: parallel request slots
    batch_size: int = 512  # -b: batch size
    ubatch_size: int = 512  # -ub: micro batch size
    flash_attn: str = "auto"  # -fa: auto/on/off
    rope_scaling: str = "linear"  # RoPE frequency scaling
    rope_freq_base: float = 0.0  # 0 = from model
    rope_freq_scale: float = 1.0
    kv_cache_type: str = "f16"  # f32/f16/bf16/q8_0/q4_0
    api_key: Optional[str] = None  # --api-key
    n_threads: int = -1  # -t: CPU threads (-1 = auto)
    verbose: bool = False
    extra_args: List[str] = field(default_factory=list)

    def to_args(self) -> List[str]:
        """Convert configuration to llama-server command-line arguments."""
        args = ["-m", self.model_path]

        if self.host:
            args.extend(["--host", self.host])
        if self.port:
            args.extend(["--port", str(self.port)])

        args.extend(["-ngl", str(self.n_gpu_layers)])

        if self.tensor_split:
            tensor_split_str = ",".join(str(x) for x in self.tensor_split)
            args.extend(["--tensor-split", tensor_split_str])

        if self.split_mode != "layer":
            args.extend(["--split-mode", self.split_mode])

        args.extend(["--main-gpu", str(self.main_gpu)])
        args.extend(["-c", str(self.ctx_size)])
        args.extend(["-np", str(self.n_parallel)])
        args.extend(["-b", str(self.batch_size)])
        args.extend(["-ub", str(self.ubatch_size)])

        if self.flash_attn != "auto":
            args.extend(["--flash-attn", self.flash_attn])

        if self.rope_scaling:
            args.extend(["--rope-scaling", self.rope_scaling])

        if self.rope_freq_base > 0:
            args.extend(["--rope-freq-base", str(self.rope_freq_base)])

        if self.rope_freq_scale != 1.0:
            args.extend(["--rope-freq-scale", str(self.rope_freq_scale)])

        if self.n_threads > 0:
            args.extend(["-t", str(self.n_threads)])

        if self.api_key:
            args.extend(["--api-key", self.api_key])

        if self.verbose:
            args.append("-v")

        args.extend(self.extra_args)

        return args

    def to_url(self) -> str:
        """Get base URL for the server."""
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    model_path: Path
    name: str = ""
    quantization: str = ""  # Q4_K_M, Q5_K_M, etc.
    context_length: int = 4096
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
