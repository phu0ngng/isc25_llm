import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class CacheConfig:
    base_cache_dir: str = os.path.expanduser("~/hf_cache")
    use_auth: bool = True
    hf_token: Optional[str] = None  # Set this via environment variable

    def __post_init__(self):
        # Try to get token from environment variable
        self.hf_token = os.environ.get("HF_TOKEN", self.hf_token)


@dataclass
class PromptConfig:
    template: str = """Context: {context}
Question: {question}
Options:
{options}
Answer: """
    options_format: str = "Option {idx}: {choice}"
    answer_format: str = "{idx}"  # Just output the number of correct answer


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    frequency: int = 1  # Save every N epochs
    keep_last_k: int = 3  # Keep last K checkpoints
    filename_template: str = "best_model.pt"


@dataclass
class HardwareConfig:
    # Device types: 'cuda' (NVIDIA), 'xpu' (Intel), 'rocm' (AMD), 'cpu'
    device_type: str = None  # If None, will be auto-detected
    precision: str = "bf16"  # bf16, fp16, or fp32
    num_devices: int = None  # If None, will use all available devices

    def __post_init__(self):
        if self.device_type is None:
            if torch.cuda.is_available():
                self.device_type = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device_type = "xpu"
            elif hasattr(torch, "rocm") and torch.rocm.is_available():
                self.device_type = "rocm"
            else:
                self.device_type = "cpu"

        if self.num_devices is None:
            if self.device_type in ["cuda", "rocm"]:
                self.num_devices = torch.cuda.device_count()
            elif self.device_type == "xpu":
                self.num_devices = torch.xpu.device_count()
            elif self.device_type == "cpu":
                self.num_devices = 1


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )


@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    speed_dataset: str = "cosmos_qa"
    accuracy_dataset: str = "lmms-lab/ScienceQA"

    precision: str = "bf16"
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4
    max_length = 512
    max_steps = -1 # no explicit limit on the number of training steps

    checkpoint_freq: int = 1
    checkpoint_dir: str = "checkpoints"

    cache: CacheConfig = field(default_factory=CacheConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
