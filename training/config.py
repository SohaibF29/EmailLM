"""
Training Configuration
======================

Centralized hyperparameters and configuration for QLoRA fine-tuning
of Llama 3.1 8B Instruct on Google Colab free tier (T4 GPU).

All settings are tuned for a T4 with 16 GB VRAM using 4-bit NF4 quantization.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_compute_dtype() -> torch.dtype:
    """Auto-detect the best compute dtype for the current GPU.

    - T4 / P100 → torch.float16  (no bfloat16 support)
    - A100 / L4  → torch.bfloat16
    - CPU        → torch.float32
    """
    if not torch.cuda.is_available():
        return torch.float32

    capability = torch.cuda.get_device_capability()
    # bfloat16 requires compute capability ≥ 8.0 (Ampere+)
    if capability[0] >= 8:
        return torch.bfloat16
    return torch.float16


def get_device_info() -> dict:
    """Return a summary dict of the current GPU."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "CPU", "vram_gb": 0}

    return {
        "device": "cuda",
        "name": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
        "capability": torch.cuda.get_device_capability(),
        "compute_dtype": str(detect_compute_dtype()),
    }


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Base model settings."""
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    trust_remote_code: bool = False
    use_auth_token: bool = True  # Llama 3.1 is a gated model


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation parameters."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit quantization settings."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    @property
    def bnb_4bit_compute_dtype(self) -> torch.dtype:
        return detect_compute_dtype()


@dataclass
class TrainingConfig:
    """HuggingFace TrainingArguments settings."""
    output_dir: str = "./outputs"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use num_train_epochs
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    logging_dir: str = "./logs"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "paged_adamw_32bit"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    report_to: str = "tensorboard"
    seed: int = 42
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False

    @property
    def fp16(self) -> bool:
        return detect_compute_dtype() == torch.float16

    @property
    def bf16(self) -> bool:
        return detect_compute_dtype() == torch.bfloat16


@dataclass
class DataConfig:
    """Dataset paths and processing settings."""
    train_file: str = "data/train.json"
    val_file: str = "data/val.json"
    max_seq_length: int = 512
    packing: bool = False


@dataclass
class PipelineConfig:
    """Top-level config that aggregates all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def summary(self) -> str:
        """Return a human-readable config summary."""
        device = get_device_info()
        lines = [
            "=" * 60,
            "TRAINING CONFIGURATION",
            "=" * 60,
            f"Model          : {self.model.model_id}",
            f"GPU            : {device.get('name', 'N/A')} ({device.get('vram_gb', 0)} GB)",
            f"Compute dtype  : {device.get('compute_dtype', 'N/A')}",
            f"Quantization   : {'4-bit NF4' if self.quantization.load_in_4bit else 'None'}",
            f"LoRA rank      : {self.lora.r}",
            f"LoRA alpha     : {self.lora.lora_alpha}",
            f"LoRA targets   : {', '.join(self.lora.target_modules)}",
            f"Batch size     : {self.training.per_device_train_batch_size}",
            f"Grad accum     : {self.training.gradient_accumulation_steps}",
            f"Effective batch: {self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps}",
            f"Learning rate  : {self.training.learning_rate}",
            f"Epochs         : {self.training.num_train_epochs}",
            f"Max seq length : {self.data.max_seq_length}",
            f"Optimizer      : {self.training.optim}",
            f"Grad checkpoint: {self.training.gradient_checkpointing}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_config() -> PipelineConfig:
    """Return the default pipeline configuration."""
    return PipelineConfig()


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
