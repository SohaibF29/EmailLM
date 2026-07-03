#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for Llama 3.1 8B Instruct
====================================================

Trains a LoRA adapter on the prepared Enron email dataset using
4-bit NF4 quantization (QLoRA) with HuggingFace TRL's SFTTrainer.

Designed for Google Colab free tier (T4 GPU, 16 GB VRAM).

Usage:
    python training/train_lora.py
    python training/train_lora.py --num_train_epochs 5 --learning_rate 1e-4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import PipelineConfig, get_config, get_device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset_from_jsonl(filepath: str) -> Dataset:
    """Load a JSONL dataset and return a HuggingFace Dataset.

    Each line is expected to have a 'messages' field with the chat format:
        [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d samples from %s", len(records), filepath)
    return Dataset.from_list(records)


def format_chat_template(
    example: dict,
    tokenizer: AutoTokenizer,
) -> dict:
    """Apply the model's chat template to the messages field.

    Returns the example with an added 'text' field containing the
    fully formatted conversation string.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def load_quantized_model(config: PipelineConfig):
    """Load the base model with 4-bit quantization."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=config.quantization.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )

    logger.info("Loading model: %s", config.model.model_id)
    logger.info("Quantization: 4-bit NF4, compute dtype: %s", config.quantization.bnb_4bit_compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
        token=config.model.use_auth_token,
    )

    # Prepare for k-bit training (freeze base, cast norms to float32)
    model = prepare_model_for_kbit_training(model)

    return model


def load_tokenizer(config: PipelineConfig) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        trust_remote_code=config.model.trust_remote_code,
        use_auth_token=config.model.use_auth_token,
    )
    # Llama models don't have a default pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def create_peft_config(config: PipelineConfig) -> LoraConfig:
    """Create the LoRA configuration."""
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def create_training_args(config: PipelineConfig) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from our config."""
    return TrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        logging_dir=config.training.logging_dir,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        optim=config.training.optim,
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs,
        report_to=config.training.report_to,
        seed=config.training.seed,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        remove_unused_columns=config.training.remove_unused_columns,
    )


def log_memory_usage() -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(
            "GPU Memory — Allocated: %.2f GB | Reserved: %.2f GB | Total: %.2f GB",
            allocated, reserved, total,
        )


def train(config: PipelineConfig) -> None:
    """Run the full QLoRA training pipeline."""
    # Print config summary
    logger.info("\n%s", config.summary())

    device_info = get_device_info()
    logger.info("Device: %s", device_info)

    # 1. Load tokenizer
    tokenizer = load_tokenizer(config)
    logger.info("Tokenizer loaded (vocab size: %d)", tokenizer.vocab_size)

    # 2. Load datasets
    train_dataset = load_dataset_from_jsonl(config.data.train_file)
    eval_dataset = load_dataset_from_jsonl(config.data.val_file)

    # 3. Format with chat template
    logger.info("Applying chat template to datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=eval_dataset.column_names,
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(eval_dataset))

    # 4. Load quantized model
    model = load_quantized_model(config)
    log_memory_usage()

    # 5. Setup LoRA
    peft_config = create_peft_config(config)
    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    log_memory_usage()

    # 6. Training arguments
    training_args = create_training_args(config)

    # 7. Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.data.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=config.data.packing,
    )

    # 8. Train
    logger.info("Starting training...")
    log_memory_usage()

    train_result = trainer.train()

    # 9. Log results
    logger.info("Training complete!")
    logger.info("Training loss: %.4f", train_result.training_loss)
    log_memory_usage()

    # 10. Save the adapter
    adapter_path = os.path.join(config.training.output_dir, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved to: %s", adapter_path)

    # 11. Save training metrics
    metrics = train_result.metrics
    metrics_path = os.path.join(config.training.output_dir, "train_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Training metrics saved to: %s", metrics_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Adapter: %s", adapter_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("TensorBoard: tensorboard --logdir %s", config.training.logging_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training overrides."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Llama 3.1 8B Instruct",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_file", type=str, help="Training data JSONL")
    parser.add_argument("--val_file", type=str, help="Validation data JSONL")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument("--seed", type=int, help="Random seed")
    return parser.parse_args()


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Apply CLI argument overrides to the config."""
    if args.train_file:
        config.data.train_file = args.train_file
    if args.val_file:
        config.data.val_file = args.val_file
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.num_train_epochs:
        config.training.num_train_epochs = args.num_train_epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.per_device_train_batch_size:
        config.training.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_seq_length:
        config.data.max_seq_length = args.max_seq_length
    if args.lora_r:
        config.lora.r = args.lora_r
    if args.lora_alpha:
        config.lora.lora_alpha = args.lora_alpha
    if args.lora_dropout:
        config.lora.lora_dropout = args.lora_dropout
    if args.seed:
        config.training.seed = args.seed
    return config


if __name__ == "__main__":
    args = parse_args()
    config = get_config()
    config = apply_cli_overrides(config, args)
    train(config)
