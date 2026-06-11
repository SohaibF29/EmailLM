#!/usr/bin/env python3
"""
Merge LoRA Adapter into Base Model
====================================

Loads the base Llama 3.1 8B Instruct model and a trained LoRA adapter,
merges the adapter weights into the base model, and saves the result.

Optionally pushes the merged model to HuggingFace Hub.

Usage:
    python training/merge_adapter.py --adapter_path outputs/final_adapter
    python training/merge_adapter.py --adapter_path outputs/final_adapter --push_to_hub --hub_model_id your-username/model-name
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def merge_and_save(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    hf_token: str | None = None,
) -> None:
    """Merge a LoRA adapter into the base model and save.

    Args:
        base_model_id: HuggingFace model ID for the base model.
        adapter_path:  Path to the trained LoRA adapter directory.
        output_dir:    Directory to save the merged model.
        push_to_hub:   If True, push the merged model to HuggingFace Hub.
        hub_model_id:  Repository ID on HuggingFace Hub (e.g. 'user/model').
        hf_token:      HuggingFace API token for pushing.
    """
    logger.info("=" * 60)
    logger.info("MERGE LoRA ADAPTER")
    logger.info("=" * 60)
    logger.info("Base model  : %s", base_model_id)
    logger.info("Adapter     : %s", adapter_path)
    logger.info("Output      : %s", output_dir)

    # Verify adapter path
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # 1. Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # 2. Load base model in float16 (for merging — no quantization needed)
    logger.info("Loading base model in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # 3. Load adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 4. Merge
    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # 5. Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged model to %s ...", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged model saved successfully!")

    # 6. Optionally push to Hub
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set")

        logger.info("Pushing to HuggingFace Hub: %s", hub_model_id)
        model.push_to_hub(hub_model_id, token=hf_token, safe_serialization=True)
        tokenizer.push_to_hub(hub_model_id, token=hf_token)
        logger.info("Pushed to Hub successfully!")

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    config = get_config()

    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=config.model.model_id,
        help="Base model HuggingFace ID",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_model",
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (e.g. 'username/model-name')",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hf_token = args.hf_token or __import__("os").environ.get("HF_TOKEN")

    merge_and_save(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hf_token=hf_token,
    )
