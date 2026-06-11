#!/usr/bin/env python3
"""
End-to-End Evaluation Runner
==============================

Loads a fine-tuned model (adapter or merged), generates emails for
validation prompts, and computes all metrics.

Supports comparison mode: base model vs fine-tuned model.

Usage:
    python evaluation/evaluate.py --model_path outputs/final_adapter
    python evaluation/evaluate.py --model_path outputs/final_adapter --compare_base
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import MetricCalculator, aggregate_results
from training.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_for_eval(
    model_path: str,
    base_model_id: str,
    is_adapter: bool = True,
    load_in_4bit: bool = True,
) -> tuple:
    """Load a model for evaluation.

    Args:
        model_path: Path to the adapter or merged model.
        base_model_id: Base model HuggingFace ID.
        is_adapter: If True, loads base model + adapter. If False, loads merged model.
        load_in_4bit: Use 4-bit quantization for memory efficiency.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info("Loading model from: %s", model_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_adapter else model_path,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    if is_adapter:
        # Load base + adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
    else:
        # Load merged model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()

    return model, tokenizer


def load_base_model_for_eval(
    base_model_id: str,
    load_in_4bit: bool = True,
) -> tuple:
    """Load the unmodified base model for comparison."""
    logger.info("Loading base model: %s", base_model_id)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_val_data(filepath: str, max_samples: int = 100) -> list[dict]:
    """Load validation samples from JSONL.

    Returns a list of dicts with 'messages' and 'task_type' fields.
    """
    samples = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    logger.info("Loaded %d validation samples from %s", len(samples), filepath)
    return samples


def extract_prompt_and_reference(sample: dict) -> tuple[list[dict], str]:
    """Extract the prompt messages and reference response from a sample.

    Returns:
        Tuple of (prompt_messages, reference_text).
        prompt_messages includes system and user turns.
        reference_text is the assistant's ground-truth response.
    """
    messages = sample["messages"]

    # Everything except the last assistant message is the prompt
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    reference = ""
    for m in messages:
        if m["role"] == "assistant":
            reference = m["content"]

    return prompt_messages, reference


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt_messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the model given prompt messages.

    Uses the tokenizer's chat template for proper formatting.
    """
    input_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    tokenizer,
    val_data: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    model_name: str = "model",
) -> dict:
    """Evaluate a model on the validation set.

    Returns:
        Dict with aggregated metrics, per-sample results, and generated texts.
    """
    logger.info("Evaluating %s on %d samples...", model_name, len(val_data))

    calc = MetricCalculator()
    predictions = []
    references = []
    per_sample_results = []

    start_time = time.time()

    for i, sample in enumerate(val_data):
        prompt_messages, reference = extract_prompt_and_reference(sample)

        # Generate
        prediction = generate_response(
            model, tokenizer, prompt_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        predictions.append(prediction)
        references.append(reference)

        # Per-sample metrics (skip BERTScore for speed)
        sample_metrics = calc.compute_single(prediction, reference, skip_bertscore=True)
        sample_metrics["task_type"] = sample.get("task_type", "unknown")
        sample_metrics["prompt"] = prompt_messages[-1]["content"][:200]
        sample_metrics["prediction"] = prediction[:500]
        sample_metrics["reference"] = reference[:500]
        per_sample_results.append(sample_metrics)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "  [%d/%d] %.1f samples/min",
                i + 1, len(val_data), (i + 1) / (elapsed / 60),
            )

    # Aggregate metrics (with BERTScore)
    logger.info("Computing aggregate metrics (including BERTScore)...")
    aggregate_metrics = calc.compute_all(predictions, references, skip_bertscore=False)

    elapsed = time.time() - start_time
    aggregate_metrics["eval_time_seconds"] = elapsed
    aggregate_metrics["samples_per_minute"] = len(val_data) / (elapsed / 60)

    return {
        "model_name": model_name,
        "num_samples": len(val_data),
        "metrics": aggregate_metrics,
        "per_sample": per_sample_results,
    }


def print_results_table(results: dict, title: str = "Evaluation Results") -> None:
    """Print a formatted results table."""
    metrics = results["metrics"]

    print(f"\n{'=' * 60}")
    print(f"  {title} — {results['model_name']}")
    print(f"  Samples: {results['num_samples']}")
    print(f"{'=' * 60}")

    # Group metrics
    groups = {
        "N-gram Overlap": ["rouge1", "rouge2", "rougeL", "bleu"],
        "Semantic Similarity": ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
        "Diversity": ["distinct_1", "distinct_2", "distinct_3"],
        "Email Format": ["format_compliance", "has_greeting_rate", "has_body_rate", "has_closing_rate"],
        "Length (Generated)": ["pred_word_count_mean", "pred_sentence_count_mean", "pred_char_count_mean"],
        "Length (Reference)": ["ref_word_count_mean", "ref_sentence_count_mean", "ref_char_count_mean"],
    }

    for group_name, keys in groups.items():
        print(f"\n  {group_name}:")
        for key in keys:
            if key in metrics:
                print(f"    {key:<30s} {metrics[key]:.4f}")

    if "eval_time_seconds" in metrics:
        print(f"\n  Time: {metrics['eval_time_seconds']:.1f}s ({metrics['samples_per_minute']:.1f} samples/min)")
    print(f"{'=' * 60}\n")


def print_comparison_table(base_results: dict, finetuned_results: dict) -> None:
    """Print a side-by-side comparison table."""
    base_m = base_results["metrics"]
    ft_m = finetuned_results["metrics"]

    print(f"\n{'=' * 72}")
    print(f"  COMPARISON: Base vs Fine-Tuned")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<30s} {'Base':>12s} {'Fine-Tuned':>12s} {'Δ':>10s}")
    print(f"  {'-' * 66}")

    compare_keys = [
        "rouge1", "rouge2", "rougeL", "bleu",
        "bertscore_f1",
        "distinct_1", "distinct_2",
        "format_compliance",
    ]

    for key in compare_keys:
        base_val = base_m.get(key, 0.0)
        ft_val = ft_m.get(key, 0.0)
        delta = ft_val - base_val
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
        print(f"  {key:<30s} {base_val:>12.4f} {ft_val:>12.4f} {delta_str:>10s}")

    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run the evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("EMAIL GENERATION EVALUATION")
    logger.info("=" * 60)

    # Load validation data
    val_data = load_val_data(args.val_data, args.max_samples)

    # Detect if model_path is an adapter directory
    adapter_config_path = Path(args.model_path) / "adapter_config.json"
    is_adapter = adapter_config_path.exists()
    logger.info("Model type: %s", "adapter" if is_adapter else "merged")

    # Evaluate fine-tuned model
    ft_model, ft_tokenizer = load_model_for_eval(
        args.model_path, args.base_model, is_adapter=is_adapter,
    )
    ft_results = evaluate_model(
        ft_model, ft_tokenizer, val_data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model_name="fine-tuned",
    )
    print_results_table(ft_results, "Fine-Tuned Model")

    # Free memory
    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Optionally evaluate base model
    base_results = None
    if args.compare_base:
        base_model, base_tokenizer = load_base_model_for_eval(args.base_model)
        base_results = evaluate_model(
            base_model, base_tokenizer, val_data,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            model_name="base",
        )
        print_results_table(base_results, "Base Model")
        print_comparison_table(base_results, ft_results)

        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    output = {
        "fine_tuned": ft_results,
        "base": base_results,
        "config": {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)

    logger.info("Results saved to: %s", args.output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    config = get_config()

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned email generation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the fine-tuned model (adapter or merged)",
    )
    parser.add_argument(
        "--base_model", type=str, default=config.model.model_id,
        help="Base model HuggingFace ID",
    )
    parser.add_argument(
        "--val_data", type=str, default="data/val.json",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation/results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max_samples", type=int, default=100,
        help="Maximum validation samples to evaluate",
    )
    parser.add_argument(
        "--compare_base", action="store_true",
        help="Also evaluate the base model for comparison",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
