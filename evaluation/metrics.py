"""
Evaluation Metrics for Email Generation
=========================================

Provides a unified MetricCalculator class with:
    - ROUGE (1, 2, L)
    - BLEU
    - BERTScore
    - Perplexity (from model loss)
    - Distinct-n (lexical diversity)
    - Email format compliance (greeting, body, closing)
    - Length statistics

Usage:
    from evaluation.metrics import MetricCalculator

    calc = MetricCalculator()
    results = calc.compute_all(predictions, references)
"""

import logging
import re
from collections import Counter
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns:
        Dict with keys rouge1, rouge2, rougeL (each is the F1 score).
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed. Skipping ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {key: float(np.mean(vals)) for key, vals in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus-level BLEU score.

    Returns:
        Dict with key 'bleu'.
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        logger.warning("nltk not installed. Skipping BLEU.")
        return {"bleu": 0.0}

    # Tokenize into word lists
    pred_tokens = [p.split() for p in predictions]
    ref_tokens = [[r.split()] for r in references]  # corpus_bleu expects list-of-list refs

    smoother = SmoothingFunction().method1
    try:
        score = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoother)
    except (ZeroDivisionError, ValueError):
        score = 0.0

    return {"bleu": float(score)}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, float]:
    """Compute BERTScore (precision, recall, F1).

    Args:
        model_type: The embedding model used for BERTScore.

    Returns:
        Dict with keys bertscore_precision, bertscore_recall, bertscore_f1.
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.warning("bert_score not installed. Skipping BERTScore.")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    try:
        P, R, F1 = bert_score(
            predictions, references,
            model_type=model_type,
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except Exception as exc:
        logger.warning("BERTScore failed: %s", exc)
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def compute_distinct_n(texts: list[str], max_n: int = 3) -> dict[str, float]:
    """Compute Distinct-n metric (lexical diversity).

    Distinct-n = number of unique n-grams / total n-grams.

    Returns:
        Dict with keys distinct_1, distinct_2, distinct_3.
    """
    results = {}
    for n in range(1, max_n + 1):
        all_ngrams = []
        for text in texts:
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if all_ngrams:
            results[f"distinct_{n}"] = len(set(all_ngrams)) / len(all_ngrams)
        else:
            results[f"distinct_{n}"] = 0.0

    return results


def compute_email_format_compliance(texts: list[str]) -> dict[str, float]:
    """Check email structural compliance (greeting, body, closing).

    Each email is scored 0–1 based on how many components are present:
        - Greeting (Dear/Hi/Hello/Hey etc.)        → 0.33
        - Body (non-trivial content, ≥ 20 chars)   → 0.34
        - Closing (Thanks/Regards/Best etc.)        → 0.33

    Returns:
        Dict with keys format_compliance (mean), has_greeting, has_body, has_closing (rates).
    """
    greeting_re = re.compile(
        r"^(dear\b|hi\b|hello\b|hey\b|good\s+(?:morning|afternoon|evening)\b|greetings\b|to whom)",
        re.IGNORECASE | re.MULTILINE,
    )
    closing_re = re.compile(
        r"(regards|sincerely|best|thanks|thank you|cheers|warm regards|kind regards|"
        r"respectfully|take care|all the best|cordially)\s*[,.]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    greetings, bodies, closings, scores = [], [], [], []

    for text in texts:
        has_greeting = bool(greeting_re.search(text))
        has_body = len(text.strip()) >= 20
        has_closing = bool(closing_re.search(text))

        greetings.append(has_greeting)
        bodies.append(has_body)
        closings.append(has_closing)

        score = (0.33 * has_greeting) + (0.34 * has_body) + (0.33 * has_closing)
        scores.append(score)

    return {
        "format_compliance": float(np.mean(scores)),
        "has_greeting_rate": float(np.mean(greetings)),
        "has_body_rate": float(np.mean(bodies)),
        "has_closing_rate": float(np.mean(closings)),
    }


def compute_length_stats(texts: list[str]) -> dict[str, float]:
    """Compute length statistics for generated texts.

    Returns:
        Dict with mean/std for word_count, sentence_count, char_count.
    """
    word_counts = [len(t.split()) for t in texts]
    sent_counts = [len(re.split(r"[.!?]+", t)) for t in texts]
    char_counts = [len(t) for t in texts]

    return {
        "word_count_mean": float(np.mean(word_counts)),
        "word_count_std": float(np.std(word_counts)),
        "sentence_count_mean": float(np.mean(sent_counts)),
        "sentence_count_std": float(np.std(sent_counts)),
        "char_count_mean": float(np.mean(char_counts)),
        "char_count_std": float(np.std(char_counts)),
    }


# ---------------------------------------------------------------------------
# MetricCalculator class
# ---------------------------------------------------------------------------


class MetricCalculator:
    """Unified metric calculator for email generation evaluation.

    Example:
        calc = MetricCalculator()
        results = calc.compute_all(predictions, references)
        print(results)
    """

    def __init__(self, bertscore_model: str = "microsoft/deberta-xlarge-mnli"):
        self.bertscore_model = bertscore_model

    def compute_all(
        self,
        predictions: list[str],
        references: list[str],
        skip_bertscore: bool = False,
    ) -> dict[str, float]:
        """Compute all metrics.

        Args:
            predictions: Model-generated emails.
            references:  Reference (ground-truth) emails.
            skip_bertscore: If True, skip BERTScore (saves time).

        Returns:
            Dict of all metric scores.
        """
        if not predictions or not references:
            logger.warning("Empty predictions or references — returning zeros.")
            return {}

        results = {}

        # N-gram overlap metrics
        logger.info("Computing ROUGE...")
        results.update(compute_rouge(predictions, references))

        logger.info("Computing BLEU...")
        results.update(compute_bleu(predictions, references))

        # Semantic similarity
        if not skip_bertscore:
            logger.info("Computing BERTScore...")
            results.update(compute_bertscore(predictions, references, self.bertscore_model))

        # Diversity
        logger.info("Computing Distinct-n...")
        results.update(compute_distinct_n(predictions))

        # Email structure
        logger.info("Computing format compliance...")
        results.update(compute_email_format_compliance(predictions))

        # Length stats
        logger.info("Computing length statistics...")
        pred_lengths = compute_length_stats(predictions)
        ref_lengths = compute_length_stats(references)

        results.update({f"pred_{k}": v for k, v in pred_lengths.items()})
        results.update({f"ref_{k}": v for k, v in ref_lengths.items()})

        return results

    def compute_single(
        self,
        prediction: str,
        reference: str,
        skip_bertscore: bool = True,
    ) -> dict[str, float]:
        """Compute metrics for a single prediction–reference pair."""
        return self.compute_all([prediction], [reference], skip_bertscore=skip_bertscore)


def aggregate_results(results_list: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-sample metric dicts into mean ± std.

    Args:
        results_list: List of dicts, one per sample.

    Returns:
        Dict mapping metric_name → {"mean": ..., "std": ...}.
    """
    if not results_list:
        return {}

    all_keys = results_list[0].keys()
    agg = {}
    for key in all_keys:
        values = [r[key] for r in results_list if key in r]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return agg
