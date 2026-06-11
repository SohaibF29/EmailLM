#!/usr/bin/env python3
"""
Enron Email Dataset Preparation Pipeline
==========================================

Parses the raw Enron maildir corpus, cleans and formats emails into
a multi-task fine-tuning dataset for Llama 3.1 8B Instruct.

Tasks:
    - compose: Generate a professional email from a subject line
    - reply:   Generate a professional reply to a received email

Output format (JSON lines, each entry):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "task_type": "compose" | "reply"
    }

Usage:
    python data/prepare_data.py --maildir_path data/raw/maildir
"""

import argparse
import email
import hashlib
import json
import logging
import os
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENT_FOLDER_NAMES = {"sent", "sent_items", "_sent_mail", "sent_mail"}
INBOX_FOLDER_NAMES = {"inbox", "notes_inbox"}

SYSTEM_PROMPT_COMPOSE = (
    "You are a professional email writing assistant. "
    "Write clear, concise, and professional emails based on the given subject and context."
)

SYSTEM_PROMPT_REPLY = (
    "You are a professional email reply assistant. "
    "Write thoughtful and professional replies to the given email."
)

# Patterns to strip from email bodies
FORWARD_PATTERNS = [
    r"-{3,}\s*Original Message\s*-{3,}",
    r"-{3,}\s*Forwarded by\s.*?-{3,}",
    r"On .+? wrote:",
]

DISCLAIMER_PATTERNS = [
    r"(?:This message|This e-?mail|This communication).*?(?:confidential|privileged).*",
    r"ENRON DISCLAIMER.*",
    r"\*{3,}.*?ECONNECTION.*",
]

GREETING_PATTERNS = re.compile(
    r"^(dear\b|hi\b|hello\b|hey\b|good\s+(?:morning|afternoon|evening)\b|greetings\b|to whom)",
    re.IGNORECASE,
)

CLOSING_PATTERNS = re.compile(
    r"(regards|sincerely|best|thanks|thank you|cheers|warm regards|kind regards|"
    r"respectfully|take care|all the best|cordially)",
    re.IGNORECASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------


def parse_email_file(filepath: str) -> Optional[dict]:
    """Parse a single raw Enron email file into a structured dict.

    Uses a fast custom text parser instead of email.message_from_file for speed.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()

        # Split headers from body at the first blank line
        parts = content.split("\n\n", 1)
        if len(parts) < 2:
            parts = content.split("\r\n\r\n", 1)
            if len(parts) < 2:
                return None

        header_text, body = parts

        # Parse headers line by line, handling folded headers (continuation lines)
        headers = {}
        current_key = None
        for line in header_text.splitlines():
            if not line.strip():
                continue
            if (line[0] == " " or line[0] == "\t") and current_key:
                headers[current_key] += " " + line.strip()
            else:
                h_parts = line.split(":", 1)
                if len(h_parts) == 2:
                    current_key = h_parts[0].strip()
                    headers[current_key] = h_parts[1].strip()

        return {
            "message_id": headers.get("Message-ID", ""),
            "date": headers.get("Date", ""),
            "from_addr": headers.get("From", ""),
            "to_addr": headers.get("To", ""),
            "cc": headers.get("Cc", ""),
            "subject": headers.get("Subject", ""),
            "body": body.strip(),
            "x_folder": headers.get("X-Folder", ""),
        }
    except Exception as exc:
        logger.debug("Failed to parse %s: %s", filepath, exc)
        return None


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_body(text: str) -> str:
    """Remove forwarded chains, disclaimers, excess whitespace."""
    # Strip forwarded / quoted blocks
    for pattern in FORWARD_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[: match.start()]

    # Remove lines that are purely quoted (> prefix)
    lines = text.split("\n")
    lines = [ln for ln in lines if not ln.strip().startswith(">")]
    text = "\n".join(lines)

    # Remove disclaimers
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    return text


def is_valid_email(
    record: dict,
    min_body_len: int = 50,
    max_body_len: int = 2000,
) -> bool:
    """Check whether a parsed email passes quality filters."""
    body = record.get("body", "")
    subject = record.get("subject", "")

    if not body or len(body) < min_body_len:
        return False
    if len(body) > max_body_len:
        return False
    if not subject or subject.strip() == "":
        return False
    # Skip calendar / automated messages
    if any(kw in subject.lower() for kw in ["out of office", "undeliverable", "delivery failure"]):
        return False
    return True


def body_hash(text: str) -> str:
    """MD5 hash for deduplication."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Maildir walking
# ---------------------------------------------------------------------------


def is_sent_folder(folder_name: str) -> bool:
    """Return True if *folder_name* is a sent-mail folder."""
    return folder_name.lower().strip("_").replace(" ", "_") in SENT_FOLDER_NAMES


def is_inbox_folder(folder_name: str) -> bool:
    """Return True if *folder_name* is an inbox folder."""
    return folder_name.lower().strip("_").replace(" ", "_") in INBOX_FOLDER_NAMES


def walk_maildir(
    maildir_path: str,
    min_body_length: int = 50,
    max_body_length: int = 2000,
) -> tuple[list[dict], list[dict]]:
    """Walk the Enron maildir and return (sent_emails, inbox_emails).

    Both lists contain parsed & cleaned email dicts.
    """
    maildir = Path(maildir_path)
    if not maildir.exists():
        raise FileNotFoundError(f"maildir not found: {maildir}")

    sent_emails: list[dict] = []
    inbox_emails: list[dict] = []
    seen_hashes: set[str] = set()

    user_dirs = sorted([d for d in maildir.iterdir() if d.is_dir()])
    logger.info("Found %d user directories", len(user_dirs))

    for idx, user_dir in enumerate(user_dirs, 1):
        # Scan immediate subfolders of this user
        try:
            subfolders = [d for d in user_dir.iterdir() if d.is_dir()]
        except OSError:
            continue

        for sub_dir in subfolders:
            folder_name = sub_dir.name
            is_sent = is_sent_folder(folder_name)
            is_inbox = is_inbox_folder(folder_name)

            # Skip folders that are neither sent nor inbox (e.g. all_documents, discussion_threads)
            if not is_sent and not is_inbox:
                continue

            for root, _, files in os.walk(sub_dir):
                for file in files:
                    filepath = os.path.join(root, file)

                    # Fast size filter: skip files larger than max_body_length + 5000 bytes (definitely too large)
                    try:
                        if os.path.getsize(filepath) > (max_body_length + 5000):
                            continue
                    except OSError:
                        continue

                    record = parse_email_file(filepath)
                    if record is None:
                        continue

                    record["body"] = clean_body(record["body"])
                    # Apply quality filters early to keep count accurate and reduce memory footprint
                    if not is_valid_email(record, min_body_length, max_body_length):
                        continue

                    bh = body_hash(record["body"])
                    if bh in seen_hashes:
                        continue
                    seen_hashes.add(bh)

                    if is_sent:
                        sent_emails.append(record)
                    else:
                        inbox_emails.append(record)

        if idx % 10 == 0 or idx == len(user_dirs):
            logger.info(
                "Progress: %d/%d users | Parsed %d sent, %d inbox emails",
                idx,
                len(user_dirs),
                len(sent_emails),
                len(inbox_emails),
            )

    logger.info(
        "Parsed %d sent emails, %d inbox/other emails (after quality filter & dedup)",
        len(sent_emails),
        len(inbox_emails),
    )
    return sent_emails, inbox_emails


# ---------------------------------------------------------------------------
# Multi-task formatting
# ---------------------------------------------------------------------------


def format_compose_sample(record: dict) -> Optional[dict]:
    """Format a sent email as a composition task.

    Prompt:  Write an email with subject: <subject>
    Target:  <email body>
    """
    subject = record["subject"].strip()
    body = record["body"].strip()

    if not subject or not body:
        return None

    # Remove RE:/FW: prefixes for composition task
    clean_subject = re.sub(r"^(?:Re|Fw|Fwd)\s*:\s*", "", subject, flags=re.IGNORECASE).strip()
    if not clean_subject:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPOSE},
            {"role": "user", "content": f"Write a professional email with the subject: {clean_subject}"},
            {"role": "assistant", "content": body},
        ],
        "task_type": "compose",
    }


def build_reply_pairs(
    sent_emails: list[dict],
    inbox_emails: list[dict],
) -> list[dict]:
    """Match replies (sent) to originals (inbox) by subject threading.

    A sent email whose subject starts with 'Re:' is matched to an inbox
    email with the same base subject.  The pair becomes a reply task.
    """
    # Index inbox emails by normalised subject
    def normalise_subject(subj: str) -> str:
        subj = re.sub(r"^(?:Re|Fw|Fwd)\s*:\s*", "", subj, flags=re.IGNORECASE)
        return subj.strip().lower()

    inbox_by_subject: dict[str, list[dict]] = defaultdict(list)
    for rec in inbox_emails:
        ns = normalise_subject(rec["subject"])
        if ns:
            inbox_by_subject[ns].append(rec)

    pairs: list[dict] = []
    for sent in sent_emails:
        subj = sent["subject"].strip()
        if not re.match(r"^Re\s*:", subj, re.IGNORECASE):
            continue

        ns = normalise_subject(subj)
        candidates = inbox_by_subject.get(ns, [])
        if not candidates:
            continue

        # Pick the first matching inbox email
        original = candidates[0]
        original_body = original["body"].strip()
        reply_body = sent["body"].strip()

        if not original_body or not reply_body:
            continue

        from_addr = original.get("from_addr", "Unknown")
        orig_subject = original["subject"].strip()

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_REPLY},
                {
                    "role": "user",
                    "content": (
                        f"Reply to this email:\n\n"
                        f"From: {from_addr}\n"
                        f"Subject: {orig_subject}\n\n"
                        f"{original_body}"
                    ),
                },
                {"role": "assistant", "content": reply_body},
            ],
            "task_type": "reply",
        })

    logger.info("Built %d reply pairs via subject threading", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Split & save
# ---------------------------------------------------------------------------


def stratified_split(
    samples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train/val, stratified by task_type."""
    rng = random.Random(seed)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_task[s["task_type"]].append(s)

    train, val = [], []
    for task_type, items in by_task.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_jsonl(records: list[dict], filepath: str) -> None:
    """Save records as JSON lines."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), filepath)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("Enron Email Data Preparation Pipeline")
    logger.info("=" * 60)

    # 1. Walk and parse emails
    sent_emails, inbox_emails = walk_maildir(
        args.maildir_path,
        min_body_length=args.min_body_length,
        max_body_length=args.max_body_length,
    )

    # 2. Filter sent emails
    sent_emails = [e for e in sent_emails if is_valid_email(e, args.min_body_length, args.max_body_length)]
    inbox_emails = [e for e in inbox_emails if is_valid_email(e, args.min_body_length, args.max_body_length)]
    logger.info("After quality filtering: %d sent, %d inbox", len(sent_emails), len(inbox_emails))

    # 3. Build composition samples
    compose_samples = []
    for rec in sent_emails:
        sample = format_compose_sample(rec)
        if sample:
            compose_samples.append(sample)
    logger.info("Compose samples: %d", len(compose_samples))

    # 4. Build reply pairs
    reply_samples = build_reply_pairs(sent_emails, inbox_emails)
    logger.info("Reply samples: %d", len(reply_samples))

    # 5. Combine & cap
    all_samples = compose_samples + reply_samples
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    if args.max_samples and len(all_samples) > args.max_samples:
        # Maintain task ratio when capping
        n_compose = int(args.max_samples * len(compose_samples) / len(all_samples))
        n_reply = args.max_samples - n_compose

        rng.shuffle(compose_samples)
        rng.shuffle(reply_samples)

        all_samples = compose_samples[:n_compose] + reply_samples[:n_reply]
        rng.shuffle(all_samples)
        logger.info("Capped to %d samples (compose=%d, reply=%d)", len(all_samples), n_compose, n_reply)

    # 6. Save raw.json (all samples)
    raw_path = os.path.join(args.output_dir, "raw.json")
    save_jsonl(all_samples, raw_path)

    # 7. Split
    train, val = stratified_split(all_samples, args.val_ratio, args.seed)
    save_jsonl(train, os.path.join(args.output_dir, "train.json"))
    save_jsonl(val, os.path.join(args.output_dir, "val.json"))

    # 8. Summary
    compose_train = sum(1 for s in train if s["task_type"] == "compose")
    reply_train = sum(1 for s in train if s["task_type"] == "reply")
    compose_val = sum(1 for s in val if s["task_type"] == "compose")
    reply_val = sum(1 for s in val if s["task_type"] == "reply")

    logger.info("-" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("-" * 60)
    logger.info("Total samples : %d", len(all_samples))
    logger.info("  Compose     : %d", sum(1 for s in all_samples if s["task_type"] == "compose"))
    logger.info("  Reply       : %d", sum(1 for s in all_samples if s["task_type"] == "reply"))
    logger.info("Train split   : %d (compose=%d, reply=%d)", len(train), compose_train, reply_train)
    logger.info("Val split     : %d (compose=%d, reply=%d)", len(val), compose_val, reply_val)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Enron email dataset for Llama 3.1 fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--maildir_path",
        type=str,
        default="data/raw/maildir",
        help="Path to the extracted Enron maildir directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Directory to save output JSON files",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum total samples (0 = no limit)",
    )
    parser.add_argument(
        "--min_body_length",
        type=int,
        default=50,
        help="Minimum email body length in characters",
    )
    parser.add_argument(
        "--max_body_length",
        type=int,
        default=2000,
        help="Maximum email body length in characters",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
