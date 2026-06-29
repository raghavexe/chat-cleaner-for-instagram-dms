"""
data_gen.py — reads cleaned conversation JSON files and generates a JSONL
training dataset of (user, assistant) pairs suitable for LLM fine-tuning.
"""

import json
import os
import sys
from typing import List, Dict

# 12 hours in milliseconds — pairs with a larger gap are skipped
DEFAULT_TIME_GAP_LIMIT = 1_000 * 60 * 60 * 12


def merge_blocks(messages: List[Dict]) -> List[Dict]:
    """
    Merges consecutive messages from the same sender into a single block.

    Args:
        messages: Cleaned message dicts with ``sender``, ``text``, ``timestamp`` keys.

    Returns:
        List of merged blocks; each block's timestamp reflects the *last* message.
    """
    blocks: List[Dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        sender = msg.get("sender")
        text = msg.get("text")
        timestamp = msg.get("timestamp")
        if sender is None or text is None or timestamp is None:
            continue
        text_str = str(text).strip()
        if not text_str:
            continue

        if not blocks or blocks[-1]["sender"] != sender:
            blocks.append({"sender": sender, "text": text_str, "timestamp": timestamp})
        else:
            blocks[-1]["text"] += "\n" + text_str
            blocks[-1]["timestamp"] = timestamp

    return blocks


def create_samples(
    blocks: List[Dict],
    time_gap_limit: int = DEFAULT_TIME_GAP_LIMIT,
) -> List[Dict]:
    """
    Extracts ``(other → me)`` adjacent pairs as training samples.

    Args:
        blocks: Output of :func:`merge_blocks`.
        time_gap_limit: Maximum ms between the user message and your reply.
                        Pairs exceeding this are dropped.

    Returns:
        List of ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`` dicts.
    """
    samples = []
    for i in range(len(blocks) - 1):
        curr, nxt = blocks[i], blocks[i + 1]
        if not (curr["sender"] == "other" and nxt["sender"] == "me"):
            continue
        try:
            curr_ts, nxt_ts = int(curr["timestamp"]), int(nxt["timestamp"])
        except (ValueError, TypeError):
            continue
        if nxt_ts - curr_ts > time_gap_limit:
            continue
        user_text = curr["text"].strip()
        assistant_text = nxt["text"].strip()
        if len(user_text) < 2 or len(assistant_text) < 2:
            continue
        samples.append({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        })
    return samples


def process_file(
    filepath: str,
    time_gap_limit: int = DEFAULT_TIME_GAP_LIMIT,
) -> List[Dict]:
    """
    Reads a single cleaned JSON file and returns training samples from it.

    Args:
        filepath: Path to a cleaned conversation JSON file.
        time_gap_limit: Forwarded to :func:`create_samples`.

    Returns:
        List of training sample dicts (may be empty).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [Skip] Error reading {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return []

    if not isinstance(data, list):
        print(
            f"  [Skip] Expected a JSON list in {os.path.basename(filepath)}, "
            f"got {type(data).__name__}",
            file=sys.stderr,
        )
        return []

    valid_data = [m for m in data if isinstance(m, dict) and "timestamp" in m]
    dropped = len(data) - len(valid_data)
    if dropped:
        print(f"  [Warning] Dropped {dropped} items missing 'timestamp'.")

    try:
        valid_data.sort(key=lambda x: x["timestamp"])
    except TypeError as e:
        print(
            f"  [Skip] Sorting failed in {os.path.basename(filepath)}: {e}",
            file=sys.stderr,
        )
        return []

    return create_samples(merge_blocks(valid_data), time_gap_limit)


def generate_dataset(
    input_folder: str = "./cleaned-text",
    output_file: str = "./train.jsonl",
    time_gap_limit: int = DEFAULT_TIME_GAP_LIMIT,
) -> int:
    """
    Iterates all cleaned JSON files in ``input_folder`` and writes a JSONL
    training dataset to ``output_file``.

    Args:
        input_folder: Directory produced by :func:`~ig_dm_cleaner.processor.process_inbox`.
        output_file: Destination path for the ``.jsonl`` file.
        time_gap_limit: Maximum reply-gap in milliseconds (default 12 hours).

    Returns:
        Total number of training samples written.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' not found.", file=sys.stderr)
        return 0

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Clear any existing dataset
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except OSError as e:
            print(f"Critical Error: Could not remove old output file: {e}", file=sys.stderr)
            return 0

    total = 0
    try:
        files = sorted(os.listdir(input_folder))
    except OSError as e:
        print(f"Critical Error: Could not list files in {input_folder}: {e}", file=sys.stderr)
        return 0

    for filename in files:
        if not filename.endswith(".json"):
            continue
        samples = process_file(os.path.join(input_folder, filename), time_gap_limit)
        if not samples:
            continue
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"{filename}: {len(samples)} samples")
            total += len(samples)
        except OSError as e:
            print(f"  [Error] Write failed for {filename}: {e.strerror}", file=sys.stderr)

    print(f"\nDataset complete. Total samples: {total}")
    return total


# Backwards-compatible shim
def main(
    input_folder: str = "./cleaned-text",
    output_file: str = "./train.jsonl",
) -> None:
    generate_dataset(input_folder, output_file)