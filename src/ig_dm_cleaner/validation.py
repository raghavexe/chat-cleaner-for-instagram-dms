"""
validation.py — splits a JSONL dataset into train and validation subsets.
"""

import os
import random
from typing import Tuple, Optional


def split_train_val(
    source_path: str = "./train.jsonl",
    train_path: str = "./train.jsonl",
    val_path: str = "./val.jsonl",
    val_fraction: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Shuffles a JSONL file and splits it into training and validation sets.

    The source file is overwritten with the training portion (``1 - val_fraction``)
    unless ``train_path`` differs from ``source_path``.

    Args:
        source_path: Path to the full ``.jsonl`` file to split.
        train_path: Destination for the training split (defaults to ``source_path``).
        val_path: Destination for the validation split.
        val_fraction: Fraction of lines to use for validation (default 0.05 = 5%).
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of ``(train_count, val_count)``.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        ValueError: If ``val_fraction`` is not in the range ``(0, 1)``.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file '{source_path}' not found.")

    with open(source_path, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]

    if seed is not None:
        random.seed(seed)
    random.shuffle(lines)

    split_idx = max(1, int(len(lines) * val_fraction))
    val_lines = lines[:split_idx]
    train_lines = lines[split_idx:]

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    print(f"Split complete → train: {len(train_lines)}, val: {len(val_lines)}")
    return len(train_lines), len(val_lines)