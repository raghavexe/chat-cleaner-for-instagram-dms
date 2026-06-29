"""
pipeline.py — high-level Pipeline class that orchestrates the full
clean → generate → validate/deduplicate workflow.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from .processor import process_inbox
from .data_gen import generate_dataset, DEFAULT_TIME_GAP_LIMIT

GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class PipelineConfig:
    """
    All tuneable knobs for a pipeline run.

    Attributes:
        my_name: Your display name inside the Instagram export.
        inbox_dir: Path to the Instagram ``inbox`` folder.
        cleaned_dir: Intermediate directory for cleaned JSON files.
        output_file: Final ``.jsonl`` dataset path.
        removed_log: Path where removed/duplicate lines are logged.
        secrets_path: Optional path to a ``secrets.json`` PII file.
        time_gap_limit: Max ms between a received message and your reply.
    """
    my_name: str
    inbox_dir: str = "./inbox"
    cleaned_dir: str = "./cleaned-text"
    output_file: str = "./train.jsonl"
    removed_log: str = "./removed_lines.jsonl"
    secrets_path: Optional[str] = None
    time_gap_limit: int = DEFAULT_TIME_GAP_LIMIT


@dataclass
class PipelineResult:
    """Summary stats returned after a completed pipeline run."""
    conversations_processed: int = 0
    total_samples: int = 0
    valid_samples: int = 0
    duplicates_removed: int = 0
    malformed_removed: int = 0
    elapsed_seconds: float = 0.0


class Pipeline:
    """
    Orchestrates the full Instagram DM → fine-tuning dataset pipeline.

    Example::

        from ig_dm_cleaner import Pipeline, PipelineConfig

        result = Pipeline(PipelineConfig(my_name="Jane")).run()
        print(f"Ready: {result.valid_samples} samples in train.jsonl")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Executes all three pipeline stages and returns summary stats."""
        c = self.config
        result = PipelineResult()
        start = time.time()

        self._banner("🚀 STARTING AI TRAINING DATA PREPARATION PIPELINE")

        # ── Step 1: Clean ──────────────────────────────────────────────
        self._step("Step 1/3", "Running text processor & cleaner…")
        conversations = process_inbox(
            my_name=c.my_name,
            inbox_dir=c.inbox_dir,
            output_dir=c.cleaned_dir,
            secrets_path=c.secrets_path,
        )
        result.conversations_processed = conversations
        self._divider()

        # ── Step 2: Generate ───────────────────────────────────────────
        self._step("Step 2/3", "Running dataset generator…")
        total = generate_dataset(
            input_folder=c.cleaned_dir,
            output_file=c.output_file,
            time_gap_limit=c.time_gap_limit,
        )
        result.total_samples = total
        self._divider()

        # ── Step 3: Validate & deduplicate ─────────────────────────────
        self._step("Step 3/3", "Deduplicating and validating JSONL dataset…")
        valid, dupes, malformed = self._validate_and_deduplicate(
            c.output_file, c.removed_log
        )
        result.valid_samples = valid
        result.duplicates_removed = dupes
        result.malformed_removed = malformed
        self._divider()

        result.elapsed_seconds = time.time() - start
        self._banner(f"🎉 PIPELINE COMPLETE IN {result.elapsed_seconds:.2f}s")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_deduplicate(
        self, file_path: str, removed_log_path: str
    ):
        """
        Reads the generated JSONL, deduplicates by conversation fingerprint,
        drops malformed lines, and overwrites the file atomically.

        Returns:
            Tuple of (valid_count, duplicate_count, malformed_count).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file '{file_path}' not found.")

        temp_path = file_path + ".tmp"
        seen: set = set()
        total = valid = dupes = malformed = 0

        with (
            open(file_path, "r", encoding="utf-8") as infile,
            open(temp_path, "w", encoding="utf-8") as outfile,
            open(removed_log_path, "w", encoding="utf-8") as logfile,
        ):
            for raw_line in infile:
                line = raw_line.strip()
                if not line:
                    continue
                total += 1

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    logfile.write(
                        json.dumps({"reason": "MALFORMED_JSON", "raw_line": line}) + "\n"
                    )
                    continue

                try:
                    fingerprint = "".join(
                        f"{m.get('role')}:{m.get('content')}||"
                        for m in data.get("messages", [])
                    )
                except (AttributeError, TypeError):
                    malformed += 1
                    logfile.write(
                        json.dumps({"reason": "STRUCTURAL_INVALID", "raw_line": line}) + "\n"
                    )
                    continue

                if fingerprint in seen:
                    dupes += 1
                    logfile.write(json.dumps({"reason": "DUPLICATE", "data": data}) + "\n")
                    continue

                seen.add(fingerprint)
                outfile.write(json.dumps(data) + "\n")
                valid += 1

        os.replace(temp_path, file_path)

        print(f"🔹 Rows processed : {total}")
        print(f"🔹 Clean rows kept: {valid}")
        print(f"⚠️  Duplicates     : {dupes}")
        print(f"⚠️  Malformed      : {malformed}")
        print(f"📂 Audit log      : {removed_log_path}")

        return valid, dupes, malformed

    @staticmethod
    def _banner(msg: str) -> None:
        print("\n" + "=" * 60)
        print(msg)
        print("=" * 60)

    @staticmethod
    def _step(label: str, msg: str) -> None:
        print(f"\n{BOLD}{GREEN} [{label}] {msg}{RESET}")

    @staticmethod
    def _divider() -> None:
        print("\n" + "-" * 50)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Instagram DM Cleaner Pipeline")
    parser.add_argument("--name", required=True, help="Your Instagram display name")
    parser.add_argument("--inbox", default="./inbox")
    parser.add_argument("--cleaned", default="./cleaned-text")
    parser.add_argument("--output", default="./train.jsonl")
    parser.add_argument("--removed-log", default="./removed_lines.jsonl")
    parser.add_argument("--time-gap", type=int, default=DEFAULT_TIME_GAP_LIMIT)

    args = parser.parse_args()

    config = PipelineConfig(
        my_name=args.name,
        inbox_dir=args.inbox,
        cleaned_dir=args.cleaned,
        output_file=args.output,
        removed_log=args.removed_log,
        time_gap_limit=args.time_gap,
    )

    result = Pipeline(config).run()

    print("\n📊 FINAL SUMMARY")
    print(result)