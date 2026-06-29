"""
tests/test_validation.py — unit tests for split_train_val
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ig_dm_cleaner.validation import split_train_val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(path, n):
    """Write n distinct JSONL lines to path."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": i}) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitTrainVal:
    def test_basic_split_counts(self, tmp_path):
        src = tmp_path / "train.jsonl"
        write_jsonl(src, 100)
        train_count, val_count = split_train_val(
            str(src), str(src), str(tmp_path / "val.jsonl"), val_fraction=0.1
        )
        assert train_count == 90
        assert val_count == 10
        assert train_count + val_count == 100

    def test_val_file_created(self, tmp_path):
        src = tmp_path / "train.jsonl"
        write_jsonl(src, 20)
        val = tmp_path / "val.jsonl"
        split_train_val(str(src), str(src), str(val))
        assert val.exists()

    def test_no_lines_lost(self, tmp_path):
        src = tmp_path / "train.jsonl"
        n = 200
        write_jsonl(src, n)
        val = tmp_path / "val.jsonl"
        train_n, val_n = split_train_val(str(src), str(src), str(val), val_fraction=0.2)
        assert train_n + val_n == n

    def test_output_is_valid_jsonl(self, tmp_path):
        src = tmp_path / "train.jsonl"
        write_jsonl(src, 40)
        val = tmp_path / "val.jsonl"
        split_train_val(str(src), str(src), str(val))
        for path in [src, val]:
            for line in path.read_text().strip().split("\n"):
                json.loads(line)  # must not raise

    def test_seed_produces_reproducible_split(self, tmp_path):
        src = tmp_path / "train.jsonl"
        write_jsonl(src, 100)
        val1 = tmp_path / "val1.jsonl"
        val2 = tmp_path / "val2.jsonl"
        train1 = tmp_path / "train1.jsonl"
        train2 = tmp_path / "train2.jsonl"

        split_train_val(str(src), str(train1), str(val1), seed=42)
        # Re-write source (it was overwritten by first call if same path — use copies)
        write_jsonl(src, 100)
        split_train_val(str(src), str(train2), str(val2), seed=42)

        assert val1.read_text() == val2.read_text()

    def test_different_seeds_produce_different_splits(self, tmp_path):
        src = tmp_path / "data.jsonl"
        write_jsonl(src, 100)
        val_a = tmp_path / "val_a.jsonl"
        split_train_val(str(src), str(src), str(val_a), seed=1)
        lines_a = val_a.read_text()

        write_jsonl(src, 100)
        val_b = tmp_path / "val_b.jsonl"
        split_train_val(str(src), str(src), str(val_b), seed=99)
        lines_b = val_b.read_text()

        assert lines_a != lines_b

    def test_invalid_val_fraction_raises(self, tmp_path):
        src = tmp_path / "train.jsonl"
        write_jsonl(src, 10)
        with pytest.raises(ValueError):
            split_train_val(str(src), str(src), str(tmp_path / "v.jsonl"), val_fraction=0)
        with pytest.raises(ValueError):
            split_train_val(str(src), str(src), str(tmp_path / "v.jsonl"), val_fraction=1.0)
        with pytest.raises(ValueError):
            split_train_val(str(src), str(src), str(tmp_path / "v.jsonl"), val_fraction=1.5)

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            split_train_val("/no/such/file.jsonl", str(tmp_path / "t.jsonl"), str(tmp_path / "v.jsonl"))

    def test_separate_train_and_source_paths(self, tmp_path):
        src = tmp_path / "source.jsonl"
        write_jsonl(src, 50)
        train_out = tmp_path / "train_out.jsonl"
        val_out = tmp_path / "val_out.jsonl"
        split_train_val(str(src), str(train_out), str(val_out), val_fraction=0.2)
        # Source file should be untouched
        assert src.exists()
        assert train_out.exists()
        assert val_out.exists()

    def test_at_least_one_val_line(self, tmp_path):
        """Even with a tiny dataset the val set gets at least 1 line."""
        src = tmp_path / "train.jsonl"
        write_jsonl(src, 5)
        val = tmp_path / "val.jsonl"
        _, val_count = split_train_val(str(src), str(src), str(val), val_fraction=0.05)
        assert val_count >= 1

    def test_blank_lines_in_source_ignored(self, tmp_path):
        src = tmp_path / "train.jsonl"
        with open(src, "w") as f:
            for i in range(10):
                f.write(json.dumps({"id": i}) + "\n")
                f.write("\n")  # blank lines between entries
        val = tmp_path / "val.jsonl"
        train_n, val_n = split_train_val(str(src), str(src), str(val), val_fraction=0.1)
        assert train_n + val_n == 10