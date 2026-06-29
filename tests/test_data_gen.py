"""
tests/test_data_gen.py — unit tests for data_gen (merge_blocks, create_samples, generate_dataset)
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ig_dm_cleaner.data_gen import merge_blocks, create_samples, generate_dataset, DEFAULT_TIME_GAP_LIMIT


# ---------------------------------------------------------------------------
# merge_blocks
# ---------------------------------------------------------------------------

class TestMergeBlocks:
    def _msg(self, sender, text, ts):
        return {"sender": sender, "text": text, "timestamp": ts}

    def test_consecutive_same_sender_merged(self):
        msgs = [self._msg("other", "hi", 1000), self._msg("other", "there", 2000)]
        result = merge_blocks(msgs)
        assert len(result) == 1
        assert result[0]["text"] == "hi\nthere"

    def test_timestamp_updates_to_latest(self):
        msgs = [self._msg("other", "a", 1000), self._msg("other", "b", 2000)]
        result = merge_blocks(msgs)
        assert result[0]["timestamp"] == 2000

    def test_different_senders_not_merged(self):
        msgs = [self._msg("other", "hi", 1000), self._msg("me", "hey", 2000)]
        result = merge_blocks(msgs)
        assert len(result) == 2

    def test_non_dict_items_skipped(self):
        msgs = [None, "bad", self._msg("other", "ok", 1000)]
        result = merge_blocks(msgs)
        assert len(result) == 1

    def test_missing_keys_skipped(self):
        msgs = [{"sender": "other"}, self._msg("other", "ok", 1000)]
        result = merge_blocks(msgs)
        assert len(result) == 1
        assert result[0]["text"] == "ok"

    def test_empty_text_skipped(self):
        msgs = [self._msg("other", "  ", 1000), self._msg("other", "ok", 2000)]
        result = merge_blocks(msgs)
        assert len(result) == 1

    def test_empty_input(self):
        assert merge_blocks([]) == []

    def test_alternating_senders_preserved(self):
        msgs = [
            self._msg("other", "a", 1000),
            self._msg("me", "b", 2000),
            self._msg("other", "c", 3000),
        ]
        result = merge_blocks(msgs)
        assert len(result) == 3
        assert [r["text"] for r in result] == ["a", "b", "c"]

    def test_text_coerced_to_string(self):
        msgs = [{"sender": "other", "text": 42, "timestamp": 1000}]
        result = merge_blocks(msgs)
        assert result[0]["text"] == "42"


# ---------------------------------------------------------------------------
# create_samples
# ---------------------------------------------------------------------------

class TestCreateSamples:
    def _block(self, sender, text, ts):
        return {"sender": sender, "text": text, "timestamp": ts}

    def test_other_then_me_creates_sample(self):
        blocks = [self._block("other", "hello", 1000), self._block("me", "hi", 2000)]
        result = create_samples(blocks)
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_me_then_other_skipped(self):
        blocks = [self._block("me", "hello", 1000), self._block("other", "hi", 2000)]
        assert create_samples(blocks) == []

    def test_same_sender_skipped(self):
        blocks = [self._block("other", "a", 1000), self._block("other", "b", 2000)]
        assert create_samples(blocks) == []

    def test_time_gap_too_large_skipped(self):
        blocks = [
            self._block("other", "hey", 1000),
            self._block("me", "hi", 1000 + DEFAULT_TIME_GAP_LIMIT + 1),
        ]
        assert create_samples(blocks) == []

    def test_time_gap_exactly_at_limit_kept(self):
        blocks = [
            self._block("other", "hey", 1000),
            self._block("me", "hi", 1000 + DEFAULT_TIME_GAP_LIMIT),
        ]
        assert len(create_samples(blocks)) == 1

    def test_single_char_user_text_skipped(self):
        blocks = [self._block("other", "x", 1000), self._block("me", "hello", 2000)]
        assert create_samples(blocks) == []

    def test_single_char_assistant_text_skipped(self):
        blocks = [self._block("other", "hello", 1000), self._block("me", "x", 2000)]
        assert create_samples(blocks) == []

    def test_invalid_timestamp_skipped(self):
        blocks = [
            self._block("other", "hello", "bad"),
            self._block("me", "hi", 2000),
        ]
        assert create_samples(blocks) == []

    def test_multiple_pairs_extracted(self):
        blocks = [
            self._block("other", "msg1", 1000),
            self._block("me", "resp1", 2000),
            self._block("other", "msg2", 3000),
            self._block("me", "resp2", 4000),
        ]
        result = create_samples(blocks)
        assert len(result) == 2

    def test_output_schema(self):
        blocks = [self._block("other", "hello", 1000), self._block("me", "hi", 2000)]
        sample = create_samples(blocks)[0]
        assert set(sample.keys()) == {"messages"}
        assert len(sample["messages"]) == 2
        assert sample["messages"][0] == {"role": "user", "content": "hello"}
        assert sample["messages"][1] == {"role": "assistant", "content": "hi"}

    def test_custom_time_gap_limit(self):
        blocks = [
            self._block("other", "hey", 1000),
            self._block("me", "hi", 6000),
        ]
        # With a limit of 4000ms the 5000ms gap should be rejected
        assert create_samples(blocks, time_gap_limit=4000) == []
        assert len(create_samples(blocks, time_gap_limit=6000)) == 1

    def test_empty_input(self):
        assert create_samples([]) == []

    def test_single_block_no_pairs(self):
        assert create_samples([self._block("other", "hi", 1000)]) == []


# ---------------------------------------------------------------------------
# generate_dataset (integration)
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def _write_cleaned(self, path, messages):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f)

    def _make_msg(self, sender, text, ts):
        return {"sender": sender, "text": text, "timestamp": ts}

    def test_produces_jsonl_file(self, tmp_path):
        conv = [
            self._make_msg("other", "hello", 1000),
            self._make_msg("me", "hi", 2000),
        ]
        self._write_cleaned(tmp_path / "conv1.json", conv)
        out = tmp_path / "train.jsonl"
        total = generate_dataset(str(tmp_path), str(out))
        assert total == 1
        assert out.exists()

    def test_output_is_valid_jsonl(self, tmp_path):
        conv = [self._make_msg("other", "hey", 1000), self._make_msg("me", "yo", 2000)]
        self._write_cleaned(tmp_path / "c.json", conv)
        out = tmp_path / "train.jsonl"
        generate_dataset(str(tmp_path), str(out))
        lines = out.read_text().strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj

    def test_multiple_files_combined(self, tmp_path):
        for i in range(3):
            conv = [
                self._make_msg("other", f"q{i}", 1000 * i + 1000),
                self._make_msg("me", f"a{i}", 1000 * i + 2000),
            ]
            self._write_cleaned(tmp_path / f"conv{i}.json", conv)
        out = tmp_path / "train.jsonl"
        total = generate_dataset(str(tmp_path), str(out))
        assert total == 3

    def test_clears_old_output_before_writing(self, tmp_path):
        old = tmp_path / "train.jsonl"
        old.write_text('{"stale": true}\n')
        conv = [self._make_msg("other", "hey", 1000), self._make_msg("me", "yo", 2000)]
        self._write_cleaned(tmp_path / "c.json", conv)
        generate_dataset(str(tmp_path), str(old))
        lines = old.read_text().strip().split("\n")
        assert all("stale" not in l for l in lines)

    def test_missing_input_folder_returns_zero(self, tmp_path):
        result = generate_dataset("/nonexistent/path", str(tmp_path / "out.jsonl"))
        assert result == 0

    def test_non_json_files_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        out = tmp_path / "train.jsonl"
        total = generate_dataset(str(tmp_path), str(out))
        assert total == 0

    def test_corrupt_json_file_skipped(self, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json")
        out = tmp_path / "train.jsonl"
        total = generate_dataset(str(tmp_path), str(out))
        assert total == 0

    def test_file_with_non_list_root_skipped(self, tmp_path):
        (tmp_path / "obj.json").write_text('{"sender": "other"}')
        out = tmp_path / "train.jsonl"
        assert generate_dataset(str(tmp_path), str(out)) == 0

    def test_items_missing_timestamp_dropped(self, tmp_path):
        conv = [
            {"sender": "other", "text": "hi"},  # no timestamp
            self._make_msg("me", "hey", 2000),
        ]
        self._write_cleaned(tmp_path / "c.json", conv)
        # No valid pair formed, so 0 samples
        assert generate_dataset(str(tmp_path), str(tmp_path / "out.jsonl")) == 0