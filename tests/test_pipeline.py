"""
tests/test_pipeline.py — integration tests for the Pipeline class
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ig_dm_cleaner.pipeline import Pipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inbox(tmp_path, conversations: dict):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for folder_name, messages in conversations.items():
        folder = inbox / folder_name
        folder.mkdir()
        (folder / "message_1.json").write_text(
            json.dumps({"messages": messages}), encoding="utf-8"
        )
    return str(inbox)


def raw_msg(sender, content, ts):
    return {"sender_name": sender, "content": content, "timestamp_ms": ts}


def default_config(tmp_path, **kwargs):
    inbox = make_inbox(tmp_path, {
        "alice_1": [
            raw_msg("Alice", "hello there", 1_000),
            raw_msg("Raghav", "hey!", 2_000),
        ]
    })
    return PipelineConfig(
        my_name="Raghav",
        inbox_dir=inbox,
        cleaned_dir=str(tmp_path / "cleaned"),
        output_file=str(tmp_path / "train.jsonl"),
        removed_log=str(tmp_path / "removed.jsonl"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig(my_name="Alice")
        assert c.inbox_dir == "./inbox"
        assert c.cleaned_dir == "./cleaned-text"
        assert c.output_file == "./train.jsonl"
        assert c.removed_log == "./removed_lines.jsonl"
        assert c.secrets_path is None

    def test_custom_values_stored(self):
        c = PipelineConfig(
            my_name="Bob",
            inbox_dir="/data/inbox",
            output_file="/out/train.jsonl",
        )
        assert c.my_name == "Bob"
        assert c.inbox_dir == "/data/inbox"


# ---------------------------------------------------------------------------
# Pipeline.run — happy path
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_run_returns_pipeline_result(self, tmp_path):
        config = default_config(tmp_path)
        result = Pipeline(config).run()
        assert result is not None

    def test_train_jsonl_created(self, tmp_path):
        config = default_config(tmp_path)
        Pipeline(config).run()
        assert os.path.exists(config.output_file)

    def test_removed_log_created(self, tmp_path):
        config = default_config(tmp_path)
        Pipeline(config).run()
        assert os.path.exists(config.removed_log)

    def test_cleaned_dir_created(self, tmp_path):
        config = default_config(tmp_path)
        Pipeline(config).run()
        assert os.path.isdir(config.cleaned_dir)

    def test_valid_samples_gt_zero(self, tmp_path):
        config = default_config(tmp_path)
        result = Pipeline(config).run()
        assert result.valid_samples > 0

    def test_conversations_processed_gt_zero(self, tmp_path):
        config = default_config(tmp_path)
        result = Pipeline(config).run()
        assert result.conversations_processed > 0

    def test_elapsed_seconds_positive(self, tmp_path):
        config = default_config(tmp_path)
        result = Pipeline(config).run()
        assert result.elapsed_seconds > 0

    def test_output_is_valid_jsonl(self, tmp_path):
        config = default_config(tmp_path)
        Pipeline(config).run()
        with open(config.output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    assert "messages" in obj

    def test_deduplication_removes_exact_dupes(self, tmp_path):
        # Two conversations with identical content
        inbox = make_inbox(tmp_path, {
            "conv_a": [
                raw_msg("Alice", "same question", 1_000),
                raw_msg("Raghav", "same answer", 2_000),
            ],
            "conv_b": [
                raw_msg("Alice", "same question", 1_000),
                raw_msg("Raghav", "same answer", 2_000),
            ],
        })
        config = PipelineConfig(
            my_name="Raghav",
            inbox_dir=inbox,
            cleaned_dir=str(tmp_path / "cleaned"),
            output_file=str(tmp_path / "train.jsonl"),
            removed_log=str(tmp_path / "removed.jsonl"),
        )
        result = Pipeline(config).run()
        assert result.valid_samples == 1
        assert result.duplicates_removed == 1


# ---------------------------------------------------------------------------
# Pipeline._validate_and_deduplicate edge cases
# ---------------------------------------------------------------------------

class TestValidateAndDeduplicate:
    """Tests the internal validation step directly via a minimal pipeline."""

    def _run_validation(self, tmp_path, lines):
        out = tmp_path / "train.jsonl"
        log = tmp_path / "removed.jsonl"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        config = PipelineConfig(my_name="X")  # won't use inbox/clean steps
        p = Pipeline(config)
        return p._validate_and_deduplicate(str(out), str(log))

    def test_valid_unique_lines_kept(self, tmp_path):
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]})
            for i in range(5)
        ]
        valid, dupes, malformed = self._run_validation(tmp_path, lines)
        assert valid == 5
        assert dupes == 0
        assert malformed == 0

    def test_duplicate_lines_counted(self, tmp_path):
        line = json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]})
        valid, dupes, malformed = self._run_validation(tmp_path, [line, line, line])
        assert valid == 1
        assert dupes == 2

    def test_malformed_json_counted(self, tmp_path):
        good = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        valid, dupes, malformed = self._run_validation(tmp_path, [good, "NOT JSON"])
        assert valid == 1
        assert malformed == 1

    def test_blank_lines_ignored(self, tmp_path):
        good = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        valid, dupes, malformed = self._run_validation(tmp_path, [good, "", "  "])
        assert valid == 1

    def test_missing_file_raises(self, tmp_path):
        config = PipelineConfig(my_name="X")
        p = Pipeline(config)
        with pytest.raises(FileNotFoundError):
            p._validate_and_deduplicate("/no/such/file.jsonl", str(tmp_path / "log.jsonl"))

    def test_removed_log_contains_duplicates(self, tmp_path):
        line = json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]})
        out = tmp_path / "train.jsonl"
        log = tmp_path / "removed.jsonl"
        out.write_text(line + "\n" + line + "\n", encoding="utf-8")
        Pipeline(PipelineConfig(my_name="X"))._validate_and_deduplicate(str(out), str(log))
        log_lines = [json.loads(l) for l in log.read_text().strip().split("\n") if l.strip()]
        assert any(entry["reason"] == "DUPLICATE" for entry in log_lines)

    def test_output_file_atomically_replaced(self, tmp_path):
        """The original file path should still exist after deduplication."""
        line = json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]})
        out = tmp_path / "train.jsonl"
        log = tmp_path / "removed.jsonl"
        out.write_text(line + "\n", encoding="utf-8")
        Pipeline(PipelineConfig(my_name="X"))._validate_and_deduplicate(str(out), str(log))
        assert out.exists()