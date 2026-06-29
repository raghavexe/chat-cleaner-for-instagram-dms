"""
tests/test_processor.py — unit tests for process_inbox
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ig_dm_cleaner.processor import process_inbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inbox(tmp_path, conversations: dict):
    """
    Creates a fake Instagram inbox structure.

    conversations: { "folder_name": [ list of message dicts ] }
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for folder_name, messages in conversations.items():
        folder = inbox / folder_name
        folder.mkdir()
        payload = {"messages": messages}
        (folder / "message_1.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return str(inbox)


def raw_msg(sender, content, ts):
    return {"sender_name": sender, "content": content, "timestamp_ms": ts}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessInbox:
    def test_basic_happy_path(self, tmp_path):
        inbox = make_inbox(tmp_path, {
            "alice_123": [
                raw_msg("Alice", "hey", 1000),
                raw_msg("Raghav", "hi", 2000),
            ]
        })
        out = str(tmp_path / "cleaned")
        count = process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        assert count == 1
        files = os.listdir(out)
        assert "alice_123.json" in files

    def test_output_json_is_valid(self, tmp_path):
        inbox = make_inbox(tmp_path, {
            "conv_1": [raw_msg("Alice", "hello", 1000)]
        })
        out = str(tmp_path / "cleaned")
        process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        data = json.loads((tmp_path / "cleaned" / "conv_1.json").read_text())
        assert isinstance(data, list)

    def test_output_schema(self, tmp_path):
        inbox = make_inbox(tmp_path, {
            "conv_1": [raw_msg("Alice", "hello", 1000)]
        })
        out = str(tmp_path / "cleaned")
        process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        data = json.loads((tmp_path / "cleaned" / "conv_1.json").read_text())
        for item in data:
            assert set(item.keys()) == {"sender", "text", "timestamp"}

    def test_sender_normalization(self, tmp_path):
        inbox = make_inbox(tmp_path, {
            "conv_1": [
                raw_msg("Alice", "hi", 1000),
                raw_msg("Raghav", "hey", 2000),
            ]
        })
        out = str(tmp_path / "cleaned")
        process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        data = json.loads((tmp_path / "cleaned" / "conv_1.json").read_text())
        senders = {m["sender"] for m in data}
        assert senders == {"me", "other"}

    def test_missing_inbox_returns_zero(self, tmp_path):
        result = process_inbox("Raghav", inbox_dir="/does/not/exist", output_dir=str(tmp_path))
        assert result == 0

    def test_output_dir_created_if_missing(self, tmp_path):
        inbox = make_inbox(tmp_path, {"conv_1": [raw_msg("Alice", "hey", 1000)]})
        out = str(tmp_path / "deep" / "nested" / "output")
        process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        assert os.path.isdir(out)

    def test_multiple_message_files_merged(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        folder = inbox / "conv_1"
        folder.mkdir()
        (folder / "message_1.json").write_text(
            json.dumps({"messages": [raw_msg("Alice", "part one", 1000)]}),
            encoding="utf-8"
        )
        (folder / "message_2.json").write_text(
            json.dumps({"messages": [raw_msg("Alice", "part two", 2000)]}),
            encoding="utf-8"
        )
        out = str(tmp_path / "cleaned")
        process_inbox("Raghav", inbox_dir=str(inbox), output_dir=out)
        data = json.loads((tmp_path / "cleaned" / "conv_1.json").read_text())
        texts = [m["text"] for m in data]
        assert "part one" in texts
        assert "part two" in texts

    def test_corrupt_json_file_skipped_gracefully(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        folder = inbox / "conv_1"
        folder.mkdir()
        (folder / "message_1.json").write_text("INVALID JSON", encoding="utf-8")
        out = str(tmp_path / "cleaned")
        # Should not raise; folder with no valid messages is skipped
        count = process_inbox("Raghav", inbox_dir=str(inbox), output_dir=out)
        assert count == 0

    def test_empty_folder_skipped(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "empty_folder").mkdir()
        out = str(tmp_path / "cleaned")
        count = process_inbox("Raghav", inbox_dir=str(inbox), output_dir=out)
        assert count == 0

    def test_non_message_json_files_ignored(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        folder = inbox / "conv_1"
        folder.mkdir()
        # Only "message_*.json" files are picked up
        (folder / "other_data.json").write_text(
            json.dumps({"messages": [raw_msg("Alice", "hey", 1000)]}),
            encoding="utf-8"
        )
        out = str(tmp_path / "cleaned")
        count = process_inbox("Raghav", inbox_dir=str(inbox), output_dir=out)
        assert count == 0

    def test_noise_messages_filtered_out(self, tmp_path):
        inbox = make_inbox(tmp_path, {
            "conv_1": [
                raw_msg("Alice", "reacted to your message", 1000),
                raw_msg("Alice", "hello", 2000),
            ]
        })
        out = str(tmp_path / "cleaned")
        process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        data = json.loads((tmp_path / "cleaned" / "conv_1.json").read_text())
        assert len(data) == 1
        assert data[0]["text"] == "hello"

    def test_multiple_conversations_processed(self, tmp_path):
        inbox = make_inbox(tmp_path, {
            "alice_1": [raw_msg("Alice", "hi", 1000)],
            "bob_2": [raw_msg("Bob", "hey", 1000)],
            "carol_3": [raw_msg("Carol", "yo", 1000)],
        })
        out = str(tmp_path / "cleaned")
        count = process_inbox("Raghav", inbox_dir=inbox, output_dir=out)
        assert count == 3

    def test_secrets_path_forwarded(self, tmp_path):
        secrets = {"alice": "[NAME]"}
        secrets_file = tmp_path / "secrets.json"
        secrets_file.write_text(json.dumps(secrets), encoding="utf-8")
        inbox = make_inbox(tmp_path, {
            "conv_1": [raw_msg("Alice", "hi from alice", 1000)]
        })
        out = str(tmp_path / "cleaned")
        process_inbox("Raghav", inbox_dir=inbox, output_dir=out, secrets_path=str(secrets_file))
        data = json.loads((tmp_path / "cleaned" / "conv_1.json").read_text())
        assert any("[NAME]" in m["text"] for m in data)