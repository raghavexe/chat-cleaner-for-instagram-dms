"""
tests/test_cleaner.py — unit tests for ChatCleaner
"""

import json
import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ig_dm_cleaner.cleaner import ChatCleaner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cleaner():
    return ChatCleaner("Raghav")


@pytest.fixture
def cleaner_with_secrets(tmp_path):
    secrets = {"john doe": "[NAME]", "mumbai": "[CITY]"}
    secrets_file = tmp_path / "secrets.json"
    secrets_file.write_text(json.dumps(secrets), encoding="utf-8")
    return ChatCleaner("Raghav", secrets_path=str(secrets_file))


# ---------------------------------------------------------------------------
# __init__ / construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_valid_construction(self):
        c = ChatCleaner("Alice")
        assert c.my_name == "Alice"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ChatCleaner("")

    def test_whitespace_only_name_raises(self):
        with pytest.raises(ValueError):
            ChatCleaner("   ")

    def test_missing_secrets_file_is_graceful(self):
        c = ChatCleaner("Alice", secrets_path="/nonexistent/secrets.json")
        assert c.secret_replacements == {}

    def test_malformed_secrets_file_is_graceful(self, tmp_path):
        bad = tmp_path / "secrets.json"
        bad.write_text("not json at all", encoding="utf-8")
        c = ChatCleaner("Alice", secrets_path=str(bad))
        assert c.secret_replacements == {}

    def test_secrets_loaded_correctly(self, tmp_path):
        data = {"Alice": "[NAME]", "London": "[CITY]"}
        f = tmp_path / "secrets.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        c = ChatCleaner("Bob", secrets_path=str(f))
        # Keys are lowercased on load
        assert "alice" in c.secret_replacements
        assert "london" in c.secret_replacements

    def test_extra_noise_extends_defaults(self):
        c = ChatCleaner("Alice", extra_noise=["custom noise phrase"])
        assert "custom noise phrase" in c.noise_indicators

    def test_custom_length_bounds(self):
        c = ChatCleaner("Alice", min_length=5, max_length=50)
        assert c.scrub_text("hi") == ""        
        assert c.scrub_text("hello") == "hello"
        assert c.scrub_text("x" * 51) == ""   


# ---------------------------------------------------------------------------
# decode_text
# ---------------------------------------------------------------------------

class TestDecodeText:
    def test_plain_ascii_unchanged(self, cleaner):
        assert cleaner.decode_text("hello world") == "hello world"

    def test_mojibake_repaired(self, cleaner):
        # "é" stored as latin-1 bytes in a UTF-8 string
        mangled = "caf\u00e9".encode("utf-8").decode("latin-1")
        result = cleaner.decode_text(mangled)
        assert "caf" in result

    def test_already_valid_utf8_unchanged(self, cleaner):
        text = "Hello 🙂 world"
        assert cleaner.decode_text(text) == text


# ---------------------------------------------------------------------------
# scrub_text
# ---------------------------------------------------------------------------

class TestScrubText:
    def test_empty_string_returns_empty(self, cleaner):
        assert cleaner.scrub_text("") == ""

    def test_none_like_falsy_returns_empty(self, cleaner):
        assert cleaner.scrub_text("") == ""

    # Platform noise
    @pytest.mark.parametrize("noise", [
        "reacted to your message",
        "liked your message",
        "sent a photo",
        "shared a link",
        "sent an attachment",
        "liked a message",
        "changed the theme",
        "@meta ai how are you",
        "shared a story",
        "sent a reel",
        "video call ended",
        "missed call",
        "sent a post",
        "started an audio call",
        "Audio call ended",
    ])
    def test_platform_noise_dropped(self, cleaner, noise):
        assert cleaner.scrub_text(noise) == ""

    def test_noise_check_is_case_insensitive(self, cleaner):
        assert cleaner.scrub_text("REACTED to your message") == ""

    def test_normal_message_kept(self, cleaner):
        assert cleaner.scrub_text("Hey, what's up?") == "Hey, what's up?"

    # Length bounds
    def test_single_char_dropped(self, cleaner):
        assert cleaner.scrub_text("x") == ""

    def test_two_char_kept(self, cleaner):
        assert cleaner.scrub_text("ok") == "ok"

    def test_over_1000_chars_dropped(self, cleaner):
        assert cleaner.scrub_text("a" * 1001) == ""

    def test_exactly_1000_chars_kept(self, cleaner):
        assert cleaner.scrub_text("a" * 1000) == "a" * 1000

    # Phone redaction
    def test_phone_number_redacted(self, cleaner):
        result = cleaner.scrub_text("call me at +44 7700 900123 ok")
        assert "[PHONE_NUMBER]" in result
        assert "7700" not in result

    def test_us_phone_redacted(self, cleaner):
        result = cleaner.scrub_text("my number is 555-867-5309")
        assert "[PHONE_NUMBER]" in result

    def test_no_false_positive_on_short_numbers(self, cleaner):
        # A plain year like "2024" should NOT be redacted
        result = cleaner.scrub_text("see you in 2024")
        assert "[PHONE_NUMBER]" not in result

    # Secret replacements
    def test_secret_term_replaced(self, cleaner_with_secrets):
        result = cleaner_with_secrets.scrub_text("ask John Doe about it")
        assert "[NAME]" in result
        assert "John Doe" not in result

    def test_secret_replacement_case_insensitive(self, cleaner_with_secrets):
        result = cleaner_with_secrets.scrub_text("ask JOHN DOE about it")
        assert "[NAME]" in result

    def test_secret_city_replaced(self, cleaner_with_secrets):
        result = cleaner_with_secrets.scrub_text("I'm flying to Mumbai tomorrow")
        assert "[CITY]" in result

    def test_message_kept_after_replacement(self, cleaner_with_secrets):
        result = cleaner_with_secrets.scrub_text("meet John Doe at the station")
        assert "meet" in result
        assert "at the station" in result


# ---------------------------------------------------------------------------
# normalize_sender
# ---------------------------------------------------------------------------

class TestNormalizeSender:
    def test_own_name_returns_me(self, cleaner):
        assert cleaner.normalize_sender("Raghav Sharma") == "me"

    def test_partial_name_match(self, cleaner):
        assert cleaner.normalize_sender("Raghav") == "me"

    def test_other_sender_returns_other(self, cleaner):
        assert cleaner.normalize_sender("Alice") == "other"

    def test_empty_sender_returns_other(self, cleaner):
        assert cleaner.normalize_sender("") == "other"


# ---------------------------------------------------------------------------
# get_other_party_name
# ---------------------------------------------------------------------------

class TestGetOtherPartyName:
    def test_returns_sanitized_name(self):
        c = ChatCleaner("Bob")
        msgs = [{"sender_name": "Alice Jones"}]
        assert c.get_other_party_name(msgs) == "alicejones"

    def test_skips_own_messages(self):
        c = ChatCleaner("Bob")
        msgs = [
            {"sender_name": "Bob"},
            {"sender_name": "Carol"},
        ]
        assert c.get_other_party_name(msgs) == "carol"

    def test_all_own_messages_returns_unknown(self):
        c = ChatCleaner("Bob")
        msgs = [{"sender_name": "Bob"}, {"sender_name": "Bob"}]
        assert c.get_other_party_name(msgs) == "unknown_sender"

    def test_empty_list_returns_unknown(self):
        c = ChatCleaner("Bob")
        assert c.get_other_party_name([]) == "unknown_sender"

    def test_no_crash_on_missing_sender_name_key(self):
        c = ChatCleaner("Bob")
        msgs = [{}]
        # Should fall back gracefully
        result = c.get_other_party_name(msgs)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# process_raw_messages
# ---------------------------------------------------------------------------

class TestProcessRawMessages:
    def _make_msg(self, sender, content, ts):
        return {"sender_name": sender, "content": content, "timestamp_ms": ts}

    def test_basic_processing(self, cleaner):
        msgs = [
            self._make_msg("Alice", "hello", 1000),
            self._make_msg("Raghav", "hi there", 2000),
        ]
        result = cleaner.process_raw_messages(msgs)
        assert len(result) == 2
        assert result[0]["sender"] == "other"
        assert result[1]["sender"] == "me"

    def test_sorted_by_timestamp(self, cleaner):
        msgs = [
            self._make_msg("Alice", "second", 2000),
            self._make_msg("Alice", "first", 1000),
        ]
        result = cleaner.process_raw_messages(msgs)
        assert result[0]["text"] == "first"
        assert result[1]["text"] == "second"

    def test_noise_messages_excluded(self, cleaner):
        msgs = [
            self._make_msg("Alice", "reacted to your message", 1000),
            self._make_msg("Alice", "hello", 2000),
        ]
        result = cleaner.process_raw_messages(msgs)
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_none_content_excluded(self, cleaner):
        msgs = [{"sender_name": "Alice", "content": None, "timestamp_ms": 1000}]
        result = cleaner.process_raw_messages(msgs)
        assert result == []

    def test_missing_content_key_excluded(self, cleaner):
        msgs = [{"sender_name": "Alice", "timestamp_ms": 1000}]
        result = cleaner.process_raw_messages(msgs)
        assert result == []

    def test_timestamp_defaults_to_zero(self, cleaner):
        msgs = [{"sender_name": "Alice", "content": "hi"}]
        result = cleaner.process_raw_messages(msgs)
        assert result[0]["timestamp"] == 0

    def test_output_schema(self, cleaner):
        msgs = [self._make_msg("Alice", "hey", 1000)]
        result = cleaner.process_raw_messages(msgs)
        assert set(result[0].keys()) == {"sender", "text", "timestamp"}


# ---------------------------------------------------------------------------
# merge_consecutive_messages
# ---------------------------------------------------------------------------

class TestMergeConsecutiveMessages:
    def test_same_sender_merged(self, cleaner):
        msgs = [
            {"sender": "other", "text": "hey"},
            {"sender": "other", "text": "how are you"},
        ]
        result = cleaner.merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert result[0]["text"] == "hey\nhow are you"

    def test_different_senders_not_merged(self, cleaner):
        msgs = [
            {"sender": "other", "text": "hey"},
            {"sender": "me", "text": "hi"},
        ]
        result = cleaner.merge_consecutive_messages(msgs)
        assert len(result) == 2

    def test_alternating_senders(self, cleaner):
        msgs = [
            {"sender": "other", "text": "a"},
            {"sender": "me", "text": "b"},
            {"sender": "other", "text": "c"},
        ]
        result = cleaner.merge_consecutive_messages(msgs)
        assert len(result) == 3

    def test_three_consecutive_merged(self, cleaner):
        msgs = [
            {"sender": "me", "text": "one"},
            {"sender": "me", "text": "two"},
            {"sender": "me", "text": "three"},
        ]
        result = cleaner.merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert result[0]["text"] == "one\ntwo\nthree"

    def test_empty_input(self, cleaner):
        assert cleaner.merge_consecutive_messages([]) == []