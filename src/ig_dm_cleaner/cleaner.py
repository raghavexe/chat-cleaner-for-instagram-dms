import os
import json
import re
from typing import List, Dict, Optional

# GLOBAL TEXT FILTERS (Throws away the ENTIRE message if found)
PLATFORM_NOISE_INDICATORS = [
    "reacted", "liked your message", "sent a photo", "shared a link",
    "wasn't notified about this message because they're in quiet mode",
    "sent an attachment", "liked a message", "changed the theme", "@meta ai",
    "shared a story", "sent a reel", "video call", "missed call", "sent a post",
    "think of me like an assistant who's here to help", "started an audio call",
    "audio call ended",
]

PHONE_PATTERN = re.compile(
    r'\+?\d{1,4}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}'
)


class ChatCleaner:
    """
    Cleans raw Instagram/Facebook DM export JSON into normalized message dicts.

    Args:
        my_name: The display name used for your account in the export
                 (e.g. "John Smith"). Used to distinguish self from others.
        secrets_path: Optional path to a secrets.json file containing
                      ``{"phrase_to_replace": "REPLACEMENT_TOKEN"}`` pairs
                      for PII scrubbing beyond phone numbers.
        min_length: Minimum character length for a message to be kept (default 2).
        max_length: Maximum character length for a message to be kept (default 1000).
        extra_noise: Additional platform-noise strings to filter on top of the defaults.
    """

    def __init__(
        self,
        my_name: str,
        secrets_path: Optional[str] = None,
        min_length: int = 2,
        max_length: int = 1000,
        extra_noise: Optional[List[str]] = None,
    ):
        if not my_name or not my_name.strip():
            raise ValueError("my_name must be a non-empty string.")

        self.my_name = my_name
        self.min_length = min_length
        self.max_length = max_length
        self.noise_indicators = list(PLATFORM_NOISE_INDICATORS)
        if extra_noise:
            self.noise_indicators.extend(n.lower() for n in extra_noise)

        self.secret_replacements = self._load_secret_replacements(secrets_path)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_secret_replacements(self, secrets_path: Optional[str]) -> Dict[str, str]:
        """
        Loads phrase→token replacement pairs from a JSON file.
        Falls back gracefully if the file is absent or malformed.
        """
        if secrets_path is None:
            return {}

        if not os.path.exists(secrets_path):
            return {}

        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            mapping = {k.lower(): v for k, v in raw.items() if k.strip()}
            print(f"[ChatCleaner] Loaded {len(mapping)} secret replacement pairs.")
            return mapping
        except Exception as e:
            print(f"[ChatCleaner] Warning: could not parse secrets file '{secrets_path}': {e}")
            return {}

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def decode_text(self, text: str) -> str:
        """Repairs common mojibake from Facebook/Instagram latin-1 JSON exports."""
        try:
            return text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    def scrub_text(self, text: str) -> str:
        """
        Master cleanup for a single message string.

        Steps:
            1. Drop entire message if it matches platform noise.
            2. Repair encoding (latin-1 → utf-8).
            3. Enforce length bounds.
            4. Replace secret phrases in-place.
            5. Redact phone numbers.

        Returns an empty string if the message should be discarded.
        """
        if not text:
            return ""

        lower_text = text.lower()
        if any(indicator in lower_text for indicator in self.noise_indicators):
            return ""

        decoded = self.decode_text(text).strip()

        if not self.min_length <= len(decoded) <= self.max_length:
            return ""

        for term, token in self.secret_replacements.items():
            if term in decoded.lower():
                decoded = re.sub(re.escape(term), token, decoded, flags=re.IGNORECASE)

        return PHONE_PATTERN.sub("[PHONE_NUMBER]", decoded).strip()

    # ------------------------------------------------------------------
    # Sender helpers
    # ------------------------------------------------------------------

    def normalize_sender(self, raw_name: str) -> str:
        """Returns ``"me"`` for your own messages, ``"other"`` for everyone else."""
        return "me" if self.my_name in raw_name else "other"

    def get_other_party_name(self, messages: List[Dict]) -> str:
        """Returns a sanitized lowercase identifier for the first non-self sender found."""
        for msg in messages:
            name = msg.get("sender_name", "")
            if self.my_name not in name:
                decoded_name = self.decode_text(name)
                # Split into tokens, join, then strip non-alphanumeric chars
                clean_name = re.sub(r"[^a-zA-Z0-9]", "", "".join(decoded_name.split()))
                return clean_name.lower() or "other_user"
        return "unknown_sender"

    # ------------------------------------------------------------------
    # High-level pipeline methods
    # ------------------------------------------------------------------

    def process_raw_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Cleans and normalizes a raw list of Instagram message dicts.

        Args:
            messages: List of message objects from ``data["messages"]`` in the export.

        Returns:
            Sorted list of ``{"sender": str, "text": str, "timestamp": int}`` dicts.
        """
        cleaned = []
        for msg in messages:
            raw_text = msg.get("content")
            processed = self.scrub_text(raw_text)
            if not processed:
                continue
            cleaned.append({
                "sender": self.normalize_sender(msg.get("sender_name", "")),
                "text": processed,
                "timestamp": msg.get("timestamp_ms", 0),
            })
        return sorted(cleaned, key=lambda x: x["timestamp"])

    def merge_consecutive_messages(self, cleaned_messages: List[Dict]) -> List[Dict]:
        """
        Merges back-to-back messages from the same sender into one block.

        Args:
            cleaned_messages: Output of :meth:`process_raw_messages`.

        Returns:
            List of merged message dicts (without timestamps).
        """
        merged: List[Dict] = []
        for msg in cleaned_messages:
            if not merged or merged[-1]["sender"] != msg["sender"]:
                merged.append({"sender": msg["sender"], "text": msg["text"]})
            else:
                merged[-1]["text"] += "\n" + msg["text"]
        return merged