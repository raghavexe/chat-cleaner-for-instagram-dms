import json
import os
import sys
from typing import Optional

from .cleaner import ChatCleaner


def process_inbox(
    my_name: str,
    inbox_dir: str = "./inbox",
    output_dir: str = "./cleaned-text",
    secrets_path: Optional[str] = None,
) -> int:
    if not os.path.exists(inbox_dir):
        print(f"Error: Inbox directory '{inbox_dir}' not found.", file=sys.stderr)
        return 0

    try:
        cleaner = ChatCleaner(my_name, secrets_path=secrets_path)
    except Exception as e:
        print(f"Failed to initialize ChatCleaner: {e}", file=sys.stderr)
        return 0

    os.makedirs(output_dir, exist_ok=True)
    processed = 0

    for root, _dirs, files in os.walk(inbox_dir):
        json_files = sorted(f for f in files if f.startswith("message_") and f.endswith(".json"))
        if not json_files:
            continue

        folder_name = os.path.basename(root)
        print(f"\nProcessing folder: {folder_name}")

        all_messages = []
        for file_name in json_files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_messages.extend(data.get("messages", []))
            except json.JSONDecodeError:
                print(f"  [Skip] Corrupt JSON: {file_name}")
            except OSError as e:
                print(f"  [Skip] Cannot read {file_name}: {e.strerror}")

        if not all_messages:
            print(f"  [Notice] No messages found in '{folder_name}'. Skipping.")
            continue

        try:
            cleaned = cleaner.process_raw_messages(all_messages)
            output_path = os.path.join(output_dir, f"{folder_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Combined {len(json_files)} file(s) → {output_path}")
            processed += 1
        except Exception as e:
            print(f"  [Error] Cleaner failed for '{folder_name}': {e}")

    return processed


def main(
    my_name: str = "Raghav",
    inbox_dir: str = "./inbox",
    output_dir: str = "./cleaned-text",
    secrets_path: Optional[str] = None,
) -> None:
    process_inbox(my_name, inbox_dir, output_dir, secrets_path)
