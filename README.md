# ig-dm-cleaner

Convert Instagram DM exports into clean JSONL datasets for LLM fine-tuning.

## Prior steps

Look at the [Google Drive Cleaner](https://github.com/raghavexe/google-drive-cleaner-utility-script-for-instagram-JSON-export) repository and add it to [Google App Scripts](https://script.google.com/home) to clean your inbox folder.
The cleaning process removes Photos, Videos, Audio Messages, Stickers and GIFs
TL;DR: it removes data that wont be required for the dataset generation steps.

## Install

```bash
pip install -e .
```

## Project layout expected

```
your-project/
├── inbox/              # unzipped Instagram export ("your_instagram_activity/messages/inbox")
└── secrets.json        # optional — PII replacements (git-ignored)
```

`secrets.json` format:

```json
{
  "real name": "[NAME]",
  "home city": "[CITY]",
  "street address": "[ADDRESS]"
}
```

---

## CLI

### Full pipeline

```bash
ig-dm-cleaner --name "Your Name" --inbox ./inbox
```

| Flag             | Default          | Description                     |
| ---------------- | ---------------- | ------------------------------- |
| `--name`         | _(required)_     | Your display name in the export |
| `--inbox`        | `./inbox`        | Instagram inbox folder          |
| `--cleaned-dir`  | `./cleaned-text` | Intermediate JSON output        |
| `--output`       | `./train.jsonl`  | Final JSONL dataset             |
| `--secrets`      | —                | Path to `secrets.json`          |
| `--split`        | off              | Auto-split after generation     |
| `--val-fraction` | `0.05`           | Validation set size             |

### Split an existing dataset

```bash
ig-dm-cleaner split --source train.jsonl --val-fraction 0.1 --seed 42
```

---

## Python API

### Full pipeline

```python
from ig_dm_cleaner import Pipeline, PipelineConfig

config = PipelineConfig(
    my_name="Your Name",
    inbox_dir="./inbox",
    secrets_path="./secrets.json",   # optional
)

result = Pipeline(config).run()
print(f"{result.valid_samples} samples written to train.jsonl")
```

### Individual components

```python
from ig_dm_cleaner import ChatCleaner, generate_dataset, split_train_val

# Clean a single batch of raw messages
cleaner = ChatCleaner("Your Name", secrets_path="secrets.json")
cleaned = cleaner.process_raw_messages(raw_messages)   # list[dict] from export JSON

# Generate JSONL from a cleaned-text folder
generate_dataset(input_folder="./cleaned-text", output_file="./train.jsonl")

# Split
split_train_val("./train.jsonl", val_fraction=0.05, seed=42)
```

---

## What the pipeline does

| Stage                         | Input                 | Output                                             |
| ----------------------------- | --------------------- | -------------------------------------------------- |
| **1 — Clean** (`processor`)   | `inbox/` folder       | `cleaned-text/*.json`                              |
| **2 — Generate** (`data_gen`) | `cleaned-text/*.json` | `train.jsonl`                                      |
| **3 — Validate** (`pipeline`) | `train.jsonl`         | deduplicated `train.jsonl` + `removed_lines.jsonl` |

Cleaning steps applied to each message:

- Drop platform-noise events (reactions, calls, theme changes, …)
- Repair latin-1 → UTF-8 mojibake from Facebook exports
- Redact phone numbers → `[PHONE_NUMBER]`
- Apply custom `secrets.json` replacements
- Enforce length bounds (2–1000 chars)

Only `other → me` adjacent pairs within a 12-hour window become training samples.

## Troubleshoot

- Run `pytest` in the root folder
- Contact me through my [website]("") (_Coming soon_)
- Create an issue in the repo for time being
