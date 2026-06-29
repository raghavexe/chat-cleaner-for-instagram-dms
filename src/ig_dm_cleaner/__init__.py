"""
ig_dm_cleaner — turn Instagram DM exports into LLM fine-tuning datasets.

Quickstart
----------
Run the full pipeline in three lines::

    from ig_dm_cleaner import Pipeline, PipelineConfig

    result = Pipeline(PipelineConfig(my_name="Your Name")).run()
    print(result.valid_samples)

Use individual components::

    from ig_dm_cleaner import ChatCleaner, generate_dataset

    cleaner = ChatCleaner("Your Name", secrets_path="secrets.json")
    cleaned = cleaner.process_raw_messages(raw_messages)
"""

from .cleaner import ChatCleaner
from .processor import process_inbox
from .data_gen import generate_dataset, merge_blocks, create_samples, DEFAULT_TIME_GAP_LIMIT
from .pipeline import Pipeline, PipelineConfig, PipelineResult
from .validation import split_train_val

__all__ = [
    "ChatCleaner",
    "process_inbox",
    "generate_dataset",
    "merge_blocks",
    "create_samples",
    "DEFAULT_TIME_GAP_LIMIT",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "split_train_val",
]