"""
CLI entry point for ig-dm-cleaner.

Usage
-----
Run the full pipeline::

    ig-dm-cleaner --name "Your Name" --inbox ./inbox

Split an existing dataset::

    ig-dm-cleaner split --source train.jsonl --val-fraction 0.05
"""

import argparse
import sys

from ig_dm_cleaner import Pipeline, PipelineConfig, split_train_val


def cmd_run(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        my_name=args.name,
        inbox_dir=args.inbox,
        cleaned_dir=args.cleaned_dir,
        output_file=args.output,
        removed_log=args.removed_log,
        secrets_path=args.secrets,
    )
    result = Pipeline(config).run()
    if args.split:
        split_train_val(
            source_path=config.output_file,
            train_path=config.output_file,
            val_path=args.val_output,
            val_fraction=args.val_fraction,
        )
    sys.exit(0 if result.valid_samples > 0 else 1)


def cmd_split(args: argparse.Namespace) -> None:
    split_train_val(
        source_path=args.source,
        train_path=args.train_output,
        val_path=args.val_output,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ig-dm-cleaner",
        description="Convert Instagram DM exports into LLM fine-tuning datasets.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ───────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run the full clean→generate→validate pipeline.")
    run_p.add_argument("--name", required=True, help="Your display name in the export.")
    run_p.add_argument("--inbox", default="./inbox", help="Path to Instagram inbox folder.")
    run_p.add_argument("--cleaned-dir", default="./cleaned-text")
    run_p.add_argument("--output", default="./train.jsonl")
    run_p.add_argument("--removed-log", default="./removed_lines.jsonl")
    run_p.add_argument("--secrets", default=None, help="Path to secrets.json.")
    run_p.add_argument("--split", action="store_true", help="Split after generation.")
    run_p.add_argument("--val-output", default="./val.jsonl")
    run_p.add_argument("--val-fraction", type=float, default=0.05)
    run_p.set_defaults(func=cmd_run)

    # ── split ─────────────────────────────────────────────────────────
    split_p = sub.add_parser("split", help="Split an existing JSONL into train/val.")
    split_p.add_argument("--source", default="./train.jsonl")
    split_p.add_argument("--train-output", default="./train.jsonl")
    split_p.add_argument("--val-output", default="./val.jsonl")
    split_p.add_argument("--val-fraction", type=float, default=0.05)
    split_p.add_argument("--seed", type=int, default=None)
    split_p.set_defaults(func=cmd_split)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()