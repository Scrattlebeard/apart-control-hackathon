"""Create trimmed variants of eval logs, excluding looping samples.

A sample is considered "looping" if it has more than --max-messages messages.
Trimmed logs are written to logs/trimmed/ with the same filenames.

Usage:
    uv run python scripts/trim_looping_logs.py logs/some_run.eval
    uv run python scripts/trim_looping_logs.py logs/run1.eval logs/run2.eval --max-messages 50
    uv run python scripts/trim_looping_logs.py logs/*.eval --output-dir logs/trimmed_custom
"""

import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

DEFAULT_MAX_MESSAGES = 30
DEFAULT_OUTPUT_DIR = "logs/trimmed"


def trim_log(log_path: str, max_messages: int, output_dir: Path) -> tuple[Path, int, int]:
    """Remove looping samples from a log and write to output_dir.

    Returns (output_path, original_count, kept_count).
    """
    log = read_eval_log(log_path)

    if not log.samples:
        out = output_dir / Path(log_path).name
        write_eval_log(log, out)
        return out, 0, 0

    original = len(log.samples)
    kept = [s for s in log.samples if len(s.messages) <= max_messages]
    removed = [s for s in log.samples if len(s.messages) > max_messages]

    for s in removed:
        print(f"  dropping {s.id} ({len(s.messages)} msgs)")

    log.samples = kept

    out = output_dir / Path(log_path).name
    write_eval_log(log, out)
    return out, original, len(kept)


def main():
    parser = argparse.ArgumentParser(
        description="Trim looping samples from eval logs"
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Eval log files to trim",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=DEFAULT_MAX_MESSAGES,
        help=f"Drop samples with more messages than this (default: {DEFAULT_MAX_MESSAGES})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for trimmed logs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    for log_path in args.logs:
        name = Path(log_path).name[:55]
        out, orig, kept = trim_log(log_path, args.max_messages, output_dir)
        removed = orig - kept
        total_removed += removed
        status = f"  {kept}/{orig} samples kept" + (f" ({removed} dropped)" if removed else "")
        print(f"{name}")
        print(status)
        print()

    print(f"Trimmed logs written to {output_dir}/")
    print(f"Total samples removed: {total_removed}")


if __name__ == "__main__":
    main()
