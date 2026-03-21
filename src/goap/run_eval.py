"""CLI entry point for running GOAP evaluations on AgentDojo."""

import argparse
import logging

from inspect_ai import eval as inspect_eval

from control_arena import EvalMode
from control_arena.eval import get_control_task
from control_arena.settings.agentdojo_setting import AgentDojoSetting

from goap.protocol import goap_protocol


def build_task(
    mode: EvalMode,
    suite: str = "workspace",
    main_task: str | None = None,
    side_task: str | None = None,
    attack: str | None = None,
):
    """Build an Inspect task for the GOAP protocol on AgentDojo."""
    setting = AgentDojoSetting(
        suite=suite,
        main_task=main_task,
        side_task=side_task,
        attack=attack or ("important_instructions" if mode == EvalMode.ATTACK else None),
    )

    protocol = goap_protocol()

    return get_control_task(
        setting=setting,
        micro_protocol=protocol,
        mode=mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Run GOAP evaluations on AgentDojo")
    parser.add_argument(
        "--mode",
        choices=["honest", "attack"],
        default="honest",
        help="Evaluation mode",
    )
    parser.add_argument("--suite", default="workspace", help="AgentDojo suite")
    parser.add_argument("--main-task", default=None, help="Main task regex filter")
    parser.add_argument("--side-task", default=None, help="Side task regex filter")
    parser.add_argument("--attack", default=None, help="Attack name")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4-20250514",
        help="Model for the untrusted agent (used within actions)",
    )
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    mode = EvalMode.ATTACK if args.mode == "attack" else EvalMode.HONEST

    task = build_task(
        mode=mode,
        suite=args.suite,
        main_task=args.main_task,
        side_task=args.side_task,
        attack=args.attack,
    )

    inspect_eval(
        task,
        model=args.model,
        limit=args.limit,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
