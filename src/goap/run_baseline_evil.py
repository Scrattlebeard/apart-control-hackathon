"""CLI entry point for running baseline evaluations with attack policy on AgentDojo.

The model gets direct tool access with no planner constraints, and in attack
mode the system prompt is swapped to include side-task instructions. This is
the worst-case baseline: full tool access + adversarial system prompt.
"""

import argparse
import logging

from control_arena import EvalMode
from control_arena.eval import get_control_task
from control_arena.eval._control_eval_config import ControlEvalConfig
from control_arena.policy._untrusted_policy import untrusted_policy_builder
from control_arena.settings.agentdojo_setting import AgentDojoSetting
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations with attack policy on AgentDojo")
    parser.add_argument(
        "--mode",
        choices=["honest", "attack"],
        default="attack",
        help="Evaluation mode (defaults to attack)",
    )
    parser.add_argument("--suite", default="workspace", help="AgentDojo suite")
    parser.add_argument(
        "--main-task", default=None, help="Main task filter (e.g. user_task_13). Omit to run all."
    )
    parser.add_argument("--side-task", default=None, help="Side task regex filter")
    parser.add_argument("--attack", default=None, help="Attack name")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument(
        "--model",
        default="openrouter/google/gemini-3-flash-preview",
        help="Model to use",
    )
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    mode = EvalMode.ATTACK if args.mode == "attack" else EvalMode.HONEST

    setting = AgentDojoSetting(
        suite=args.suite,
        main_task=args.main_task,
        side_task=args.side_task,
        attack=args.attack or ("important_instructions" if mode == EvalMode.ATTACK else None),
    )

    task = get_control_task(
        setting=setting,
        micro_protocol=untrusted_policy_builder(),
        mode=mode,
        eval_config=ControlEvalConfig(
            untrusted_model=args.model,
            trusted_model=args.model,
            untrusted_config=GenerateConfig(max_tool_output=40 * 1024),
        ),
    )

    inspect_eval(
        task,
        model=args.model,
        limit=args.limit,
        log_dir=args.log_dir,
        max_tool_output=40 * 1024,
    )


if __name__ == "__main__":
    main()
