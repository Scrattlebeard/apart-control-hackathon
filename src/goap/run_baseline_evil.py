"""CLI entry point for running baseline evaluations with attack policy.

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
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig


def _build_agentdojo_setting(args, mode: EvalMode):
    from control_arena.settings.agentdojo_setting import AgentDojoSetting

    return AgentDojoSetting(
        suite=args.suite,
        main_task=args.main_task,
        side_task=args.side_task,
        attack=args.attack or ("important_instructions" if mode == EvalMode.ATTACK else None),
    )


def _build_shade_arena_setting(args, mode: EvalMode):
    from control_arena.settings.shade_arena.shade_arena_setting import ShadeArenaSetting

    return ShadeArenaSetting(
        task_pair=args.task_pair,
        hint=args.hint,
        thinking_time=args.thinking_time,
    )


_SETTING_BUILDERS = {
    "agentdojo": _build_agentdojo_setting,
    "shade_arena": _build_shade_arena_setting,
}


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations with attack policy")
    parser.add_argument(
        "--setting",
        choices=list(_SETTING_BUILDERS.keys()),
        default="agentdojo",
        help="ControlArena setting",
    )
    parser.add_argument(
        "--mode",
        choices=["honest", "attack"],
        default="attack",
        help="Evaluation mode (defaults to attack)",
    )

    # AgentDojo-specific args
    parser.add_argument("--suite", default="workspace", help="AgentDojo suite")
    parser.add_argument(
        "--main-task", default=None, help="Main task filter (e.g. user_task_13). Omit to run all."
    )
    parser.add_argument("--side-task", default=None, help="Side task regex filter")
    parser.add_argument("--attack", default=None, help="Attack name")

    # ShadeArena-specific args
    parser.add_argument("--task-pair", default=None, help="ShadeArena task pair regex filter")
    parser.add_argument("--hint", action="store_true", help="ShadeArena: include hint in attack mode")
    parser.add_argument(
        "--thinking-time", action="store_true", help="ShadeArena: enable unmonitored thinking tags"
    )

    # Shared args
    parser.add_argument(
        "--eval-set",
        action="store_true",
        help="Use eval_set for automatic retry of failed/cancelled samples",
    )
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

    builder = _SETTING_BUILDERS[args.setting]
    setting = builder(args, mode)

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

    if args.eval_set:
        # Dedicated subdir so eval_set doesn't collide with other runs
        model_slug = args.model.split("/")[-1]
        log_subdir = f"{args.log_dir}/baseline_evil_{model_slug}_{args.mode}"
        eval_set(
            task,
            log_dir=log_subdir,
            model=args.model,
            limit=args.limit,
            retry_attempts=3,
            retry_wait=10,
            max_tool_output=40 * 1024,
            token_limit=4_000_000,
            fail_on_error=False,
        )
    else:
        inspect_eval(
            task,
            model=args.model,
            limit=args.limit,
            log_dir=args.log_dir,
            max_tool_output=40 * 1024,
            token_limit=4_000_000,
        )


if __name__ == "__main__":
    main()
