"""CLI entry point for running GOAP evaluations on ControlArena settings."""

import argparse
import logging

from control_arena import EvalMode
from control_arena.eval import get_control_task
from control_arena.eval._control_eval_config import ControlEvalConfig
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

from goap.protocol import goap_protocol


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


def build_task(
    mode: EvalMode,
    setting_name: str = "agentdojo",
    untrusted_model: str | None = None,
    trusted_model: str | None = None,
    **setting_kwargs,
):
    """Build an Inspect task for the GOAP protocol on any supported setting."""

    class _NS:
        pass

    ns = _NS()
    for k, v in setting_kwargs.items():
        setattr(ns, k, v)

    builder = _SETTING_BUILDERS[setting_name]
    setting = builder(ns, mode)

    protocol = goap_protocol(setting_name=setting_name)

    eval_config = ControlEvalConfig(
        untrusted_model=untrusted_model,
        trusted_model=trusted_model,
        untrusted_config=GenerateConfig(max_tool_output=40 * 1024),
    )

    return get_control_task(
        setting=setting,
        micro_protocol=protocol,
        mode=mode,
        eval_config=eval_config,
    )


def main():
    parser = argparse.ArgumentParser(description="Run GOAP evaluations on ControlArena")
    parser.add_argument(
        "--setting",
        choices=list(_SETTING_BUILDERS.keys()),
        default="agentdojo",
        help="ControlArena setting",
    )
    parser.add_argument(
        "--mode",
        choices=["honest", "attack"],
        default="honest",
        help="Evaluation mode",
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
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument(
        "--model",
        default="openrouter/google/gemini-3-flash-preview",
        help="Untrusted model (content generation within actions)",
    )
    parser.add_argument(
        "--trusted-model",
        default=None,
        help="Trusted model (defaults to --model since GOAP planner is deterministic)",
    )
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    mode = EvalMode.ATTACK if args.mode == "attack" else EvalMode.HONEST
    trusted = args.trusted_model or args.model

    task = build_task(
        mode=mode,
        setting_name=args.setting,
        untrusted_model=args.model,
        trusted_model=trusted,
        # Pass all setting-specific args through
        suite=args.suite,
        main_task=args.main_task,
        side_task=args.side_task,
        attack=args.attack,
        task_pair=args.task_pair,
        hint=args.hint,
        thinking_time=args.thinking_time,
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
