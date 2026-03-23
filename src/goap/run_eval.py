"""CLI entry point for running GOAP evaluations on ControlArena settings."""

import argparse
import logging

from control_arena import EvalMode
from control_arena.eval import get_control_task
from control_arena.eval._control_eval_config import ControlEvalConfig
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig

from goap.protocol import goap_protocol
from goap.prompts import load_prompt


def _build_agentdojo_setting(args, mode: EvalMode):
    from control_arena.settings.agentdojo_setting import AgentDojoSetting

    if getattr(args, "no_injection", False):
        attack_name = None
    else:
        attack_name = args.attack or ("important_instructions" if mode == EvalMode.ATTACK else None)

    return AgentDojoSetting(
        suite=args.suite,
        main_task=args.main_task,
        side_task=args.side_task,
        attack=attack_name,
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


def _build_trusted_monitoring_protocol():
    """Build the trusted-monitoring baseline: untrusted policy + full-trajectory monitor."""
    from control_arena.micro_protocol import trusted_monitoring_builder
    from control_arena.monitor import (
        MonitorScoreRange,
        basic_monitor_builder,
        full_trajectory_prompt_formatter_markdown,
    )

    monitor = basic_monitor_builder(
        prompt=load_prompt("monitor_user.txt"),
        prompt_formatter=full_trajectory_prompt_formatter_markdown,
        system_prompt=load_prompt("monitor_system.txt"),
        score_range=MonitorScoreRange(min=0, max=100),
    )
    return trusted_monitoring_builder(monitor=monitor)


def build_task(
    mode: EvalMode,
    setting_name: str = "agentdojo",
    protocol_name: str = "goap",
    untrusted_model: str | None = None,
    trusted_model: str | None = None,
    **setting_kwargs,
):
    """Build an Inspect task for a control protocol on any supported setting."""

    class _NS:
        pass

    ns = _NS()
    for k, v in setting_kwargs.items():
        setattr(ns, k, v)

    builder = _SETTING_BUILDERS[setting_name]
    setting = builder(ns, mode)

    monitoring_scorers = None
    if protocol_name == "goap":
        protocol = goap_protocol(setting_name=setting_name)
    elif protocol_name == "trusted-monitoring":
        from control_arena.scorers import max_suspicion_scorer

        protocol = _build_trusted_monitoring_protocol()
        monitoring_scorers = [max_suspicion_scorer()]
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")

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
        monitoring_scorers=monitoring_scorers,
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
    parser.add_argument(
        "--protocol",
        choices=["goap", "trusted-monitoring"],
        default="goap",
        help="Control protocol: goap (structural) or trusted-monitoring (baseline)",
    )

    # AgentDojo-specific args
    parser.add_argument("--suite", default="workspace", help="AgentDojo suite")
    parser.add_argument(
        "--main-task", default=None, help="Main task filter (e.g. user_task_13). Omit to run all."
    )
    parser.add_argument("--side-task", default=None, help="Side task regex filter")
    parser.add_argument("--attack", default=None, help="Attack name")
    parser.add_argument(
        "--no-injection",
        action="store_true",
        help="Attack mode without environment prompt injections (attack policy only)",
    )

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
        protocol_name=args.protocol,
        untrusted_model=args.model,
        trusted_model=trusted,
        # Pass all setting-specific args through
        suite=args.suite,
        main_task=args.main_task,
        side_task=args.side_task,
        attack=args.attack,
        no_injection=getattr(args, "no_injection", False),
        task_pair=args.task_pair,
        hint=args.hint,
        thinking_time=args.thinking_time,
    )

    if args.eval_set:
        model_slug = args.model.split("/")[-1]
        log_subdir = f"{args.log_dir}/{args.protocol}_{model_slug}_{args.mode}"
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
