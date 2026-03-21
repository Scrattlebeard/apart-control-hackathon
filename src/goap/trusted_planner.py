"""Trusted LLM planner for tool-typed GOAP.

The trusted model's job is small and well-scoped:
1. Define the goal (which output satisfies the task)
2. Specify tool instances with fixed_args and content_params
3. Add content-generation bridges where A* can't proceed without LLM help

A* searches the combined action space (tool instances + bridges).
The model does NOT specify ordering or preconditions/effects — those
are derived mechanically from tool signatures and bridge declarations.
"""

import json
import logging
from typing import Any

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import Tool
from pydantic import BaseModel, ValidationError

from goap.catalog import build_catalog, format_catalog
from goap.planner import a_star_plan, diagnose_failure
from goap.prompts import load_prompt
from goap.types import GoalState, GoapAction, Plan, ToolTemplate, WorldState

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Raised when the trusted planner fails to produce a valid plan."""


# ---------------------------------------------------------------------------
# Pydantic response models — what the trusted model outputs
# ---------------------------------------------------------------------------


class ToolInstance(BaseModel):
    """A specific use of a tool, with structural args locked down."""

    id: str  # unique instance name, e.g. "search_team_file"
    tool: str  # tool template name from the catalog
    fixed_args: dict[str, Any] = {}
    content_params: list[str] = []  # params needing LLM generation


class Bridge(BaseModel):
    """A content-generation step connecting tool outputs to tool inputs."""

    name: str
    inputs: list[str]  # state conditions required (instance output types)
    outputs: list[str]  # state conditions produced (one or more)
    prompt: str  # generation prompt for untrusted model


class SubAgent(BaseModel):
    """A recursive GOAP sub-plan for dynamic iteration."""

    name: str
    state_input: str  # state condition providing context
    sub_goal: str  # what the sub-agent should achieve
    effect: str  # state condition produced when done
    prompt: str  # planning prompt for the trusted model's sub-plan


class PlannerResponse(BaseModel):
    """The trusted model's complete output for a task."""

    goal: dict[str, bool]
    world_state: dict[str, bool] = {}
    tool_instances: list[ToolInstance]
    bridges: list[Bridge] = []
    sub_agents: list[SubAgent] = []


# ---------------------------------------------------------------------------
# Build action space from model response + catalog
# ---------------------------------------------------------------------------


def _find_template(catalog: list[ToolTemplate], tool_name: str) -> ToolTemplate | None:
    for t in catalog:
        if t.name == tool_name:
            return t
    return None


def build_action_space(
    response: PlannerResponse,
    catalog: list[ToolTemplate],
) -> list[GoapAction]:
    """Convert model response + catalog into GoapActions for A*.

    Tool instances become tool actions with:
    - preconditions: one per content_param (named ``{instance_id}:{param_name}``)
    - effects: ``{instance_id}_result``

    Bridges become bridge actions with:
    - preconditions: from inputs
    - effects: from output
    """
    actions: list[GoapAction] = []

    for inst in response.tool_instances:
        template = _find_template(catalog, inst.tool)
        if template is None:
            raise PlanningError(
                f"Tool instance '{inst.id}' references unknown tool '{inst.tool}'"
            )

        # Preconditions: each content_param needs a state condition to be satisfied
        preconds: dict[str, bool] = {}
        for param in inst.content_params:
            preconds[f"{inst.id}:{param}"] = True

        # Effect: this instance's result
        effect_name = f"{inst.id}_result"
        effects: dict[str, bool] = {effect_name: True}

        # Build the llm_prompt from content_params
        llm_prompt = ""
        if inst.content_params:
            llm_prompt = (
                f"Generate values for: {', '.join(inst.content_params)}. "
                f"Use the tool results in context."
            )

        actions.append(
            GoapAction(
                name=inst.id,
                action_type="tool",
                tool_name=inst.tool,
                preconditions=preconds,
                effects=effects,
                fixed_args=inst.fixed_args,
                llm_args=inst.content_params,
                llm_prompt=llm_prompt,
            )
        )

    for bridge in response.bridges:
        preconds = {inp: True for inp in bridge.inputs}
        effects = {out: True for out in bridge.outputs}

        actions.append(
            GoapAction(
                name=bridge.name,
                action_type="bridge",
                preconditions=preconds,
                effects=effects,
                llm_prompt=bridge.prompt,
            )
        )

    for sub in response.sub_agents:
        actions.append(
            GoapAction(
                name=sub.name,
                action_type="run_agent",
                preconditions={sub.state_input: True},
                effects={sub.effect: True},
                sub_goal=sub.sub_goal,
                sub_prompt=sub.prompt,
            )
        )

    return actions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_response(
    response: PlannerResponse,
    catalog: list[ToolTemplate],
) -> None:
    """Validate the model response against the catalog."""
    catalog_names = {t.name for t in catalog}

    # Check tool instances reference valid tools
    for inst in response.tool_instances:
        if inst.tool not in catalog_names:
            raise PlanningError(
                f"Tool instance '{inst.id}' references unknown tool '{inst.tool}'. "
                f"Available: {sorted(catalog_names)}"
            )

        template = _find_template(catalog, inst.tool)
        assert template is not None
        param_names = {p.name for p in template.params}

        # fixed_args must be valid param names
        for key in inst.fixed_args:
            if key not in param_names:
                raise PlanningError(
                    f"Instance '{inst.id}': fixed_arg '{key}' is not a parameter of '{inst.tool}'. "
                    f"Valid: {sorted(param_names)}"
                )

        # content_params must be valid param names
        for cp in inst.content_params:
            if cp not in param_names:
                raise PlanningError(
                    f"Instance '{inst.id}': content_param '{cp}' is not a parameter of '{inst.tool}'. "
                    f"Valid: {sorted(param_names)}"
                )

        # Type-check fixed_args against catalog schema
        for key, value in inst.fixed_args.items():
            param = next((p for p in template.params if p.name == key), None)
            if param and param.type.startswith("array") and not isinstance(value, list):
                raise PlanningError(
                    f"Instance '{inst.id}': fixed_arg '{key}' must be a list "
                    f"(type: {param.type}), got {type(value).__name__}: {value!r}"
                )

        # No overlap
        overlap = set(inst.fixed_args.keys()) & set(inst.content_params)
        if overlap:
            raise PlanningError(
                f"Instance '{inst.id}': params in both fixed_args and content_params: {overlap}"
            )

    # Unique instance IDs
    ids = [inst.id for inst in response.tool_instances]
    ids += [b.name for b in response.bridges]
    ids += [s.name for s in response.sub_agents]
    if len(set(ids)) != len(ids):
        dupes = [x for x in ids if ids.count(x) > 1]
        raise PlanningError(f"Duplicate action IDs: {set(dupes)}")

    # Must have exactly one submit instance
    submit_instances = [i for i in response.tool_instances if i.tool == "submit"]
    if len(submit_instances) != 1:
        raise PlanningError(
            f"Must have exactly one submit tool instance, found {len(submit_instances)}"
        )

    # Goal must include the submit instance's effect
    submit_id = submit_instances[0].id
    submit_effect = f"{submit_id}_result"
    if submit_effect not in response.goal:
        raise PlanningError(
            f"Goal must include '{submit_effect}: true' (the submit instance's effect)"
        )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    """Load and assemble the trusted planner system prompt."""
    template = load_prompt("trusted_planner.txt")
    example = load_prompt("few_shot_example.txt")
    return template.replace("__EXAMPLE__", example)


# ---------------------------------------------------------------------------
# Trusted model call
# ---------------------------------------------------------------------------


async def _call_trusted_model(
    task_description: str,
    catalog: list[ToolTemplate],
    context: str = "",
    error_context: str = "",
) -> PlannerResponse:
    """Single trusted model call. Returns parsed PlannerResponse."""
    system_content = _build_system_prompt()

    user_content = f"## Task\n\n{task_description}\n\n"
    if context:
        user_content += f"## Context from previous steps\n\n{context}\n\n"
    if error_context:
        user_content += f"## Error from previous attempt\n\n{error_context}\n\n"
    user_content += f"## Tool catalog\n\n{format_catalog(catalog)}"

    system_msg = ChatMessageSystem(content=system_content)
    user_msg = ChatMessageUser(content=user_content)

    model = get_model(role="trusted")
    logger.info("Calling trusted model for plan generation")

    result = await model.generate(
        [system_msg, user_msg],
        config=GenerateConfig(temperature=0.0, max_tokens=4096),
    )

    raw = result.choices[0].message.text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlanningError(f"Trusted model returned invalid JSON: {e}\nRaw: {raw[:500]}") from e

    try:
        response = PlannerResponse.model_validate(data)
    except ValidationError as e:
        raise PlanningError(f"Trusted model response failed validation: {e}") from e

    if not response.tool_instances:
        raise PlanningError("Trusted model returned no tool instances")

    return response


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_plan(
    task_description: str,
    tools: list[Tool],
    context: str = "",
) -> tuple[Plan, list[ToolTemplate]]:
    """Use the trusted model to generate a GOAP plan for the given task.

    Returns (plan, catalog). The plan's actions have action_type set
    to "tool", "bridge", or "run_agent". The catalog is needed by the
    protocol for sub-agent planning.

    Raises PlanningError on failure.
    """
    catalog = build_catalog(tools)

    response = await _call_trusted_model(task_description, catalog, context)
    _validate_response(response, catalog)

    actions = build_action_space(response, catalog)

    for a in actions:
        pre = f" pre={a.preconditions}" if a.preconditions else ""
        eff = f" eff={a.effects}" if a.effects else ""
        logger.info("  Action: %s [%s]%s%s", a.name, a.action_type, pre, eff)

    start = WorldState(conditions=dict(response.world_state))
    goal = GoalState(conditions=response.goal, description=task_description)

    plan = a_star_plan(start=start, goal=goal, available_actions=actions)

    # Fallback: if A* fails, diagnose and retry once
    if plan is None:
        unreachable = diagnose_failure(start, goal, actions)
        if not unreachable:
            raise PlanningError("A* failed but no unreachable conditions found — likely a cycle")

        error_msg = (
            f"A* could not find a plan. These conditions are unreachable: {sorted(unreachable)}. "
            f"You likely need to add bridges producing these conditions. "
            f"Each content_param creates precondition '{{instance_id}}:{{param_name}}' — "
            f"add a bridge with that exact output."
        )
        logger.warning("A* failed, retrying with error context: %s", error_msg)

        response2 = await _call_trusted_model(
            task_description, catalog, context, error_context=error_msg
        )
        _validate_response(response2, catalog)
        actions2 = build_action_space(response2, catalog)

        for a in actions2:
            pre = f" pre={a.preconditions}" if a.preconditions else ""
            eff = f" eff={a.effects}" if a.effects else ""
            logger.info("  Retry action: %s [%s]%s%s", a.name, a.action_type, pre, eff)

        plan = a_star_plan(start=start, goal=GoalState(conditions=response2.goal), available_actions=actions2)
        if plan is None:
            raise PlanningError(
                "A* failed even after retry. Unreachable conditions in retry: "
                f"{sorted(diagnose_failure(start, GoalState(conditions=response2.goal), actions2))}"
            )

    logger.info(
        "Plan: %d steps — %s",
        len(plan.actions),
        " → ".join(a.name for a in plan.actions),
    )
    return plan, catalog
