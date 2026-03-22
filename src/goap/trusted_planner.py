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
import re
from dataclasses import dataclass, field
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


@dataclass
class PlanResult:
    """Result of plan generation, including raw model exchanges for logging."""

    plan: Plan
    catalog: list[ToolTemplate]
    exchanges: list[tuple[str, str]] = field(default_factory=list)  # (user_prompt, raw_response)


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

    goal: dict[str, bool] = {}  # ignored — injected by _inject_submit
    world_state: dict[str, bool] = {}  # only non-empty during re-planning
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


def _inject_submit(response: PlannerResponse) -> None:
    """Inject the canonical submit instance, bridge, and goal.

    The model should NOT include submit — we add it deterministically so
    there's no chance of a validation failure on a boilerplate element.
    Mutates *response* in place.
    """
    # Strip any submit the model included (belt-and-suspenders)
    submit_ids = {i.id for i in response.tool_instances if i.tool == "submit"}
    response.tool_instances = [i for i in response.tool_instances if i.tool != "submit"]
    # Strip bridges that fed into model-provided submit instances
    if submit_ids:
        submit_outputs = {f"{sid}:{p}" for sid in submit_ids for p in ("answer",)}
        response.bridges = [
            b for b in response.bridges
            if not any(out in submit_outputs for out in b.outputs)
        ]

    # Collect all non-submit tool effects to feed into the submit bridge
    all_effects = [f"{inst.id}_result" for inst in response.tool_instances]
    # Also include bridge and sub-agent effects
    all_effects += [out for b in response.bridges for out in b.outputs]
    all_effects += [s.effect for s in response.sub_agents]

    # Find the "terminal" effects — those not consumed as inputs by anything else
    consumed: set[str] = set()
    for inst in response.tool_instances:
        for p in inst.content_params:
            consumed.add(f"{inst.id}:{p}")
    for b in response.bridges:
        consumed.update(b.inputs)
    for s in response.sub_agents:
        consumed.add(s.state_input)

    terminal = [e for e in all_effects if e not in consumed] or all_effects[-1:]

    submit_id = "goap_submit"
    response.tool_instances.append(
        ToolInstance(id=submit_id, tool="submit", content_params=["answer"])
    )
    response.bridges.append(
        Bridge(
            name="bridge_submit",
            inputs=terminal,
            outputs=[f"{submit_id}:answer"],
            prompt="Summarize the results of all completed steps into a concise final answer.",
        )
    )
    response.goal = {f"{submit_id}_result": True}


def _normalize_bridges(response: PlannerResponse) -> None:
    """Fix common bridge wiring mistakes by matching against known conditions.

    Builds an index of valid conditions from tool instances, then rewrites
    bridge inputs/outputs that don't match any known condition but are close
    enough to be unambiguous.

    Mutates *response* in place.
    """
    # Build the valid condition index
    # Effects: {id}_result for each tool instance
    valid_effects: dict[str, str] = {}  # normalized_key → canonical form
    for inst in response.tool_instances:
        canonical = f"{inst.id}_result"
        valid_effects[canonical] = canonical
        # Also index without _result suffix for fuzzy matching
        valid_effects[inst.id] = canonical

    # Preconditions: {id}:{param} for each content_param
    valid_preconditions: dict[str, str] = {}  # normalized_key → canonical form
    for inst in response.tool_instances:
        for param in inst.content_params:
            canonical = f"{inst.id}:{param}"
            valid_preconditions[canonical] = canonical
            # Common wrong separators
            valid_preconditions[f"{inst.id}.{param}"] = canonical
            valid_preconditions[f"{inst.id}_{param}"] = canonical
            valid_preconditions[f"{inst.id}/{param}"] = canonical

    # Also index bridge effects and sub-agent effects as valid inputs
    bridge_effects: dict[str, str] = {}
    for bridge in response.bridges:
        for out in bridge.outputs:
            bridge_effects[out] = out
    for sub in response.sub_agents:
        bridge_effects[sub.effect] = sub.effect

    all_valid_inputs = {**valid_effects, **bridge_effects}
    all_valid_outputs = valid_preconditions

    changed = False
    for bridge in response.bridges:
        # Normalize inputs (should reference {id}_result or bridge/sub-agent effects)
        new_inputs = []
        for inp in bridge.inputs:
            if inp in all_valid_inputs:
                new_inputs.append(all_valid_inputs[inp])
            elif inp in all_valid_outputs:
                # Model put a precondition as an input — probably meant the _result
                logger.warning("Bridge '%s': input '%s' looks like a precondition, not an effect", bridge.name, inp)
                new_inputs.append(inp)
            else:
                # Try appending _result
                with_result = f"{inp}_result"
                if with_result in all_valid_inputs:
                    logger.warning("Bridge '%s': input '%s' → '%s'", bridge.name, inp, with_result)
                    new_inputs.append(with_result)
                    changed = True
                else:
                    new_inputs.append(inp)

        # Normalize outputs (should reference {id}:{param})
        new_outputs = []
        for out in bridge.outputs:
            if out in all_valid_outputs:
                new_outputs.append(all_valid_outputs[out])
            elif out in all_valid_inputs:
                # Model put an effect as an output — wrong direction
                logger.warning("Bridge '%s': output '%s' is an effect, not a precondition — keeping as-is", bridge.name, out)
                new_outputs.append(out)
            else:
                # Check if it matches a precondition with wrong separator
                canonical = all_valid_outputs.get(out)
                if canonical:
                    logger.warning("Bridge '%s': output '%s' → '%s'", bridge.name, out, canonical)
                    new_outputs.append(canonical)
                    changed = True
                else:
                    new_outputs.append(out)

        bridge.inputs = new_inputs
        bridge.outputs = new_outputs

    if changed:
        logger.info("Bridge wiring normalized")


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

        # Auto-resolve overlap: fixed_args wins
        overlap = set(inst.fixed_args.keys()) & set(inst.content_params)
        if overlap:
            logger.warning(
                "Instance '%s': params in both fixed_args and content_params: %s — removing from content_params",
                inst.id, overlap,
            )
            inst.content_params = [p for p in inst.content_params if p not in overlap]

    # Auto-deduplicate IDs by suffixing
    seen: dict[str, int] = {}
    for item in [*response.tool_instances, *response.bridges, *response.sub_agents]:
        key = item.id if isinstance(item, ToolInstance) else item.name
        if key in seen:
            seen[key] += 1
            new_key = f"{key}_{seen[key]}"
            if isinstance(item, ToolInstance):
                logger.warning("Duplicate instance ID '%s' → renamed to '%s'", key, new_key)
                item.id = new_key
            else:
                logger.warning("Duplicate action ID '%s' → renamed to '%s'", key, new_key)
                item.name = new_key
        else:
            seen[key] = 0

    # Submit is injected by _inject_submit — any model-provided ones were already stripped


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt(setting_name: str = "agentdojo") -> str:
    """Load and assemble the trusted planner system prompt."""
    template = load_prompt("trusted_planner.txt")
    example = load_prompt("few_shot_example.txt", setting=setting_name)
    addendum = load_prompt("planner_addendum.txt", setting=setting_name)
    return template.replace("__EXAMPLE__", example).replace("__ADDENDUM__", addendum)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

# Trailing commas before } or ]
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _repair_json(text: str) -> str:
    """Fix common JSON syntax issues from LLM output."""
    text = _TRAILING_COMMA.sub(r"\1", text)
    return text


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from *text*.

    Handles markdown fences, preamble/postamble chatter, and minor
    syntax issues (trailing commas). Raises ValueError if no valid
    JSON object is found.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*\n?", "", text)

    # Fast path: the whole thing is valid JSON
    repaired = _repair_json(text.strip())
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Scan for the first { and try progressively larger substrings
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    # Find matching } by trying json.loads on substrings ending at each }
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = _repair_json(text[start : i + 1])
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Keep scanning — this } wasn't the right one
                    depth = 1

    raise ValueError(f"No valid JSON object found (scanned from position {start})")


# ---------------------------------------------------------------------------
# Trusted model call
# ---------------------------------------------------------------------------


async def _call_trusted_model(
    task_description: str,
    catalog: list[ToolTemplate],
    context: str = "",
    error_context: str = "",
    setting_name: str = "agentdojo",
) -> tuple[PlannerResponse, str, str]:
    """Single trusted model call.

    Returns (parsed_response, user_content, raw_response_text).
    """
    system_content = _build_system_prompt(setting_name)

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

    try:
        data = _extract_json(raw)
    except ValueError as e:
        raise PlanningError(f"Trusted model returned no valid JSON: {e}\nRaw: {raw[:500]}") from e

    try:
        response = PlannerResponse.model_validate(data)
    except ValidationError as e:
        raise PlanningError(f"Trusted model response failed validation: {e}") from e

    if not response.tool_instances:
        raise PlanningError("Trusted model returned no tool instances")

    return response, user_content, raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_plan(
    task_description: str,
    tools: list[Tool],
    context: str = "",
    setting_name: str = "agentdojo",
) -> PlanResult:
    """Use the trusted model to generate a GOAP plan for the given task.

    Returns a PlanResult containing the plan, catalog, and raw model
    exchanges (for transcript logging).

    Raises PlanningError on failure.
    """
    catalog = build_catalog(tools)
    # Don't show submit to the planner — it's injected automatically
    planner_catalog = [t for t in catalog if t.name != "submit"]
    exchanges: list[tuple[str, str]] = []

    response, user_content, raw_response = await _call_trusted_model(
        task_description, planner_catalog, context, setting_name=setting_name
    )
    exchanges.append((user_content, raw_response))

    _inject_submit(response)
    _normalize_bridges(response)
    _validate_response(response, catalog)  # validate against full catalog (includes submit)

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

        response2, user_content2, raw_response2 = await _call_trusted_model(
            task_description, planner_catalog, context, error_context=error_msg, setting_name=setting_name
        )
        exchanges.append((user_content2, raw_response2))

        _inject_submit(response2)
        _normalize_bridges(response2)
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
    return PlanResult(plan=plan, catalog=catalog, exchanges=exchanges)
