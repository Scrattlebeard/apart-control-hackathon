"""Type system for GOAP preconditions and effects.

Conditions are string keys mapped to boolean values. World state is a
dict of these conditions. Actions have preconditions (conditions that
must be true) and effects (conditions set after execution).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolParam:
    """A single parameter from a tool's signature."""

    name: str
    type: str
    required: bool
    description: str = ""
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ToolTemplate:
    """A tool's typed signature for GOAP planning.

    Built mechanically from inspect_ai tool signatures. The output_type
    is derived as ``{tool_name}_result`` — it becomes the state condition
    produced when the tool executes.
    """

    name: str
    description: str
    params: list[ToolParam]
    output_type: str  # e.g. "search_emails_result"


@dataclass(frozen=True)
class GoapAction:
    """A single action in the GOAP action space."""

    name: str
    preconditions: dict[str, bool]
    effects: dict[str, bool]
    # "tool" | "bridge" | "create_state" | "run_agent"
    action_type: str = "tool"
    # The tool name to call (for action_type="tool")
    tool_name: str | None = None
    # Fixed arguments set by the planner (not by the LLM)
    fixed_args: dict[str, Any] = field(default_factory=dict)
    # Description for the untrusted LLM when filling parameters
    description: str = ""
    # Args the LLM generates (e.g. ["content"], ["body"])
    llm_args: list[str] = field(default_factory=list)
    # Prompt guiding the LLM's generation for llm_args
    llm_prompt: str = ""
    # Cost for A* (lower = preferred)
    cost: float = 1.0
    # For run_agent: the sub-goal description and planning prompt
    sub_goal: str = ""
    sub_prompt: str = ""


@dataclass
class WorldState:
    """Current state of the world as a set of boolean conditions."""

    conditions: dict[str, bool] = field(default_factory=dict)

    def satisfies(self, preconditions: dict[str, bool]) -> bool:
        """Check if this world state satisfies all preconditions."""
        return all(self.conditions.get(k) == v for k, v in preconditions.items())

    def apply(self, effects: dict[str, bool]) -> "WorldState":
        """Return a new world state with effects applied."""
        new_conditions = dict(self.conditions)
        new_conditions.update(effects)
        return WorldState(conditions=new_conditions)

    def unsatisfied_count(self, goal: dict[str, bool]) -> int:
        """Count how many goal conditions are not yet satisfied."""
        return sum(1 for k, v in goal.items() if self.conditions.get(k) != v)


@dataclass
class GoalState:
    """Goal conditions the planner tries to achieve."""

    conditions: dict[str, bool]
    # Human-readable description of what we're trying to do
    description: str = ""


@dataclass
class Plan:
    """A sequence of actions to reach the goal state."""

    actions: list[GoapAction]
    # The goal this plan achieves
    goal: GoalState | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.actions) == 0
