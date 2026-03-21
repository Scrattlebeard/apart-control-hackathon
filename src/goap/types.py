"""Type system for GOAP preconditions and effects.

Conditions are string keys mapped to boolean values. World state is a
dict of these conditions. Actions have preconditions (conditions that
must be true) and effects (conditions set after execution).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoapAction:
    """A single action in the GOAP action space."""

    name: str
    preconditions: dict[str, bool]
    effects: dict[str, bool]
    # The tool name to call (may differ from name for compound actions)
    tool_name: str | None = None
    # Fixed arguments set by the planner (not by the LLM)
    fixed_args: dict[str, Any] = field(default_factory=dict)
    # Description for the untrusted LLM when filling parameters
    description: str = ""
    # Cost for A* (lower = preferred)
    cost: float = 1.0


@dataclass
class WorldState:
    """Current state of the world as a set of boolean conditions."""

    conditions: dict[str, bool] = field(default_factory=dict)

    def satisfies(self, preconditions: dict[str, bool]) -> bool:
        """Check if this world state satisfies all preconditions."""
        return all(
            self.conditions.get(k) == v for k, v in preconditions.items()
        )

    def apply(self, effects: dict[str, bool]) -> "WorldState":
        """Return a new world state with effects applied."""
        new_conditions = dict(self.conditions)
        new_conditions.update(effects)
        return WorldState(conditions=new_conditions)

    def unsatisfied_count(self, goal: dict[str, bool]) -> int:
        """Count how many goal conditions are not yet satisfied."""
        return sum(
            1 for k, v in goal.items() if self.conditions.get(k) != v
        )


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
