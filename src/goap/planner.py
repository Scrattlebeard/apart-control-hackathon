"""A* GOAP planner.

Given a world state, a goal state, and a set of actions, finds the
cheapest action sequence that transforms the world state into one
satisfying the goal. The heuristic is the count of unsatisfied goal
conditions (admissible — never overestimates).
"""

import heapq
import logging
from dataclasses import dataclass, field

from goap.types import GoalState, GoapAction, Plan, WorldState

logger = logging.getLogger(__name__)

# Safety bound on search iterations
MAX_ITERATIONS = 10_000


@dataclass(order=True)
class SearchNode:
    f_score: float
    # Tiebreaker: prefer nodes explored later (deeper in the graph)
    counter: int = field(compare=True)
    # Not compared — carried for state tracking
    g_score: float = field(compare=False)
    state: WorldState = field(compare=False)
    actions: list[GoapAction] = field(compare=False)


def a_star_plan(
    start: WorldState,
    goal: GoalState,
    available_actions: list[GoapAction],
    max_iterations: int = MAX_ITERATIONS,
) -> Plan | None:
    """Find the cheapest plan from start to goal using A* search.

    Returns None if no plan exists within the iteration budget.
    """
    logger.info(
        "A* search started — %d available actions, goal conditions: %s",
        len(available_actions),
        dict(goal.conditions),
    )

    if start.satisfies(goal.conditions):
        logger.info("Goal already satisfied, returning empty plan")
        return Plan(actions=[], goal=goal)

    counter = 0
    h = start.unsatisfied_count(goal.conditions)
    open_set: list[SearchNode] = [
        SearchNode(f_score=h, counter=counter, g_score=0.0, state=start, actions=[])
    ]
    # Track visited states to avoid cycles
    visited: set[frozenset[tuple[str, bool]]] = set()

    iterations = 0
    while open_set and iterations < max_iterations:
        iterations += 1
        node = heapq.heappop(open_set)

        state_key = frozenset(node.state.conditions.items())
        if state_key in visited:
            continue
        visited.add(state_key)

        for action in available_actions:
            if not node.state.satisfies(action.preconditions):
                continue

            new_state = node.state.apply(action.effects)
            new_state_key = frozenset(new_state.conditions.items())
            if new_state_key in visited:
                continue

            new_actions = [*node.actions, action]
            new_g = node.g_score + action.cost

            if new_state.satisfies(goal.conditions):
                action_seq = " → ".join(a.name for a in new_actions)
                logger.info(
                    "Plan found in %d iterations, %d actions: %s",
                    iterations,
                    len(new_actions),
                    action_seq,
                )
                return Plan(actions=new_actions, goal=goal)

            new_h = new_state.unsatisfied_count(goal.conditions)
            counter += 1
            heapq.heappush(
                open_set,
                SearchNode(
                    f_score=new_g + new_h,
                    counter=counter,
                    g_score=new_g,
                    state=new_state,
                    actions=new_actions,
                ),
            )

    logger.warning(
        "No plan found after %d iterations (visited %d states)",
        iterations,
        len(visited),
    )
    return None


def diagnose_failure(
    start: WorldState,
    goal: GoalState,
    available_actions: list[GoapAction],
) -> set[str]:
    """Find preconditions that are unreachable from the start state.

    Computes the forward-reachable state (all conditions that *could*
    become true by chaining actions from the start), then returns goal
    and precondition keys that never become reachable. Used by the
    fallback loop to tell the trusted model what bridges are missing.
    """
    # Forward-reachable: keep applying any applicable action until fixed point
    reachable = dict(start.conditions)
    changed = True
    while changed:
        changed = False
        for action in available_actions:
            # Check if this action's preconditions are all reachable
            if all(reachable.get(k) == v for k, v in action.preconditions.items()):
                for k, v in action.effects.items():
                    if reachable.get(k) != v:
                        reachable[k] = v
                        changed = True

    # Which goal conditions are unreachable?
    unreachable_goals = {
        k for k, v in goal.conditions.items() if reachable.get(k) != v
    }

    # Which preconditions across all actions are unreachable?
    unreachable_preconds: set[str] = set()
    for action in available_actions:
        for k, v in action.preconditions.items():
            if reachable.get(k) != v:
                unreachable_preconds.add(k)

    result = unreachable_goals | unreachable_preconds
    if result:
        logger.info("Unreachable conditions: %s", result)
    return result
