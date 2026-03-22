"""GOAP planner — topological sort with A* fallback.

Primary strategy: build a dependency DAG from action preconditions/effects
and topologically sort it. This handles the common case (independent
parallel tracks feeding into bottlenecks) in linear time.

Fallback: A* search for cases where the DAG approach fails (shouldn't
happen with well-formed plans, but kept as a safety net).
"""

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field

from goap.types import GoalState, GoapAction, Plan, WorldState

logger = logging.getLogger(__name__)

# Safety bound on A* search iterations
MAX_ITERATIONS = 10_000


def topological_plan(
    start: WorldState,
    goal: GoalState,
    available_actions: list[GoapAction],
) -> Plan | None:
    """Order actions via dependency-aware topological sort.

    Builds a DAG: action A depends on action B if B produces a condition
    that A requires. Actions whose preconditions are already satisfied in
    the start state have no dependencies. Uses Kahn's algorithm (BFS).

    Returns None if the goal is unreachable or there's a cycle.
    """
    if start.satisfies(goal.conditions):
        return Plan(actions=[], goal=goal)

    # Map: condition → action that produces it
    producers: dict[str, GoapAction] = {}
    for action in available_actions:
        for cond in action.effects:
            producers[cond] = action

    # Check goal reachability
    for cond, val in goal.conditions.items():
        if not start.conditions.get(cond) == val and cond not in producers:
            logger.warning("Topo: goal condition '%s' has no producer", cond)
            return None

    # Build dependency edges: action → set of actions it depends on
    action_set = set(id(a) for a in available_actions)
    deps: dict[int, set[int]] = {id(a): set() for a in available_actions}
    rdeps: dict[int, set[int]] = {id(a): set() for a in available_actions}  # reverse

    for action in available_actions:
        for cond in action.preconditions:
            if start.conditions.get(cond) == action.preconditions[cond]:
                continue  # already satisfied
            producer = producers.get(cond)
            if producer is None:
                logger.warning("Topo: precondition '%s' of '%s' has no producer", cond, action.name)
                return None
            if id(producer) != id(action):
                deps[id(action)].add(id(producer))
                rdeps[id(producer)].add(id(action))

    # Kahn's algorithm
    id_to_action = {id(a): a for a in available_actions}
    in_degree = {aid: len(d) for aid, d in deps.items()}
    queue: deque[int] = deque(aid for aid, deg in in_degree.items() if deg == 0)
    ordered: list[GoapAction] = []

    while queue:
        aid = queue.popleft()
        ordered.append(id_to_action[aid])
        for dependent in rdeps.get(aid, set()):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(available_actions):
        cycle_actions = [id_to_action[aid].name for aid, deg in in_degree.items() if deg > 0]
        logger.warning("Topo: cycle detected involving %s — falling back to A*", cycle_actions)
        return None

    # Verify the ordering actually reaches the goal
    state = WorldState(conditions=dict(start.conditions))
    for action in ordered:
        if not state.satisfies(action.preconditions):
            logger.warning("Topo: ordering broken at '%s' — falling back to A*", action.name)
            return None
        state = state.apply(action.effects)

    if not state.satisfies(goal.conditions):
        logger.warning("Topo: ordered plan doesn't reach goal — falling back to A*")
        return None

    logger.info(
        "Topo plan: %d actions — %s",
        len(ordered),
        " → ".join(a.name for a in ordered),
    )
    return Plan(actions=ordered, goal=goal)


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
    """Find the cheapest plan from start to goal.

    Tries topological sort first (O(V+E), handles the common case of
    independent parallel tracks). Falls back to A* search if topo sort
    fails (cycles, broken ordering).

    Returns None if no plan exists within the iteration budget.
    """
    # Fast path: topological sort
    topo_result = topological_plan(start, goal, available_actions)
    if topo_result is not None:
        return topo_result

    logger.info(
        "A* search started (topo fallback) — %d available actions, goal conditions: %s",
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
