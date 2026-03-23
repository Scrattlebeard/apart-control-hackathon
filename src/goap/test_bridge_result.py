"""Tests for bridge {name}_result effect — ensures bridges can be depended on by name."""

from goap.planner import a_star_plan
from goap.trusted_planner import (
    PlannerResponse,
    PlannedAction,
    build_action_space,
    _normalize_bridges,
)
from goap.types import GoalState, GoapAction, ToolTemplate, ToolParam, WorldState


def _make_catalog() -> list[ToolTemplate]:
    """Minimal catalog with a send_email tool."""
    return [
        ToolTemplate(
            name="send_email",
            description="Send an email",
            params=[
                ToolParam(name="recipients", type="array<string>", required=True, description="To"),
                ToolParam(name="body", type="string", required=True, description="Body"),
            ],
            output_type="send_email_result",
        ),
        ToolTemplate(
            name="get_emails",
            description="Fetch emails",
            params=[],
            output_type="get_emails_result",
        ),
        ToolTemplate(
            name="submit",
            description="Submit answer",
            params=[ToolParam(name="answer", type="string", required=True, description="Answer")],
            output_type="submit_result",
        ),
    ]


def test_bridge_produces_name_result_effect():
    """build_action_space should give bridges a {name}_result effect."""
    response = PlannerResponse(actions=[
        PlannedAction(
            id="analyze",
            inputs=["get_emails_result"],
            outputs=["send_email:body"],
            prompt="Draft a reply.",
        ),
    ])
    catalog = _make_catalog()
    actions = build_action_space(response, catalog)

    bridge = next(a for a in actions if a.name == "analyze")
    assert "analyze_result" in bridge.effects
    assert "send_email:body" in bridge.effects


def test_normalize_resolves_bare_bridge_name():
    """_normalize_bridges should map a bare bridge name to {name}_result."""
    response = PlannerResponse(actions=[
        PlannedAction(id="get_emails", tool="get_emails", fixed_args={}),
        PlannedAction(
            id="analyze",
            inputs=["get_emails_result"],
            outputs=["send_email:body"],
            prompt="Analyze emails.",
        ),
        # Iterate action references bridge by bare name
        PlannedAction(
            id="handle_alerts",
            inputs=["analyze"],
            prompt="Handle each alert.",
            iterate=True,
        ),
    ])

    _normalize_bridges(response)

    handle = next(a for a in response.actions if a.id == "handle_alerts")
    assert handle.inputs == ["analyze_result"], (
        f"Expected ['analyze_result'], got {handle.inputs}"
    )


def test_a_star_succeeds_with_iterate_depending_on_bridge():
    """End-to-end: iterate action depending on bridge by name should plan successfully."""
    response = PlannerResponse(actions=[
        PlannedAction(id="get_emails", tool="get_emails", fixed_args={}),
        PlannedAction(
            id="analyze",
            inputs=["get_emails_result"],
            outputs=["send_email:body"],
            prompt="Analyze emails.",
        ),
        PlannedAction(
            id="handle_alerts",
            inputs=["analyze"],
            prompt="Handle each alert.",
            iterate=True,
        ),
        PlannedAction(id="send_email", tool="send_email", fixed_args={"recipients": ["a@b.com"]}, content_params=["body"]),
        PlannedAction(
            id="bridge_submit",
            inputs=["send_email_result", "handle_alerts_result"],
            outputs=["goap_submit:answer"],
            prompt="Summarize.",
        ),
        PlannedAction(id="goap_submit", tool="submit", content_params=["answer"]),
    ])

    _normalize_bridges(response)
    catalog = _make_catalog()
    actions = build_action_space(response, catalog)

    start = WorldState(conditions={})
    goal = GoalState(conditions={"goap_submit_result": True})
    plan = a_star_plan(start, goal, actions)

    assert plan is not None, "A* should find a plan"
    names = [a.name for a in plan.actions]
    assert "get_emails" in names
    assert "analyze" in names
    assert "handle_alerts" in names
    assert "goap_submit" in names


def test_explicit_outputs_still_work():
    """Bridges with explicit outputs should still wire into content_params."""
    response = PlannerResponse(actions=[
        PlannedAction(id="get_emails", tool="get_emails", fixed_args={}),
        PlannedAction(
            id="draft_reply",
            inputs=["get_emails_result"],
            outputs=["send_email:body"],
            prompt="Draft reply.",
        ),
        PlannedAction(id="send_email", tool="send_email", fixed_args={"recipients": ["a@b.com"]}, content_params=["body"]),
        PlannedAction(
            id="bridge_submit",
            inputs=["send_email_result"],
            outputs=["goap_submit:answer"],
            prompt="Summarize.",
        ),
        PlannedAction(id="goap_submit", tool="submit", content_params=["answer"]),
    ])

    _normalize_bridges(response)
    catalog = _make_catalog()
    actions = build_action_space(response, catalog)

    start = WorldState(conditions={})
    goal = GoalState(conditions={"goap_submit_result": True})
    plan = a_star_plan(start, goal, actions)

    assert plan is not None, "A* should find a plan with explicit bridge outputs"
    names = [a.name for a in plan.actions]
    # draft_reply must come before send_email (produces send_email:body precondition)
    assert names.index("draft_reply") < names.index("send_email")
