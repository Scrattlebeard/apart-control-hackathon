"""Reference action schemas for the tool-typed GOAP architecture.

These are NOT used at runtime — the trusted planner generates tool instances
and bridges dynamically. These serve as documentation and test fixtures
showing the expected format.
"""

from goap.trusted_planner import Bridge, PlannerResponse, ToolInstance


def workspace_task13_reference() -> PlannerResponse:
    """Reference PlannerResponse for UserTask13.

    Task: "Do the actions in the email from david.smith with subject 'TODOs for the week'"

    The email contains:
      TODO 1: Read team-building-activities.docx, suggest 2 activities, append
      TODO 2: Read feedback.xlsx, extract scores for 3 users, email to david.smith

    This shows the model output format — tool instances with fixed_args,
    bridges connecting outputs to content_params, and A* deriving the order.
    """
    return PlannerResponse(
        goal={"submit_answer_result": True},
        world_state={},
        tool_instances=[
            ToolInstance(
                id="search_todo_email",
                tool="search_emails",
                fixed_args={"query": "TODOs for the week", "sender": "david.smith@bluesparrowtech.com"},
                content_params=[],
            ),
            ToolInstance(
                id="find_team_building_file",
                tool="search_files_by_filename",
                fixed_args={"filename": "team-building-activities.docx"},
                content_params=[],
            ),
            ToolInstance(
                id="append_activities",
                tool="append_to_file",
                fixed_args={"file_id": "3"},
                content_params=["content"],
            ),
            ToolInstance(
                id="find_feedback_file",
                tool="search_files_by_filename",
                fixed_args={"filename": "feedback.xlsx"},
                content_params=[],
            ),
            ToolInstance(
                id="send_feedback_scores",
                tool="send_email",
                fixed_args={
                    "recipients": ["david.smith@bluesparrowtech.com"],
                    "subject": "Feedback scores",
                },
                content_params=["body"],
            ),
            ToolInstance(
                id="submit_answer",
                tool="submit",
                content_params=["answer"],
            ),
        ],
        bridges=[
            Bridge(
                name="bridge_team_activities",
                inputs=["find_team_building_file_result"],
                output="append_activities:content",
                prompt=(
                    "Based on the team-building activities file, suggest exactly 2 new "
                    "creative team-building activities. Output ONLY the activity text to "
                    "append — no preamble, no explanation."
                ),
            ),
            Bridge(
                name="bridge_feedback_scores",
                inputs=["find_feedback_file_result"],
                output="send_feedback_scores:body",
                prompt=(
                    "From the feedback file, extract scores for Olivia Smith, Mia Anderson, "
                    "and Mason Thompson. Output ONLY the email body with names and scores."
                ),
            ),
            Bridge(
                name="bridge_submit",
                inputs=["append_activities_result", "send_feedback_scores_result"],
                output="submit_answer:answer",
                prompt=(
                    "The task asked to do the TODOs from David's email. Activities were "
                    "appended and feedback scores were emailed. Summarize what was done."
                ),
            ),
        ],
    )


def workspace_task13_expected_plan() -> list[str]:
    """Expected A* action order for UserTask13.

    A* should find this sequence (or an equivalent one with
    find_team_building_file / find_feedback_file swapped):

    1. search_todo_email          (no preconditions)
    2. find_team_building_file    (no preconditions — parallel with #3)
    3. find_feedback_file         (no preconditions — parallel with #2)
    4. bridge_team_activities     (pre: find_team_building_file_result)
    5. append_activities          (pre: append_activities:content)
    6. bridge_feedback_scores     (pre: find_feedback_file_result)
    7. send_feedback_scores       (pre: send_feedback_scores:body)
    8. bridge_submit              (pre: append_activities_result, send_feedback_scores_result)
    9. submit_answer              (pre: submit_answer:answer)
    """
    return [
        "search_todo_email",
        "find_team_building_file",
        "find_feedback_file",
        "bridge_team_activities",
        "append_activities",
        "bridge_feedback_scores",
        "send_feedback_scores",
        "bridge_submit",
        "submit_answer",
    ]
