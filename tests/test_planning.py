from __future__ import annotations

import pytest

from customer_service_agent.graph.planning import split_tool_calls
from customer_service_agent.graph.policy import verify_complaint, verify_memory_write
from customer_service_agent.reasoning import ToolPlan


def test_split_tool_calls_rejects_missing_plan() -> None:
    with pytest.raises(TypeError, match="must return ToolPlan"):
        split_tool_calls(None)  # type: ignore[arg-type]


def test_continue_after_read_string_is_not_normalized() -> None:
    plan = ToolPlan(
        tool_calls=[
            {
                "name": "order_lookup",
                "args": {"order_id": 7890, "continue_after_read": "true"},
                "id": "lookup-1",
            }
        ]
    )

    result = split_tool_calls(plan)

    assert result.continue_after_read is False
    assert result.tool_calls == [
        {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
    ]


def test_verifier_preserves_complaint_action_id() -> None:
    result = verify_complaint(
        [
            {
                "name": "propose_log_complaint",
                "args": {"order_id": 7890, "issue": "package damaged"},
                "id": "complaint-1",
            }
        ],
        order_id=7890,
        authenticated_customer_id=7,
        tool_results={},
    )

    assert result["requested_actions"][0]["id"] == "complaint-1"


def test_verifier_preserves_memory_action_id() -> None:
    result = verify_memory_write(
        [
            {
                "name": "propose_write_memory",
                "args": {
                    "key": "contact_preference",
                    "value": "prefers email",
                    "memory_type": "preference",
                },
                "id": "memory-1",
            }
        ],
        authenticated_customer_id=7,
        tool_results={},
    )

    assert result["requested_actions"][0]["id"] == "memory-1"
