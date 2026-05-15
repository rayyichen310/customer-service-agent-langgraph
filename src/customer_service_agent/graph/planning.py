from __future__ import annotations

from typing import Any

from customer_service_agent.graph.tools import ACTION_TOOL_NAMES, CONTROL_TOOL_NAMES
from customer_service_agent.models import (
    MEMORY_TYPES,
    WRITABLE_MEMORY_TYPES,
    MemoryWriteCandidate,
    OrderReference,
)
from customer_service_agent.reasoning import ToolPlan


def prepare_plan(
    plan,
    state_snapshot: dict[str, Any],
):
    if plan is None:
        plan = empty_tool_plan()

    plan.customer_id = first_not_none(
        plan.customer_id,
        first_int_arg(plan.tool_calls, "customer_id"),
        first_int_arg(plan.requested_actions, "customer_id"),
        state_snapshot.get("active_customer_id"),
    )
    plan.order_reference = infer_order_reference(plan)

    plan.issue = plan.issue or first_str_arg(plan.requested_actions, "issue") or first_str_arg(
        plan.tool_calls, "issue"
    )
    plan.memory_candidate = normalize_memory_candidate(plan)

    plan.requested_actions = normalize_requested_actions(plan)
    plan.tool_calls = normalize_read_tools(plan)

    return plan


def empty_tool_plan() -> ToolPlan:
    return ToolPlan()


def normalize_requested_actions(
    plan: ToolPlan,
) -> list[dict[str, Any]]:
    raw_actions = [
        *plan.requested_actions,
        *[call for call in plan.tool_calls if call.get("name") in ACTION_TOOL_NAMES],
    ]
    actions_by_name = {str(action.get("name")): dict(action) for action in raw_actions}

    normalized: list[dict[str, Any]] = []
    for action in actions_by_name.values():
        name = str(action.get("name") or "")
        args = dict(action.get("args") or {})
        if name in {"propose_refund", "propose_cancel_order", "propose_log_complaint"}:
            if args.get("order_id") is None and plan.order_reference.order_id is not None:
                args["order_id"] = plan.order_reference.order_id
        if name == "propose_log_complaint":
            if args.get("customer_id") is None and plan.customer_id is not None:
                args["customer_id"] = plan.customer_id
            if not args.get("issue") and plan.issue:
                args["issue"] = plan.issue
        if name == "propose_write_memory":
            if args.get("customer_id") is None and plan.customer_id is not None:
                args["customer_id"] = plan.customer_id
            if not args.get("key") and plan.memory_candidate.key:
                args["key"] = plan.memory_candidate.key
            if not args.get("value") and plan.memory_candidate.value:
                args["value"] = plan.memory_candidate.value
        normalized.append({"name": name, "args": args, "id": action.get("id")})
    return normalized


def normalize_read_tools(plan: ToolPlan) -> list[dict[str, Any]]:
    read_tools = [
        call
        for call in plan.tool_calls
        if call.get("name") not in ACTION_TOOL_NAMES
        and call.get("name") not in CONTROL_TOOL_NAMES
    ]
    for call in read_tools:
        args = call.setdefault("args", {})
        if (
            call.get("name") == "order_lookup"
            and args.get("order_id") is None
            and plan.order_reference.order_id is not None
        ):
            args["order_id"] = plan.order_reference.order_id
    return read_tools


def infer_order_reference(plan: ToolPlan) -> OrderReference:
    if plan.order_reference.order_id is not None and plan.order_reference.confidence == "high":
        return plan.order_reference

    explicit_order_id = first_not_none(
        first_int_arg(plan.tool_calls, "order_id"),
        first_int_arg(plan.requested_actions, "order_id"),
    )
    if explicit_order_id is not None:
        return OrderReference(order_id=explicit_order_id, source="explicit", confidence="high")

    if plan.order_reference.order_id is not None:
        return plan.order_reference

    return OrderReference(source="none", confidence="low")


def normalize_memory_candidate(plan: ToolPlan) -> MemoryWriteCandidate:
    candidate = plan.memory_candidate
    memory_type = (
        candidate.memory_type
        if candidate.memory_type != "unclear"
        else first_str_arg(plan.requested_actions, "memory_type")
        or first_str_arg(plan.tool_calls, "memory_type")
        or "unclear"
    )
    if memory_type not in MEMORY_TYPES:
        memory_type = "unclear"
    key = (
        candidate.key
        or first_str_arg(plan.requested_actions, "key")
        or first_str_arg(plan.tool_calls, "key")
    )
    value = (
        candidate.value
        or first_str_arg(plan.requested_actions, "value")
        or first_str_arg(plan.tool_calls, "value")
    )
    return MemoryWriteCandidate(
        should_write=(
            candidate.should_write or memory_type in WRITABLE_MEMORY_TYPES and bool(key and value)
        ),
        memory_type=memory_type,
        key=key,
        value=value,
        reason_code=candidate.reason_code,
    )


def first_int_arg(calls: list[dict[str, Any]], key: str) -> int | None:
    for call in calls:
        value = int_or_none((call.get("args") or {}).get(key))
        if value is not None:
            return value
    return None


def first_str_arg(calls: list[dict[str, Any]], key: str) -> str | None:
    for call in calls:
        value = (call.get("args") or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None
