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
    plan.order_id = first_not_none(
        plan.order_id,
        first_int_arg(plan.tool_calls, "order_id"),
        first_int_arg(plan.requested_actions, "order_id"),
    )
    plan.order_reference = infer_order_reference(plan)
    if plan.order_reference.order_id is not None and plan.order_id is None:
        plan.order_id = plan.order_reference.order_id

    plan.issue = plan.issue or first_str_arg(plan.requested_actions, "issue") or first_str_arg(
        plan.tool_calls, "issue"
    )
    plan.memory_candidate = normalize_memory_candidate(plan)
    if plan.memory_candidate.should_write:
        plan.memory_key = plan.memory_key or plan.memory_candidate.key
        plan.memory_value = plan.memory_value or plan.memory_candidate.value

    has_control_replan = any(call.get("name") in CONTROL_TOOL_NAMES for call in plan.tool_calls)
    plan.requested_actions = normalize_requested_actions(plan)
    plan.tool_calls = normalize_read_tools(plan)
    plan.requires_replan_after_tools = plan.requires_replan_after_tools or has_control_replan
    plan.missing_slots = missing_slots_for_plan(plan)

    if plan.needs_user_clarification and not plan.missing_slots:
        plan.missing_slots.append("user_clarification")

    return plan


def empty_tool_plan() -> ToolPlan:
    return ToolPlan(needs_user_clarification=True, missing_slots=["planner_action"])


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
        if name in {"request_refund", "request_cancel_order", "request_log_complaint"}:
            if args.get("order_id") is None and plan.order_reference.order_id is not None:
                args["order_id"] = plan.order_reference.order_id
        if name == "request_log_complaint":
            if args.get("customer_id") is None and plan.customer_id is not None:
                args["customer_id"] = plan.customer_id
            if not args.get("issue") and plan.issue:
                args["issue"] = plan.issue
        if name == "request_write_memory":
            if args.get("customer_id") is None and plan.customer_id is not None:
                args["customer_id"] = plan.customer_id
            if not args.get("key") and plan.memory_key:
                args["key"] = plan.memory_key
            if not args.get("value") and plan.memory_value:
                args["value"] = plan.memory_value
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
        plan.order_id,
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
    if candidate.should_write or candidate.memory_type != "unclear":
        return candidate
    memory_type = first_str_arg(plan.requested_actions, "memory_type") or first_str_arg(
        plan.tool_calls, "memory_type"
    )
    if memory_type not in MEMORY_TYPES:
        return candidate
    key = plan.memory_key or first_str_arg(plan.requested_actions, "key") or first_str_arg(
        plan.tool_calls, "key"
    )
    value = plan.memory_value or first_str_arg(plan.requested_actions, "value") or first_str_arg(
        plan.tool_calls, "value"
    )
    return MemoryWriteCandidate(
        should_write=memory_type in WRITABLE_MEMORY_TYPES and bool(key and value),
        memory_type=memory_type,
        key=key,
        value=value,
    )


def missing_slots_for_plan(plan: ToolPlan) -> list[str]:
    missing: list[str] = list(plan.missing_slots)
    action_names = {action.get("name") for action in plan.requested_actions}
    if action_names & {"request_refund", "request_cancel_order"}:
        if plan.customer_id is None:
            missing.append("customer_id")
        if plan.order_reference.order_id is None or plan.order_reference.confidence != "high":
            missing.append("order_id")
        missing.extend(["order_status", "order_customer_id"])
    if "request_log_complaint" in action_names:
        if plan.customer_id is None:
            missing.append("customer_id")
        if plan.order_reference.order_id is None or plan.order_reference.confidence != "high":
            missing.append("order_id")
        if not plan.issue:
            missing.append("complaint_issue")
    if "request_write_memory" in action_names:
        if plan.customer_id is None:
            missing.append("customer_id")
        if not plan.memory_candidate.should_write:
            missing.append("long_term_write_allowed")
        if not plan.memory_key:
            missing.append("memory_key")
        if not plan.memory_value:
            missing.append("memory_value")
    return dedupe(missing)


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


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
