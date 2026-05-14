from __future__ import annotations

import re
from typing import Any

from customer_service_agent.reasoning import ACTION_TOOL_NAMES, ToolPlan


def apply_pending_intent_to_plan(
    plan,
    user_message: str,
    state_snapshot: dict[str, Any],
):
    if plan is None:
        plan = empty_tool_plan()

    pending_intent = pending_intent_from_context(plan, user_message)
    if not pending_intent:
        return plan

    active_order_id = state_snapshot.get("active_order_id")
    tool_results = state_snapshot.get("tool_results", {})
    order = tool_results.get("order")
    plan.pending_intent = pending_intent
    if hasattr(plan, "pending_action"):
        plan.pending_action = None
    strip_planned_mutation_actions(plan)

    if pending_intent in {"refund", "cancel"}:
        order_id = first_not_none(
            int_from_message(user_message),
            plan.order_id,
            state_snapshot.get("pending_order_id"),
            active_order_id,
        )
        if order:
            order_id = order["order_id"]
        plan.pending_order_id = order_id
        if plan.order_id is None:
            plan.order_id = order_id
        if not plan.tool_calls and not order and order_id is not None:
            call = {
                "name": "order_lookup",
                "args": {"order_id": order_id},
                "id": "pending-order-lookup",
            }
            plan.tool_calls = [call]
            plan.requested_actions = []
            plan.steps = [call["name"]]
            plan.reasoning = (
                f"Resolved pending intent {pending_intent} into order_lookup before transaction verification."
            )
        if not plan.tool_calls:
            plan.requires_follow_up = False
            plan.follow_up_question = None
        return plan

    if pending_intent == "memory_write":
        key = plan.memory_key
        value = plan.memory_value
        if not key or not value:
            key, value = memory_preference(user_message)
        plan.memory_key = key
        plan.memory_value = value
        plan.pending_order_id = None
        if not plan.tool_calls:
            plan.requires_follow_up = False
            plan.follow_up_question = None
            plan.reasoning = "Resolved pending intent memory_write for verifier continuation."
        return plan

    if pending_intent == "complaint":
        order_id = first_not_none(
            int_from_message(user_message),
            plan.order_id,
            state_snapshot.get("pending_order_id"),
            active_order_id,
        )
        plan.pending_order_id = order_id
        if plan.order_id is None:
            plan.order_id = order_id
        if not plan.issue:
            plan.issue = complaint_issue(user_message)
        if not plan.tool_calls:
            plan.requires_follow_up = False
            plan.follow_up_question = None
            plan.reasoning = "Resolved pending intent complaint for verifier continuation."
        return plan

    return plan


def empty_tool_plan() -> ToolPlan:
    return ToolPlan(
        requires_follow_up=True,
        follow_up_question="I need a little more detail to help with that.",
    )


def pending_intent_from_context(plan, user_message: str) -> str | None:
    if plan.pending_intent in {"refund", "cancel", "complaint", "memory_write"}:
        return str(plan.pending_intent)

    action_names = {action.get("name") for action in plan.requested_actions}
    tool_names = {call.get("name") for call in plan.tool_calls}
    requested_mutation = requested_order_mutation(user_message)
    if "request_refund" in action_names or requested_mutation == "refund":
        return "refund"
    if "request_cancel_order" in action_names or requested_mutation == "cancel":
        return "cancel"
    if "request_write_memory" in action_names or requested_memory_write(user_message):
        return "memory_write"
    if (
        "request_log_complaint" in action_names
        or "request_log_complaint" in tool_names
        or requested_complaint(user_message)
    ):
        return "complaint"
    return None


def strip_planned_mutation_actions(plan) -> None:
    plan.tool_calls = [
        call for call in plan.tool_calls if call.get("name") not in ACTION_TOOL_NAMES
    ]
    plan.requested_actions = []
    plan.steps = [str(call.get("name") or "") for call in plan.tool_calls]


def requested_order_mutation(message: str) -> str | None:
    normalized = message.lower()
    if "refund" in normalized and (
        "order" in normalized
        or "refund it" in normalized
        or normalized.strip().startswith("refund")
    ):
        return "refund"
    if "cancel" in normalized or "cancellation" in normalized:
        return "cancel"
    return None


def requested_memory_write(message: str) -> bool:
    normalized = message.lower()
    return "remember" in normalized or "preference" in normalized


def requested_complaint(message: str) -> bool:
    normalized = message.lower()
    complaint_words = {"complain", "complaint", "late", "damaged", "broken"}
    return any(word in normalized for word in complaint_words)


def memory_preference(message: str) -> tuple[str | None, str | None]:
    normalized = message.lower()
    if "refund" in normalized:
        return "refund_preference", "prefers refunds"
    if "email" in normalized:
        return "contact_preference", "prefers email"
    if "remember" in normalized:
        value = re.sub(r"^\s*remember\s+", "", message, flags=re.IGNORECASE).strip()
        if value:
            return "customer_preference", value
    return None, None


def complaint_issue(message: str) -> str:
    normalized = message.lower()
    if "late" in normalized:
        return "Order is late again" if "again" in normalized else "Order is late"
    if "damaged" in normalized:
        return "Package damaged"
    if "broken" in normalized:
        return "Item is broken"
    return "customer requested to file a complaint"


def int_from_message(message: str) -> int | None:
    match = re.search(r"\b\d+\b", message)
    if not match:
        return None
    return int(match.group(0))


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None
