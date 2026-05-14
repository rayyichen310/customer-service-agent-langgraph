from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from customer_service_agent.models import MemoryWriteCandidate, OrderReference, VerifierOutput


@dataclass(frozen=True)
class OrderMutationRule:
    action_name: str
    allowed_status: str | None = None
    blocked_statuses: tuple[str, ...] = ()

    def validate_order(self, order: dict[str, Any]) -> dict[str, Any] | None:
        status = order["status"]
        if self.allowed_status is not None and status != self.allowed_status:
            return order_policy_error(self.action_name, order)
        if status in self.blocked_statuses:
            return order_policy_error(self.action_name, order)
        return None


ORDER_MUTATION_RULES = {
    "request_refund": OrderMutationRule(
        action_name="request_refund",
        allowed_status="delivered",
    ),
    "request_cancel_order": OrderMutationRule(
        action_name="request_cancel_order",
        blocked_statuses=("delivered", "refund_requested"),
    ),
}

ORDER_ACTION_NAMES = set(ORDER_MUTATION_RULES)


def verify_policy(state: dict[str, Any], *, max_react_iterations: int) -> dict[str, Any]:
    requested_actions = list(state.get("requested_actions", []))
    action_names = {action.get("name") for action in requested_actions}
    tool_results = dict(state.get("tool_results", {}))
    order = tool_results.get("order")
    order_reference = OrderReference(**(state.get("order_reference") or {}))
    active_customer_id = state.get("active_customer_id")
    called_order_lookup = any(
        call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
    )
    can_replan = int(state.get("react_iterations") or 0) < int(
        state.get("max_react_iterations") or max_react_iterations
    )

    if action_names & ORDER_ACTION_NAMES:
        return verify_order_mutation(
            requested_actions,
            order=order,
            order_reference=order_reference,
            active_customer_id=active_customer_id,
            called_order_lookup=called_order_lookup,
            can_replan=can_replan,
            tool_results=tool_results,
        )

    if "request_log_complaint" in action_names:
        return verify_complaint(
            requested_actions,
            planned_issue=state.get("issue"),
            order_reference=order_reference,
            active_customer_id=active_customer_id,
            tool_results=tool_results,
        )

    if "request_write_memory" in action_names:
        return verify_memory_write(
            requested_actions,
            state=state,
            tool_results=tool_results,
        )

    called_customer_profile = any(
        call.get("name") == "customer_profile" for call in state.get("tool_calls", [])
    )
    if called_customer_profile and not tool_results.get("customer"):
        return _decision(
            "block",
            ["customer_id"],
            [],
            [],
            "Customer profile was not found.",
            tool_results,
        )

    if state.get("follow_up_question"):
        return _decision(
            "ask_user",
            state.get("missing_slots", []),
            [],
            [],
            state["follow_up_question"],
            tool_results,
        )

    return _decision("proceed_to_response", [], [], [], None, tool_results)


def verify_order_mutation(
    requested_actions: list[dict[str, Any]],
    *,
    order: dict[str, Any] | None,
    order_reference: OrderReference,
    active_customer_id: int | None,
    called_order_lookup: bool,
    can_replan: bool,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    action_names = {action.get("name") for action in requested_actions}
    missing_slots: list[str] = []
    if active_customer_id is None:
        missing_slots.append("customer_id")
    if order_reference.order_id is None or order_reference.confidence != "high":
        missing_slots.append("order_id")
    if not order:
        missing_slots.extend(["order_status", "order_customer_id"])

    if "order_id" in missing_slots:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            list(action_names),
            "Which order do you mean?",
            tool_results,
            planner_feedback="The user's order reference is ambiguous.",
        )

    if active_customer_id is None:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            list(action_names),
            "I need a customer ID before changing an order.",
            tool_results,
        )

    if not order:
        if called_order_lookup:
            return _decision(
            "block",
            missing_slots,
            [],
            list(action_names),
            None,
            tool_results,
            policy_errors=[
                policy_error(
                    "ORDER_NOT_FOUND",
                    blocked_action=first_action_name(action_names),
                    order_id=order_reference.order_id,
                    reason="order not found",
                )
            ],
        )
        if can_replan:
            return _decision(
                "replan",
                missing_slots,
                [],
                list(action_names),
                None,
                tool_results,
                planner_feedback="Need order lookup before mutation.",
            )
        return _decision(
            "ask_user",
            missing_slots,
            [],
            list(action_names),
            "I need verified order details before changing that order.",
            tool_results,
            planner_feedback="Replan limit reached before required order details were available.",
        )

    if order["customer_id"] != active_customer_id:
        return _decision(
            "block",
            ["order_customer_id"],
            [],
            list(action_names),
            None,
            tool_results,
            policy_errors=[
                policy_error(
                    "ORDER_CUSTOMER_MISMATCH",
                    blocked_action=first_action_name(action_names),
                    order_id=order["order_id"],
                    customer_id=active_customer_id,
                    reason="order belongs to a different customer",
                )
            ],
        )

    policy_errors = verify_action_policy(action_names, order)
    if policy_errors:
        return _decision(
            "block",
            [],
            [],
            list(action_names),
            None,
            tool_results,
            policy_errors=policy_errors,
        )

    safe_actions = [
        normalize_action(action, order_id=order["order_id"], customer_id=active_customer_id)
        for action in requested_actions
        if action.get("name") in ORDER_ACTION_NAMES
    ]
    return _decision(
        "proceed_to_action",
        [],
        [action["name"] for action in safe_actions],
        [],
        None,
        tool_results,
        requested_actions=safe_actions,
        reasoning="Verified order ownership and status before approving requested action.",
    )


def verify_complaint(
    requested_actions: list[dict[str, Any]],
    *,
    planned_issue: str | None,
    order_reference: OrderReference,
    active_customer_id: int | None,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    issue = first_action_arg(requested_actions, "issue") or planned_issue
    missing_slots: list[str] = []
    if active_customer_id is None:
        missing_slots.append("customer_id")
    if order_reference.order_id is None or order_reference.confidence != "high":
        missing_slots.append("order_id")
    if not issue:
        missing_slots.append("complaint_issue")

    if "order_id" in missing_slots:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            ["request_log_complaint"],
            "Which order would you like to report an issue for?",
            tool_results,
            planner_feedback="The user's order reference is ambiguous.",
        )
    if "complaint_issue" in missing_slots:
        order_phrase = (
            f" for order {order_reference.order_id}" if order_reference.order_id is not None else ""
        )
        return _decision(
            "ask_user",
            missing_slots,
            [],
            ["request_log_complaint"],
            f"What issue would you like to report{order_phrase}?",
            tool_results,
            planner_feedback=(
                "The user wants to complain about an order, but did not provide the issue details."
            ),
        )
    if active_customer_id is None:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            ["request_log_complaint"],
            "I need a customer ID before logging that complaint.",
            tool_results,
        )

    action = {
        "name": "request_log_complaint",
        "args": {
            "customer_id": active_customer_id,
            "order_id": order_reference.order_id,
            "issue": issue,
        },
        "id": "verified-request_log_complaint",
    }
    return _decision(
        "proceed_to_action",
        [],
        ["request_log_complaint"],
        [],
        None,
        tool_results,
        requested_actions=[action],
        reasoning="Verified complaint order reference and issue before approving requested action.",
    )


def verify_memory_write(
    requested_actions: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    candidate = MemoryWriteCandidate(**(state.get("memory_candidate") or {}))
    active_customer_id = state.get("active_customer_id")
    if not candidate.should_write:
        reason = (
            "I can remember durable preferences, but this sounds like an order issue or transaction request."
            if candidate.memory_type in {"temporary_issue", "transaction_request"}
            else "What would you like me to remember as a long-term preference or profile note?"
        )
        return _decision(
            "ask_user",
            ["long_term_write_allowed"],
            [],
            ["request_write_memory"],
            reason,
            tool_results,
            planner_feedback=candidate.reason,
        )

    if active_customer_id is None:
        return _decision(
            "ask_user",
            ["customer_id"],
            [],
            ["request_write_memory"],
            "I need a customer ID before storing that preference.",
            tool_results,
        )

    key = state.get("memory_key") or candidate.key
    value = state.get("memory_value") or candidate.value
    if not key or not value:
        return _decision(
            "ask_user",
            ["memory_candidate"],
            [],
            ["request_write_memory"],
            "What would you like me to remember?",
            tool_results,
            planner_feedback="A durable memory write needs a key and value.",
        )

    action = {
        "name": "request_write_memory",
        "args": {"customer_id": active_customer_id, "key": key, "value": value},
        "id": "verified-request_write_memory",
    }
    return _decision(
        "proceed_to_action",
        [],
        ["request_write_memory"],
        [],
        None,
        tool_results,
        requested_actions=[action],
        reasoning="Verified durable memory candidate before approving memory write.",
    )


def verify_action_policy(action_names: set[str], order: dict[str, Any] | None) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not order:
        return errors

    for rule in ORDER_MUTATION_RULES.values():
        if rule.action_name in action_names:
            error = rule.validate_order(order)
            if error:
                errors.append(error)
    return errors


def normalize_action(
    action: dict[str, Any],
    *,
    order_id: int | None,
    customer_id: int | None,
) -> dict[str, Any]:
    args = dict(action.get("args") or {})
    if action.get("name") in ORDER_ACTION_NAMES:
        args["order_id"] = order_id
    if action.get("name") == "request_log_complaint":
        args["customer_id"] = customer_id
    return {"name": action.get("name"), "args": args, "id": action.get("id")}


def first_action_arg(actions: list[dict[str, Any]], key: str) -> str | None:
    for action in actions:
        value = (action.get("args") or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return None


def order_policy_error(action_name: str, order: dict[str, Any]) -> dict[str, Any]:
    if action_name == "request_cancel_order":
        return policy_error(
            "ORDER_NOT_CANCELLABLE",
            blocked_action=action_name,
            order_id=order["order_id"],
            current_status=order["status"],
            reason=status_reason(order["status"]),
        )
    return policy_error(
        "ORDER_NOT_REFUNDABLE",
        blocked_action=action_name,
        order_id=order["order_id"],
        current_status=order["status"],
        reason=status_reason(order["status"]),
    )


def policy_error(
    error_code: str,
    *,
    blocked_action: str | None = None,
    order_id: int | None = None,
    customer_id: int | None = None,
    current_status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "blocked_action": blocked_action,
        "order_id": order_id,
        "customer_id": customer_id,
        "current_status": current_status,
        "reason": reason,
    }


def status_reason(status: str) -> str:
    if status == "refund_requested":
        return "refund already requested"
    if status == "cancel_requested":
        return "cancellation already requested"
    if status == "delivered":
        return "order already delivered"
    return f"current status is {status}"


def first_action_name(action_names: set[str]) -> str | None:
    return sorted(action_names)[0] if action_names else None


def prefer_explicit_int(value: Any, fallback: int | None) -> int | None:
    parsed = int_or_none(value)
    return parsed if parsed is not None else fallback


def int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _decision(
    decision: str,
    missing_slots: list[str],
    safe_actions: list[str],
    blocked_actions: list[str],
    reason: str | None,
    tool_results: dict[str, Any],
    *,
    requested_actions: list[dict[str, Any]] | None = None,
    planner_feedback: str | None = None,
    reasoning: str | None = None,
    policy_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = VerifierOutput(
        decision=decision,
        missing_slots=dedupe(missing_slots),
        safe_actions=dedupe(safe_actions),
        blocked_actions=dedupe(blocked_actions),
        policy_errors=policy_errors or [],
        reason=reason,
        planner_feedback=planner_feedback,
    )
    result: dict[str, Any] = {
        "verifier_decision": decision,
        "verification_decision": output.model_dump(),
        "missing_slots": output.missing_slots,
        "policy_errors": output.policy_errors,
        "verification_errors": [reason] if reason and decision == "ask_user" else [],
        "tool_results": tool_results,
    }
    if requested_actions is not None:
        result["requested_actions"] = requested_actions
    if reasoning:
        result["reasoning"] = reasoning
    return result


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
