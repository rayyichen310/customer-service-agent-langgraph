from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from customer_service_agent.graph.planning import complaint_issue, requested_order_mutation


@dataclass(frozen=True)
class PendingIntentResolution:
    action: dict[str, Any] | None = None
    pending_order_id: int | None = None
    reasoning: str | None = None
    errors: tuple[str, ...] = ()
    ask_user: bool = False


@dataclass(frozen=True)
class OrderMutationRule:
    action_name: str
    missing_order_error: str
    resolved_reasoning: str
    allowed_status: str | None = None
    blocked_statuses: tuple[str, ...] = ()
    status_error: str = ""

    def validate_order(self, order: dict[str, Any]) -> str | None:
        status = order["status"]
        if self.allowed_status is not None and status != self.allowed_status:
            return self._format_status_error(order)
        if status in self.blocked_statuses:
            return self._format_status_error(order)
        return None

    def _format_status_error(self, order: dict[str, Any]) -> str:
        return self.status_error.format(
            order_id=order["order_id"],
            status=customer_status_phrase(order["status"]),
        )


ORDER_MUTATION_RULES = {
    "refund": OrderMutationRule(
        action_name="request_refund",
        missing_order_error="I could not complete the refund request within the reasoning step limit.",
        resolved_reasoning=(
            "Resolved pending intent refund into request_refund after verified order observation."
        ),
        allowed_status="delivered",
        status_error="Order {order_id} is {status} and is not eligible for refund yet.",
    ),
    "cancel": OrderMutationRule(
        action_name="request_cancel_order",
        missing_order_error="I could not complete the cancellation request within the reasoning step limit.",
        resolved_reasoning=(
            "Resolved pending intent cancel into request_cancel_order after verified order observation."
        ),
        blocked_statuses=("delivered", "refund_requested"),
        status_error="Order {order_id} is {status} and cannot be cancelled.",
    ),
}


ORDER_ACTION_NAMES = {rule.action_name for rule in ORDER_MUTATION_RULES.values()}


def verify_policy(state: dict[str, Any], *, max_react_iterations: int) -> dict[str, Any]:
    if state.get("follow_up_question") and not state.get("pending_intent"):
        return {
            "verifier_decision": "ask_user",
            "verification_errors": [state["follow_up_question"]],
        }

    errors: list[str] = []
    tool_results = dict(state.get("tool_results", {}))
    order = tool_results.get("order")
    active_order_id = state.get("active_order_id")
    active_customer_id = state.get("active_customer_id")
    last_message = state["messages"][-1].content if state.get("messages") else ""
    pending_intent = state.get("pending_intent")
    pending_order_id = state.get("pending_order_id")
    requested_mutation = (
        str(pending_intent)
        if pending_intent in {"refund", "cancel"}
        else requested_order_mutation(str(last_message))
    )
    requested_actions = list(state.get("requested_actions", []))
    action_names = {action.get("name") for action in requested_actions}
    called_order_lookup = any(
        call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
    )
    needs_order = bool(
        action_names & ORDER_ACTION_NAMES
        or requested_mutation
        or called_order_lookup
    )
    resolved_reasoning: str | None = None
    can_replan = int(state.get("react_iterations") or 0) < int(
        state.get("max_react_iterations") or max_react_iterations
    )

    if needs_order and active_order_id is None:
        errors.append("I need an order ID to continue.")
        return _decision("ask_user", errors, tool_results)

    if needs_order and active_order_id is not None and not order:
        if not called_order_lookup and can_replan:
            return _decision("replan", [], tool_results)
        errors.append(f"Order {active_order_id} does not exist.")
        return _decision("blocked", errors, tool_results)

    if active_customer_id is not None and order and order["customer_id"] != active_customer_id:
        if action_names & ORDER_ACTION_NAMES or requested_mutation in {"refund", "cancel"}:
            errors.append(
                f"Order {order['order_id']} does not belong to customer {active_customer_id}."
            )
            return _decision("blocked", errors, tool_results)

    if pending_intent:
        resolution = resolve_pending_intent(
            str(pending_intent),
            state=state,
            order=order,
            active_order_id=active_order_id,
            active_customer_id=active_customer_id,
            last_message=str(last_message),
        )
        if resolution.errors:
            errors.extend(resolution.errors)
            return _decision(
                "ask_user" if resolution.ask_user else "blocked",
                errors,
                tool_results,
            )
        if resolution.action and resolution.action["name"] not in action_names:
            requested_actions.append(resolution.action)
            action_names.add(resolution.action["name"])
            pending_order_id = resolution.pending_order_id
            resolved_reasoning = resolution.reasoning

    errors.extend(verify_action_policy(action_names, order))

    called_customer_profile = any(
        call.get("name") == "customer_profile" for call in state.get("tool_calls", [])
    )
    if called_customer_profile and not tool_results.get("customer"):
        errors.append("Customer profile was not found.")

    if "request_log_complaint" in action_names and not active_customer_id:
        errors.append("I could not log the complaint without a valid customer or order.")

    if "request_write_memory" in action_names and not active_customer_id:
        errors.append("I could not store that preference without a valid customer ID.")

    result = {
        "verifier_decision": "blocked" if errors else "approved",
        "verification_errors": errors,
        "tool_results": tool_results,
        "requested_actions": requested_actions,
        "pending_order_id": pending_order_id,
    }
    if resolved_reasoning:
        result["reasoning"] = resolved_reasoning
    return result


def resolve_pending_intent(
    pending_intent: str,
    *,
    state: dict[str, Any],
    order: dict[str, Any] | None,
    active_order_id: int | None,
    active_customer_id: int | None,
    last_message: str,
) -> PendingIntentResolution:
    if pending_intent in ORDER_MUTATION_RULES:
        return _resolve_order_mutation(pending_intent, order, active_customer_id)
    if pending_intent == "complaint":
        order_id = prefer_explicit_int(state.get("pending_order_id"), active_order_id)
        return PendingIntentResolution(
            action=_resolved_action(
                "complaint",
                order_id=order_id,
                customer_id=active_customer_id,
                issue=state.get("issue") or complaint_issue(last_message),
            ),
            pending_order_id=order_id,
            reasoning="Resolved pending intent complaint into request_log_complaint.",
        )
    if pending_intent == "memory_write":
        if not state.get("memory_key") or not state.get("memory_value"):
            return PendingIntentResolution(
                errors=("I need a little more detail to help with that.",),
                ask_user=True,
            )
        return PendingIntentResolution(
            action=_resolved_action(
                "memory_write",
                customer_id=active_customer_id,
                memory_key=state.get("memory_key"),
                memory_value=state.get("memory_value"),
            ),
            reasoning="Resolved pending intent memory_write into request_write_memory.",
        )
    return PendingIntentResolution()


def verify_action_policy(action_names: set[str], order: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if not order:
        return errors

    for rule in ORDER_MUTATION_RULES.values():
        if rule.action_name in action_names:
            error = rule.validate_order(order)
            if error:
                errors.append(error)
    return errors


def _resolve_order_mutation(
    pending_intent: str,
    order: dict[str, Any] | None,
    active_customer_id: int | None,
) -> PendingIntentResolution:
    rule = ORDER_MUTATION_RULES[pending_intent]
    if not order:
        return PendingIntentResolution(errors=(rule.missing_order_error,))

    error = rule.validate_order(order)
    if error:
        return PendingIntentResolution(errors=(error,))

    return PendingIntentResolution(
        action=_resolved_action(
            pending_intent,
            order_id=order["order_id"],
            customer_id=active_customer_id,
        ),
        pending_order_id=order["order_id"],
        reasoning=rule.resolved_reasoning,
    )


def _resolved_action(
    pending_intent: str,
    *,
    order_id: int | None = None,
    customer_id: int | None = None,
    issue: str | None = None,
    memory_key: str | None = None,
    memory_value: str | None = None,
) -> dict[str, Any]:
    if pending_intent == "refund":
        return {
            "name": "request_refund",
            "args": {"order_id": order_id},
            "id": "resolved-request_refund",
        }
    if pending_intent == "cancel":
        return {
            "name": "request_cancel_order",
            "args": {"order_id": order_id},
            "id": "resolved-request_cancel_order",
        }
    if pending_intent == "complaint":
        return {
            "name": "request_log_complaint",
            "args": {
                "customer_id": customer_id,
                "order_id": order_id,
                "issue": issue or "customer requested to file a complaint",
            },
            "id": "resolved-request_log_complaint",
        }
    return {
        "name": "request_write_memory",
        "args": {
            "customer_id": customer_id,
            "key": memory_key,
            "value": memory_value,
        },
        "id": "resolved-request_write_memory",
    }


def customer_status_phrase(status: str) -> str:
    if status == "refund_requested":
        return "already marked with a submitted refund request"
    if status == "cancel_requested":
        return "already marked with a submitted cancellation request"
    return status.replace("_", " ")


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
    verifier_decision: str,
    errors: list[str],
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verifier_decision": verifier_decision,
        "verification_errors": errors,
        "tool_results": tool_results,
    }
