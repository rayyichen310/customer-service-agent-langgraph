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
    "propose_refund": OrderMutationRule(
        action_name="propose_refund",
        allowed_status="delivered",
    ),
    "propose_cancel_order": OrderMutationRule(
        action_name="propose_cancel_order",
        blocked_statuses=("delivered", "refund_requested"),
    ),
}

ORDER_ACTION_NAMES = set(ORDER_MUTATION_RULES)

ACTION_SLOT_REQUIREMENTS = {
    "order_mutation": ("customer_id", "order_id", "order_record"),
    "propose_log_complaint": ("customer_id", "order_id", "complaint_issue"),
    "propose_write_memory": ("customer_id", "long_term_write_allowed", "memory_key_value"),
}

CUSTOMER_PROFILE_NOT_FOUND = "CUSTOMER_PROFILE_NOT_FOUND"
ORDER_REFERENCE_AMBIGUOUS = "ORDER_REFERENCE_AMBIGUOUS"
CUSTOMER_ID_MISSING = "CUSTOMER_ID_MISSING"
ORDER_LOOKUP_REQUIRED_BEFORE_MUTATION = "ORDER_LOOKUP_REQUIRED_BEFORE_MUTATION"
ORDER_DETAILS_MISSING_AFTER_LOOKUP = "ORDER_DETAILS_MISSING_AFTER_LOOKUP"
ORDER_DETAILS_MISSING_AFTER_REPLAN_LIMIT = "ORDER_DETAILS_MISSING_AFTER_REPLAN_LIMIT"
COMPLAINT_ISSUE_MISSING = "COMPLAINT_ISSUE_MISSING"
MEMORY_WRITE_NOT_ALLOWED = "MEMORY_WRITE_NOT_ALLOWED"
MEMORY_KEY_VALUE_MISSING = "MEMORY_KEY_VALUE_MISSING"
ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
ORDER_CUSTOMER_MISMATCH = "ORDER_CUSTOMER_MISMATCH"
REFUND_ALREADY_REQUESTED = "REFUND_ALREADY_REQUESTED"
CANCELLATION_ALREADY_REQUESTED = "CANCELLATION_ALREADY_REQUESTED"
ORDER_ALREADY_DELIVERED = "ORDER_ALREADY_DELIVERED"
STATUS_NOT_ALLOWED = "STATUS_NOT_ALLOWED"


def verify_policy(state: dict[str, Any], *, max_react_iterations: int) -> dict[str, Any]:
    requested_actions = list(state.get("requested_actions", []))
    action_names = {action.get("name") for action in requested_actions}
    tool_results = dict(state.get("tool_results", {}))
    order = tool_results.get("order")
    order_reference = OrderReference(**(state.get("order_reference") or {}))
    active_customer_id = state.get("active_customer_id")
    order_lookup_result = tool_results.get("order_lookup")
    called_order_lookup = any(
        call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
    ) or bool(order_lookup_result)
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

    if "propose_log_complaint" in action_names:
        return verify_complaint(
            requested_actions,
            planned_issue=state.get("issue"),
            order_reference=order_reference,
            active_customer_id=active_customer_id,
            tool_results=tool_results,
        )

    if "propose_write_memory" in action_names:
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
            CUSTOMER_PROFILE_NOT_FOUND,
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
    missing_slots = missing_slots_for_requirements(
        ACTION_SLOT_REQUIREMENTS["order_mutation"],
        active_customer_id=active_customer_id,
        order_reference=order_reference,
        order=order,
    )

    if "order_id" in missing_slots:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            list(action_names),
            ORDER_REFERENCE_AMBIGUOUS,
            tool_results,
            context={"order_reference": order_reference.model_dump()},
            planner_feedback_code=ORDER_REFERENCE_AMBIGUOUS,
        )

    if active_customer_id is None:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            list(action_names),
            CUSTOMER_ID_MISSING,
            tool_results,
        )

    if not order:
        if called_order_lookup:
            return _decision(
                "block",
                missing_slots,
                [],
                list(action_names),
                ORDER_DETAILS_MISSING_AFTER_LOOKUP,
                tool_results,
                policy_errors=[
                    policy_error(
                        "ORDER_NOT_FOUND",
                        blocked_action=first_action_name(action_names),
                        order_id=order_reference.order_id,
                        reason_code=ORDER_NOT_FOUND,
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
                planner_feedback_code=ORDER_LOOKUP_REQUIRED_BEFORE_MUTATION,
                context={"order_id": order_reference.order_id},
            )
        return _decision(
            "ask_user",
            missing_slots,
            [],
            list(action_names),
            ORDER_DETAILS_MISSING_AFTER_REPLAN_LIMIT,
            tool_results,
            planner_feedback_code=ORDER_DETAILS_MISSING_AFTER_REPLAN_LIMIT,
            context={"order_id": order_reference.order_id},
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
                    reason_code=ORDER_CUSTOMER_MISMATCH,
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
    missing_slots = missing_slots_for_requirements(
        ACTION_SLOT_REQUIREMENTS["propose_log_complaint"],
        active_customer_id=active_customer_id,
        order_reference=order_reference,
        issue=issue,
    )

    if "order_id" in missing_slots:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            ["propose_log_complaint"],
            ORDER_REFERENCE_AMBIGUOUS,
            tool_results,
            context={"order_reference": order_reference.model_dump()},
            planner_feedback_code=ORDER_REFERENCE_AMBIGUOUS,
        )
    if "complaint_issue" in missing_slots:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            ["propose_log_complaint"],
            COMPLAINT_ISSUE_MISSING,
            tool_results,
            context={"order_id": order_reference.order_id},
            planner_feedback_code=COMPLAINT_ISSUE_MISSING,
        )
    if active_customer_id is None:
        return _decision(
            "ask_user",
            missing_slots,
            [],
            ["propose_log_complaint"],
            CUSTOMER_ID_MISSING,
            tool_results,
        )

    action = {
        "name": "propose_log_complaint",
        "args": {
            "customer_id": active_customer_id,
            "order_id": order_reference.order_id,
            "issue": issue,
        },
        "id": "verified-propose_log_complaint",
    }
    return _decision(
        "proceed_to_action",
        [],
        ["propose_log_complaint"],
        [],
        None,
        tool_results,
        requested_actions=[action],
    )


def verify_memory_write(
    requested_actions: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    candidate = MemoryWriteCandidate(**(state.get("memory_candidate") or {}))
    active_customer_id = state.get("active_customer_id")
    key = candidate.key
    value = candidate.value
    missing_slots = missing_slots_for_requirements(
        ACTION_SLOT_REQUIREMENTS["propose_write_memory"],
        active_customer_id=active_customer_id,
        memory_candidate=candidate,
        memory_key=key,
        memory_value=value,
    )

    if "long_term_write_allowed" in missing_slots:
        return _decision(
            "ask_user",
            ["long_term_write_allowed"],
            [],
            ["propose_write_memory"],
            MEMORY_WRITE_NOT_ALLOWED,
            tool_results,
            context={"memory_type": candidate.memory_type},
            planner_feedback_code=MEMORY_WRITE_NOT_ALLOWED,
        )

    if "customer_id" in missing_slots:
        return _decision(
            "ask_user",
            ["customer_id"],
            [],
            ["propose_write_memory"],
            CUSTOMER_ID_MISSING,
            tool_results,
        )

    if "memory_key_value" in missing_slots:
        return _decision(
            "ask_user",
            ["memory_candidate"],
            [],
            ["propose_write_memory"],
            MEMORY_KEY_VALUE_MISSING,
            tool_results,
            planner_feedback_code=MEMORY_KEY_VALUE_MISSING,
        )

    action = {
        "name": "propose_write_memory",
        "args": {"customer_id": active_customer_id, "key": key, "value": value},
        "id": "verified-propose_write_memory",
    }
    return _decision(
        "proceed_to_action",
        [],
        ["propose_write_memory"],
        [],
        None,
        tool_results,
        requested_actions=[action],
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


def missing_slots_for_requirements(
    requirements: tuple[str, ...],
    *,
    active_customer_id: int | None = None,
    order_reference: OrderReference | None = None,
    order: dict[str, Any] | None = None,
    issue: str | None = None,
    memory_candidate: MemoryWriteCandidate | None = None,
    memory_key: str | None = None,
    memory_value: str | None = None,
) -> list[str]:
    missing: list[str] = []
    for requirement in requirements:
        if requirement == "customer_id" and active_customer_id is None:
            missing.append("customer_id")
        elif requirement == "order_id" and (
            order_reference is None
            or order_reference.order_id is None
            or order_reference.confidence != "high"
        ):
            missing.append("order_id")
        elif requirement == "order_record" and not order:
            missing.extend(["order_status", "order_customer_id"])
        elif requirement == "complaint_issue" and not issue:
            missing.append("complaint_issue")
        elif requirement == "long_term_write_allowed" and (
            memory_candidate is None or not memory_candidate.should_write
        ):
            missing.append("long_term_write_allowed")
        elif requirement == "memory_key_value" and (not memory_key or not memory_value):
            missing.append("memory_key_value")
    return dedupe(missing)


def normalize_action(
    action: dict[str, Any],
    *,
    order_id: int | None,
    customer_id: int | None,
) -> dict[str, Any]:
    args = dict(action.get("args") or {})
    if action.get("name") in ORDER_ACTION_NAMES:
        args["order_id"] = order_id
    if action.get("name") == "propose_log_complaint":
        args["customer_id"] = customer_id
    return {"name": action.get("name"), "args": args, "id": action.get("id")}


def first_action_arg(actions: list[dict[str, Any]], key: str) -> str | None:
    for action in actions:
        value = (action.get("args") or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return None


def order_policy_error(action_name: str, order: dict[str, Any]) -> dict[str, Any]:
    if action_name == "propose_cancel_order":
        return policy_error(
            "ORDER_NOT_CANCELLABLE",
            blocked_action=action_name,
            order_id=order["order_id"],
            current_status=order["status"],
            reason_code=status_reason_code(order["status"]),
        )
    return policy_error(
        "ORDER_NOT_REFUNDABLE",
        blocked_action=action_name,
        order_id=order["order_id"],
        current_status=order["status"],
        reason_code=status_reason_code(order["status"]),
    )


def policy_error(
    error_code: str,
    *,
    blocked_action: str | None = None,
    order_id: int | None = None,
    customer_id: int | None = None,
    current_status: str | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "blocked_action": blocked_action,
        "order_id": order_id,
        "customer_id": customer_id,
        "current_status": current_status,
        "reason_code": reason_code,
    }


def status_reason_code(status: str) -> str:
    if status == "refund_requested":
        return REFUND_ALREADY_REQUESTED
    if status == "cancel_requested":
        return CANCELLATION_ALREADY_REQUESTED
    if status == "delivered":
        return ORDER_ALREADY_DELIVERED
    return STATUS_NOT_ALLOWED


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
    reason_code: str | None,
    tool_results: dict[str, Any],
    *,
    requested_actions: list[dict[str, Any]] | None = None,
    planner_feedback_code: str | None = None,
    context: dict[str, Any] | None = None,
    policy_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = VerifierOutput(
        decision=decision,
        missing_slots=dedupe(missing_slots),
        safe_actions=dedupe(safe_actions),
        blocked_actions=dedupe(blocked_actions),
        policy_errors=policy_errors or [],
        reason_code=reason_code,
        planner_feedback_code=planner_feedback_code,
        context=context or {},
    )
    result: dict[str, Any] = {
        "verification_decision": output.model_dump(exclude_defaults=True, exclude_none=True),
        "tool_results": tool_results,
    }
    if requested_actions is not None:
        result["requested_actions"] = requested_actions
    return result


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
