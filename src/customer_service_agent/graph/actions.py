from __future__ import annotations

from typing import Any

from customer_service_agent.graph.policy import prefer_explicit_int


def execute_requested_actions(state: dict[str, Any], repository) -> dict[str, Any]:
    tool_results = dict(state.get("tool_results", {}))
    if state.get("verifier_decision") != "approved":
        return tool_results

    active_customer_id = state.get("active_customer_id")
    active_order_id = state.get("active_order_id")

    handlers = {
        "request_refund": _request_refund,
        "request_cancel_order": _request_cancel_order,
        "request_log_complaint": _request_log_complaint,
        "request_write_memory": _request_write_memory,
    }

    for action in state.get("requested_actions", []):
        handler = handlers.get(action.get("name"))
        if handler:
            handler(
                tool_results=tool_results,
                repository=repository,
                state=state,
                args=action.get("args", {}),
                active_customer_id=active_customer_id,
                active_order_id=active_order_id,
            )

    return tool_results


def _request_refund(
    *,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
    active_customer_id: int | None,
    active_order_id: int | None,
) -> None:
    order_id = prefer_explicit_int(args.get("order_id"), active_order_id)
    if order_id is not None:
        tool_results["refund"] = repository.request_refund(
            order_id,
            customer_id=active_customer_id,
        )


def _request_cancel_order(
    *,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
    active_customer_id: int | None,
    active_order_id: int | None,
) -> None:
    order_id = prefer_explicit_int(args.get("order_id"), active_order_id)
    if order_id is not None:
        tool_results["cancelled_order"] = repository.cancel_order(
            order_id,
            customer_id=active_customer_id,
        )


def _request_log_complaint(
    *,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
    active_customer_id: int | None,
    active_order_id: int | None,
) -> None:
    customer_id = prefer_explicit_int(args.get("customer_id"), active_customer_id)
    order_id = prefer_explicit_int(args.get("order_id"), active_order_id)
    issue = args.get("issue") or state.get("issue") or "customer requested to file a complaint"
    if customer_id is not None:
        tool_results["complaint"] = repository.log_complaint(
            customer_id=customer_id,
            order_id=order_id,
            issue=str(issue),
        )


def _request_write_memory(
    *,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
    active_customer_id: int | None,
    active_order_id: int | None,
) -> None:
    customer_id = prefer_explicit_int(args.get("customer_id"), active_customer_id)
    key = args.get("key") or state.get("memory_key")
    value = args.get("value") or state.get("memory_value")
    if customer_id is not None and key and value:
        tool_results["memory_write"] = repository.write_memory(
            customer_id=customer_id,
            key=str(key),
            value=str(value),
        )
