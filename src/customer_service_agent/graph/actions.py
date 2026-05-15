from __future__ import annotations

from typing import Any

from customer_service_agent.graph.policy import prefer_explicit_int


ORDER_MUTATION_ACTIONS = {
    "propose_refund": ("refund", "request_refund"),
    "propose_cancel_order": ("cancelled_order", "cancel_order"),
}


def execute_requested_actions(state: dict[str, Any], repository) -> dict[str, Any]:
    tool_results = dict(state.get("tool_results", {}))
    if (state.get("verification_decision") or {}).get("decision") != "proceed_to_action":
        return tool_results

    handlers = {
        "propose_log_complaint": _propose_log_complaint,
        "propose_write_memory": _propose_write_memory,
    }

    for action in state.get("requested_actions", []):
        name = action.get("name")
        args = action.get("args", {})
        if name in ORDER_MUTATION_ACTIONS:
            _request_order_mutation(name, tool_results, repository, state, args)
            continue

        handler = handlers.get(name)
        if handler:
            handler(
                tool_results=tool_results,
                repository=repository,
                state=state,
                args=args,
            )

    return tool_results


def _request_order_mutation(
    action_name: str,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
) -> None:
    result_key, repository_method = ORDER_MUTATION_ACTIONS[action_name]
    order_id = _order_id(args, state)
    if order_id is None:
        return

    tool_results[result_key] = getattr(repository, repository_method)(
        order_id,
        customer_id=state.get("active_customer_id"),
    )


def _propose_log_complaint(
    *,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
) -> None:
    customer_id = _customer_id(args, state)
    order_id = _order_id(args, state)
    issue = args.get("issue") or state.get("issue")
    if customer_id is not None and issue:
        tool_results["complaint"] = repository.log_complaint(
            customer_id=customer_id,
            order_id=order_id,
            issue=str(issue),
        )


def _propose_write_memory(
    *,
    tool_results: dict[str, Any],
    repository,
    state: dict[str, Any],
    args: dict[str, Any],
) -> None:
    customer_id = _customer_id(args, state)
    memory_candidate = state.get("memory_candidate") or {}
    key = args.get("key") or memory_candidate.get("key")
    value = args.get("value") or memory_candidate.get("value")
    if customer_id is not None and key and value:
        tool_results["memory_write"] = repository.write_memory(
            customer_id=customer_id,
            key=str(key),
            value=str(value),
        )


def _customer_id(args: dict[str, Any], state: dict[str, Any]) -> int | None:
    return prefer_explicit_int(args.get("customer_id"), state.get("active_customer_id"))


def _order_id(args: dict[str, Any], state: dict[str, Any]) -> int | None:
    return prefer_explicit_int(args.get("order_id"), state.get("active_order_id"))
