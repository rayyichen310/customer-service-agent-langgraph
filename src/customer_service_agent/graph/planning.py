from __future__ import annotations

from typing import Any

from customer_service_agent.graph.tools import ACTION_TOOL_NAMES
from customer_service_agent.reasoning import ToolPlan


def split_tool_calls(plan):
    if plan is None:
        return ToolPlan()

    action_calls: list[dict[str, Any]] = []
    read_calls: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str | None, tuple[tuple[str, str], ...]]] = set()
    explicit_continue_after_read: list[bool] = []

    for action in plan.requested_actions:
        add_action(action_calls, seen_actions, action)

    for call in plan.tool_calls:
        if call.get("name") in ACTION_TOOL_NAMES:
            add_action(action_calls, seen_actions, call)
            continue

        args = call.setdefault("args", {})
        flag = bool_or_none(args.pop("continue_after_read", None))
        if flag is not None:
            explicit_continue_after_read.append(flag)
        read_calls.append(call)

    plan.tool_calls = read_calls
    plan.requested_actions = action_calls
    plan.continue_after_read = continue_after_read(
        plan.continue_after_read,
        explicit_continue_after_read,
    )
    return plan


def add_action(
    action_calls: list[dict[str, Any]],
    seen_actions: set[tuple[str, str | None, tuple[tuple[str, str], ...]]],
    action: dict[str, Any],
) -> None:
    args = dict(action.get("args") or {})
    normalized_action = {
        "name": str(action.get("name") or ""),
        "args": args,
        "id": action.get("id"),
    }
    signature = (
        normalized_action["name"],
        normalized_action["id"],
        tuple(sorted((str(key), str(value)) for key, value in args.items())),
    )
    if signature in seen_actions:
        return
    seen_actions.add(signature)
    action_calls.append(normalized_action)


def continue_after_read(default: bool, explicit_flags: list[bool]) -> bool:
    if True in explicit_flags:
        return True
    if False in explicit_flags:
        return False
    return default


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None
