from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from customer_service_agent.service import build_agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive customer service agent chat with per-turn node trace."
    )
    parser.add_argument("--thread-id", default=None, help="Conversation thread ID")
    parser.add_argument("--customer-id", type=int, default=None, help="Customer ID")
    parser.add_argument(
        "--json-trace",
        action="store_true",
        help="Print the full summarized trace state as JSON after each turn.",
    )
    args = parser.parse_args()

    agent = build_agent()
    thread_id = args.thread_id or str(uuid.uuid4())

    print(f"thread_id={thread_id}")
    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_input or user_input.lower() in {"exit", "quit"}:
            break

        result, node_trace = agent.trace(
            thread_id=thread_id,
            message=user_input,
            customer_id=args.customer_id,
        )
        print_node_trace(node_trace, json_trace=args.json_trace)
        print(f"agent> {result.response}")


def print_node_trace(node_trace: list[dict[str, Any]], json_trace: bool = False) -> None:
    node_path = " -> ".join(item.get("node", "<unknown>") for item in node_trace)
    print(f"trace> {node_path or '<no nodes>'}")

    if json_trace:
        print(json.dumps(node_trace, ensure_ascii=False, indent=2, default=str))
        return

    for index, item in enumerate(node_trace, start=1):
        node = item.get("node", "<unknown>")
        state = item.get("state", {})
        print(f"  {index}. {node}: {_summary_for_node(node, state)}")


def _summary_for_node(node: str, state: dict[str, Any]) -> str:
    if node == "planner":
        parts = [
            _format_value("auth_customer", state.get("authenticated_customer_id")),
            _format_list("tool_calls", _names(state.get("tool_calls", []))),
            _format_list("requested_actions", _names(state.get("requested_actions", []))),
        ]
        return _join_parts(parts)

    if node in {"read_tools", "actions"}:
        parts = [
            _format_list("tools", state.get("tool_result_keys", [])),
            _format_value("auth_customer", state.get("authenticated_customer_id")),
        ]
        return _join_parts(parts)

    if node == "verifier":
        decision = state.get("verification_decision", {}) or {}
        parts = [
            _format_value(
                "decision",
                decision.get("decision") or state.get("verifier_decision"),
            ),
            _format_list("missing", decision.get("missing_slots", [])),
            _format_list("policy_errors", decision.get("policy_errors", [])),
            _format_list("tool_keys", state.get("tool_result_keys", [])),
        ]
        return _join_parts(parts)

    if node == "respond":
        return _join_parts(
            [
                _format_value("final_response", state.get("final_response")),
                _format_value("long_term_memory_count", state.get("long_term_memory_count")),
            ]
        )

    return json.dumps(state, ensure_ascii=False, default=str)


def _names(items: list[Any]) -> list[str]:
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("tool") or item.get("action")
            if name:
                names.append(str(name))
                continue
        names.append(str(item))
    return names


def _format_value(label: str, value: Any) -> str | None:
    if value is None or value == "":
        return None
    return f"{label}={value}"


def _format_list(label: str, value: list[Any] | None) -> str | None:
    if not value:
        return None
    return f"{label}={value}"


def _join_parts(parts: list[str | None]) -> str:
    visible_parts = [part for part in parts if part]
    return " ".join(visible_parts) if visible_parts else "-"


if __name__ == "__main__":
    main()
