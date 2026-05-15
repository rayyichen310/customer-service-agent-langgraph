from __future__ import annotations

from typing import Any


def summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    if node_name == "planner":
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "current_turn_order_id": update.get("current_turn_order_id"),
            "order_reference": update.get("order_reference"),
            "issue": update.get("issue"),
            "memory_key": update.get("memory_key"),
            "memory_value": update.get("memory_value"),
            "memory_candidate": update.get("memory_candidate"),
            "missing_slots": update.get("missing_slots", []),
            "tool_calls": update.get("tool_calls", []),
            "requested_actions": update.get("requested_actions", []),
            "requires_replan_after_tools": update.get("requires_replan_after_tools", False),
        }
    if node_name in {"read_tools", "actions"}:
        tool_results = update.get("tool_results", {})
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "tool_result_keys": list(tool_results.keys()),
            "tool_results": tool_results,
        }
    if node_name in {"memory", "memory_update"}:
        long_term_memory = update.get("long_term_memory", [])
        return {
            "long_term_memory_count": len(long_term_memory),
            "long_term_memory": long_term_memory,
        }
    if node_name == "verifier":
        tool_results = update.get("tool_results", {})
        return {
            "verifier_decision": update.get("verifier_decision"),
            "verification_decision": update.get("verification_decision", {}),
            "missing_slots": update.get("missing_slots", []),
            "policy_errors": update.get("policy_errors", []),
            "tool_result_keys": list(tool_results.keys()),
            "requested_actions": update.get("requested_actions", []),
        }
    if node_name == "respond":
        return {
            "final_response": update.get("final_response"),
            "verified_facts": update.get("verified_facts", {}),
        }
    return update
