from __future__ import annotations

from typing import Any


def summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    if node_name == "planner":
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "order_reference": update.get("order_reference"),
            "issue": update.get("issue"),
            "memory_candidate": update.get("memory_candidate"),
            "tool_calls": update.get("tool_calls", []),
            "requested_actions": update.get("requested_actions", []),
        }
    if node_name == "read_tools":
        tool_results = update.get("tool_results", {})
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "tool_result_keys": list(tool_results.keys()),
            "tool_results": tool_results,
        }
    if node_name == "actions":
        tool_results = update.get("tool_results", {})
        return {
            "tool_result_keys": list(tool_results.keys()),
            "tool_results": tool_results,
        }
    if node_name == "verifier":
        tool_results = update.get("tool_results", {})
        verification_decision = update.get("verification_decision", {})
        return {
            "verifier_decision": verification_decision.get("decision"),
            "verification_decision": verification_decision,
            "tool_result_keys": list(tool_results.keys()),
            "requested_actions": update.get("requested_actions", []),
        }
    if node_name == "respond":
        long_term_memory = update.get("long_term_memory", [])
        return {
            "final_response": update.get("final_response"),
            "verified_facts": update.get("verified_facts", {}),
            "long_term_memory_count": len(long_term_memory),
        }
    return update
