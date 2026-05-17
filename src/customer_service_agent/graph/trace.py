from __future__ import annotations

from typing import Any


def summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    if node_name == "planner":
        return {
            "authenticated_customer_id": update.get("authenticated_customer_id"),
            "turn_history_count": len(update.get("turn_history", [])),
            "tool_calls": update.get("tool_calls", []),
            "requested_actions": update.get("requested_actions", []),
        }
    if node_name == "read_tools":
        tool_results = update.get("tool_results", {})
        return {
            "authenticated_customer_id": update.get("authenticated_customer_id"),
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
            "turn_history_count": len(update.get("turn_history", [])),
            "long_term_memory_count": len(long_term_memory),
        }
    return update
