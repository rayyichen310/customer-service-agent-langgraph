from __future__ import annotations

from typing import Any


def summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    if node_name == "planner":
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "issue": update.get("issue"),
            "memory_key": update.get("memory_key"),
            "memory_value": update.get("memory_value"),
            "pending_intent": update.get("pending_intent"),
            "pending_order_id": update.get("pending_order_id"),
            "tool_calls": update.get("tool_calls", []),
            "requested_actions": update.get("requested_actions", []),
            "follow_up_question": update.get("follow_up_question"),
            "plan_steps": update.get("plan_steps", []),
            "reasoning": update.get("reasoning"),
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
            "verification_errors": update.get("verification_errors", []),
            "tool_result_keys": list(tool_results.keys()),
            "pending_order_id": update.get("pending_order_id"),
            "requested_actions": update.get("requested_actions", []),
            "reasoning": update.get("reasoning"),
        }
    if node_name == "respond":
        return {
            "final_response": update.get("final_response"),
            "verified_facts": update.get("verified_facts", {}),
            "response_constraints": update.get("response_constraints", []),
        }
    return update
