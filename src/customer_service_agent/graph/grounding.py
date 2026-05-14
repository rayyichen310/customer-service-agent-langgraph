from __future__ import annotations

from typing import Any


def build_response_grounding(
    tool_results: dict[str, Any],
    verification_errors: list[str],
    verification_decision: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    verified_facts = build_verified_facts(
        tool_results,
        verification_errors,
        verification_decision or {},
    )
    return verified_facts, build_response_constraints(verified_facts, verification_errors)


def build_verified_facts(
    tool_results: dict[str, Any],
    verification_errors: list[str],
    verification_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verified_facts: dict[str, Any] = {}
    verification_decision = verification_decision or {}

    if verification_errors:
        verified_facts["verification_errors"] = list(verification_errors)

    policy_errors = verification_decision.get("policy_errors") or []
    if policy_errors:
        verified_facts["policy_errors"] = list(policy_errors)

    order = tool_results.get("order")
    if order:
        verified_facts["order"] = {
            "order_id": order.get("order_id"),
            "customer_id": order.get("customer_id"),
            "product_name": order.get("product_name"),
            "status": order.get("status"),
            "order_date": order.get("order_date"),
            "delivery_date": order.get("delivery_date"),
        }

    customer = tool_results.get("customer")
    if customer:
        verified_facts["customer"] = {
            "customer_id": customer.get("customer_id"),
            "name": customer.get("name"),
            "email": customer.get("email"),
        }

    refund = tool_results.get("refund")
    if refund:
        verified_facts["refund_request"] = {
            "order_id": refund.get("order_id"),
            "status": refund.get("status"),
            "created_this_turn": True,
        }

    cancelled = tool_results.get("cancelled_order")
    if cancelled:
        verified_facts["cancellation_request"] = {
            "order_id": cancelled.get("order_id"),
            "status": cancelled.get("status"),
            "created_this_turn": True,
        }

    complaint = tool_results.get("complaint")
    if complaint:
        verified_facts["complaint_logged"] = {
            "complaint_id": complaint.get("complaint_id"),
            "customer_id": complaint.get("customer_id"),
            "order_id": complaint.get("order_id"),
            "issue": complaint.get("issue"),
            "status": complaint.get("status"),
            "created_this_turn": True,
        }

    memory_write = tool_results.get("memory_write")
    if memory_write:
        verified_facts["memory_written"] = {
            "customer_id": memory_write.get("customer_id"),
            "key": memory_write.get("key"),
            "value": memory_write.get("value"),
            "created_this_turn": True,
        }

    memories = tool_results.get("memories")
    if memories:
        verified_facts["customer_memories"] = [
            {
                "customer_id": memory.get("customer_id"),
                "key": memory.get("key"),
                "value": memory.get("value"),
            }
            for memory in memories
        ]

    complaints = tool_results.get("complaints")
    if complaints:
        verified_facts["customer_complaints"] = [
            {
                "complaint_id": complaint.get("complaint_id"),
                "order_id": complaint.get("order_id"),
                "issue": complaint.get("issue"),
                "status": complaint.get("status"),
            }
            for complaint in complaints
        ]

    issue_patterns = tool_results.get("issue_patterns")
    if issue_patterns:
        verified_facts["issue_patterns"] = {
            "total_complaints": issue_patterns.get("total_complaints"),
            "issue_counts": issue_patterns.get("issue_counts", {}),
            "repeated_issues": issue_patterns.get("repeated_issues", {}),
        }

    return verified_facts


def build_response_constraints(
    verified_facts: dict[str, Any],
    verification_errors: list[str],
) -> list[str]:
    return []
