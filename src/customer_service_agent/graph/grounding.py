from __future__ import annotations

from typing import Any


def build_response_grounding(
    tool_results: dict[str, Any],
    verification_errors: list[str],
) -> tuple[dict[str, Any], list[str]]:
    verified_facts = build_verified_facts(tool_results, verification_errors)
    return verified_facts, build_response_constraints(verified_facts, verification_errors)


def build_verified_facts(
    tool_results: dict[str, Any],
    verification_errors: list[str],
) -> dict[str, Any]:
    verified_facts: dict[str, Any] = {}

    if verification_errors:
        verified_facts["verification_errors"] = list(verification_errors)

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
            "repeated_late_delivery": issue_patterns.get("repeated_late_delivery"),
        }

    return verified_facts


def build_response_constraints(
    verified_facts: dict[str, Any],
    verification_errors: list[str],
) -> list[str]:
    constraints: list[str] = []
    if verification_errors:
        constraints.append(
            "For verifier errors, communicate the first error without adding unsupported action claims."
        )
    if "refund_request" in verified_facts:
        constraints.append(
            "For the refund, describe only the request status shown in verified_facts and use natural customer-facing wording instead of raw status values."
        )
        if verified_facts["refund_request"].get("created_this_turn"):
            constraints.append(
                "For this current-turn refund result, confirm the refund request was submitted or requested in this turn; do not say it was already requested or already submitted."
            )
    if "cancellation_request" in verified_facts:
        constraints.append(
            "For the cancellation, describe only the request status shown in verified_facts and use natural customer-facing wording instead of raw status values."
        )
        if verified_facts["cancellation_request"].get("created_this_turn"):
            constraints.append(
                "For this current-turn cancellation result, confirm the cancellation request was submitted or requested in this turn; do not say it was already requested or already submitted."
            )
    if "complaint_logged" in verified_facts:
        constraints.append(
            "For the complaint, use 2-3 concise sentences, include a brief empathy phrase, and mention only the complaint ID, order, issue, or customer-facing complaint status present in verified_facts."
        )
        constraints.append(
            "For complaint responses, do not mention unrelated refund or cancellation status unless a refund_request or cancellation_request verified fact is present."
        )
    if "issue_patterns" in verified_facts:
        constraints.append(
            "If repeated_late_delivery is true, mention the repeated late-delivery pattern; "
            "otherwise do not mention repeated late-delivery history."
        )
    if "customer_memories" in verified_facts or "customer_complaints" in verified_facts:
        constraints.append(
            "Mention customer history only from customer_memories or customer_complaints."
        )
    if "memory_written" in verified_facts:
        constraints.append(
            "For this memory write, use 2 concise sentences and confirm only the saved key or value shown in verified_facts."
        )
    return constraints
