from __future__ import annotations

from langchain_core.tools import tool

from customer_service_agent.models import MemoryType


ORDER_TOOL_NAMES = {
    "order_lookup",
    "request_refund",
    "request_cancel_order",
    "request_log_complaint",
}
ACTION_TOOL_NAMES = {
    "request_refund",
    "request_cancel_order",
    "request_log_complaint",
    "request_write_memory",
}
CONTROL_TOOL_NAMES = {
    "continue_after_read",
}


@tool("order_lookup")
def order_lookup(order_id: int) -> str:
    """Read order details by order ID."""
    return "schema only"


@tool("customer_profile")
def customer_profile(customer_id: int) -> str:
    """Read customer profile details by customer ID."""
    return "schema only"


@tool("read_customer_memory")
def read_customer_memory(customer_id: int) -> str:
    """Read long-term customer memory records."""
    return "schema only"


@tool("list_customer_complaints")
def list_customer_complaints(customer_id: int) -> str:
    """Read previous customer complaints."""
    return "schema only"


@tool("summarize_issue_patterns")
def summarize_issue_patterns(customer_id: int) -> str:
    """Read summarized complaint issue patterns for a customer."""
    return "schema only"


@tool("continue_after_read")
def continue_after_read() -> str:
    """Mark that planning should continue after requested read tools return observations."""
    return "schema only"


@tool("request_refund")
def request_refund(order_id: int) -> str:
    """Request a refund action for an order. Execution is gated by verifier policy."""
    return "schema only"


@tool("request_cancel_order")
def request_cancel_order(order_id: int) -> str:
    """Request a cancel action for an order. Execution is gated by verifier policy."""
    return "schema only"


@tool("request_log_complaint")
def request_log_complaint(
    customer_id: int | None = None,
    order_id: int | None = None,
    issue: str = "",
) -> str:
    """Request logging a complaint. Leave issue empty when the customer has not provided it."""
    return "schema only"


@tool("request_write_memory")
def request_write_memory(
    customer_id: int | None = None,
    key: str = "",
    value: str = "",
    memory_type: MemoryType = "unclear",
) -> str:
    """Request writing a long-term memory item with an explicit memory type."""
    return "schema only"


PLANNER_TOOLS = [
    order_lookup,
    customer_profile,
    read_customer_memory,
    list_customer_complaints,
    summarize_issue_patterns,
    continue_after_read,
    request_refund,
    request_cancel_order,
    request_log_complaint,
    request_write_memory,
]
