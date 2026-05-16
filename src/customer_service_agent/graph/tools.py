from __future__ import annotations

from langchain_core.tools import tool

from customer_service_agent.models import MemoryType


ACTION_TOOL_NAMES = {
    "propose_refund",
    "propose_cancel_order",
    "propose_log_complaint",
    "propose_write_memory",
}


@tool("order_lookup")
def order_lookup(order_id: int, continue_after_read: bool = False) -> str:
    """Read order details such as customer, product, status, and dates by order ID."""
    return "schema only"


@tool("customer_profile")
def customer_profile(customer_id: int, continue_after_read: bool = False) -> str:
    """Read customer identity and contact details by customer ID."""
    return "schema only"


@tool("read_customer_memory")
def read_customer_memory(customer_id: int, continue_after_read: bool = False) -> str:
    """Read durable customer preferences and profile notes by customer ID."""
    return "schema only"


@tool("read_customer_issue_history")
def read_customer_issue_history(customer_id: int, continue_after_read: bool = False) -> str:
    """Read previous customer complaint records and summarized issue patterns by customer ID."""
    return "schema only"


@tool("propose_refund")
def propose_refund(order_id: int) -> str:
    """Propose a refund when the customer asks for one. The verifier checks order safety before execution."""
    return "schema only"


@tool("propose_cancel_order")
def propose_cancel_order(order_id: int) -> str:
    """Propose cancellation when the customer asks for one. The verifier checks order safety before execution."""
    return "schema only"


@tool("propose_log_complaint")
def propose_log_complaint(
    customer_id: int | None = None,
    order_id: int | None = None,
    issue: str = "",
) -> str:
    """Propose logging a customer complaint with known customer, order, and issue details."""
    return "schema only"


@tool("propose_write_memory")
def propose_write_memory(
    customer_id: int | None = None,
    key: str = "",
    value: str = "",
    memory_type: MemoryType = "unclear",
) -> str:
    """Propose writing durable customer memory with a key, value, and explicit memory type."""
    return "schema only"


PLANNER_TOOLS = [
    order_lookup,
    customer_profile,
    read_customer_memory,
    read_customer_issue_history,
    propose_refund,
    propose_cancel_order,
    propose_log_complaint,
    propose_write_memory,
]
