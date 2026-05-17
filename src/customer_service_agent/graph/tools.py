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
def customer_profile(continue_after_read: bool = False) -> str:
    """Read identity and contact details for the authenticated customer."""
    return "schema only"


@tool("read_customer_memory")
def read_customer_memory(continue_after_read: bool = False) -> str:
    """Read durable preferences and profile notes for the authenticated customer."""
    return "schema only"


@tool("read_customer_issue_history")
def read_customer_issue_history(continue_after_read: bool = False) -> str:
    """Read complaint records and summarized issue patterns for the authenticated customer."""
    return "schema only"


@tool("list_my_orders")
def list_my_orders(continue_after_read: bool = False) -> str:
    """List orders belonging to the authenticated customer."""
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
    order_id: int | None = None,
    issue: str = "",
) -> str:
    """Propose logging a complaint for the authenticated customer with known order and issue details."""
    return "schema only"


@tool("propose_write_memory")
def propose_write_memory(
    key: str = "",
    value: str = "",
    memory_type: MemoryType = "unclear",
) -> str:
    """Propose writing durable memory for the authenticated customer with a key, value, and type."""
    return "schema only"


PLANNER_TOOLS = [
    order_lookup,
    customer_profile,
    read_customer_memory,
    read_customer_issue_history,
    list_my_orders,
    propose_refund,
    propose_cancel_order,
    propose_log_complaint,
    propose_write_memory,
]
