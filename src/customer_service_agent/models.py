from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    product_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    order_date: Mapped[datetime | None] = mapped_column(DateTime)
    delivery_date: Mapped[datetime | None] = mapped_column(DateTime)


class Complaint(Base):
    __tablename__ = "complaints"

    complaint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.order_id"))
    issue: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )


class CustomerMemory(Base):
    __tablename__ = "customer_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )


MemoryType = Literal[
    "preference",
    "profile_note",
    "temporary_issue",
    "transaction_request",
    "unclear",
]
MEMORY_TYPES: set[str] = {
    "preference",
    "profile_note",
    "temporary_issue",
    "transaction_request",
    "unclear",
}
WRITABLE_MEMORY_TYPES: set[str] = {"preference", "profile_note"}


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    customer_id: int | None = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    order_id: int | None = None
    customer_id: int | None = None
    tool_results: dict[str, Any] = Field(default_factory=dict)
    verified_facts: dict[str, Any] = Field(default_factory=dict)
    verification_decision: dict[str, Any] = Field(default_factory=dict)


class OrderReference(BaseModel):
    order_id: int | None = None
    source: Literal["explicit", "none"] = "none"
    confidence: Literal["high", "medium", "low"] = "low"


class MemoryWriteCandidate(BaseModel):
    should_write: bool = False
    memory_type: MemoryType = "unclear"
    key: str | None = None
    value: str | None = None
    reason_code: str | None = None


class VerifierOutput(BaseModel):
    decision: Literal[
        "proceed_to_action",
        "proceed_to_response",
        "replan",
        "ask_user",
        "block",
    ]
    missing_slots: list[str] = Field(default_factory=list)
    safe_actions: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)
    policy_errors: list[dict[str, Any]] = Field(default_factory=list)
    reason_code: str | None = None
    planner_feedback_code: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class AgentState(MessagesState):
    active_customer_id: int | None
    active_order_id: int | None
    order_reference: dict[str, Any]
    issue: str | None
    memory_candidate: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    requested_actions: list[dict[str, Any]]
    tool_results: dict[str, Any]
    verification_decision: dict[str, Any]
    verified_facts: dict[str, Any]
    react_iterations: int
    max_react_iterations: int
    long_term_memory: list[dict[str, Any]]
    final_response: str | None


OrderRecord = Annotated[dict[str, Any], "OrderRecord"]
CustomerRecord = Annotated[dict[str, Any], "CustomerRecord"]
