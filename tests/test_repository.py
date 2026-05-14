from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from customer_service_agent.models import Base, Complaint, Customer, CustomerMemory, Order
from customer_service_agent.repository import CustomerServiceRepository


def build_repository() -> CustomerServiceRepository:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with session_factory() as session:
        session.add_all(
            [
                Customer(customer_id=1, name="Alice Chen", email="alice@example.com"),
                Customer(customer_id=2, name="Bob Lin", email="bob@example.com"),
            ]
        )
        session.add_all(
            [
                Order(
                    order_id=12345,
                    customer_id=1,
                    product_name="Wireless Earbuds",
                    status="in_transit",
                    order_date=datetime.fromisoformat("2026-05-01T10:00:00"),
                ),
                Order(
                    order_id=5678,
                    customer_id=2,
                    product_name="Gaming Keyboard",
                    status="delivered",
                    order_date=datetime.fromisoformat("2026-04-20T09:00:00"),
                    delivery_date=datetime.fromisoformat("2026-04-24T15:00:00"),
                ),
            ]
        )
        session.add_all(
            [
                Complaint(customer_id=1, order_id=12345, issue="delivery was late last month", status="closed"),
                Complaint(customer_id=1, order_id=12345, issue="late delivery again", status="open"),
            ]
        )
        session.add_all(
            [
                CustomerMemory(customer_id=1, key="refund_preference", value="Remember I prefer refunds"),
                CustomerMemory(customer_id=1, key="issue_history", value="Repeated late delivery complaints"),
            ]
        )
        session.commit()
    return CustomerServiceRepository(session_factory)


def test_get_order_and_customer() -> None:
    repository = build_repository()

    order = repository.get_order(12345)
    customer = repository.get_customer(1)

    assert order is not None
    assert order["status"] == "in_transit"
    assert customer is not None
    assert customer["email"] == "alice@example.com"


def test_request_refund_updates_status() -> None:
    repository = build_repository()

    refund = repository.request_refund(5678)

    assert refund is not None
    assert refund["status"] == "refund_requested"


def test_request_refund_requires_matching_customer_when_provided() -> None:
    repository = build_repository()

    refund = repository.request_refund(5678, customer_id=1)

    assert refund is None
    assert repository.get_order(5678)["status"] == "delivered"


def test_cancel_order_updates_status() -> None:
    repository = build_repository()

    cancelled = repository.cancel_order(12345)

    assert cancelled is not None
    assert cancelled["status"] == "cancel_requested"


def test_cancel_order_requires_matching_customer_when_provided() -> None:
    repository = build_repository()

    cancelled = repository.cancel_order(12345, customer_id=2)

    assert cancelled is None
    assert repository.get_order(12345)["status"] == "in_transit"


def test_log_complaint_and_write_memory() -> None:
    repository = build_repository()

    complaint = repository.log_complaint(customer_id=1, order_id=12345, issue="package damaged")
    memory = repository.write_memory(customer_id=1, key="service_note", value="Prefers email follow-up")

    assert complaint["issue"] == "package damaged"
    assert memory["key"] == "service_note"


def test_write_memory_upserts_by_customer_and_key() -> None:
    repository = build_repository()

    first = repository.write_memory(customer_id=1, key="refund_preference", value="prefers refunds")
    second = repository.write_memory(customer_id=1, key="refund_preference", value="strongly prefers refunds")
    memories = repository.read_memories(1, key="refund_preference")

    assert first["id"] == second["id"]
    assert len(memories) == 1
    assert memories[0]["value"] == "strongly prefers refunds"


def test_log_complaint_requires_order_to_belong_to_customer() -> None:
    repository = build_repository()

    complaint = repository.log_complaint(customer_id=2, order_id=12345, issue="package damaged")

    assert complaint is None
    assert repository.list_complaints(2) == []


def test_read_memory_and_issue_patterns() -> None:
    repository = build_repository()

    memories = repository.read_memories(1)
    patterns = repository.summarize_issue_patterns(1)

    assert len(memories) >= 2
    assert patterns["issue_counts"] == {
        "delivery was late last month": 1,
        "late delivery again": 1,
    }
    assert patterns["repeated_issues"] == {}
