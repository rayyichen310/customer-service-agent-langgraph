from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from customer_service_agent.graph.agent import CustomerServiceAgent
from customer_service_agent.models import Base, Complaint, Customer, CustomerMemory, Order
from customer_service_agent.reasoning import HeuristicReasoner
from customer_service_agent.repository import CustomerServiceRepository


def seed_repository(session_factory) -> None:
    with session_factory() as session:
        session.add_all(
            [
                Customer(customer_id=1, name="Alice Chen", email="alice@example.com"),
                Customer(customer_id=2, name="Bob Lin", email="bob@example.com"),
                Customer(customer_id=3, name="Carol Wang", email="carol@example.com"),
            ]
        )
        session.add_all(
            [
                Order(order_id=12345, customer_id=1, product_name="Wireless Earbuds", status="in_transit"),
                Order(order_id=1001, customer_id=1, product_name="Smart Lamp", status="processing"),
                Order(order_id=5678, customer_id=2, product_name="Gaming Keyboard", status="delivered"),
                Order(order_id=2222, customer_id=3, product_name="Laptop Stand", status="delivered"),
                Order(order_id=7890, customer_id=1, product_name="USB-C Dock", status="delivered"),
            ]
        )
        session.add_all(
            [
                Complaint(customer_id=1, order_id=12345, issue="delivery was late last month", status="closed"),
                Complaint(customer_id=1, order_id=1001, issue="late delivery again", status="open"),
            ]
        )
        session.add_all(
            [
                CustomerMemory(customer_id=1, key="refund_preference", value="Remember I prefer refunds"),
                CustomerMemory(customer_id=1, key="issue_history", value="Repeated late delivery complaints"),
            ]
        )
        session.commit()


def build_test_agent() -> CustomerServiceAgent:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    seed_repository(session_factory)
    repository = CustomerServiceRepository(session_factory)
    return CustomerServiceAgent(HeuristicReasoner(), repository)


def test_intent_parsing_for_status_query() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-0", "Check status of order 1001")
    assert result.intent == "order_status"
    assert result.order_id == 1001


def test_order_lookup() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-1", "Where is my order 12345?")
    assert result.intent == "order_status"
    assert result.order_id == 12345
    assert "in_transit" in result.response


def test_customer_profile_lookup() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-profile", "Show my profile", customer_id=1)
    assert result.intent == "customer_profile"
    assert "alice@example.com" in result.response.lower()


def test_refund_updates_order() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-2", "Refund order 5678")
    assert result.intent == "refund_request"
    assert result.tool_results["refund"]["status"] == "refund_requested"


def test_conditional_refund_if_delivered() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-conditional", "Refund order 7890 if delivered")
    assert result.intent == "refund_request"
    assert result.tool_results["refund"]["order_id"] == 7890


def test_complaint_logging() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-3", "I want to complain about order 2222")
    assert result.intent == "complaint"
    assert "complaint" in result.tool_results


def test_short_term_memory_uses_previous_order() -> None:
    agent = build_test_agent()
    first = agent.invoke("thread-4", "Where is my order 12345?")
    second = agent.invoke("thread-4", "Cancel it")
    assert first.order_id == 12345
    assert second.intent == "cancel_order"
    assert second.order_id == 12345


def test_long_term_memory_and_personalization() -> None:
    agent = build_test_agent()
    history = agent.invoke("thread-5", "What issues have I had before?", customer_id=1)
    personalized = agent.invoke("thread-6", "My order is late again", customer_id=1)
    assert "complaint" in history.response.lower() or "preference" in history.response.lower()
    assert "repeated late-delivery issues" in personalized.response.lower()


def test_memory_write() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-memory", "Remember I prefer refunds", customer_id=1)
    assert result.intent == "memory_write"
    assert result.tool_results["memory_write"]["key"] == "refund_preference"


def test_invalid_order_rejected() -> None:
    agent = build_test_agent()
    result = agent.invoke("thread-7", "Refund order 0000")
    assert result.verification_errors
    assert "does not exist" in result.verification_errors[0]
