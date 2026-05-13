from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from customer_service_agent.graph.agent import CustomerServiceAgent
from customer_service_agent.models import Base, Customer, CustomerMemory, Order
from customer_service_agent.reasoning import ResponseContext, ToolPlan
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
                Customer(customer_id=7, name="Dana Lee", email="dana@example.com"),
                Customer(customer_id=8, name="Evan Wu", email="evan@example.com"),
            ]
        )
        session.add(
            Order(
                order_id=7890,
                customer_id=7,
                product_name="Standing Desk",
                status="delivered",
                order_date=datetime.fromisoformat("2026-04-20T09:00:00"),
                delivery_date=datetime.fromisoformat("2026-04-24T15:00:00"),
            )
        )
        session.add(CustomerMemory(customer_id=7, key="preference", value="Prefers email"))
        session.commit()
    return CustomerServiceRepository(session_factory)


class RefundAfterObservationReasoner:
    def __init__(self) -> None:
        self.plan_snapshots: list[dict[str, Any]] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_snapshots.append(state_snapshot)
        order = state_snapshot.get("tool_results", {}).get("order")
        if not order:
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
                ],
                order_id=7890,
                steps=["order_lookup"],
                reasoning="Need to observe the order before requesting a refund.",
            )

        action = {"name": "request_refund", "args": {"order_id": 7890}, "id": "refund-1"}
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            order_id=7890,
            steps=["request_refund"],
            reasoning="The observed order is delivered, so request a refund.",
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return f"The refund request was submitted for order {context.active_order_id}."


class LookupOnlyReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        return ToolPlan(
            tool_calls=[
                {
                    "name": "order_lookup",
                    "args": {"order_id": 7890},
                    "id": f"lookup-{self.plan_calls}",
                }
            ],
            order_id=7890,
            steps=["order_lookup"],
            reasoning="Keep looking up the order without requesting the mutation.",
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return "Done."


class CombinedThenActionReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        action = {"name": "request_refund", "args": {"order_id": 7890}, "id": "refund-1"}
        if self.plan_calls == 1:
            lookup = {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
            return ToolPlan(
                tool_calls=[lookup, action],
                requested_actions=[action],
                order_id=7890,
                steps=["order_lookup", "request_refund"],
                reasoning="Incorrectly combined read and action.",
            )
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            order_id=7890,
            steps=["request_refund"],
            reasoning="Request refund after the previous observation.",
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return "The refund request was submitted."


class CustomerReadReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        if "order" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
                ],
                order_id=7890,
                steps=["order_lookup"],
            )
        if "memory" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "read_customer_memory",
                        "args": {"customer_id": state_snapshot.get("active_customer_id")},
                        "id": "memory-1",
                    }
                ],
                customer_id=state_snapshot.get("active_customer_id"),
                steps=["read_customer_memory"],
            )
        return ToolPlan(
            tool_calls=[
                {
                    "name": "customer_profile",
                    "args": {"customer_id": state_snapshot.get("active_customer_id")},
                    "id": "profile-1",
                }
            ],
            customer_id=state_snapshot.get("active_customer_id"),
            steps=["customer_profile"],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return ",".join(sorted(context.tool_results))


class ComplaintThenProfileReasoner:
    def __init__(self) -> None:
        self.plan_snapshots: list[dict[str, Any]] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_snapshots.append(state_snapshot)
        if "complaint" in user_message.lower():
            action = {
                "name": "request_log_complaint",
                "args": {
                    "customer_id": state_snapshot.get("active_customer_id"),
                    "issue": "package damaged",
                },
                "id": "complaint-1",
            }
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
                customer_id=state_snapshot.get("active_customer_id"),
                issue="package damaged",
                steps=["request_log_complaint"],
            )
        return ToolPlan(
            tool_calls=[
                {
                    "name": "customer_profile",
                    "args": {"customer_id": state_snapshot.get("active_customer_id")},
                    "id": "profile-1",
                }
            ],
            customer_id=state_snapshot.get("active_customer_id"),
            steps=["customer_profile"],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return str(context.tool_results)


def test_refund_uses_react_loop_before_action() -> None:
    repository = build_repository()
    reasoner = RefundAfterObservationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-refund", "Refund order 7890 if delivered")

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert len(planner_updates) == 2
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert planner_updates[0]["requested_actions"] == []
    assert [action["name"] for action in planner_updates[1]["requested_actions"]] == ["request_refund"]

    assert reasoner.plan_snapshots[0]["tool_results"] == {}
    assert reasoner.plan_snapshots[1]["tool_results"]["order"]["status"] == "delivered"

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert verifier_updates[-1]["verifier_decision"] == "approved"
    assert response.verifier_decision == "approved"
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert response.response == "The refund request was submitted for order 7890."
    assert repository.get_order(7890)["status"] == "refund_requested"


def test_verifier_replans_combined_read_and_refund_action() -> None:
    repository = build_repository()
    reasoner = CombinedThenActionReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-combined", "Refund order 7890 if delivered")

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [update["verifier_decision"] for update in verifier_updates] == [
        "replan",
        "approved",
    ]
    assert [update["node"] for update in updates].count("actions") == 1
    assert reasoner.plan_calls == 2
    assert response.verifier_decision == "approved"
    assert response.tool_results["refund"]["status"] == "refund_requested"


def test_react_loop_stops_at_iteration_limit() -> None:
    repository = build_repository()
    reasoner = LookupOnlyReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-limit", "Refund order 7890 if delivered")

    assert reasoner.plan_calls == 3
    assert [update["node"] for update in updates].count("planner") == 3
    assert response.verifier_decision == "blocked"
    assert response.verification_errors == [
        "I could not complete the refund request within the reasoning step limit."
    ]
    assert "refund" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


def test_profile_query_does_not_include_stale_order_result() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(CustomerReadReasoner(), repository)

    first_response, _ = agent.trace("profile-pollution", "Check order 7890", customer_id=7)
    second_response, _ = agent.trace("profile-pollution", "Show my profile", customer_id=7)

    assert "order" in first_response.tool_results
    assert set(second_response.tool_results) == {"customer"}
    assert second_response.response == "customer"


def test_memory_read_does_not_include_stale_order_result() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(CustomerReadReasoner(), repository)

    first_response, _ = agent.trace("memory-pollution", "Check order 7890", customer_id=7)
    second_response, _ = agent.trace("memory-pollution", "Read my memory", customer_id=7)

    assert "order" in first_response.tool_results
    assert set(second_response.tool_results) == {"memories"}
    assert second_response.response == "memories"


def test_customer_cannot_refund_another_customers_order() -> None:
    repository = build_repository()
    reasoner = RefundAfterObservationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("ownership-refund", "Refund order 7890 if delivered", customer_id=8)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert verifier_updates[-1]["verifier_decision"] == "blocked"
    assert response.verifier_decision == "blocked"
    assert response.verification_errors == ["Order 7890 does not belong to customer 8."]
    assert "refund" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


def test_stale_complaint_issue_does_not_reach_later_unrelated_request() -> None:
    repository = build_repository()
    reasoner = ComplaintThenProfileReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    complaint_response, _ = agent.trace("issue-pollution", "Log a complaint", customer_id=7)
    profile_response, profile_updates = agent.trace("issue-pollution", "Show my profile", customer_id=7)

    planner_updates = [update["state"] for update in profile_updates if update["node"] == "planner"]
    assert complaint_response.tool_results["complaint"]["issue"] == "package damaged"
    assert reasoner.plan_snapshots[-1]["issue"] is None
    assert planner_updates[-1]["issue"] is None
    assert "package damaged" not in profile_response.response
