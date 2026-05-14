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
                Customer(customer_id=2, name="Bob Lin", email="bob@example.com"),
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
        session.add(
            Order(
                order_id=5678,
                customer_id=2,
                product_name="Gaming Keyboard",
                status="delivered",
                order_date=datetime.fromisoformat("2026-04-20T09:00:00"),
                delivery_date=datetime.fromisoformat("2026-04-24T15:00:00"),
            )
        )
        session.add(
            Order(
                order_id=2468,
                customer_id=7,
                product_name="Monitor Arm",
                status="processing",
                order_date=datetime.fromisoformat("2026-04-21T09:00:00"),
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


class RefundByMessageReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        order_id = 5678 if "5678" in user_message else 7890
        order = state_snapshot.get("tool_results", {}).get("order")
        if not order:
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": order_id}, "id": "lookup-1"}
                ],
                order_id=order_id,
                steps=["order_lookup"],
            )
        action = {"name": "request_refund", "args": {"order_id": order["order_id"]}, "id": "refund-1"}
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            order_id=order["order_id"],
            steps=["request_refund"],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed refund response"


class CancelAfterObservationReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        order = state_snapshot.get("tool_results", {}).get("order")
        if not order:
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 2468}, "id": "lookup-1"}
                ],
                order_id=2468,
                steps=["order_lookup"],
            )
        action = {"name": "request_cancel_order", "args": {"order_id": 2468}, "id": "cancel-1"}
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            order_id=2468,
            steps=["request_cancel_order"],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed cancel response"


class NoToolMemoryWriteReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        return ToolPlan(reasoning="I should remember the refund preference.")

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed memory response"


class PendingIntentOnlyReasoner:
    def __init__(
        self,
        intent: str,
        *,
        order_id: int | None = None,
        issue: str | None = None,
        memory_key: str | None = None,
        memory_value: str | None = None,
        response: str = "Done.",
    ) -> None:
        self.intent = intent
        self.order_id = order_id
        self.issue = issue
        self.memory_key = memory_key
        self.memory_value = memory_value
        self.response = response

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        return ToolPlan(
            customer_id=state_snapshot.get("active_customer_id"),
            order_id=self.order_id,
            issue=self.issue,
            memory_key=self.memory_key,
            memory_value=self.memory_value,
            pending_intent=self.intent,
            pending_order_id=self.order_id,
            reasoning=f"Planner identified {self.intent} intent only.",
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return self.response


class DirectMutationReasoner:
    def __init__(self, action_name: str, args: dict[str, Any]):
        self.action_name = action_name
        self.args = args

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        args = dict(self.args)
        if "customer_id" not in args and state_snapshot.get("active_customer_id") is not None:
            args["customer_id"] = state_snapshot["active_customer_id"]
        action = {"name": self.action_name, "args": args, "id": f"{self.action_name}-1"}
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            customer_id=args.get("customer_id"),
            order_id=args.get("order_id"),
            issue=args.get("issue"),
            memory_key=args.get("key"),
            memory_value=args.get("value"),
            steps=[self.action_name],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed mutation response"


def test_refund_uses_react_loop_before_action() -> None:
    repository = build_repository()
    reasoner = RefundAfterObservationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-refund", "Refund order 7890 if delivered")

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert len(planner_updates) == 1
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert planner_updates[0]["requested_actions"] == []
    assert planner_updates[0]["pending_intent"] == "refund"
    assert "pending_action" not in planner_updates[0]
    assert planner_updates[0]["pending_order_id"] == 7890

    assert reasoner.plan_snapshots[0]["tool_results"] == {}

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "request_refund"
    ]
    assert verifier_updates[-1]["reasoning"] == (
        "Resolved pending intent refund into request_refund after verified order observation."
    )
    assert verifier_updates[-1]["verifier_decision"] == "approved"
    assert response.verifier_decision == "approved"
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert response.verified_facts["refund_request"] == {
        "order_id": 7890,
        "status": "refund_requested",
        "created_this_turn": True,
    }
    assert response.response == "The refund request was submitted for order 7890."
    assert repository.get_order(7890)["status"] == "refund_requested"


def test_planner_action_calls_are_normalized_to_pending_intent() -> None:
    repository = build_repository()
    reasoner = CombinedThenActionReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-combined", "Refund order 7890 if delivered")

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert planner_updates[0]["requested_actions"] == []
    assert planner_updates[0]["pending_intent"] == "refund"
    assert "pending_action" not in planner_updates[0]
    assert [update["verifier_decision"] for update in verifier_updates] == ["approved"]
    assert [update["node"] for update in updates].count("actions") == 1
    assert reasoner.plan_calls == 1
    assert response.verifier_decision == "approved"
    assert response.tool_results["refund"]["status"] == "refund_requested"


def test_pending_intent_continues_transaction_after_observation() -> None:
    repository = build_repository()
    reasoner = LookupOnlyReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-limit", "Refund order 7890 if delivered")

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]

    assert reasoner.plan_calls == 1
    assert [update["node"] for update in updates].count("planner") == 1
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "request_refund"
    ]
    assert verifier_updates[-1]["reasoning"] == (
        "Resolved pending intent refund into request_refund after verified order observation."
    )
    assert response.verifier_decision == "approved"
    assert response.verification_errors == []
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert repository.get_order(7890)["status"] == "refund_requested"


def test_refund_owned_delivered_order_routes_success_through_responder() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(RefundByMessageReasoner(), repository)

    response, _ = agent.trace("refund-5678", "Refund order 5678", customer_id=2)

    assert response.verification_errors == []
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert response.verified_facts["refund_request"] == {
        "order_id": 5678,
        "status": "refund_requested",
        "created_this_turn": True,
    }
    constraints = " ".join(response.response_constraints)
    assert "current-turn refund" in constraints
    assert "raw status values" in constraints
    assert "Do not invent refund status" not in constraints
    assert "Do not promise future handling" not in constraints
    assert response.response == "LLM guessed refund response"
    assert repository.get_order(5678)["status"] == "refund_requested"


def test_memory_write_resolves_pending_intent_and_routes_success_through_responder() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(NoToolMemoryWriteReasoner(), repository)

    response, updates = agent.trace("memory-write", "Remember I prefer refunds", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert planner_updates[0]["pending_intent"] == "memory_write"
    assert "pending_action" not in planner_updates[0]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "request_write_memory"
    ]
    assert verifier_updates[-1]["reasoning"] == (
        "Resolved pending intent memory_write into request_write_memory."
    )
    assert response.verification_errors == []
    assert response.tool_results["memory_write"]["key"] == "refund_preference"
    assert response.verified_facts["memory_written"]["key"] == "refund_preference"
    assert response.verified_facts["memory_written"]["created_this_turn"] is True
    assert "saved key or value" in " ".join(response.response_constraints)
    assert response.response == "LLM guessed memory response"
    assert len(repository.read_memories(7, key="refund_preference")) == 1


def test_pending_action_not_required_from_planner_for_transaction_intents() -> None:
    cases = [
        (
            "refund",
            PendingIntentOnlyReasoner("refund", order_id=7890),
            "request_refund",
            "Refund order 7890",
            7,
        ),
        (
            "cancel",
            PendingIntentOnlyReasoner("cancel", order_id=2468),
            "request_cancel_order",
            "Cancel order 2468",
            7,
        ),
        (
            "complaint",
            PendingIntentOnlyReasoner("complaint", order_id=7890, issue="package damaged"),
            "request_log_complaint",
            "I want to complain about order 7890",
            7,
        ),
        (
            "memory_write",
            PendingIntentOnlyReasoner(
                "memory_write",
                memory_key="contact_preference",
                memory_value="prefers email",
            ),
            "request_write_memory",
            "Remember I prefer email",
            7,
        ),
    ]

    for intent, reasoner, expected_action, message, customer_id in cases:
        repository = build_repository()
        agent = CustomerServiceAgent(reasoner, repository)
        response, updates = agent.trace(f"pending-{intent}", message, customer_id=customer_id)

        planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
        verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
        assert planner_updates[-1]["pending_intent"] == intent
        assert "pending_action" not in planner_updates[-1]
        assert planner_updates[-1]["requested_actions"] == []
        assert expected_action in [
            action["name"] for action in verifier_updates[-1]["requested_actions"]
        ]
        assert response.verification_errors == []


def test_pending_intent_refund_delivered_order_resolves_to_refund_action() -> None:
    repository = build_repository()
    reasoner = PendingIntentOnlyReasoner("refund", order_id=7890)
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("pending-refund-resolve", "Refund order 7890", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert verifier_updates[-1]["reasoning"] == (
        "Resolved pending intent refund into request_refund after verified order observation."
    )
    assert verifier_updates[-1]["requested_actions"] == [
        {"name": "request_refund", "args": {"order_id": 7890}, "id": "resolved-request_refund"}
    ]
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert repository.get_order(7890)["status"] == "refund_requested"


def test_pending_intent_cancel_blocked_status_does_not_mutate() -> None:
    repository = build_repository()
    reasoner = PendingIntentOnlyReasoner("cancel", order_id=7890)
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("pending-cancel-blocked", "Cancel order 7890", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert verifier_updates[-1]["verifier_decision"] == "blocked"
    assert response.verification_errors == ["Order 7890 is delivered and cannot be cancelled."]
    assert "cancelled_order" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


def test_refund_response_uses_customer_facing_status_wording() -> None:
    repository = build_repository()
    reasoner = PendingIntentOnlyReasoner(
        "refund",
        order_id=7890,
        response="The status is now refund_requested for order 7890.",
    )
    agent = CustomerServiceAgent(reasoner, repository)

    response, _ = agent.trace("refund-status-wording", "Refund order 7890", customer_id=7)

    assert response.verified_facts["refund_request"]["status"] == "refund_requested"
    assert "refund_requested" not in response.response
    assert "refund requested" in response.response.lower()


def test_current_turn_refund_response_does_not_say_already_requested() -> None:
    repository = build_repository()
    reasoner = PendingIntentOnlyReasoner(
        "refund",
        order_id=7890,
        response="The refund request is already submitted for order 7890.",
    )
    agent = CustomerServiceAgent(reasoner, repository)

    response, _ = agent.trace("refund-current-turn-not-already", "Refund order 7890", customer_id=7)

    assert response.verified_facts["refund_request"]["created_this_turn"] is True
    assert "already submitted" not in response.response.lower()
    assert "already requested" not in response.response.lower()


def test_responder_future_promise_sentence_is_removed() -> None:
    repository = build_repository()
    reasoner = PendingIntentOnlyReasoner(
        "refund",
        order_id=7890,
        response="I've submitted the refund request for order 7890. We will follow up soon.",
    )
    agent = CustomerServiceAgent(reasoner, repository)

    response, _ = agent.trace("refund-no-future-promise", "Refund order 7890", customer_id=7)

    assert "submitted the refund request" in response.response
    assert "will follow up" not in response.response.lower()


def test_successful_mutations_use_responder_instead_of_deterministic_templates() -> None:
    repository = build_repository()

    cancel_agent = CustomerServiceAgent(CancelAfterObservationReasoner(), repository)
    cancel_response, _ = cancel_agent.trace("cancel-template", "Cancel order 2468", customer_id=7)

    complaint_agent = CustomerServiceAgent(
        DirectMutationReasoner(
            "request_log_complaint",
            {"order_id": 7890, "issue": "package damaged"},
        ),
        repository,
    )
    complaint_response, _ = complaint_agent.trace(
        "complaint-template",
        "I want to complain about order 7890",
        customer_id=7,
    )

    memory_agent = CustomerServiceAgent(
        DirectMutationReasoner(
            "request_write_memory",
            {"key": "contact_preference", "value": "prefers email"},
        ),
        repository,
    )
    memory_response, _ = memory_agent.trace(
        "memory-template",
        "Remember I prefer email",
        customer_id=7,
    )

    assert cancel_response.verified_facts["cancellation_request"] == {
        "order_id": 2468,
        "status": "cancel_requested",
        "created_this_turn": True,
    }
    assert complaint_response.verified_facts["complaint_logged"]["order_id"] == 7890
    assert complaint_response.verified_facts["complaint_logged"]["issue"] == "package damaged"
    assert complaint_response.verified_facts["complaint_logged"]["created_this_turn"] is True
    assert memory_response.verified_facts["memory_written"]["key"] == "contact_preference"

    assert cancel_response.response == "LLM guessed cancel response"
    assert complaint_response.response == "LLM guessed mutation response"
    assert memory_response.response == "LLM guessed mutation response"
    assert cancel_response.response != "Cancellation request submitted for order 2468."
    assert complaint_response.response != "Complaint logged for order 7890."
    assert memory_response.response != "Memory updated: contact_preference."
    assert "current-turn cancellation" in " ".join(cancel_response.response_constraints)
    assert "brief empathy phrase" in " ".join(complaint_response.response_constraints)


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
