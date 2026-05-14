from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from customer_service_agent.graph.agent import CustomerServiceAgent
from customer_service_agent.models import Base, Customer, CustomerMemory, MemoryWriteCandidate, Order
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
        session.add(
            Order(
                order_id=2222,
                customer_id=7,
                product_name="Desk Lamp",
                status="delivered",
                order_date=datetime.fromisoformat("2026-04-18T09:00:00"),
                delivery_date=datetime.fromisoformat("2026-04-20T12:00:00"),
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
                requires_replan_after_tools=True,
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
                requires_replan_after_tools=False,
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


class ActionFirstThenLookupReasoner:
    def __init__(self) -> None:
        self.plan_snapshots: list[dict[str, Any]] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_snapshots.append(state_snapshot)
        action = {"name": "request_refund", "args": {"order_id": 7890}, "id": "refund-1"}
        if len(self.plan_snapshots) == 1:
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
                order_id=7890,
                steps=["request_refund"],
                reasoning="Incorrectly requested refund before reading order details.",
            )
        if not state_snapshot.get("tool_results", {}).get("order"):
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
                ],
                order_id=7890,
                requires_replan_after_tools=True,
                steps=["order_lookup"],
                reasoning="Verifier requested order lookup before refund.",
            )
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            order_id=7890,
            steps=["request_refund"],
            reasoning="Request refund after verified order observation.",
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


class PassiveReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        if "check order" in user_message.lower():
            order_id = 2468 if "2468" in user_message else 7890
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": order_id}, "id": "lookup-1"}
                ],
                order_id=order_id,
                requires_replan_after_tools=True,
                steps=["order_lookup"],
            )
        if "profile" in user_message.lower():
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
        if "issues" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "list_customer_complaints",
                        "args": {"customer_id": state_snapshot.get("active_customer_id")},
                        "id": "complaints-1",
                    }
                ],
                customer_id=state_snapshot.get("active_customer_id"),
                steps=["list_customer_complaints"],
            )
        return ToolPlan(customer_id=state_snapshot.get("active_customer_id"))

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return "Done."


class SequentialAmbiguousComplaintReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        if self.plan_calls == 1:
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
                ],
                order_id=7890,
                steps=["order_lookup"],
            )
        if self.plan_calls == 2:
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
        if self.plan_calls == 3:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "list_customer_complaints",
                        "args": {"customer_id": state_snapshot.get("active_customer_id")},
                        "id": "complaints-1",
                    }
                ],
                customer_id=state_snapshot.get("active_customer_id"),
                steps=["list_customer_complaints"],
            )
        action = {
            "name": "request_log_complaint",
            "args": {
                "customer_id": state_snapshot.get("active_customer_id"),
                "issue": "late again",
            },
            "id": "complaint-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            customer_id=state_snapshot.get("active_customer_id"),
            issue="late again",
            steps=["request_log_complaint"],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return "Done."


class PronounCancelReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        normalized = user_message.lower()
        if "check order" in normalized:
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 2468}, "id": "lookup-1"}
                ],
                order_id=2468,
                requires_replan_after_tools=True,
                steps=["order_lookup"],
            )
        if (
            "cancel" in normalized
            and state_snapshot.get("planner_feedback") == "Need order lookup before mutation."
        ):
            return ToolPlan(
                tool_calls=[{"name": "order_lookup", "args": {}, "id": "lookup-2"}],
                requires_replan_after_tools=True,
                steps=["order_lookup"],
            )
        if "cancel" in normalized:
            action = {"name": "request_cancel_order", "args": {}, "id": "cancel-1"}
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
                steps=["request_cancel_order"],
            )
        return ToolPlan(customer_id=state_snapshot.get("active_customer_id"))

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return "Done."


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
                    "order_id": 7890,
                    "issue": "package damaged",
                },
                "id": "complaint-1",
            }
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
                customer_id=state_snapshot.get("active_customer_id"),
                order_id=7890,
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
                requires_replan_after_tools=True,
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
                requires_replan_after_tools=True,
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


class ExplicitMemoryWriteReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        action = {
            "name": "request_write_memory",
            "args": {
                "customer_id": state_snapshot.get("active_customer_id"),
                "key": "refund_preference",
                "value": "prefers refunds",
            },
            "id": "memory-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            customer_id=state_snapshot.get("active_customer_id"),
            memory_key="refund_preference",
            memory_value="prefers refunds",
            memory_candidate=MemoryWriteCandidate(
                should_write=True,
                memory_type="preference",
                key="refund_preference",
                value="prefers refunds",
                reason="Planner classified this as a durable preference.",
            ),
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed memory response"


class MemoryCandidateReasoner:
    def __init__(self, memory_type: str, reason: str) -> None:
        self.memory_type = memory_type
        self.reason = reason

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        action = {
            "name": "request_write_memory",
            "args": {"customer_id": state_snapshot.get("active_customer_id")},
            "id": "memory-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
            customer_id=state_snapshot.get("active_customer_id"),
            memory_candidate=MemoryWriteCandidate(
                should_write=False,
                memory_type=self.memory_type,
                reason=self.reason,
            ),
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return "Done."


class ActionRequestReasoner:
    def __init__(
        self,
        request_type: str,
        *,
        order_id: int | None = None,
        issue: str | None = None,
        memory_key: str | None = None,
        memory_value: str | None = None,
        response: str = "Done.",
    ) -> None:
        self.request_type = request_type
        self.order_id = order_id
        self.issue = issue
        self.memory_key = memory_key
        self.memory_value = memory_value
        self.response = response

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        order = state_snapshot.get("tool_results", {}).get("order")
        if (
            self.request_type in {"refund", "cancel"}
            and state_snapshot.get("planner_feedback") == "Need order lookup before mutation."
            and not order
        ):
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": self.order_id} if self.order_id is not None else {},
                        "id": "order_lookup-1",
                    }
                ],
                customer_id=state_snapshot.get("active_customer_id"),
                order_id=self.order_id,
                requires_replan_after_tools=True,
                steps=["order_lookup"],
                reasoning="Verifier requested order lookup before mutation.",
            )

        tool_calls = self._tool_calls(state_snapshot.get("active_customer_id"))
        return ToolPlan(
            customer_id=state_snapshot.get("active_customer_id"),
            order_id=self.order_id,
            issue=self.issue,
            memory_key=self.memory_key,
            memory_value=self.memory_value,
            tool_calls=tool_calls,
            requested_actions=tool_calls,
            reasoning=f"Planner requested {self.request_type}.",
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
        return self.response

    def _tool_calls(self, customer_id: int | None) -> list[dict[str, Any]]:
        if self.request_type == "refund":
            return [
                {
                    "name": "request_refund",
                    "args": {"order_id": self.order_id},
                    "id": "request_refund-1",
                }
            ]
        if self.request_type == "cancel":
            return [
                {
                    "name": "request_cancel_order",
                    "args": {"order_id": self.order_id},
                    "id": "request_cancel_order-1",
                }
            ]
        if self.request_type == "complaint":
            return [
                {
                    "name": "request_log_complaint",
                    "args": {
                        "customer_id": customer_id,
                        "order_id": self.order_id,
                        "issue": self.issue,
                    },
                    "id": "request_log_complaint-1",
                }
            ]
        if self.request_type == "memory_write":
            return [
                {
                    "name": "request_write_memory",
                    "args": {
                        "customer_id": customer_id,
                        "key": self.memory_key,
                        "value": self.memory_value,
                    },
                    "id": "request_write_memory-1",
                }
            ]
        return []


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
    assert len(planner_updates) == 2
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert planner_updates[0]["requested_actions"] == []
    assert [action["name"] for action in planner_updates[1]["requested_actions"]] == [
        "request_refund"
    ]
    assert planner_updates[0]["order_reference"] == {
        "order_id": 7890,
        "source": "explicit",
        "confidence": "high",
    }

    assert reasoner.plan_snapshots[0]["tool_results"] == {}

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "request_refund"
    ]
    assert verifier_updates[-1]["verifier_decision"] == "proceed_to_action"
    assert response.verifier_decision == "proceed_to_action"
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert response.verified_facts["refund_request"] == {
        "order_id": 7890,
        "status": "refund_requested",
        "created_this_turn": True,
    }
    assert response.response == "The refund request was submitted for order 7890."
    assert repository.get_order(7890)["status"] == "refund_requested"


def test_planner_action_calls_are_verifier_gated() -> None:
    repository = build_repository()
    reasoner = CombinedThenActionReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-combined", "Refund order 7890 if delivered")

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert [action["name"] for action in planner_updates[0]["requested_actions"]] == [
        "request_refund"
    ]
    assert [update["verifier_decision"] for update in verifier_updates] == ["proceed_to_action"]
    assert [update["node"] for update in updates].count("actions") == 1
    assert reasoner.plan_calls == 1
    assert response.verifier_decision == "proceed_to_action"
    assert response.tool_results["refund"]["status"] == "refund_requested"


def test_verifier_replans_when_mutation_lacks_order_observation() -> None:
    repository = build_repository()
    reasoner = ActionFirstThenLookupReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("verifier-replan", "Refund order 7890", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [update["verifier_decision"] for update in verifier_updates] == [
        "replan",
        "proceed_to_action",
    ]
    assert verifier_updates[0]["verification_decision"]["missing_slots"] == [
        "order_status",
        "order_customer_id",
    ]
    assert verifier_updates[0]["verification_decision"]["planner_feedback"] == (
        "Need order lookup before mutation."
    )
    assert [call["name"] for call in planner_updates[1]["tool_calls"]] == ["order_lookup"]
    assert [action["name"] for action in planner_updates[-1]["requested_actions"]] == [
        "request_refund"
    ]
    assert response.verifier_decision == "proceed_to_action"
    assert response.tool_results["refund"]["status"] == "refund_requested"


def test_lookup_only_planner_does_not_fabricate_refund_action() -> None:
    repository = build_repository()
    reasoner = LookupOnlyReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-limit", "Refund order 7890 if delivered")

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]

    assert reasoner.plan_calls == 1
    assert [update["node"] for update in updates].count("planner") == 1
    assert verifier_updates[-1]["requested_actions"] == []
    assert response.verifier_decision == "proceed_to_response"
    assert response.verification_errors == []
    assert "refund" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


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


def test_explicit_memory_write_candidate_routes_success_through_responder() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(ExplicitMemoryWriteReasoner(), repository)

    response, updates = agent.trace("memory-write", "Remember I prefer refunds", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert planner_updates[0]["memory_candidate"]["memory_type"] == "preference"
    assert planner_updates[0]["memory_candidate"]["should_write"] is True
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "request_write_memory"
    ]
    assert response.verification_errors == []
    assert response.tool_results["memory_write"]["key"] == "refund_preference"
    assert response.verified_facts["memory_written"]["key"] == "refund_preference"
    assert response.verified_facts["memory_written"]["created_this_turn"] is True
    assert "saved key or value" in " ".join(response.response_constraints)
    assert response.response == "LLM guessed memory response"
    assert len(repository.read_memories(7, key="refund_preference")) == 1


def test_no_tool_memory_write_is_not_inferred_from_message_text() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(NoToolMemoryWriteReasoner(), repository)

    response, updates = agent.trace("memory-no-tool", "Remember I prefer refunds", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[0]["requested_actions"] == []
    assert planner_updates[0]["memory_candidate"]["should_write"] is False
    assert "memory_write" not in response.tool_results
    assert len(repository.read_memories(7, key="refund_preference")) == 0


def test_action_request_resolve_to_verifier_actions() -> None:
    cases = [
        (
            "refund",
            ActionRequestReasoner("refund", order_id=7890),
            "request_refund",
            "Refund order 7890",
            7,
        ),
        (
            "cancel",
            ActionRequestReasoner("cancel", order_id=2468),
            "request_cancel_order",
            "Cancel order 2468",
            7,
        ),
        (
            "complaint",
            ActionRequestReasoner("complaint", order_id=7890, issue="package damaged"),
            "request_log_complaint",
            "I want to complain about order 7890",
            7,
        ),
        (
            "memory_write",
            ActionRequestReasoner(
                "memory_write",
                memory_key="contact_preference",
                memory_value="prefers email",
            ),
            "request_write_memory",
            "Remember I prefer email",
            7,
        ),
    ]

    for request_type, reasoner, expected_action, message, customer_id in cases:
        repository = build_repository()
        agent = CustomerServiceAgent(reasoner, repository)
        response, updates = agent.trace(f"request-{request_type}", message, customer_id=customer_id)

        verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
        assert expected_action in [
            action["name"] for action in verifier_updates[-1]["requested_actions"]
        ]
        assert response.verification_errors == []


def test_action_request_refund_delivered_order_resolves_to_refund_action() -> None:
    repository = build_repository()
    reasoner = ActionRequestReasoner("refund", order_id=7890)
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("request-refund-resolve", "Refund order 7890", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "request_refund"
    ]
    assert verifier_updates[-1]["requested_actions"][0]["args"] == {"order_id": 7890}
    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert repository.get_order(7890)["status"] == "refund_requested"


def test_action_request_cancel_block_status_does_not_mutate() -> None:
    repository = build_repository()
    reasoner = ActionRequestReasoner("cancel", order_id=7890)
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("request-cancel-block", "Cancel order 7890", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert verifier_updates[-1]["verifier_decision"] == "block"
    assert response.verification_errors == []
    assert response.policy_errors == [
        {
            "error_code": "ORDER_NOT_CANCELLABLE",
            "blocked_action": "request_cancel_order",
            "order_id": 7890,
            "customer_id": None,
            "current_status": "delivered",
            "reason": "order already delivered",
        }
    ]
    assert response.verified_facts["policy_errors"] == response.policy_errors
    assert "cancelled_order" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


def test_complaint_missing_issue_asks_before_logging() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(ActionRequestReasoner("complaint", order_id=2222), repository)

    response, updates = agent.trace(
        "complaint-missing-issue",
        "I want to complain about order 2222",
        customer_id=7,
    )

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert response.verifier_decision == "ask_user"
    assert "complaint" not in response.tool_results
    assert "complaint_issue" in response.missing_slots
    assert verifier_updates[-1]["verification_decision"]["decision"] == "ask_user"
    assert verifier_updates[-1]["verification_decision"]["missing_slots"] == ["complaint_issue"]


def test_complaint_with_issue_logs_complaint() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(
        ActionRequestReasoner("complaint", order_id=2222, issue="arrived damaged"),
        repository,
    )

    response, _ = agent.trace(
        "complaint-with-issue",
        "I want to complain about order 2222. It arrived damaged.",
        customer_id=7,
    )

    assert response.verifier_decision == "proceed_to_action"
    assert response.tool_results["complaint"]["order_id"] == 2222
    assert "damaged" in response.tool_results["complaint"]["issue"].lower()


def test_ambiguous_order_reference_asks_before_complaint() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(SequentialAmbiguousComplaintReasoner(), repository)

    agent.trace("ambiguous-order", "Check order 7890", customer_id=7)
    agent.trace("ambiguous-order", "Show my profile", customer_id=7)
    agent.trace("ambiguous-order", "What issues have I had before?", customer_id=7)
    response, updates = agent.trace("ambiguous-order", "My order is late again", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[-1]["order_reference"]["confidence"] == "low"
    assert response.verifier_decision == "ask_user"
    assert "complaint" not in response.tool_results
    assert "order_id" in response.missing_slots


def test_strong_pronoun_reference_runs_lookup_before_cancel() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(PronounCancelReasoner(), repository)

    agent.trace("pronoun-cancel", "Check order 2468", customer_id=7)
    response, updates = agent.trace("pronoun-cancel", "Cancel it", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[0]["order_reference"] == {
        "order_id": 2468,
        "source": "pronoun",
        "confidence": "high",
    }
    assert [call["name"] for call in planner_updates[1]["tool_calls"]] == ["order_lookup"]
    assert response.verifier_decision == "proceed_to_action"
    assert response.tool_results["cancelled_order"]["order_id"] == 2468


def test_temporary_issue_is_not_written_to_long_term_memory() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(
        MemoryCandidateReasoner(
            "temporary_issue",
            "Planner classified this as a temporary order issue.",
        ),
        repository,
    )

    response, updates = agent.trace(
        "temporary-memory",
        "Remember this order is late",
        customer_id=7,
    )

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[-1]["memory_candidate"]["memory_type"] == "temporary_issue"
    assert "memory_write" not in response.tool_results
    assert response.verifier_decision == "ask_user"
    assert not repository.read_memories(7, key="customer_note")


def test_transaction_request_is_not_written_to_long_term_memory() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(
        MemoryCandidateReasoner(
            "transaction_request",
            "Planner classified this as a transaction request.",
        ),
        repository,
    )

    response, updates = agent.trace(
        "transaction-memory",
        "Remember I want a refund",
        customer_id=7,
    )

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[-1]["memory_candidate"]["memory_type"] == "transaction_request"
    assert "memory_write" not in response.tool_results
    assert response.verifier_decision == "ask_user"
    assert response.missing_slots == ["long_term_write_allowed"]


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
    assert verifier_updates[-1]["verifier_decision"] == "block"
    assert response.verifier_decision == "block"
    assert response.verification_errors == []
    assert response.policy_errors == [
        {
            "error_code": "ORDER_CUSTOMER_MISMATCH",
            "blocked_action": "request_refund",
            "order_id": 7890,
            "customer_id": 8,
            "current_status": None,
            "reason": "order belongs to a different customer",
        }
    ]
    assert "refund" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


def test_stale_complaint_issue_does_not_reach_later_unrelated_request() -> None:
    repository = build_repository()
    reasoner = ComplaintThenProfileReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    complaint_response, _ = agent.trace(
        "issue-pollution",
        "Log a complaint about order 7890",
        customer_id=7,
    )
    profile_response, profile_updates = agent.trace("issue-pollution", "Show my profile", customer_id=7)

    planner_updates = [update["state"] for update in profile_updates if update["node"] == "planner"]
    assert complaint_response.tool_results["complaint"]["issue"] == "package damaged"
    assert reasoner.plan_snapshots[-1]["issue"] is None
    assert planner_updates[-1]["issue"] is None
    assert "package damaged" not in profile_response.response
