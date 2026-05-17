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
        action = {"name": "propose_refund", "args": {"order_id": 7890}, "id": "refund-1"}
        if not order:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": 7890},
                        "id": "lookup-1",
                    },
                    action,
                ],
                requested_actions=[action],
            )

        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        order_id = context.verified_facts.get("order", {}).get("order_id")
        return f"The refund request was submitted for order {order_id}."


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
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "Done."


class ReadThenStopReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        if self.plan_calls == 1:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": 7890},
                        "id": "lookup-1",
                    }
                ],
            )
        return ToolPlan()

    def respond(self, context: ResponseContext) -> str:
        return str(context.verified_facts.get("order", {}).get("order_id"))


class ReadWithoutContinuationReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        return ToolPlan(
            tool_calls=[
                {
                    "name": "order_lookup",
                    "args": {"order_id": 7890, "continue_after_read": False},
                    "id": "lookup-1",
                },
            ],
        )

    def respond(self, context: ResponseContext) -> str:
        return str(context.verified_facts.get("order", {}).get("order_id"))


class CombinedThenActionReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        action = {"name": "propose_refund", "args": {"order_id": 7890}, "id": "refund-1"}
        if self.plan_calls == 1:
            lookup = {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
            return ToolPlan(
                tool_calls=[lookup, action],
                requested_actions=[action],
            )
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "The refund request was submitted."


class ActionFirstThenLookupReasoner:
    def __init__(self) -> None:
        self.plan_snapshots: list[dict[str, Any]] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_snapshots.append(state_snapshot)
        action = {"name": "propose_refund", "args": {"order_id": 7890}, "id": "refund-1"}
        if len(self.plan_snapshots) == 1:
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
            )
        if not state_snapshot.get("tool_results", {}).get("order"):
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": 7890},
                        "id": "lookup-1",
                    },
                    action,
                ],
                requested_actions=[action],
            )
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "The refund request was submitted."


class CustomerReadReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        if "order" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": 7890},
                        "id": "lookup-1",
                    }
                ],
            )
        if "memory" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "read_customer_memory",
                        "args": {"customer_id": state_snapshot.get("authenticated_customer_id")},
                        "id": "memory-1",
                    }
                ],
            )
        return ToolPlan(
            tool_calls=[
                {
                    "name": "customer_profile",
                    "args": {"customer_id": state_snapshot.get("authenticated_customer_id")},
                    "id": "profile-1",
                }
            ],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return ",".join(sorted(context.verified_facts))


class PassiveReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        if "check order" in user_message.lower():
            order_id = 2468 if "2468" in user_message else 7890
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": order_id}, "id": "lookup-1"}
                ],
            )
        if "profile" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "customer_profile",
                        "args": {"customer_id": state_snapshot.get("authenticated_customer_id")},
                        "id": "profile-1",
                    }
                ],
            )
        if "issues" in user_message.lower():
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "read_customer_issue_history",
                        "args": {"customer_id": state_snapshot.get("authenticated_customer_id")},
                        "id": "issue-history-1",
                    }
                ],
            )
        return ToolPlan()

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "Done."


class SequentialAmbiguousComplaintReasoner:
    def __init__(self) -> None:
        self.plan_calls = 0

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_calls += 1
        if self.plan_calls == 1:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": 7890},
                        "id": "lookup-1",
                    }
                ],
            )
        if self.plan_calls == 2:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "customer_profile",
                        "args": {
                            "customer_id": state_snapshot.get("authenticated_customer_id"),
                        },
                        "id": "profile-1",
                    }
                ],
            )
        if self.plan_calls == 3:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "read_customer_issue_history",
                        "args": {
                            "customer_id": state_snapshot.get("authenticated_customer_id"),
                        },
                        "id": "issue-history-1",
                    }
                ],
            )
        action = {
            "name": "propose_log_complaint",
            "args": {
                "customer_id": state_snapshot.get("authenticated_customer_id"),
                "issue": "late again",
            },
            "id": "complaint-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "Done."


class ActiveOrderCancelReasoner:
    def __init__(self) -> None:
        self.last_order_id: int | None = None

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        normalized = user_message.lower()
        if "check order" in normalized:
            self.last_order_id = 2468
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 2468}, "id": "lookup-1"}
                ],
            )
        if (
            "cancel" in normalized
            and (state_snapshot.get("planner_feedback") or {}).get("code") == "ORDER_LOOKUP_REQUIRED_BEFORE_MUTATION"
        ):
            action = {
                "name": "propose_cancel_order",
                "args": {"order_id": self.last_order_id},
                "id": "cancel-1",
            }
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {
                            "order_id": self.last_order_id,
                        },
                        "id": "lookup-2",
                    },
                    action,
                ],
                requested_actions=[action],
            )
        if "cancel" in normalized:
            action = {
                "name": "propose_cancel_order",
                "args": {"order_id": self.last_order_id},
                "id": "cancel-1",
            }
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
            )
        return ToolPlan()

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "Done."


class ComplaintThenProfileReasoner:
    def __init__(self) -> None:
        self.plan_snapshots: list[dict[str, Any]] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_snapshots.append(state_snapshot)
        if "complaint" in user_message.lower():
            action = {
                "name": "propose_log_complaint",
                "args": {
                    "customer_id": state_snapshot.get("authenticated_customer_id"),
                    "order_id": 7890,
                    "issue": "package damaged",
                },
                "id": "complaint-1",
            }
            return ToolPlan(
                tool_calls=[action],
                requested_actions=[action],
            )
        return ToolPlan(
            tool_calls=[
                {
                    "name": "customer_profile",
                    "args": {"customer_id": state_snapshot.get("authenticated_customer_id")},
                    "id": "profile-1",
                }
            ],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return str(context.verified_facts)


class RefundByMessageReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        order_id = 5678 if "5678" in user_message else 7890
        order = state_snapshot.get("tool_results", {}).get("order")
        action = {"name": "propose_refund", "args": {"order_id": order_id}, "id": "refund-1"}
        if not order:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": order_id},
                        "id": "lookup-1",
                    },
                    action,
                ],
                requested_actions=[action],
            )
        return ToolPlan(
            tool_calls=[
                {"name": action["name"], "args": {"order_id": order["order_id"]}, "id": action["id"]}
            ],
            requested_actions=[
                {"name": action["name"], "args": {"order_id": order["order_id"]}, "id": action["id"]}
            ],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed refund response"


class CancelAfterObservationReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        order = state_snapshot.get("tool_results", {}).get("order")
        action = {"name": "propose_cancel_order", "args": {"order_id": 2468}, "id": "cancel-1"}
        if not order:
            return ToolPlan(
                tool_calls=[
                    {
                        "name": "order_lookup",
                        "args": {"order_id": 2468},
                        "id": "lookup-1",
                    },
                    action,
                ],
                requested_actions=[action],
            )
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed cancel response"


class NoToolMemoryWriteReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        return ToolPlan()

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed memory response"


class ExplicitMemoryWriteReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        action = {
            "name": "propose_write_memory",
            "args": {
                "customer_id": state_snapshot.get("authenticated_customer_id"),
                "key": "refund_preference",
                "value": "prefers refunds",
                "memory_type": "preference",
            },
            "id": "memory-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed memory response"


class MemoryTypeActionReasoner:
    def __init__(self, memory_type: str) -> None:
        self.memory_type = memory_type

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        action = {
            "name": "propose_write_memory",
            "args": {
                "customer_id": state_snapshot.get("authenticated_customer_id"),
                "key": "customer_note",
                "value": "not durable preference",
                "memory_type": self.memory_type,
            },
            "id": "memory-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return "Done."


class WarmClarificationReasoner:
    def __init__(self) -> None:
        self.response_contexts: list[ResponseContext] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        action = {
            "name": "propose_log_complaint",
            "args": {
                "customer_id": state_snapshot.get("authenticated_customer_id"),
                "order_id": 2222,
            },
            "id": "complaint-1",
        }
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        self.response_contexts.append(context)
        return "Could you tell me what issue you'd like to report for order 2222?"


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
            and (state_snapshot.get("planner_feedback") or {}).get("code") == "ORDER_LOOKUP_REQUIRED_BEFORE_MUTATION"
            and not order
        ):
            action_calls = self._tool_calls(state_snapshot.get("authenticated_customer_id"))
            lookup = {
                "name": "order_lookup",
                "args": {"order_id": self.order_id} if self.order_id is not None else {},
                "id": "order_lookup-1",
            }
            return ToolPlan(
                tool_calls=[lookup, *action_calls],
                requested_actions=action_calls,
            )

        tool_calls = self._tool_calls(state_snapshot.get("authenticated_customer_id"))
        return ToolPlan(
            tool_calls=tool_calls,
            requested_actions=tool_calls,
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_decision.get("decision") == "ask_user":
            return context.verification_decision.get("reason_code") or "NEEDS_USER_INPUT"
        return self.response

    def _tool_calls(self, customer_id: int | None) -> list[dict[str, Any]]:
        if self.request_type == "refund":
            return [
                {
                    "name": "propose_refund",
                    "args": {"order_id": self.order_id},
                    "id": "propose_refund-1",
                }
            ]
        if self.request_type == "cancel":
            return [
                {
                    "name": "propose_cancel_order",
                    "args": {"order_id": self.order_id},
                    "id": "propose_cancel_order-1",
                }
            ]
        if self.request_type == "complaint":
            return [
                {
                    "name": "propose_log_complaint",
                    "args": {
                        "customer_id": customer_id,
                        "order_id": self.order_id,
                        "issue": self.issue,
                    },
                    "id": "propose_log_complaint-1",
                }
            ]
        if self.request_type == "memory_write":
            return [
                {
                    "name": "propose_write_memory",
                    "args": {
                        "customer_id": customer_id,
                        "key": self.memory_key,
                        "value": self.memory_value,
                        "memory_type": "preference",
                    },
                    "id": "propose_write_memory-1",
                }
            ]
        return []


class DirectMutationReasoner:
    def __init__(self, action_name: str, args: dict[str, Any]):
        self.action_name = action_name
        self.args = args

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        args = dict(self.args)
        action = {"name": self.action_name, "args": args, "id": f"{self.action_name}-1"}
        return ToolPlan(
            tool_calls=[action],
            requested_actions=[action],
        )

    def respond(self, context: ResponseContext) -> str:
        return "LLM guessed mutation response"


class CrossCustomerReadReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        return ToolPlan(
            tool_calls=[
                {
                    "name": "customer_profile",
                    "args": {"customer_id": 2},
                    "id": "profile-1",
                },
                {
                    "name": "read_customer_memory",
                    "args": {"customer_id": 2},
                    "id": "memory-1",
                },
            ],
        )

    def respond(self, context: ResponseContext) -> str:
        return str(context.verified_facts)


class ListOrdersReasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        return ToolPlan(
            tool_calls=[
                {
                    "name": "list_my_orders",
                    "args": {},
                    "id": "orders-1",
                }
            ],
        )

    def respond(self, context: ResponseContext) -> str:
        return str(context.verified_facts.get("orders", []))


class TurnHistoryReasoner:
    def __init__(self) -> None:
        self.plan_snapshots: list[dict[str, Any]] = []

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        self.plan_snapshots.append(state_snapshot)
        if "7890" in user_message:
            return ToolPlan(
                tool_calls=[
                    {"name": "order_lookup", "args": {"order_id": 7890}, "id": "lookup-1"}
                ],
            )
        return ToolPlan()

    def respond(self, context: ResponseContext) -> str:
        order = context.verified_facts.get("order", {})
        if order:
            return f"Order {order['order_id']} is {order['status']}."
        return "No current facts."


def test_refund_uses_react_loop_before_action() -> None:
    repository = build_repository()
    reasoner = RefundAfterObservationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-refund", "Refund order 7890 if delivered", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert len(planner_updates) == 1
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert [action["name"] for action in planner_updates[0]["requested_actions"]] == [
        "propose_refund"
    ]
    assert reasoner.plan_snapshots[0]["tool_results"] == {}

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "propose_refund"
    ]
    assert verifier_updates[-1]["verifier_decision"] == "proceed_to_action"
    assert response.verification_decision["decision"] == "proceed_to_action"
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

    response, updates = agent.trace(
        "react-combined",
        "Refund order 7890 if delivered",
        customer_id=7,
    )

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [call["name"] for call in planner_updates[0]["tool_calls"]] == ["order_lookup"]
    assert [action["name"] for action in planner_updates[0]["requested_actions"]] == [
        "propose_refund"
    ]
    assert [update["verifier_decision"] for update in verifier_updates] == ["proceed_to_action"]
    assert [update["node"] for update in updates].count("actions") == 1
    assert reasoner.plan_calls == 1
    assert response.verification_decision["decision"] == "proceed_to_action"
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
    assert verifier_updates[0]["verification_decision"]["planner_feedback_code"] == (
        "ORDER_LOOKUP_REQUIRED_BEFORE_MUTATION"
    )
    assert [call["name"] for call in planner_updates[1]["tool_calls"]] == ["order_lookup"]
    assert [action["name"] for action in planner_updates[-1]["requested_actions"]] == [
        "propose_refund"
    ]
    assert response.verification_decision["decision"] == "proceed_to_action"
    assert response.tool_results["refund"]["status"] == "refund_requested"


def test_lookup_only_planner_does_not_fabricate_refund_action() -> None:
    repository = build_repository()
    reasoner = LookupOnlyReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("react-limit", "Refund order 7890 if delivered", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]

    assert reasoner.plan_calls == 1
    assert [update["node"] for update in updates].count("planner") == 1
    assert verifier_updates[-1]["requested_actions"] == []
    assert response.verification_decision["decision"] == "proceed_to_response"
    assert "refund" not in response.tool_results
    assert repository.get_order(7890)["status"] == "delivered"


def test_read_then_stop_returns_verified_order_without_action() -> None:
    repository = build_repository()
    reasoner = ReadThenStopReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("read-then-stop", "Where is my order 7890?", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert reasoner.plan_calls == 1
    assert planner_updates[-1]["tool_calls"][0]["args"] == {"order_id": 7890}
    assert response.order_id == 7890
    assert response.response == "7890"


def test_continue_after_read_false_routes_to_response_without_replanning() -> None:
    repository = build_repository()
    reasoner = ReadWithoutContinuationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("read-then-answer", "Where is my order 7890?", customer_id=7)

    assert reasoner.plan_calls == 1
    assert [update["node"] for update in updates].count("planner") == 1
    assert [update["node"] for update in updates] == [
        "planner",
        "read_tools",
        "verifier",
        "respond",
    ]
    assert response.order_id == 7890
    assert response.response == "7890"


def test_refund_owned_delivered_order_routes_success_through_responder() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(RefundByMessageReasoner(), repository)

    response, _ = agent.trace("refund-5678", "Refund order 5678", customer_id=2)

    assert response.tool_results["refund"]["status"] == "refund_requested"
    assert response.verified_facts["refund_request"] == {
        "order_id": 5678,
        "status": "refund_requested",
        "created_this_turn": True,
    }
    assert response.response == "LLM guessed refund response"
    assert repository.get_order(5678)["status"] == "refund_requested"


def test_explicit_memory_write_action_routes_success_through_responder() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(ExplicitMemoryWriteReasoner(), repository)

    response, updates = agent.trace("memory-write", "Remember I prefer refunds", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "propose_write_memory"
    ]
    assert response.tool_results["memory_write"]["key"] == "refund_preference"
    assert response.verified_facts["memory_written"]["key"] == "refund_preference"
    assert response.verified_facts["memory_written"]["created_this_turn"] is True
    assert response.response == "LLM guessed memory response"
    assert len(repository.read_memories(7, key="refund_preference")) == 1


def test_no_tool_memory_write_is_not_inferred_from_message_text() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(NoToolMemoryWriteReasoner(), repository)

    response, updates = agent.trace("memory-no-tool", "Remember I prefer refunds", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[0]["requested_actions"] == []
    assert "memory_write" not in response.tool_results
    assert len(repository.read_memories(7, key="refund_preference")) == 0


def test_action_request_resolve_to_verifier_actions() -> None:
    cases = [
        (
            "refund",
            ActionRequestReasoner("refund", order_id=7890),
            "propose_refund",
            "Refund order 7890",
            7,
        ),
        (
            "cancel",
            ActionRequestReasoner("cancel", order_id=2468),
            "propose_cancel_order",
            "Cancel order 2468",
            7,
        ),
        (
            "complaint",
            ActionRequestReasoner("complaint", order_id=7890, issue="package damaged"),
            "propose_log_complaint",
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
            "propose_write_memory",
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


def test_action_propose_refund_delivered_order_resolves_to_refund_action() -> None:
    repository = build_repository()
    reasoner = ActionRequestReasoner("refund", order_id=7890)
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("request-refund-resolve", "Refund order 7890", customer_id=7)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert [action["name"] for action in verifier_updates[-1]["requested_actions"]] == [
        "propose_refund"
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
    assert response.verification_decision.get("policy_errors", []) == [
        {
            "error_code": "ORDER_NOT_CANCELLABLE",
            "blocked_action": "propose_cancel_order",
            "order_id": 7890,
            "customer_id": None,
            "current_status": "delivered",
            "reason_code": "ORDER_ALREADY_DELIVERED",
        }
    ]
    assert response.verified_facts["policy_errors"] == response.verification_decision.get("policy_errors", [])
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
    assert response.verification_decision["decision"] == "ask_user"
    assert "complaint" not in response.tool_results
    assert "complaint_issue" in response.verification_decision.get("missing_slots", [])
    assert verifier_updates[-1]["verification_decision"]["decision"] == "ask_user"
    assert verifier_updates[-1]["verification_decision"]["missing_slots"] == ["complaint_issue"]


def test_ask_user_response_routes_through_responder() -> None:
    repository = build_repository()
    reasoner = WarmClarificationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, _ = agent.trace(
        "warm-clarification",
        "I want to complain about order 2222",
        customer_id=7,
    )

    assert reasoner.response_contexts
    verification_decision = reasoner.response_contexts[-1].verification_decision
    assert verification_decision["decision"] == "ask_user"
    assert verification_decision["reason_code"] == "COMPLAINT_ISSUE_MISSING"
    assert verification_decision["context"] == {"order_id": 2222}
    assert response.verification_decision["decision"] == "ask_user"
    assert response.response == "Could you tell me what issue you'd like to report for order 2222?"
    assert "complaint" not in response.tool_results


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

    assert response.verification_decision["decision"] == "proceed_to_action"
    assert response.tool_results["complaint"]["order_id"] == 2222
    assert "damaged" in response.tool_results["complaint"]["issue"].lower()


def test_ambiguous_order_asks_before_complaint() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(SequentialAmbiguousComplaintReasoner(), repository)

    agent.trace("ambiguous-order", "Check order 7890", customer_id=7)
    agent.trace("ambiguous-order", "Show my profile", customer_id=7)
    agent.trace("ambiguous-order", "What issues have I had before?", customer_id=7)
    response, updates = agent.trace("ambiguous-order", "My order is late again", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[-1]["requested_actions"][0]["args"].get("order_id") is None
    assert response.verification_decision["decision"] == "ask_user"
    assert "complaint" not in response.tool_results
    assert "order_id" in response.verification_decision.get("missing_slots", [])


def test_context_order_action_runs_lookup_before_cancel() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(ActiveOrderCancelReasoner(), repository)

    agent.trace("active-order-cancel", "Check order 2468", customer_id=7)
    response, updates = agent.trace("active-order-cancel", "Cancel it", customer_id=7)

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert [call["name"] for call in planner_updates[1]["tool_calls"]] == ["order_lookup"]
    assert response.verification_decision["decision"] == "proceed_to_action"
    assert response.tool_results["cancelled_order"]["order_id"] == 2468


def test_temporary_issue_is_not_written_to_long_term_memory() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(
        MemoryTypeActionReasoner(
            "temporary_issue",
        ),
        repository,
    )

    response, updates = agent.trace(
        "temporary-memory",
        "Remember this order is late",
        customer_id=7,
    )

    assert "memory_write" not in response.tool_results
    assert response.verification_decision["decision"] == "ask_user"
    assert response.verification_decision.get("missing_slots", []) == ["long_term_write_allowed"]
    assert not repository.read_memories(7, key="customer_note")


def test_transaction_request_is_not_written_to_long_term_memory() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(
        MemoryTypeActionReasoner(
            "transaction_request",
        ),
        repository,
    )

    response, updates = agent.trace(
        "transaction-memory",
        "Remember I want a refund",
        customer_id=7,
    )

    assert "memory_write" not in response.tool_results
    assert response.verification_decision["decision"] == "ask_user"
    assert response.verification_decision.get("missing_slots", []) == ["long_term_write_allowed"]


def test_successful_mutations_use_responder_instead_of_deterministic_templates() -> None:
    repository = build_repository()

    cancel_agent = CustomerServiceAgent(CancelAfterObservationReasoner(), repository)
    cancel_response, _ = cancel_agent.trace("cancel-template", "Cancel order 2468", customer_id=7)

    complaint_agent = CustomerServiceAgent(
        DirectMutationReasoner(
            "propose_log_complaint",
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
            "propose_write_memory",
            {
                "key": "contact_preference",
                "value": "prefers email",
                "memory_type": "preference",
            },
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
    assert second_response.response == "customer_memories"


def test_customer_scoped_reads_ignore_planner_customer_id() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(CrossCustomerReadReasoner(), repository)

    response, updates = agent.trace(
        "cross-customer-profile",
        "Show customer id 2",
        customer_id=7,
    )

    assert response.customer_id == 7
    assert response.tool_results["customer"]["customer_id"] == 7
    assert response.tool_results["customer"]["name"] == "Dana Lee"
    assert response.verified_facts["customer"]["customer_id"] == 7
    assert "Bob Lin" not in response.response


def test_list_my_orders_only_returns_authenticated_customer_orders() -> None:
    repository = build_repository()
    agent = CustomerServiceAgent(ListOrdersReasoner(), repository)

    response, updates = agent.trace(
        "list-my-orders",
        "Show customer id 2 orders",
        customer_id=7,
    )

    planner_updates = [update["state"] for update in updates if update["node"] == "planner"]
    assert planner_updates[-1]["tool_calls"] == [
        {"name": "list_my_orders", "args": {}, "id": "orders-1"}
    ]
    assert [order["order_id"] for order in response.tool_results["orders"]] == [
        2468,
        7890,
        2222,
    ]
    assert response.verified_facts["orders"] == [
        {
            "order_id": 2468,
            "product_name": "Monitor Arm",
            "status": "processing",
            "order_date": "2026-04-21T09:00:00",
            "delivery_date": None,
        },
        {
            "order_id": 7890,
            "product_name": "Standing Desk",
            "status": "delivered",
            "order_date": "2026-04-20T09:00:00",
            "delivery_date": "2026-04-24T15:00:00",
        },
        {
            "order_id": 2222,
            "product_name": "Desk Lamp",
            "status": "delivered",
            "order_date": "2026-04-18T09:00:00",
            "delivery_date": "2026-04-20T12:00:00",
        },
    ]
    assert "5678" not in response.response
    assert "Gaming Keyboard" not in response.response


def test_turn_history_carries_previous_decision_facts_response_and_message() -> None:
    repository = build_repository()
    reasoner = TurnHistoryReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    first_response, _ = agent.trace(
        "turn-history",
        "Check order 7890",
        customer_id=7,
    )
    second_response, _ = agent.trace(
        "turn-history",
        "Cancel it",
        customer_id=7,
    )

    assert first_response.verified_facts["order"]["order_id"] == 7890
    assert reasoner.plan_snapshots[0]["recent_turns"] == []
    assert reasoner.plan_snapshots[1]["tool_results"] == {}
    assert reasoner.plan_snapshots[1]["recent_turns"] == [
        {
            "user_message": "Check order 7890",
            "assistant_response": "Order 7890 is delivered.",
            "verification_decision": {"decision": "proceed_to_response"},
            "verified_facts": {
                "order": {
                    "order_id": 7890,
                    "customer_id": 7,
                    "product_name": "Standing Desk",
                    "status": "delivered",
                    "order_date": "2026-04-20T09:00:00",
                    "delivery_date": "2026-04-24T15:00:00",
                }
            },
        }
    ]
    assert second_response.response == "No current facts."


def test_customer_cannot_refund_another_customers_order() -> None:
    repository = build_repository()
    reasoner = RefundAfterObservationReasoner()
    agent = CustomerServiceAgent(reasoner, repository)

    response, updates = agent.trace("ownership-refund", "Refund order 7890 if delivered", customer_id=8)

    verifier_updates = [update["state"] for update in updates if update["node"] == "verifier"]
    assert verifier_updates[-1]["verifier_decision"] == "block"
    assert response.verification_decision["decision"] == "block"
    assert response.tool_results["order_lookup"] == {"order_id": 7890, "found": False}
    assert "order" not in response.tool_results
    assert response.verification_decision.get("policy_errors", []) == [
        {
            "error_code": "ORDER_NOT_FOUND",
            "blocked_action": "propose_refund",
            "order_id": 7890,
            "customer_id": None,
            "current_status": None,
            "reason_code": "ORDER_NOT_FOUND",
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

    assert complaint_response.tool_results["complaint"]["issue"] == "package damaged"
    assert "issue" not in reasoner.plan_snapshots[-1]
    assert "package damaged" not in profile_response.response
