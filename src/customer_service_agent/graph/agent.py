from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service_agent.models import AgentState, ChatResponse
from customer_service_agent.reasoning import ResponseContext


DEFAULT_MAX_REACT_ITERATIONS = 3


def build_graph(reasoner, repository, max_react_iterations: int = DEFAULT_MAX_REACT_ITERATIONS):
    def planner_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1].content if state.get("messages") else ""
        react_iterations = int(state.get("react_iterations") or 0) + 1
        state_snapshot = {
            "active_customer_id": state.get("active_customer_id"),
            "active_order_id": state.get("active_order_id"),
            "issue": state.get("issue"),
            "memory_key": state.get("memory_key"),
            "memory_value": state.get("memory_value"),
            "pending_intent": state.get("pending_intent"),
            "pending_action": state.get("pending_action"),
            "pending_order_id": state.get("pending_order_id"),
            "tool_results": state.get("tool_results", {}),
            "react_iterations": react_iterations,
            "max_react_iterations": state.get("max_react_iterations") or max_react_iterations,
        }
        plan = reasoner.plan(str(last_message), state_snapshot)
        plan = _apply_pending_intent_to_plan(plan, str(last_message), state_snapshot)
        next_customer_id = (
            plan.customer_id if plan.customer_id is not None else state.get("active_customer_id")
        )
        next_order_id = plan.order_id if plan.order_id is not None else state.get("active_order_id")
        pending_intent = plan.pending_intent
        pending_action = plan.pending_action
        pending_order_id = (
            plan.pending_order_id if plan.pending_order_id is not None else next_order_id
        )
        return {
            "plan_steps": plan.steps,
            "reasoning": plan.reasoning,
            "active_customer_id": next_customer_id,
            "active_order_id": next_order_id,
            "issue": plan.issue,
            "memory_key": plan.memory_key,
            "memory_value": plan.memory_value,
            "pending_intent": pending_intent,
            "pending_action": pending_action,
            "pending_order_id": pending_order_id if pending_intent else None,
            "tool_calls": plan.tool_calls,
            "requested_actions": plan.requested_actions,
            "requires_follow_up": plan.requires_follow_up,
            "follow_up_question": plan.follow_up_question,
            "tool_results": state.get("tool_results", {}),
            "verifier_decision": None,
            "verification_errors": [],
            "verified_facts": {},
            "response_constraints": [],
            "react_iterations": react_iterations,
            "max_react_iterations": state.get("max_react_iterations") or max_react_iterations,
            "long_term_memory": [],
            "final_response": None,
        }

    def read_tool_node(state: AgentState) -> dict[str, Any]:
        if state.get("requires_follow_up"):
            return {"tool_results": state.get("tool_results", {})}

        tool_results: dict[str, Any] = dict(state.get("tool_results", {}))
        active_customer_id = state.get("active_customer_id")
        active_order_id = state.get("active_order_id")

        for tool_call in state.get("tool_calls", []):
            name = tool_call.get("name")
            args = tool_call.get("args", {})

            if name == "order_lookup":
                order_id = _prefer_explicit_int(args.get("order_id"), active_order_id)
                if order_id is not None:
                    active_order_id = order_id
                    order = repository.get_order(order_id)
                    if order:
                        tool_results["order"] = order
                        if active_customer_id is None:
                            active_customer_id = order["customer_id"]

            elif name == "customer_profile":
                customer_id = _prefer_explicit_int(args.get("customer_id"), active_customer_id)
                if customer_id is not None:
                    active_customer_id = customer_id
                    tool_results["customer"] = repository.get_customer(customer_id)

            elif name == "read_customer_memory":
                customer_id = _prefer_explicit_int(args.get("customer_id"), active_customer_id)
                if customer_id is not None:
                    active_customer_id = customer_id
                    tool_results["memories"] = repository.read_memories(customer_id)

            elif name == "list_customer_complaints":
                customer_id = _prefer_explicit_int(args.get("customer_id"), active_customer_id)
                if customer_id is not None:
                    active_customer_id = customer_id
                    tool_results["complaints"] = repository.list_complaints(customer_id)

            elif name == "summarize_issue_patterns":
                customer_id = _prefer_explicit_int(args.get("customer_id"), active_customer_id)
                if customer_id is not None:
                    active_customer_id = customer_id
                    tool_results["issue_patterns"] = repository.summarize_issue_patterns(customer_id)

        return {
            "tool_results": tool_results,
            "active_customer_id": active_customer_id,
            "active_order_id": active_order_id,
        }

    def memory_node(state: AgentState) -> dict[str, Any]:
        if state.get("requires_follow_up"):
            return {"long_term_memory": []}

        active_customer_id = state.get("active_customer_id")
        if not active_customer_id:
            return {"long_term_memory": []}

        memories = repository.read_memories(active_customer_id)
        long_term_memory = memories[:5]

        complaint = state.get("tool_results", {}).get("complaint")
        if complaint:
            long_term_memory = repository.read_memories(active_customer_id)[:5]

        return {"long_term_memory": long_term_memory}

    def verifier_node(state: AgentState) -> dict[str, Any]:
        if state.get("requires_follow_up") and not state.get("pending_intent"):
            return {
                "verifier_decision": "ask_user",
                "verification_errors": [
                    state.get("follow_up_question")
                    or "I need a little more detail to help with that."
                ],
            }

        errors: list[str] = []
        tool_results = dict(state.get("tool_results", {}))
        order = tool_results.get("order")
        active_order_id = state.get("active_order_id")
        active_customer_id = state.get("active_customer_id")
        last_message = state["messages"][-1].content if state.get("messages") else ""
        pending_intent = state.get("pending_intent")
        pending_action = state.get("pending_action")
        pending_order_id = state.get("pending_order_id")
        requested_mutation = (
            str(pending_intent)
            if pending_intent in {"refund", "cancel"}
            else _requested_order_mutation(str(last_message))
        )
        requested_actions = list(state.get("requested_actions", []))
        action_names = {action.get("name") for action in requested_actions}
        called_order_lookup = any(
            call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
        )
        needs_order = bool(
            action_names & {"request_refund", "request_cancel_order"}
            or requested_mutation
            or called_order_lookup
        )
        resolved_reasoning: str | None = None
        can_replan = int(state.get("react_iterations") or 0) < int(
            state.get("max_react_iterations") or max_react_iterations
        )

        if needs_order and active_order_id is None:
            errors.append("I need an order ID to continue.")
            return {
                "verifier_decision": "ask_user",
                "verification_errors": errors,
                "tool_results": tool_results,
            }

        if needs_order and active_order_id is not None and not order:
            if not called_order_lookup and can_replan:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": tool_results,
                }
            errors.append(f"Order {active_order_id} does not exist.")
            return {
                "verifier_decision": "blocked",
                "verification_errors": errors,
                "tool_results": tool_results,
            }

        order_action_names = {"request_refund", "request_cancel_order"}
        if active_customer_id is not None and order and order["customer_id"] != active_customer_id:
            if action_names & order_action_names or requested_mutation in {"refund", "cancel"}:
                errors.append(
                    f"Order {order['order_id']} does not belong to customer {active_customer_id}."
                )
                return {
                    "verifier_decision": "blocked",
                    "verification_errors": errors,
                    "tool_results": tool_results,
                }

        combined_read_and_order_action = bool(
            action_names & order_action_names and called_order_lookup
        )
        if combined_read_and_order_action:
            if can_replan:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": tool_results,
                }
            errors.append(
                "I could not complete the order action within the reasoning step limit."
            )
            return {
                "verifier_decision": "blocked",
                "verification_errors": errors,
                "tool_results": tool_results,
            }

        if requested_mutation == "refund" and "request_refund" not in action_names:
            if order and order["status"] != "delivered":
                errors.append(
                    f"Order {order['order_id']} is currently `{order['status']}` and is not eligible for refund yet."
                )
                return {
                    "verifier_decision": "blocked",
                    "verification_errors": errors,
                    "tool_results": tool_results,
                }
            if order:
                action = _resolved_pending_action(
                    "refund",
                    order_id=order["order_id"],
                    customer_id=active_customer_id,
                )
                requested_actions.append(action)
                action_names.add(action["name"])
                pending_action = action["name"]
                pending_order_id = order["order_id"]
                resolved_reasoning = (
                    "Resolved pending intent refund into request_refund after verified order observation."
                )
            elif can_replan:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": tool_results,
                }
            else:
                errors.append("I could not complete the refund request within the reasoning step limit.")
                return {
                    "verifier_decision": "blocked",
                    "verification_errors": errors,
                    "tool_results": tool_results,
                }

        if requested_mutation == "cancel" and "request_cancel_order" not in action_names:
            if order and order["status"] in {"delivered", "refund_requested"}:
                errors.append(
                    f"Order {order['order_id']} is currently `{order['status']}` and cannot be cancelled."
                )
                return {
                    "verifier_decision": "blocked",
                    "verification_errors": errors,
                    "tool_results": tool_results,
                }
            if order:
                action = _resolved_pending_action(
                    "cancel",
                    order_id=order["order_id"],
                    customer_id=active_customer_id,
                )
                requested_actions.append(action)
                action_names.add(action["name"])
                pending_action = action["name"]
                pending_order_id = order["order_id"]
                resolved_reasoning = (
                    "Resolved pending intent cancel into request_cancel_order after verified order observation."
                )
            elif can_replan:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": tool_results,
                }
            else:
                errors.append("I could not complete the cancellation request within the reasoning step limit.")
                return {
                    "verifier_decision": "blocked",
                    "verification_errors": errors,
                    "tool_results": tool_results,
                }

        if pending_intent == "complaint" and "request_log_complaint" not in action_names:
            action = _resolved_pending_action(
                "complaint",
                order_id=_prefer_explicit_int(pending_order_id, active_order_id),
                customer_id=active_customer_id,
                issue=state.get("issue") or _complaint_issue(str(last_message)),
            )
            requested_actions.append(action)
            action_names.add(action["name"])
            pending_action = action["name"]
            resolved_reasoning = "Resolved pending intent complaint into request_log_complaint."

        if pending_intent == "memory_write" and "request_write_memory" not in action_names:
            if not state.get("memory_key") or not state.get("memory_value"):
                errors.append("I need a little more detail to help with that.")
                return {
                    "verifier_decision": "ask_user",
                    "verification_errors": errors,
                    "tool_results": tool_results,
                }
            action = _resolved_pending_action(
                "memory_write",
                customer_id=active_customer_id,
                memory_key=state.get("memory_key"),
                memory_value=state.get("memory_value"),
            )
            requested_actions.append(action)
            action_names.add(action["name"])
            pending_action = action["name"]
            resolved_reasoning = "Resolved pending intent memory_write into request_write_memory."

        if "request_refund" in action_names and order:
            if order["status"] != "delivered":
                errors.append(
                    f"Order {order['order_id']} is currently `{order['status']}` and is not eligible for refund yet."
                )

        if "request_cancel_order" in action_names and order:
            if order["status"] in {"delivered", "refund_requested"}:
                errors.append(
                    f"Order {order['order_id']} is currently `{order['status']}` and cannot be cancelled."
                )

        called_customer_profile = any(
            call.get("name") == "customer_profile" for call in state.get("tool_calls", [])
        )
        if called_customer_profile and not tool_results.get("customer"):
            errors.append("Customer profile was not found.")

        if "request_log_complaint" in action_names and not active_customer_id:
            errors.append("I could not log the complaint without a valid customer or order.")

        if "request_write_memory" in action_names and not active_customer_id:
            errors.append("I could not store that preference without a valid customer ID.")

        decision = "blocked" if errors else "approved"
        result = {
            "verifier_decision": decision,
            "verification_errors": errors,
            "tool_results": tool_results,
            "requested_actions": requested_actions,
            "pending_action": pending_action,
            "pending_order_id": pending_order_id,
        }
        if resolved_reasoning:
            result["reasoning"] = resolved_reasoning
        return result

    def action_tool_node(state: AgentState) -> dict[str, Any]:
        tool_results = dict(state.get("tool_results", {}))
        if state.get("verifier_decision") != "approved":
            return {"tool_results": tool_results}

        active_customer_id = state.get("active_customer_id")
        active_order_id = state.get("active_order_id")

        for action in state.get("requested_actions", []):
            name = action.get("name")
            args = action.get("args", {})

            if name == "request_refund":
                order_id = _prefer_explicit_int(args.get("order_id"), active_order_id)
                if order_id is not None:
                    tool_results["refund"] = repository.request_refund(
                        order_id,
                        customer_id=active_customer_id,
                    )

            elif name == "request_cancel_order":
                order_id = _prefer_explicit_int(args.get("order_id"), active_order_id)
                if order_id is not None:
                    tool_results["cancelled_order"] = repository.cancel_order(
                        order_id,
                        customer_id=active_customer_id,
                    )

            elif name == "request_log_complaint":
                customer_id = _prefer_explicit_int(args.get("customer_id"), active_customer_id)
                order_id = _prefer_explicit_int(args.get("order_id"), active_order_id)
                issue = args.get("issue") or state.get("issue") or "customer requested to file a complaint"
                if customer_id is not None:
                    tool_results["complaint"] = repository.log_complaint(
                        customer_id=customer_id,
                        order_id=order_id,
                        issue=str(issue),
                    )

            elif name == "request_write_memory":
                customer_id = _prefer_explicit_int(args.get("customer_id"), active_customer_id)
                key = args.get("key") or state.get("memory_key")
                value = args.get("value") or state.get("memory_value")
                if customer_id is not None and key and value:
                    tool_results["memory_write"] = repository.write_memory(
                        customer_id=customer_id,
                        key=str(key),
                        value=str(value),
                    )

        return {"tool_results": tool_results}

    def memory_update_node(state: AgentState) -> dict[str, Any]:
        active_customer_id = state.get("active_customer_id")
        if not active_customer_id:
            return {"long_term_memory": state.get("long_term_memory", [])}

        complaint = state.get("tool_results", {}).get("complaint")
        if complaint:
            repository.write_memory(active_customer_id, "issue_history", complaint["issue"])

        return {"long_term_memory": repository.read_memories(active_customer_id)[:5]}

    def response_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1].content if state.get("messages") else ""
        verified_facts, response_constraints = _build_response_grounding(
            state.get("tool_results", {}),
            state.get("verification_errors", []),
        )
        if state.get("verification_errors"):
            response = state["verification_errors"][0]
            return {
                "messages": [("assistant", response)],
                "final_response": response,
                "verified_facts": verified_facts,
                "response_constraints": response_constraints,
            }

        response = reasoner.respond(
            ResponseContext(
                user_message=str(last_message),
                tool_results=state.get("tool_results", {}),
                verification_errors=state.get("verification_errors", []),
                long_term_memory=state.get("long_term_memory", []),
                active_customer_id=state.get("active_customer_id"),
                active_order_id=state.get("active_order_id"),
                verified_facts=verified_facts,
                response_constraints=response_constraints,
            )
        )
        return {
            "messages": [("assistant", response)],
            "final_response": response,
            "verified_facts": verified_facts,
            "response_constraints": response_constraints,
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("read_tools", read_tool_node)
    graph.add_node("memory", memory_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("actions", action_tool_node)
    graph.add_node("memory_update", memory_update_node)
    graph.add_node("respond", response_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "read_tools")
    graph.add_conditional_edges(
        "read_tools",
        _route_after_read_tools,
        {"planner": "planner", "memory": "memory"},
    )
    graph.add_edge("memory", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"planner": "planner", "actions": "actions", "respond": "respond"},
    )
    graph.add_edge("actions", "memory_update")
    graph.add_edge("memory_update", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())


def _route_after_read_tools(state: AgentState) -> str:
    if _needs_replan_after_read_tools(state):
        return "planner"
    return "memory"


def _route_after_verifier(state: AgentState) -> str:
    decision = state.get("verifier_decision")
    if decision == "replan":
        return "planner"
    if decision == "approved":
        return "actions"
    return "respond"


def _needs_replan_after_read_tools(state: AgentState) -> bool:
    if state.get("requires_follow_up"):
        return False
    if state.get("pending_intent") in {"refund", "cancel"}:
        return False
    if int(state.get("react_iterations") or 0) >= int(
        state.get("max_react_iterations") or DEFAULT_MAX_REACT_ITERATIONS
    ):
        return False

    last_message = state["messages"][-1].content if state.get("messages") else ""
    if not _requested_order_mutation(str(last_message)):
        return False
    if state.get("requested_actions"):
        return False

    called_order_lookup = any(
        call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
    )
    return called_order_lookup and bool(state.get("tool_results", {}).get("order"))


def _apply_pending_intent_to_plan(
    plan,
    user_message: str,
    state_snapshot: dict[str, Any],
):
    if plan is None:
        plan = _empty_tool_plan()

    pending_intent = _pending_intent_from_context(plan, user_message)
    if not pending_intent:
        return plan

    active_order_id = state_snapshot.get("active_order_id")
    tool_results = state_snapshot.get("tool_results", {})
    order = tool_results.get("order")
    plan.pending_intent = pending_intent
    plan.pending_action = _pending_action_name(pending_intent)

    if pending_intent in {"refund", "cancel"}:
        order_id = _first_not_none(
            _int_from_message(user_message),
            plan.order_id,
            state_snapshot.get("pending_order_id"),
            active_order_id,
        )
        if order:
            order_id = order["order_id"]
        plan.pending_order_id = order_id
        if plan.order_id is None:
            plan.order_id = order_id
        if not plan.tool_calls and not order and order_id is not None:
            call = {
                "name": "order_lookup",
                "args": {"order_id": order_id},
                "id": "pending-order-lookup",
            }
            plan.tool_calls = [call]
            plan.requested_actions = []
            plan.steps = [call["name"]]
            plan.reasoning = (
                f"Resolved pending intent {pending_intent} into order_lookup before transaction verification."
            )
        if not plan.tool_calls:
            plan.requires_follow_up = False
            plan.follow_up_question = None
        return plan

    if pending_intent == "memory_write":
        key = plan.memory_key
        value = plan.memory_value
        if not key or not value:
            key, value = _memory_preference(user_message)
        plan.memory_key = key
        plan.memory_value = value
        plan.pending_order_id = None
        if not plan.tool_calls:
            plan.requires_follow_up = False
            plan.follow_up_question = None
            plan.reasoning = "Resolved pending intent memory_write for verifier continuation."
        return plan

    if pending_intent == "complaint":
        order_id = _first_not_none(
            _int_from_message(user_message),
            plan.order_id,
            state_snapshot.get("pending_order_id"),
            active_order_id,
        )
        plan.pending_order_id = order_id
        if plan.order_id is None:
            plan.order_id = order_id
        if not plan.issue:
            plan.issue = _complaint_issue(user_message)
        if not plan.tool_calls:
            plan.requires_follow_up = False
            plan.follow_up_question = None
            plan.reasoning = "Resolved pending intent complaint for verifier continuation."
        return plan

    return plan


def _empty_tool_plan():
    from customer_service_agent.reasoning import ToolPlan

    return ToolPlan(
        requires_follow_up=True,
        follow_up_question="I need a little more detail to help with that.",
    )


def _pending_intent_from_context(plan, user_message: str) -> str | None:
    action_names = {action.get("name") for action in plan.requested_actions}
    tool_names = {call.get("name") for call in plan.tool_calls}
    requested_mutation = _requested_order_mutation(user_message)
    if "request_refund" in action_names or requested_mutation == "refund":
        return "refund"
    if "request_cancel_order" in action_names or requested_mutation == "cancel":
        return "cancel"
    if "request_write_memory" in action_names or _requested_memory_write(user_message):
        return "memory_write"
    if (
        "request_log_complaint" in action_names
        or "request_log_complaint" in tool_names
        or _requested_complaint(user_message)
    ):
        return "complaint"
    return None


def _pending_action_name(pending_intent: str | None) -> str | None:
    return {
        "refund": "request_refund",
        "cancel": "request_cancel_order",
        "complaint": "request_log_complaint",
        "memory_write": "request_write_memory",
    }.get(str(pending_intent))


def _resolved_pending_action(
    pending_intent: str,
    *,
    order_id: int | None = None,
    customer_id: int | None = None,
    issue: str | None = None,
    memory_key: str | None = None,
    memory_value: str | None = None,
) -> dict[str, Any]:
    if pending_intent == "refund":
        return {
            "name": "request_refund",
            "args": {"order_id": order_id},
            "id": "resolved-request_refund",
        }
    if pending_intent == "cancel":
        return {
            "name": "request_cancel_order",
            "args": {"order_id": order_id},
            "id": "resolved-request_cancel_order",
        }
    if pending_intent == "complaint":
        return {
            "name": "request_log_complaint",
            "args": {
                "customer_id": customer_id,
                "order_id": order_id,
                "issue": issue or "customer requested to file a complaint",
            },
            "id": "resolved-request_log_complaint",
        }
    return {
        "name": "request_write_memory",
        "args": {
            "customer_id": customer_id,
            "key": memory_key,
            "value": memory_value,
        },
        "id": "resolved-request_write_memory",
    }


class CustomerServiceAgent:
    def __init__(self, reasoner, repository):
        self._reasoner = reasoner
        self._repository = repository
        self._graph = build_graph(reasoner, repository)

    def invoke(
        self,
        thread_id: str,
        message: str,
        customer_id: int | None = None,
    ) -> ChatResponse:
        state = _new_turn_state(message, DEFAULT_MAX_REACT_ITERATIONS)
        if customer_id is not None:
            state["active_customer_id"] = customer_id

        result = self._graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return ChatResponse(
            thread_id=thread_id,
            response=result.get("final_response") or "",
            order_id=result.get("active_order_id"),
            customer_id=result.get("active_customer_id"),
            tool_results=result.get("tool_results", {}),
            verified_facts=result.get("verified_facts", {}),
            response_constraints=result.get("response_constraints", []),
            verifier_decision=result.get("verifier_decision"),
            verification_errors=result.get("verification_errors", []),
        )

    def trace(
        self,
        thread_id: str,
        message: str,
        customer_id: int | None = None,
    ) -> tuple[ChatResponse, list[dict[str, Any]]]:
        state = _new_turn_state(message, DEFAULT_MAX_REACT_ITERATIONS)
        if customer_id is not None:
            state["active_customer_id"] = customer_id

        updates: list[dict[str, Any]] = []
        result: dict[str, Any] = {}
        config = {"configurable": {"thread_id": thread_id}}

        for update in self._graph.stream(state, config=config, stream_mode="updates"):
            for node_name, node_update in update.items():
                updates.append(
                    {
                        "node": node_name,
                        "state": _summarize_node_update(node_name, node_update),
                    }
                )

        snapshot = self._graph.get_state(config)
        if isinstance(snapshot.values, dict):
            result = snapshot.values

        return (
            ChatResponse(
                thread_id=thread_id,
                response=result.get("final_response") or "",
                order_id=result.get("active_order_id"),
                customer_id=result.get("active_customer_id"),
                tool_results=result.get("tool_results", {}),
                verified_facts=result.get("verified_facts", {}),
                response_constraints=result.get("response_constraints", []),
                verifier_decision=result.get("verifier_decision"),
                verification_errors=result.get("verification_errors", []),
            ),
            updates,
        )


def _summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    if node_name == "planner":
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "issue": update.get("issue"),
            "memory_key": update.get("memory_key"),
            "memory_value": update.get("memory_value"),
            "pending_intent": update.get("pending_intent"),
            "pending_action": update.get("pending_action"),
            "pending_order_id": update.get("pending_order_id"),
            "tool_calls": update.get("tool_calls", []),
            "requested_actions": update.get("requested_actions", []),
            "requires_follow_up": update.get("requires_follow_up"),
            "plan_steps": update.get("plan_steps", []),
            "reasoning": update.get("reasoning"),
        }
    if node_name in {"read_tools", "actions"}:
        tool_results = update.get("tool_results", {})
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "tool_result_keys": list(tool_results.keys()),
            "tool_results": tool_results,
        }
    if node_name in {"memory", "memory_update"}:
        long_term_memory = update.get("long_term_memory", [])
        return {
            "long_term_memory_count": len(long_term_memory),
            "long_term_memory": long_term_memory,
        }
    if node_name == "verifier":
        tool_results = update.get("tool_results", {})
        return {
            "verifier_decision": update.get("verifier_decision"),
            "verification_errors": update.get("verification_errors", []),
            "tool_result_keys": list(tool_results.keys()),
            "pending_action": update.get("pending_action"),
            "pending_order_id": update.get("pending_order_id"),
            "requested_actions": update.get("requested_actions", []),
            "reasoning": update.get("reasoning"),
        }
    if node_name == "respond":
        return {
            "final_response": update.get("final_response"),
            "verified_facts": update.get("verified_facts", {}),
            "response_constraints": update.get("response_constraints", []),
        }
    return update


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _prefer_explicit_int(value: Any, fallback: int | None) -> int | None:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else fallback


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _requested_order_mutation(message: str) -> str | None:
    normalized = message.lower()
    if "refund" in normalized and (
        "order" in normalized
        or "refund it" in normalized
        or normalized.strip().startswith("refund")
    ):
        return "refund"
    if "cancel" in normalized or "cancellation" in normalized:
        return "cancel"
    return None


def _requested_memory_write(message: str) -> bool:
    normalized = message.lower()
    return "remember" in normalized or "preference" in normalized


def _requested_complaint(message: str) -> bool:
    normalized = message.lower()
    complaint_words = {"complain", "complaint", "late", "damaged", "broken"}
    return any(word in normalized for word in complaint_words)


def _memory_preference(message: str) -> tuple[str | None, str | None]:
    normalized = message.lower()
    if "refund" in normalized:
        return "refund_preference", "prefers refunds"
    if "email" in normalized:
        return "contact_preference", "prefers email"
    if "remember" in normalized:
        value = re.sub(r"^\s*remember\s+", "", message, flags=re.IGNORECASE).strip()
        if value:
            return "customer_preference", value
    return None, None


def _complaint_issue(message: str) -> str:
    normalized = message.lower()
    if "late" in normalized:
        return "Order is late again" if "again" in normalized else "Order is late"
    if "damaged" in normalized:
        return "Package damaged"
    if "broken" in normalized:
        return "Item is broken"
    return "customer requested to file a complaint"


def _int_from_message(message: str) -> int | None:
    match = re.search(r"\b\d+\b", message)
    if not match:
        return None
    return int(match.group(0))


def _last_user_message(state: dict[str, Any]) -> str:
    if not state.get("messages"):
        return ""
    return str(state["messages"][-1].content)


def _build_response_grounding(
    tool_results: dict[str, Any],
    verification_errors: list[str],
) -> tuple[dict[str, Any], list[str]]:
    verified_facts: dict[str, Any] = {}

    if verification_errors:
        verified_facts["verification_errors"] = list(verification_errors)

    order = tool_results.get("order")
    if order:
        verified_facts["order"] = {
            "order_id": order.get("order_id"),
            "customer_id": order.get("customer_id"),
            "product_name": order.get("product_name"),
            "status": order.get("status"),
            "order_date": order.get("order_date"),
            "delivery_date": order.get("delivery_date"),
        }

    customer = tool_results.get("customer")
    if customer:
        verified_facts["customer"] = {
            "customer_id": customer.get("customer_id"),
            "name": customer.get("name"),
            "email": customer.get("email"),
        }

    refund = tool_results.get("refund")
    if refund:
        verified_facts["refund_request"] = {
            "order_id": refund.get("order_id"),
            "status": refund.get("status"),
            "created_this_turn": True,
        }

    cancelled = tool_results.get("cancelled_order")
    if cancelled:
        verified_facts["cancellation_request"] = {
            "order_id": cancelled.get("order_id"),
            "status": cancelled.get("status"),
            "created_this_turn": True,
        }

    complaint = tool_results.get("complaint")
    if complaint:
        verified_facts["complaint_logged"] = {
            "complaint_id": complaint.get("complaint_id"),
            "customer_id": complaint.get("customer_id"),
            "order_id": complaint.get("order_id"),
            "issue": complaint.get("issue"),
            "status": complaint.get("status"),
        }

    memory_write = tool_results.get("memory_write")
    if memory_write:
        verified_facts["memory_written"] = {
            "customer_id": memory_write.get("customer_id"),
            "key": memory_write.get("key"),
            "value": memory_write.get("value"),
        }

    memories = tool_results.get("memories")
    if memories:
        verified_facts["customer_memories"] = [
            {
                "customer_id": memory.get("customer_id"),
                "key": memory.get("key"),
                "value": memory.get("value"),
            }
            for memory in memories
        ]

    complaints = tool_results.get("complaints")
    if complaints:
        verified_facts["customer_complaints"] = [
            {
                "complaint_id": complaint.get("complaint_id"),
                "order_id": complaint.get("order_id"),
                "issue": complaint.get("issue"),
                "status": complaint.get("status"),
            }
            for complaint in complaints
        ]

    issue_patterns = tool_results.get("issue_patterns")
    if issue_patterns:
        verified_facts["issue_patterns"] = {
            "total_complaints": issue_patterns.get("total_complaints"),
            "issue_counts": issue_patterns.get("issue_counts", {}),
            "repeated_late_delivery": issue_patterns.get("repeated_late_delivery"),
        }

    constraints = [
        "Use only verified_facts and tool_results as ground truth.",
        "Do not invent refund status, complaint IDs, delivery dates, or customer history.",
        "Do not claim a mutation succeeded unless the matching verified fact is present.",
        "Do not promise future handling, follow-up, investigation, escalation, or resolution "
        "unless verified_facts or tool_results explicitly support that action or status.",
        "Use a warm customer-service tone while staying concise.",
    ]
    if verification_errors:
        constraints.append(
            "For verifier errors, communicate the first error without adding unsupported action claims."
        )
    if "refund_request" in verified_facts:
        constraints.append(
            "For the refund, only describe the request status shown in verified_facts."
        )
        if verified_facts["refund_request"].get("created_this_turn"):
            constraints.append(
                "For this current-turn refund result, say the request was submitted or requested; "
                "do not say it was already requested or already submitted."
            )
    if "cancellation_request" in verified_facts:
        constraints.append(
            "For the cancellation, only describe the request status shown in verified_facts."
        )
        if verified_facts["cancellation_request"].get("created_this_turn"):
            constraints.append(
                "For this current-turn cancellation result, say the request was submitted or "
                "requested; do not say it was already requested or already submitted."
            )
    if "complaint_logged" in verified_facts:
        constraints.append(
            "For the complaint, only mention the order, issue, status, or complaint ID if present in verified_facts."
        )
    if "issue_patterns" in verified_facts:
        constraints.append(
            "If repeated_late_delivery is true, mention the repeated late-delivery pattern; "
            "otherwise do not mention repeated late-delivery history."
        )
    if "customer_memories" in verified_facts or "customer_complaints" in verified_facts:
        constraints.append(
            "Mention customer history only from customer_memories or customer_complaints."
        )
    if "memory_written" in verified_facts:
        constraints.append(
            "For memory writes, only confirm the saved key or value shown in verified_facts."
        )

    return verified_facts, constraints


def _new_turn_state(message: str, max_react_iterations: int) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=message)],
        "plan_steps": [],
        "reasoning": None,
        "issue": None,
        "memory_key": None,
        "memory_value": None,
        "pending_intent": None,
        "pending_action": None,
        "pending_order_id": None,
        "tool_calls": [],
        "requested_actions": [],
        "requires_follow_up": False,
        "follow_up_question": None,
        "tool_results": {},
        "verification_errors": [],
        "verified_facts": {},
        "response_constraints": [],
        "verifier_decision": None,
        "react_iterations": 0,
        "max_react_iterations": max_react_iterations,
        "long_term_memory": [],
        "final_response": None,
    }
