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
            "tool_results": state.get("tool_results", {}),
            "react_iterations": react_iterations,
            "max_react_iterations": state.get("max_react_iterations") or max_react_iterations,
        }
        plan = reasoner.plan(str(last_message), state_snapshot)
        plan = _repair_plan_from_context(plan, str(last_message), state_snapshot)
        next_customer_id = (
            plan.customer_id if plan.customer_id is not None else state.get("active_customer_id")
        )
        next_order_id = plan.order_id if plan.order_id is not None else state.get("active_order_id")
        return {
            "plan_steps": plan.steps,
            "reasoning": plan.reasoning,
            "active_customer_id": next_customer_id,
            "active_order_id": next_order_id,
            "issue": plan.issue,
            "memory_key": plan.memory_key,
            "memory_value": plan.memory_value,
            "tool_calls": plan.tool_calls,
            "requested_actions": plan.requested_actions,
            "requires_follow_up": plan.requires_follow_up,
            "follow_up_question": plan.follow_up_question,
            "tool_results": state.get("tool_results", {}),
            "verifier_decision": None,
            "verification_errors": [],
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
        if state.get("requires_follow_up"):
            repaired = _repair_plan_from_context(None, _last_user_message(state), state)
            if repaired.tool_calls:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": state.get("tool_results", {}),
                }
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
        requested_mutation = _requested_order_mutation(str(last_message))
        action_names = {action.get("name") for action in state.get("requested_actions", [])}
        called_order_lookup = any(
            call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
        )
        needs_order = bool(
            action_names & {"request_refund", "request_cancel_order"}
            or requested_mutation
            or called_order_lookup
        )
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
            if can_replan:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": tool_results,
                }
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
            if can_replan:
                return {
                    "verifier_decision": "replan",
                    "verification_errors": [],
                    "tool_results": tool_results,
                }
            errors.append("I could not complete the cancellation request within the reasoning step limit.")
            return {
                "verifier_decision": "blocked",
                "verification_errors": errors,
                "tool_results": tool_results,
            }

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
        return {
            "verifier_decision": decision,
            "verification_errors": errors,
            "tool_results": tool_results,
        }

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
        if state.get("verification_errors"):
            response = state["verification_errors"][0]
            return {"messages": [("assistant", response)], "final_response": response}

        template_response = _mutation_response(state.get("tool_results", {}))
        if template_response:
            return {"messages": [("assistant", template_response)], "final_response": template_response}

        response = reasoner.respond(
            ResponseContext(
                user_message=str(last_message),
                tool_results=state.get("tool_results", {}),
                verification_errors=state.get("verification_errors", []),
                long_term_memory=state.get("long_term_memory", []),
                active_customer_id=state.get("active_customer_id"),
                active_order_id=state.get("active_order_id"),
            )
        )
        return {"messages": [("assistant", response)], "final_response": response}

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


def _repair_plan_from_context(
    plan,
    user_message: str,
    state_snapshot: dict[str, Any],
):
    if plan is not None and plan.tool_calls:
        return plan

    active_customer_id = state_snapshot.get("active_customer_id")
    active_order_id = state_snapshot.get("active_order_id")
    tool_results = state_snapshot.get("tool_results", {})
    order = tool_results.get("order")
    requested_mutation = _requested_order_mutation(user_message)

    if requested_mutation in {"refund", "cancel"}:
        order_id = _int_from_message(user_message) or active_order_id
        if order:
            order_id = order["order_id"]
            action_name = "request_refund" if requested_mutation == "refund" else "request_cancel_order"
            action = {"name": action_name, "args": {"order_id": order_id}, "id": f"repaired-{action_name}"}
            return _repaired_tool_plan(
                action,
                customer_id=active_customer_id,
                order_id=order_id,
                reasoning=f"Rule-based repair selected {action_name} from available order observation.",
            )
        if order_id is not None:
            call = {"name": "order_lookup", "args": {"order_id": order_id}, "id": "repaired-order-lookup"}
            return _repaired_tool_plan(
                call,
                customer_id=active_customer_id,
                order_id=order_id,
                reasoning="Rule-based repair selected order_lookup before an order mutation.",
            )

    if _requested_memory_write(user_message) and active_customer_id is not None:
        key, value = _memory_preference(user_message)
        if key and value:
            action = {
                "name": "request_write_memory",
                "args": {"customer_id": active_customer_id, "key": key, "value": value},
                "id": "repaired-memory-write",
            }
            return _repaired_tool_plan(
                action,
                customer_id=active_customer_id,
                memory_key=key,
                memory_value=value,
                reasoning="Rule-based repair selected request_write_memory from the stated preference.",
            )

    if _requested_complaint(user_message) and active_customer_id is not None:
        order_id = _int_from_message(user_message) or active_order_id
        action = {
            "name": "request_log_complaint",
            "args": {
                "customer_id": active_customer_id,
                "order_id": order_id,
                "issue": _complaint_issue(user_message),
            },
            "id": "repaired-complaint",
        }
        return _repaired_tool_plan(
            action,
            customer_id=active_customer_id,
            order_id=order_id,
            issue=action["args"]["issue"],
            reasoning="Rule-based repair selected request_log_complaint from the complaint request.",
        )

    if plan is not None:
        return plan
    return _repaired_tool_plan(None)


def _repaired_tool_plan(
    call: dict[str, Any] | None,
    *,
    customer_id: int | None = None,
    order_id: int | None = None,
    issue: str | None = None,
    memory_key: str | None = None,
    memory_value: str | None = None,
    reasoning: str = "",
):
    from customer_service_agent.reasoning import ACTION_TOOL_NAMES, ToolPlan

    tool_calls = [call] if call else []
    requested_actions = [call] if call and call["name"] in ACTION_TOOL_NAMES else []
    return ToolPlan(
        tool_calls=tool_calls,
        requested_actions=requested_actions,
        customer_id=customer_id,
        order_id=order_id,
        issue=issue,
        memory_key=memory_key,
        memory_value=memory_value,
        requires_follow_up=not tool_calls,
        follow_up_question="I need a little more detail to help with that." if not tool_calls else None,
        steps=[call["name"]] if call else [],
        reasoning=reasoning,
    )


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
        }
    if node_name == "respond":
        return {"final_response": update.get("final_response")}
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


def _mutation_response(tool_results: dict[str, Any]) -> str | None:
    refund = tool_results.get("refund")
    if refund:
        return f"Refund request submitted for order {refund['order_id']}."

    cancelled = tool_results.get("cancelled_order")
    if cancelled:
        return f"Cancellation request submitted for order {cancelled['order_id']}."

    complaint = tool_results.get("complaint")
    if complaint:
        order_id = complaint.get("order_id")
        if order_id is not None:
            return f"Complaint logged for order {order_id}."
        return f"Complaint logged for customer {complaint['customer_id']}."

    memory_write = tool_results.get("memory_write")
    if memory_write:
        return f"Memory updated: {memory_write['key']}."

    return None


def _new_turn_state(message: str, max_react_iterations: int) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=message)],
        "plan_steps": [],
        "reasoning": None,
        "issue": None,
        "memory_key": None,
        "memory_value": None,
        "tool_calls": [],
        "requested_actions": [],
        "requires_follow_up": False,
        "follow_up_question": None,
        "tool_results": {},
        "verification_errors": [],
        "verifier_decision": None,
        "react_iterations": 0,
        "max_react_iterations": max_react_iterations,
        "long_term_memory": [],
        "final_response": None,
    }
