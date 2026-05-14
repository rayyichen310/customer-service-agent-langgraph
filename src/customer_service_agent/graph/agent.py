from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service_agent.models import AgentState, ChatResponse
from customer_service_agent.reasoning import ResponseContext
from customer_service_agent.graph.actions import execute_requested_actions
from customer_service_agent.graph.grounding import build_response_grounding
from customer_service_agent.graph.planning import (
    apply_pending_intent_to_plan,
    requested_order_mutation,
)
from customer_service_agent.graph.policy import prefer_explicit_int, verify_policy
from customer_service_agent.graph.response_style import polish_customer_response
from customer_service_agent.graph.trace import summarize_node_update


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
            "pending_order_id": state.get("pending_order_id"),
            "tool_results": state.get("tool_results", {}),
            "react_iterations": react_iterations,
            "max_react_iterations": state.get("max_react_iterations") or max_react_iterations,
        }
        plan = reasoner.plan(str(last_message), state_snapshot)
        plan = apply_pending_intent_to_plan(plan, str(last_message), state_snapshot)
        next_customer_id = (
            plan.customer_id if plan.customer_id is not None else state.get("active_customer_id")
        )
        next_order_id = plan.order_id if plan.order_id is not None else state.get("active_order_id")
        pending_intent = plan.pending_intent
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
            "pending_order_id": pending_order_id if pending_intent else None,
            "tool_calls": plan.tool_calls,
            "requested_actions": plan.requested_actions,
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
        if state.get("follow_up_question"):
            return {"tool_results": state.get("tool_results", {})}

        tool_results: dict[str, Any] = dict(state.get("tool_results", {}))
        active_customer_id = state.get("active_customer_id")
        active_order_id = state.get("active_order_id")
        customer_read_tools = {
            "customer_profile": ("customer", repository.get_customer),
            "read_customer_memory": ("memories", repository.read_memories),
            "list_customer_complaints": ("complaints", repository.list_complaints),
            "summarize_issue_patterns": ("issue_patterns", repository.summarize_issue_patterns),
        }

        for tool_call in state.get("tool_calls", []):
            name = tool_call.get("name")
            args = tool_call.get("args", {})

            if name == "order_lookup":
                order_id = prefer_explicit_int(args.get("order_id"), active_order_id)
                if order_id is not None:
                    active_order_id = order_id
                    order = repository.get_order(order_id)
                    if order:
                        tool_results["order"] = order
                        if active_customer_id is None:
                            active_customer_id = order["customer_id"]

            elif name in customer_read_tools:
                result_key, read_customer_data = customer_read_tools[name]
                customer_id = prefer_explicit_int(args.get("customer_id"), active_customer_id)
                if customer_id is not None:
                    active_customer_id = customer_id
                    tool_results[result_key] = read_customer_data(customer_id)

        return {
            "tool_results": tool_results,
            "active_customer_id": active_customer_id,
            "active_order_id": active_order_id,
        }

    def memory_node(state: AgentState) -> dict[str, Any]:
        if state.get("follow_up_question"):
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
        return verify_policy(state, max_react_iterations=max_react_iterations)

    def action_tool_node(state: AgentState) -> dict[str, Any]:
        return {"tool_results": execute_requested_actions(state, repository)}

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
        verified_facts, response_constraints = build_response_grounding(
            state.get("tool_results", {}),
            state.get("verification_errors", []),
        )
        if state.get("verification_errors"):
            response = polish_customer_response(state["verification_errors"][0], verified_facts)
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
        response = polish_customer_response(response, verified_facts)
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
    if state.get("follow_up_question"):
        return False
    if state.get("pending_intent") in {"refund", "cancel"}:
        return False
    if int(state.get("react_iterations") or 0) >= int(
        state.get("max_react_iterations") or DEFAULT_MAX_REACT_ITERATIONS
    ):
        return False

    last_message = state["messages"][-1].content if state.get("messages") else ""
    if not requested_order_mutation(str(last_message)):
        return False
    if state.get("requested_actions"):
        return False

    called_order_lookup = any(
        call.get("name") == "order_lookup" for call in state.get("tool_calls", [])
    )
    return called_order_lookup and bool(state.get("tool_results", {}).get("order"))


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
        state = _new_turn_state(message, DEFAULT_MAX_REACT_ITERATIONS, customer_id)
        result = self._graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return _chat_response(thread_id, result)

    def trace(
        self,
        thread_id: str,
        message: str,
        customer_id: int | None = None,
    ) -> tuple[ChatResponse, list[dict[str, Any]]]:
        state = _new_turn_state(message, DEFAULT_MAX_REACT_ITERATIONS, customer_id)
        updates: list[dict[str, Any]] = []
        result: dict[str, Any] = {}
        config = {"configurable": {"thread_id": thread_id}}

        for update in self._graph.stream(state, config=config, stream_mode="updates"):
            for node_name, node_update in update.items():
                updates.append(
                    {
                        "node": node_name,
                        "state": summarize_node_update(node_name, node_update),
                    }
                )

        snapshot = self._graph.get_state(config)
        if isinstance(snapshot.values, dict):
            result = snapshot.values

        return (
            _chat_response(thread_id, result),
            updates,
        )


def _new_turn_state(
    message: str,
    max_react_iterations: int,
    customer_id: int | None = None,
) -> dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=message)],
        "plan_steps": [],
        "reasoning": None,
        "issue": None,
        "memory_key": None,
        "memory_value": None,
        "pending_intent": None,
        "pending_order_id": None,
        "tool_calls": [],
        "requested_actions": [],
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
    if customer_id is not None:
        state["active_customer_id"] = customer_id
    return state


def _chat_response(thread_id: str, result: dict[str, Any]) -> ChatResponse:
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
