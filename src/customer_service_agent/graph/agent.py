from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service_agent.models import AgentState, ChatResponse
from customer_service_agent.reasoning import ResponseContext
from customer_service_agent.graph.actions import execute_requested_actions
from customer_service_agent.graph.response_facts import build_verified_facts
from customer_service_agent.graph.planning import prepare_plan
from customer_service_agent.graph.policy import prefer_explicit_int, verify_policy
from customer_service_agent.graph.trace import summarize_node_update


DEFAULT_MAX_REACT_ITERATIONS = 3


def build_graph(reasoner, repository, max_react_iterations: int = DEFAULT_MAX_REACT_ITERATIONS):
    def planner_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1].content if state.get("messages") else ""
        react_iterations = int(state.get("react_iterations") or 0) + 1
        planner_feedback = _planner_feedback(state.get("verification_decision", {}))
        state_snapshot = {
            "active_customer_id": state.get("active_customer_id"),
            "active_order_id": state.get("active_order_id"),
            "issue": state.get("issue"),
            "tool_results": state.get("tool_results", {}),
            "planner_feedback": planner_feedback,
            "react_iterations": react_iterations,
            "max_react_iterations": state.get("max_react_iterations") or max_react_iterations,
        }
        plan = reasoner.plan(str(last_message), state_snapshot)
        plan = prepare_plan(plan, state_snapshot)
        next_customer_id = (
            plan.customer_id if plan.customer_id is not None else state.get("active_customer_id")
        )
        next_order_id = plan.order_id if plan.order_id is not None else state.get("active_order_id")
        return {
            **_reset_turn_outputs(
                max_react_iterations=state.get("max_react_iterations") or max_react_iterations,
            ),
            "active_customer_id": next_customer_id,
            "active_order_id": next_order_id,
            "issue": plan.issue,
            "memory_candidate": plan.memory_candidate.model_dump(),
            "tool_calls": plan.tool_calls,
            "requested_actions": plan.requested_actions,
            "continue_after_read": plan.continue_after_read,
            "tool_results": state.get("tool_results", {}),
            "react_iterations": react_iterations,
        }

    def read_tool_node(state: AgentState) -> dict[str, Any]:
        tool_results: dict[str, Any] = dict(state.get("tool_results", {}))
        active_customer_id = state.get("active_customer_id")
        active_order_id = state.get("active_order_id")
        customer_read_tools = {
            "customer_profile": ("customer", repository.get_customer),
            "read_customer_memory": ("memories", repository.read_memories),
        }

        for tool_call in state.get("tool_calls", []):
            name = tool_call.get("name")
            args = tool_call.get("args", {})

            if name == "order_lookup":
                order_id = prefer_explicit_int(args.get("order_id"), state.get("active_order_id"))
                if order_id is not None:
                    active_order_id = order_id
                    tool_results.pop("order", None)
                    order = repository.get_order(order_id)
                    tool_results["order_lookup"] = {"order_id": order_id, "found": bool(order)}
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

            elif name == "read_customer_issue_history":
                customer_id = prefer_explicit_int(args.get("customer_id"), active_customer_id)
                if customer_id is not None:
                    active_customer_id = customer_id
                    tool_results["complaints"] = repository.list_complaints(customer_id)
                    tool_results["issue_patterns"] = repository.summarize_issue_patterns(customer_id)

        return {
            "tool_results": tool_results,
            "active_customer_id": active_customer_id,
            "active_order_id": active_order_id,
        }

    def verifier_node(state: AgentState) -> dict[str, Any]:
        return verify_policy(state, max_react_iterations=max_react_iterations)

    def action_tool_node(state: AgentState) -> dict[str, Any]:
        return {"tool_results": execute_requested_actions(state, repository)}

    def response_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1].content if state.get("messages") else ""
        long_term_memory = _read_recent_memory(repository, state.get("active_customer_id"))
        verified_facts = build_verified_facts(
            state.get("tool_results", {}),
            state.get("verification_decision", {}),
        )
        response = reasoner.respond(
            ResponseContext(
                user_message=str(last_message),
                verification_decision=state.get("verification_decision", {}),
                long_term_memory=long_term_memory,
                active_customer_id=state.get("active_customer_id"),
                active_order_id=state.get("active_order_id"),
                verified_facts=verified_facts,
            )
        )
        response = response.strip()
        return {
            "messages": [("assistant", response)],
            "final_response": response,
            "verified_facts": verified_facts,
            "long_term_memory": long_term_memory,
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("read_tools", read_tool_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("actions", action_tool_node)
    graph.add_node("respond", response_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"read_tools": "read_tools", "verifier": "verifier"},
    )
    graph.add_conditional_edges(
        "read_tools",
        _route_after_read_tools,
        {"planner": "planner", "verifier": "verifier"},
    )
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"planner": "planner", "actions": "actions", "respond": "respond"},
    )
    graph.add_edge("actions", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())


def _route_after_planner(state: AgentState) -> str:
    if state.get("tool_calls"):
        return "read_tools"
    return "verifier"


def _route_after_read_tools(state: AgentState) -> str:
    if _needs_replan_after_read_tools(state):
        return "planner"
    return "verifier"


def _route_after_verifier(state: AgentState) -> str:
    decision = (state.get("verification_decision") or {}).get("decision")
    if decision == "replan":
        return "planner"
    if decision == "proceed_to_action":
        return "actions"
    return "respond"


def _needs_replan_after_read_tools(state: AgentState) -> bool:
    if int(state.get("react_iterations") or 0) >= int(
        state.get("max_react_iterations") or DEFAULT_MAX_REACT_ITERATIONS
    ):
        return False

    if state.get("requested_actions"):
        return False

    return bool(state.get("tool_calls")) and bool(state.get("continue_after_read", True))


def _read_recent_memory(repository, customer_id: int | None) -> list[dict[str, Any]]:
    if customer_id is None:
        return []
    return repository.read_memories(customer_id)[:5]


def _reset_turn_outputs(*, max_react_iterations: int) -> dict[str, Any]:
    return {
        "verification_decision": {},
        "verified_facts": {},
        "long_term_memory": [],
        "final_response": None,
        "max_react_iterations": max_react_iterations,
    }


def _planner_feedback(verification_decision: dict[str, Any]) -> dict[str, Any]:
    code = verification_decision.get("planner_feedback_code")
    if not code:
        return {}
    feedback = {"code": code}
    context = verification_decision.get("context")
    if context:
        feedback["context"] = context
    return feedback


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
        "issue": None,
        "memory_candidate": {},
        "tool_calls": [],
        "requested_actions": [],
        "continue_after_read": True,
        "tool_results": {},
        "verification_decision": {},
        "verified_facts": {},
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
        verification_decision=result.get("verification_decision", {}),
    )
