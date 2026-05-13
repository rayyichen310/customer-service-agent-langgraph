from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service_agent.models import AgentState, ChatResponse
from customer_service_agent.reasoning import ResponseContext


def build_graph(reasoner, repository):
    def planner_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1].content if state.get("messages") else ""
        state_snapshot = {
            "active_customer_id": state.get("active_customer_id"),
            "active_order_id": state.get("active_order_id"),
            "issue": state.get("issue"),
        }
        plan = reasoner.plan(str(last_message), state_snapshot)
        next_customer_id = (
            plan.customer_id if plan.customer_id is not None else state.get("active_customer_id")
        )
        next_order_id = plan.order_id if plan.order_id is not None else state.get("active_order_id")
        return {
            "intent": plan.intent,
            "plan_steps": plan.steps,
            "reasoning": plan.reasoning,
            "active_customer_id": next_customer_id,
            "active_order_id": next_order_id,
            "issue": plan.issue or state.get("issue"),
            "memory_key": plan.memory_key,
            "memory_value": plan.memory_value,
            "requires_follow_up": plan.requires_follow_up,
            "follow_up_question": plan.follow_up_question,
            "tool_results": {},
            "verification_errors": [],
            "long_term_memory": [],
            "final_response": None,
        }

    def tool_node(state: AgentState) -> dict[str, Any]:
        if state.get("requires_follow_up"):
            return {"tool_results": {}}

        intent = state.get("intent")
        tool_results: dict[str, Any] = {}
        active_customer_id = state.get("active_customer_id")
        active_order_id = state.get("active_order_id")

        if active_order_id is not None:
            order = repository.get_order(active_order_id)
            if order:
                tool_results["order"] = order
                if active_customer_id is None:
                    active_customer_id = order["customer_id"]

        if intent == "customer_profile" and active_customer_id:
            tool_results["customer"] = repository.get_customer(active_customer_id)

        elif intent == "refund_request":
            order = tool_results.get("order")
            if order and order["status"] == "delivered":
                tool_results["refund"] = repository.request_refund(order["order_id"])

        elif intent == "cancel_order":
            order = tool_results.get("order")
            if order and order["status"] not in {"delivered", "refund_requested"}:
                tool_results["cancelled_order"] = repository.cancel_order(order["order_id"])

        elif intent == "complaint":
            issue = state.get("issue") or "Unspecified issue"
            if active_order_id is not None and "order" not in tool_results:
                tool_results["order"] = repository.get_order(active_order_id)
                if tool_results["order"]:
                    if active_customer_id is None:
                        active_customer_id = tool_results["order"]["customer_id"]
            if active_customer_id:
                complaint = repository.log_complaint(
                    customer_id=active_customer_id,
                    order_id=active_order_id,
                    issue=issue,
                )
                tool_results["complaint"] = complaint

        elif intent == "memory_write":
            if active_customer_id and state.get("memory_key") and state.get("memory_value"):
                tool_results["memory_write"] = repository.write_memory(
                    customer_id=active_customer_id,
                    key=state["memory_key"],
                    value=state["memory_value"],
                )

        elif intent in {"memory_read", "general_support"}:
            if active_customer_id:
                tool_results["memories"] = repository.read_memories(active_customer_id)
                tool_results["complaints"] = repository.list_complaints(active_customer_id)
                tool_results["issue_patterns"] = repository.summarize_issue_patterns(active_customer_id)

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
            repository.write_memory(
                active_customer_id,
                "issue_history",
                complaint["issue"],
            )
            long_term_memory = repository.read_memories(active_customer_id)[:5]

        return {"long_term_memory": long_term_memory}

    def verifier_node(state: AgentState) -> dict[str, Any]:
        if state.get("requires_follow_up"):
            return {"verification_errors": [state.get("follow_up_question")]}

        errors: list[str] = []
        intent = state.get("intent")
        tool_results = dict(state.get("tool_results", {}))
        order = tool_results.get("order")
        active_order_id = state.get("active_order_id")

        if intent in {"order_status", "refund_request", "cancel_order"} and active_order_id is None:
            errors.append("I need an order ID to continue.")

        if (
            intent in {"order_status", "refund_request", "cancel_order"}
            and active_order_id is not None
            and not order
        ):
            errors.append(f"Order {active_order_id} does not exist.")

        if intent == "refund_request" and order:
            if order["status"] != "delivered":
                errors.append(
                    f"Order {order['order_id']} is currently `{order['status']}` and is not eligible for refund yet."
                )
            elif not tool_results.get("refund"):
                errors.append(f"I could not initiate a refund for order {order['order_id']}.")

        if intent == "cancel_order" and order:
            if order["status"] in {"delivered", "refund_requested"}:
                errors.append(
                    f"Order {order['order_id']} is currently `{order['status']}` and cannot be cancelled."
                )
            elif not tool_results.get("cancelled_order"):
                errors.append(f"I could not cancel order {order['order_id']}.")

        if intent == "customer_profile" and not tool_results.get("customer"):
            errors.append("Customer profile was not found.")

        if intent == "complaint" and not tool_results.get("complaint"):
            errors.append("I could not log the complaint without a valid customer or order.")

        if intent == "memory_write" and not tool_results.get("memory_write"):
            errors.append("I could not store that preference without a valid customer ID.")

        return {
            "verification_errors": errors,
            "tool_results": tool_results,
        }

    def response_node(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1].content if state.get("messages") else ""
        response = reasoner.respond(
            ResponseContext(
                user_message=str(last_message),
                intent=state.get("intent"),
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
    graph.add_node("tools", tool_node)
    graph.add_node("memory", memory_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("respond", response_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tools")
    graph.add_edge("tools", "memory")
    graph.add_edge("memory", "verifier")
    graph.add_edge("verifier", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())


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
        state: dict[str, Any] = {"messages": [HumanMessage(content=message)]}
        if customer_id is not None:
            state["active_customer_id"] = customer_id

        result = self._graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return ChatResponse(
            thread_id=thread_id,
            response=result.get("final_response") or "",
            intent=result.get("intent") or "general_support",
            order_id=result.get("active_order_id"),
            customer_id=result.get("active_customer_id"),
            tool_results=result.get("tool_results", {}),
            verification_errors=result.get("verification_errors", []),
        )

    def trace(
        self,
        thread_id: str,
        message: str,
        customer_id: int | None = None,
    ) -> tuple[ChatResponse, list[dict[str, Any]]]:
        state: dict[str, Any] = {"messages": [HumanMessage(content=message)]}
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
                intent=result.get("intent") or "general_support",
                order_id=result.get("active_order_id"),
                customer_id=result.get("active_customer_id"),
                tool_results=result.get("tool_results", {}),
                verification_errors=result.get("verification_errors", []),
            ),
            updates,
        )


def _summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    if node_name == "planner":
        return {
            "intent": update.get("intent"),
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "issue": update.get("issue"),
            "memory_key": update.get("memory_key"),
            "memory_value": update.get("memory_value"),
            "requires_follow_up": update.get("requires_follow_up"),
            "plan_steps": update.get("plan_steps", []),
            "reasoning": update.get("reasoning"),
        }
    if node_name == "tools":
        tool_results = update.get("tool_results", {})
        return {
            "active_customer_id": update.get("active_customer_id"),
            "active_order_id": update.get("active_order_id"),
            "tool_result_keys": list(tool_results.keys()),
            "tool_results": tool_results,
        }
    if node_name == "memory":
        long_term_memory = update.get("long_term_memory", [])
        return {
            "long_term_memory_count": len(long_term_memory),
            "long_term_memory": long_term_memory,
        }
    if node_name == "verifier":
        tool_results = update.get("tool_results", {})
        return {
            "verification_errors": update.get("verification_errors", []),
            "tool_result_keys": list(tool_results.keys()),
        }
    if node_name == "respond":
        return {"final_response": update.get("final_response")}
    return update
