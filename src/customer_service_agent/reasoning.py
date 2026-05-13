from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from customer_service_agent.config import Settings


READ_TOOL_NAMES = {
    "order_lookup",
    "customer_profile",
    "read_customer_memory",
    "list_customer_complaints",
    "summarize_issue_patterns",
}
ORDER_TOOL_NAMES = {
    "order_lookup",
    "request_refund",
    "request_cancel_order",
    "request_log_complaint",
}
ACTION_TOOL_NAMES = {
    "request_refund",
    "request_cancel_order",
    "request_log_complaint",
    "request_write_memory",
}


@tool("order_lookup")
def order_lookup(order_id: int) -> str:
    """Read order details by order ID."""
    return "schema only"


@tool("customer_profile")
def customer_profile(customer_id: int) -> str:
    """Read customer profile details by customer ID."""
    return "schema only"


@tool("read_customer_memory")
def read_customer_memory(customer_id: int) -> str:
    """Read long-term customer memory records."""
    return "schema only"


@tool("list_customer_complaints")
def list_customer_complaints(customer_id: int) -> str:
    """Read previous customer complaints."""
    return "schema only"


@tool("summarize_issue_patterns")
def summarize_issue_patterns(customer_id: int) -> str:
    """Read summarized complaint issue patterns for a customer."""
    return "schema only"


@tool("request_refund")
def request_refund(order_id: int) -> str:
    """Request a refund action for an order. Execution is gated by verifier policy."""
    return "schema only"


@tool("request_cancel_order")
def request_cancel_order(order_id: int) -> str:
    """Request a cancel action for an order. Execution is gated by verifier policy."""
    return "schema only"


@tool("request_log_complaint")
def request_log_complaint(
    customer_id: int | None = None,
    order_id: int | None = None,
    issue: str = "customer requested to file a complaint",
) -> str:
    """Request logging a customer complaint. Execution is gated by verifier policy."""
    return "schema only"


@tool("request_write_memory")
def request_write_memory(customer_id: int | None = None, key: str = "", value: str = "") -> str:
    """Request writing a long-term customer memory preference or note."""
    return "schema only"


PLANNER_TOOLS = [
    order_lookup,
    customer_profile,
    read_customer_memory,
    list_customer_complaints,
    summarize_issue_patterns,
    request_refund,
    request_cancel_order,
    request_log_complaint,
    request_write_memory,
]


@dataclass
class ResponseContext:
    user_message: str
    tool_results: dict[str, Any]
    verification_errors: list[str]
    long_term_memory: list[dict[str, Any]]
    active_customer_id: int | None
    active_order_id: int | None
    verified_facts: dict[str, Any] = field(default_factory=dict)
    response_constraints: list[str] = field(default_factory=list)


@dataclass
class ToolPlan:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    requested_actions: list[dict[str, Any]] = field(default_factory=list)
    customer_id: int | None = None
    order_id: int | None = None
    issue: str | None = None
    memory_key: str | None = None
    memory_value: str | None = None
    requires_follow_up: bool = False
    follow_up_question: str | None = None
    steps: list[str] = field(default_factory=list)
    reasoning: str = ""


class Reasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        raise NotImplementedError

    def respond(self, context: ResponseContext) -> str:
        raise NotImplementedError


class StructuredChatReasoner(Reasoner):
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> ToolPlan:
        system_prompt = (
            "You are a tool-calling planner for a customer service agent. "
            "Call tools instead of answering directly. Follow a Reason-Act-Observe loop: use read "
            "tools to gather facts before requesting mutations, then use action request tools only "
            "after the needed facts are available in State.observations. Action request tools do "
            "not execute until a verifier approves them. For refunds and cancellations, if the "
            "order status is unknown, call order_lookup first and do not request the mutation in "
            "the same response. After State.observations contains the order status, call "
            "request_refund for eligible refund requests or request_cancel_order for eligible "
            "cancel requests. For customer profile, call customer_profile. "
            "For issue history, call read_customer_memory, list_customer_complaints, and "
            "summarize_issue_patterns. For complaints, call request_log_complaint; if details are "
            "missing, use issue='customer requested to file a complaint'. For memory writes, call "
            "request_write_memory with a key and value. Use the current active IDs when the user "
            "refers to 'it' or omits the order number."
        )
        state_prompt = json.dumps(
            {
                "active_customer_id": state_snapshot.get("active_customer_id"),
                "active_order_id": state_snapshot.get("active_order_id"),
                "known_issue": state_snapshot.get("issue"),
                "observations": state_snapshot.get("tool_results", {}),
                "react_iterations": state_snapshot.get("react_iterations"),
                "max_react_iterations": state_snapshot.get("max_react_iterations"),
            }
        )
        response = self._planner.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"State: {state_prompt}"),
                HumanMessage(content=f"User query: {user_message}"),
            ]
        )
        return _build_tool_plan(response, state_snapshot)

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]

        system_prompt = (
            "You are a warm customer service agent. "
            "Use verified_facts and tool_results exactly as ground truth. "
            "Follow response_constraints. "
            "Only mention facts supported by verified_facts or tool_results. "
            "Do not mention provided memory unless it is also represented in verified_facts "
            "or tool_results. "
            "Do not invent refund status, complaint IDs, delivery dates, customer history, "
            "or any action that is not present in the verified facts. "
            "Be concise, accurate, and personalized."
        )
        payload = {
            "verified_facts": context.verified_facts,
            "response_constraints": context.response_constraints,
            "tool_results": context.tool_results,
            "long_term_memory": context.long_term_memory[:5],
            "active_customer_id": context.active_customer_id,
            "active_order_id": context.active_order_id,
            "user_message": context.user_message,
        }
        response = self._model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=True)),
            ]
        )
        return _message_text(response)


class OpenAIReasoner(StructuredChatReasoner):
    def __init__(self, settings: Settings):
        kwargs = {
            "model": settings.openai_model,
            "temperature": 0,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._model = ChatOpenAI(**kwargs)
        self._planner = self._model.bind_tools(PLANNER_TOOLS)


class GoogleReasoner(StructuredChatReasoner):
    def __init__(self, settings: Settings):
        self._model = ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=0,
            api_key=settings.google_api_key,
        )
        self._planner = self._model.bind_tools(PLANNER_TOOLS)


def build_reasoner(settings: Settings) -> Reasoner:
    backend = settings.llm_backend.lower()
    if backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_BACKEND=openai.")
        return OpenAIReasoner(settings)
    if backend == "google":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_BACKEND=google.")
        return GoogleReasoner(settings)
    raise ValueError(f"Unsupported LLM_BACKEND: {settings.llm_backend}")


def _build_tool_plan(response: BaseMessage, state_snapshot: dict[str, Any]) -> ToolPlan:
    raw_tool_calls = getattr(response, "tool_calls", []) or []
    tool_calls = [_normalize_tool_call(tool_call) for tool_call in raw_tool_calls]
    requested_actions = [call for call in tool_calls if call["name"] in ACTION_TOOL_NAMES]

    planned_customer_id = _first_int_arg(tool_calls, "customer_id")
    planned_order_id = _first_int_arg(tool_calls, "order_id")
    customer_id = (
        planned_customer_id
        if planned_customer_id is not None
        else state_snapshot.get("active_customer_id")
    )
    order_id = planned_order_id if planned_order_id is not None else state_snapshot.get("active_order_id")

    for call in tool_calls:
        args = call["args"]
        if args.get("customer_id") is None and customer_id is not None:
            args["customer_id"] = customer_id
        if (
            call["name"] in ORDER_TOOL_NAMES
            and args.get("order_id") is None
            and order_id is not None
        ):
            args["order_id"] = order_id

    issue = _first_str_arg(tool_calls, "issue")
    memory_key = _first_str_arg(tool_calls, "key")
    memory_value = _first_str_arg(tool_calls, "value")
    steps = [call["name"] for call in tool_calls]
    reasoning = _message_text(response)
    if reasoning == str(getattr(response, "content", "")) and not reasoning:
        reasoning = "Tool calls selected by the model."

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
        steps=steps,
        reasoning=reasoning,
    )


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        name = str(tool_call.get("name") or "")
        args = tool_call.get("args") or {}
        tool_call_id = tool_call.get("id")
    else:
        name = str(getattr(tool_call, "name", ""))
        args = getattr(tool_call, "args", {}) or {}
        tool_call_id = getattr(tool_call, "id", None)
    return {"name": name, "args": dict(args), "id": tool_call_id}


def _first_int_arg(tool_calls: list[dict[str, Any]], key: str) -> int | None:
    for call in tool_calls:
        value = call["args"].get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _first_str_arg(tool_calls: list[dict[str, Any]], key: str) -> str | None:
    for call in tool_calls:
        value = call["args"].get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _message_text(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_blocks: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_blocks.append(str(block["text"]))
                elif isinstance(block, str):
                    text_blocks.append(block)
            if text_blocks:
                return "\n".join(text_blocks)
            return str(content)
    return str(message.content)
