from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from customer_service_agent.config import Settings
from customer_service_agent.models import MemoryWriteCandidate, OrderReference
from customer_service_agent.graph.tools import (
    ACTION_TOOL_NAMES,
    CONTROL_TOOL_NAMES,
    ORDER_TOOL_NAMES,
    PLANNER_TOOLS,
)
from customer_service_agent.prompts import BASE_RESPONDER_INSTRUCTIONS, PLANNER_INSTRUCTIONS


DEFAULT_FOLLOW_UP_QUESTION = "I need a little more detail to help with that."


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
    order_reference: OrderReference = field(default_factory=OrderReference)
    issue: str | None = None
    memory_key: str | None = None
    memory_value: str | None = None
    memory_candidate: MemoryWriteCandidate = field(default_factory=MemoryWriteCandidate)
    missing_slots: list[str] = field(default_factory=list)
    requires_replan_after_tools: bool = False
    confidence: str = "low"
    needs_user_clarification: bool = False
    clarification_question: str | None = None
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
        state_prompt = json.dumps(
            {
                "active_customer_id": state_snapshot.get("active_customer_id"),
                "active_order_id": state_snapshot.get("active_order_id"),
                "order_reference": state_snapshot.get("order_reference"),
                "known_issue": state_snapshot.get("issue"),
                "observations": state_snapshot.get("tool_results", {}),
                "missing_slots": state_snapshot.get("missing_slots", []),
                "planner_feedback": state_snapshot.get("planner_feedback"),
                "verification_decision": state_snapshot.get("verification_decision", {}),
                "react_iterations": state_snapshot.get("react_iterations"),
                "max_react_iterations": state_snapshot.get("max_react_iterations"),
            }
        )
        response = self._planner.invoke(
            [
                SystemMessage(content=PLANNER_INSTRUCTIONS),
                HumanMessage(content=f"State: {state_prompt}"),
                HumanMessage(content=f"User query: {user_message}"),
            ]
        )
        return _build_tool_plan(response, state_snapshot)

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]
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
                SystemMessage(content=BASE_RESPONDER_INSTRUCTIONS),
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
    requires_replan_after_tools = any(call["name"] in CONTROL_TOOL_NAMES for call in tool_calls)
    customer_id = _planned_id(tool_calls, "customer_id", state_snapshot.get("active_customer_id"))
    order_id = _planned_id(tool_calls, "order_id", None)
    _fill_missing_ids(tool_calls, customer_id=customer_id, order_id=order_id)

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
        requires_replan_after_tools=requires_replan_after_tools,
        follow_up_question=None if tool_calls else DEFAULT_FOLLOW_UP_QUESTION,
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


def _planned_id(
    tool_calls: list[dict[str, Any]],
    key: str,
    fallback: int | None,
) -> int | None:
    value = _first_int_arg(tool_calls, key)
    return value if value is not None else fallback


def _fill_missing_ids(
    tool_calls: list[dict[str, Any]],
    *,
    customer_id: int | None,
    order_id: int | None,
) -> None:
    for call in tool_calls:
        args = call["args"]
        if args.get("customer_id") is None and customer_id is not None:
            args["customer_id"] = customer_id
        if call["name"] in ORDER_TOOL_NAMES and args.get("order_id") is None and order_id is not None:
            args["order_id"] = order_id


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
