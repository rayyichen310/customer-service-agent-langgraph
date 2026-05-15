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
    PLANNER_TOOLS,
)
from customer_service_agent.prompts import BASE_RESPONDER_INSTRUCTIONS, PLANNER_INSTRUCTIONS


@dataclass
class ResponseContext:
    user_message: str
    verification_decision: dict[str, Any]
    long_term_memory: list[dict[str, Any]]
    active_customer_id: int | None
    active_order_id: int | None
    verified_facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolPlan:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    requested_actions: list[dict[str, Any]] = field(default_factory=list)
    customer_id: int | None = None
    order_id: int | None = None
    order_reference: OrderReference = field(default_factory=OrderReference)
    issue: str | None = None
    memory_candidate: MemoryWriteCandidate = field(default_factory=MemoryWriteCandidate)


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
                "planner_feedback": state_snapshot.get("planner_feedback", {}),
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
        return _build_tool_plan(response)

    def respond(self, context: ResponseContext) -> str:
        payload = {
            "verified_facts": context.verified_facts,
            "verification_decision": context.verification_decision,
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


def _build_tool_plan(response: BaseMessage) -> ToolPlan:
    raw_tool_calls = getattr(response, "tool_calls", []) or []
    tool_calls = [_normalize_tool_call(tool_call) for tool_call in raw_tool_calls]

    return ToolPlan(
        tool_calls=tool_calls,
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
