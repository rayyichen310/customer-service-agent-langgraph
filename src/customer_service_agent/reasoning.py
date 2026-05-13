from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from customer_service_agent.config import Settings
from customer_service_agent.models import QueryPlan


@dataclass
class ResponseContext:
    user_message: str
    intent: str | None
    tool_results: dict[str, Any]
    verification_errors: list[str]
    long_term_memory: list[dict[str, Any]]
    active_customer_id: int | None
    active_order_id: int | None


class Reasoner:
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> QueryPlan:
        raise NotImplementedError

    def respond(self, context: ResponseContext) -> str:
        raise NotImplementedError


class StructuredChatReasoner(Reasoner):
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> QueryPlan:
        system_prompt = (
            "You are planning actions for a customer service agent. "
            "Return a structured plan only. Supported intents: order_status, customer_profile, "
            "refund_request, complaint, memory_read, memory_write, cancel_order, general_support. "
            "Use the current active IDs when the user refers to 'it' or omits the order number. "
            "For a complaint without details, set intent=complaint, requires_follow_up=false, "
            "and use a concise generic issue such as 'customer requested to file a complaint'. "
            "For memory writes, fill both memory_key and memory_value."
        )
        state_prompt = json.dumps(
            {
                "active_customer_id": state_snapshot.get("active_customer_id"),
                "active_order_id": state_snapshot.get("active_order_id"),
                "known_issue": state_snapshot.get("issue"),
            }
        )
        result = self._planner.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"State: {state_prompt}"),
                HumanMessage(content=f"User query: {user_message}"),
            ]
        )
        result = _coerce_query_plan(result)
        if result.order_id is None:
            result.order_id = state_snapshot.get("active_order_id")
        if result.customer_id is None:
            result.customer_id = state_snapshot.get("active_customer_id")
        return result

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]

        system_prompt = (
            "You are a helpful customer service agent. "
            "Use the tool outputs exactly as ground truth. "
            "Be concise, accurate, and personalized."
        )
        payload = {
            "intent": context.intent,
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
        self._planner = self._model.with_structured_output(QueryPlan)


class GoogleReasoner(StructuredChatReasoner):
    def __init__(self, settings: Settings):
        self._model = ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=0,
            api_key=settings.google_api_key,
        )
        self._planner = self._model.with_structured_output(
            schema=QueryPlan.model_json_schema(),
            method="json_schema",
        )


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


def _coerce_query_plan(result: Any) -> QueryPlan:
    if isinstance(result, QueryPlan):
        return result
    if isinstance(result, dict):
        return QueryPlan.model_validate(result)
    if isinstance(result, str):
        return QueryPlan.model_validate_json(result)
    raise TypeError(f"Planner returned unsupported result type: {type(result).__name__}")


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
