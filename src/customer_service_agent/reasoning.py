from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from customer_service_agent.config import Settings
from customer_service_agent.models import QueryPlan


ORDER_ID_PATTERN = re.compile(r"\b(\d{1,10})\b")


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


class HeuristicReasoner(Reasoner):
    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> QueryPlan:
        lowered = user_message.lower()
        extracted_order_id = _extract_order_id(user_message)
        active_order_id = state_snapshot.get("active_order_id")
        active_customer_id = state_snapshot.get("active_customer_id")
        order_id = extracted_order_id if extracted_order_id is not None else active_order_id
        customer_id = active_customer_id

        if "profile" in lowered:
            return QueryPlan(
                intent="customer_profile",
                customer_id=customer_id,
                steps=["load_customer_profile"],
                reasoning="Customer profile request detected.",
                requires_follow_up=customer_id is None,
                follow_up_question="Please share your customer ID so I can retrieve your profile."
                if customer_id is None
                else None,
            )

        if "remember" in lowered or "prefer" in lowered:
            memory_key = "preference"
            memory_value = user_message.strip()
            if "refund" in lowered:
                memory_key = "refund_preference"
            return QueryPlan(
                intent="memory_write",
                customer_id=customer_id,
                memory_key=memory_key,
                memory_value=memory_value,
                steps=["write_memory"],
                reasoning="Memory write request detected.",
                requires_follow_up=customer_id is None,
                follow_up_question="Please share your customer ID so I can save that preference."
                if customer_id is None
                else None,
            )

        if "refund" in lowered:
            return QueryPlan(
                intent="refund_request",
                order_id=order_id,
                customer_id=customer_id,
                steps=["lookup_order", "request_refund"],
                reasoning="Refund request detected.",
                requires_follow_up=order_id is None,
                follow_up_question="Please provide the order ID you want refunded."
                if order_id is None
                else None,
            )

        if "cancel" in lowered:
            return QueryPlan(
                intent="cancel_order",
                order_id=order_id,
                customer_id=customer_id,
                steps=["lookup_order", "cancel_order"],
                reasoning="Cancellation request detected.",
                requires_follow_up=order_id is None,
                follow_up_question="Please provide the order ID you want cancelled."
                if order_id is None
                else None,
            )

        if "complain" in lowered or "complaint" in lowered:
            issue = _extract_issue(user_message)
            return QueryPlan(
                intent="complaint",
                order_id=order_id,
                customer_id=customer_id,
                issue=issue,
                steps=["lookup_order", "log_complaint"] if order_id else ["log_complaint"],
                reasoning="Complaint request detected.",
                requires_follow_up=customer_id is None and order_id is None,
                follow_up_question="Please provide your customer ID or order ID so I can log the complaint."
                if customer_id is None and order_id is None
                else None,
            )

        if "what issues" in lowered or ("before" in lowered and "issue" in lowered):
            return QueryPlan(
                intent="memory_read",
                customer_id=customer_id,
                memory_key="issue_history",
                steps=["read_memory", "read_complaints"],
                reasoning="Long-term memory retrieval request detected.",
                requires_follow_up=customer_id is None,
                follow_up_question="Please share your customer ID so I can review your history."
                if customer_id is None
                else None,
            )

        if "late again" in lowered or ("late" in lowered and "again" in lowered):
            return QueryPlan(
                intent="general_support",
                customer_id=customer_id if customer_id is not None else active_customer_id,
                issue=_extract_issue(user_message),
                steps=["read_complaints", "read_memory"],
                reasoning="Repeated issue detected and should be personalized.",
                requires_follow_up=(customer_id if customer_id is not None else active_customer_id) is None,
                follow_up_question="Please share your customer ID or order ID so I can review the repeated issue."
                if (customer_id if customer_id is not None else active_customer_id) is None and order_id is None
                else None,
            )

        if "where is my order" in lowered or "track" in lowered or "status" in lowered:
            return QueryPlan(
                intent="order_status",
                order_id=order_id,
                customer_id=customer_id,
                steps=["lookup_order"],
                reasoning="Order tracking request detected.",
                requires_follow_up=order_id is None,
                follow_up_question="Please provide the order ID you want to track."
                if order_id is None
                else None,
            )

        return QueryPlan(
            intent="general_support",
            order_id=order_id,
            customer_id=customer_id,
            steps=["lookup_order"] if order_id else [],
            reasoning="Falling back to general support.",
        )

    def respond(self, context: ResponseContext) -> str:
        if context.verification_errors:
            return context.verification_errors[0]

        if context.intent == "order_status":
            order = context.tool_results.get("order")
            if not order:
                return "I could not find that order. Please verify the order ID."
            return (
                f"Order {order['order_id']} for {order['product_name']} is currently "
                f"`{order['status']}`."
            )

        if context.intent == "customer_profile":
            customer = context.tool_results.get("customer")
            if not customer:
                return "I could not find that customer profile. Please verify the customer ID."
            return (
                f"Your profile shows customer ID {customer['customer_id']}, name {customer['name']}, "
                f"and email {customer['email']}."
            )

        if context.intent == "refund_request":
            refund = context.tool_results.get("refund")
            if not refund:
                return "I could not submit the refund request."
            return (
                f"Your refund request for order {refund['order_id']} has been initiated. "
                f"The order status is now `{refund['status']}`."
            )

        if context.intent == "cancel_order":
            cancelled = context.tool_results.get("cancelled_order")
            if not cancelled:
                return "I could not submit the cancellation request."
            return (
                f"Order {cancelled['order_id']} has been marked as `{cancelled['status']}`."
            )

        if context.intent == "complaint":
            complaint = context.tool_results.get("complaint")
            if not complaint:
                return "I could not log the complaint."
            return (
                f"I logged complaint {complaint['complaint_id']} for "
                f"{complaint['issue']}."
            )

        if context.intent == "memory_write":
            saved = context.tool_results.get("memory_write")
            if not saved:
                return "I could not save that preference."
            return f"I remembered `{saved['value']}` for future interactions."

        if context.intent == "memory_read":
            complaints = context.tool_results.get("complaints", [])
            memories = context.tool_results.get("memories", [])
            if not complaints and not memories:
                return "I did not find any prior issues or stored preferences."
            parts = []
            if complaints:
                parts.append(f"I found {len(complaints)} prior complaint(s).")
            if memories:
                recent = ", ".join(memory["value"] for memory in memories[:2])
                parts.append(f"Stored preferences include: {recent}.")
            return " ".join(parts)

        if context.intent == "general_support":
            patterns = context.tool_results.get("issue_patterns")
            if patterns and patterns.get("repeated_late_delivery"):
                return (
                    "I can see repeated late-delivery issues in your history. "
                    "I will prioritize this case and can help log a complaint or start a refund request."
                )
            return "I can help with order status, refunds, complaints, and customer history."

        return "I processed your request."


class OpenAIReasoner(Reasoner):
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

    def plan(self, user_message: str, state_snapshot: dict[str, Any]) -> QueryPlan:
        system_prompt = (
            "You are planning actions for a customer service agent. "
            "Return a structured plan only. Supported intents: order_status, customer_profile, "
            "refund_request, complaint, memory_read, memory_write, cancel_order, general_support. "
            "Use the current active IDs when the user refers to 'it' or omits the order number."
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


def build_reasoner(settings: Settings) -> Reasoner:
    if settings.llm_backend == "openai" and settings.openai_api_key:
        return OpenAIReasoner(settings)
    return HeuristicReasoner()


def _extract_order_id(user_message: str) -> int | None:
    match = ORDER_ID_PATTERN.search(user_message)
    if not match:
        return None
    return int(match.group(1))


def _extract_issue(user_message: str) -> str:
    lowered = user_message.strip()
    lowered = re.sub(r"^i want to complain about\s*", "", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"^my\s*", "", lowered, flags=re.IGNORECASE)
    return lowered.rstrip("?")


def _message_text(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        if isinstance(message.content, str):
            return message.content
        return str(message.content)
    return str(message.content)
