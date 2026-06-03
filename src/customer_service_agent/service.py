from __future__ import annotations

from customer_service_agent.config import get_settings
from customer_service_agent.db import create_session_factory
from customer_service_agent.graph.agent import CustomerServiceAgent
from customer_service_agent.reasoning import build_reasoner
from customer_service_agent.repository import CustomerServiceRepository


def build_repository() -> CustomerServiceRepository:
    settings = get_settings()
    session_factory = create_session_factory(settings)
    return CustomerServiceRepository(session_factory)


def build_agent() -> CustomerServiceAgent:
    settings = get_settings()
    repository = build_repository()
    reasoner = build_reasoner(settings)
    return CustomerServiceAgent(reasoner=reasoner, repository=repository)
