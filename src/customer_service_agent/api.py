from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from customer_service_agent.models import ChatRequest, ChatResponse
from customer_service_agent.service import build_agent

app = FastAPI(title="Customer Service Agent", version="0.1.0")
agent = build_agent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return agent.invoke(
        thread_id=payload.thread_id,
        message=payload.message,
        customer_id=payload.customer_id,
    )


def run() -> None:
    uvicorn.run("customer_service_agent.api:app", host="0.0.0.0", port=8000, reload=False)

