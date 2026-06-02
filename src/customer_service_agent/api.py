from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from customer_service_agent.config import PROJECT_ROOT, get_settings
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


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    def event_stream():
        yield _sse_event(
            "started",
            {
                "thread_id": payload.thread_id,
                "customer_id": payload.customer_id,
            },
        )
        try:
            for event in agent.stream_trace(
                thread_id=payload.thread_id,
                message=payload.message,
                customer_id=payload.customer_id,
            ):
                yield _sse_event(str(event["event"]), event["data"])
        except Exception as exc:
            yield _sse_event("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/demo/reset")
def reset_demo() -> dict[str, str]:
    settings = get_settings()
    if settings.app_env != "development":
        raise HTTPException(status_code=403, detail="Demo reset is only available in development.")

    script_path = PROJECT_ROOT / "frontend" / "dev" / "reset_demo_data.py"
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Demo reset script was not found.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Demo reset failed.").strip()
        raise HTTPException(status_code=500, detail=detail) from exc

    return {"status": "reset"}


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def run() -> None:
    uvicorn.run("customer_service_agent.api:app", host="0.0.0.0", port=8000, reload=False)
