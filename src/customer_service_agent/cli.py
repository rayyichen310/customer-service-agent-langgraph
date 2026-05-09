from __future__ import annotations

import argparse
import json
import uuid

from customer_service_agent.service import build_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer service agent CLI")
    parser.add_argument("message", nargs="?", help="User message")
    parser.add_argument("--thread-id", default=None, help="Conversation thread ID")
    parser.add_argument("--customer-id", type=int, default=None, help="Customer ID")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive chat loop.",
    )
    args = parser.parse_args()

    agent = build_agent()
    thread_id = args.thread_id or str(uuid.uuid4())

    if args.interactive:
        print(f"thread_id={thread_id}")
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in {"exit", "quit"}:
                break
            result = agent.invoke(thread_id=thread_id, message=user_input, customer_id=args.customer_id)
            print(f"agent> {result.response}")
        return

    if not args.message:
        parser.error("message is required unless --interactive is used")

    result = agent.invoke(thread_id=thread_id, message=args.message, customer_id=args.customer_id)
    print(json.dumps(result.model_dump(), ensure_ascii=True, indent=2))

