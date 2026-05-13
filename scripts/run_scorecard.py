from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_service_agent.config import get_settings  # noqa: E402
from customer_service_agent.service import build_agent  # noqa: E402


SCORECARD_CASES: list[dict[str, Any]] = [
    {
        "number": 1,
        "function": "Intent Parsing",
        "query": "Where is my order 12345?",
        "customer_id": 1,
    },
    {
        "number": 2,
        "function": "OrderLookupTool",
        "query": "Check status of order 1001",
        "customer_id": 1,
    },
    {
        "number": 3,
        "function": "CustomerProfileTool",
        "query": "Show my profile",
        "customer_id": 1,
    },
    {
        "number": 4,
        "function": "RefundTool",
        "query": "Refund order 5678",
        "customer_id": 2,
    },
    {
        "number": 5,
        "function": "ComplaintLoggerTool",
        "query": "I want to complain about order 2222",
        "customer_id": 3,
    },
    {
        "number": 6,
        "function": "Multi-step Reasoning",
        "query": "Refund order 7890 if delivered",
        "customer_id": 1,
    },
    {
        "number": 7,
        "function": "Short-Term Memory (STM)",
        "query": "Cancel it",
        "customer_id": None,
    },
    {
        "number": 8,
        "function": "Long-Term Memory (Read)",
        "query": "What issues have I had before?",
        "customer_id": 1,
    },
    {
        "number": 9,
        "function": "Long-Term Memory (Write)",
        "query": "Remember I prefer refunds",
        "customer_id": 1,
    },
    {
        "number": 10,
        "function": "Personalization",
        "query": "My order is late again",
        "customer_id": 1,
    },
    {
        "number": 11,
        "function": "Verifier",
        "query": "Refund order 0000",
        "customer_id": 1,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PDF natural-language scorecard.")
    parser.add_argument(
        "--thread-id",
        default=f"scorecard-{uuid.uuid4()}",
        help="Thread ID reused across all queries to exercise STM.",
    )
    parser.add_argument(
        "--reset-demo-data",
        action="store_true",
        help="Reset known seeded orders and remove scorecard-created complaint/memory rows first.",
    )
    parser.add_argument(
        "--keep-demo-data",
        action="store_true",
        help="Do not clean up scorecard-created database rows after the run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final JSON output; suppress per-case progress messages.",
    )
    parser.add_argument(
        "--show-node-trace",
        action="store_true",
        help="Print planner/read_tools/memory/verifier/actions/respond updates for each query.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=int,
        choices=range(1, len(SCORECARD_CASES) + 1),
        metavar="N",
        help="Only run one scorecard case number. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if args.reset_demo_data:
        log_progress(args.quiet, "Resetting demo data before scorecard...")
        reset_demo_data()

    log_progress(args.quiet, "Building agent...")
    agent = build_agent()
    records: list[dict[str, Any]] = []
    cases = [case for case in SCORECARD_CASES if not args.case or case["number"] in args.case]

    try:
        for case in cases:
            log_progress(
                args.quiet,
                f"[{case['number']}/{len(SCORECARD_CASES)}] {case['function']}: {case['query']}",
            )
            try:
                if args.show_node_trace:
                    result, node_trace = agent.trace(
                        thread_id=args.thread_id,
                        message=case["query"],
                        customer_id=case["customer_id"],
                    )
                else:
                    result = agent.invoke(
                        thread_id=args.thread_id,
                        message=case["query"],
                        customer_id=case["customer_id"],
                    )
                    node_trace = []
                records.append(
                    {
                        "number": case["number"],
                        "function": case["function"],
                        "query": case["query"],
                        "input_customer_id": case["customer_id"],
                        "intent": result.intent,
                        "order_id": result.order_id,
                        "customer_id": result.customer_id,
                        "tool_results": result.tool_results,
                        "verification_errors": result.verification_errors,
                        "response": result.response,
                        "node_trace": node_trace,
                    }
                )
                if args.show_node_trace:
                    log_node_trace(args.quiet, node_trace)
                log_progress(args.quiet, f"  intent: {result.intent}")
                if result.verification_errors:
                    log_progress(args.quiet, f"  verifier: {result.verification_errors[0]}")
                log_progress(args.quiet, f"  response: {result.response}")
            except Exception as exc:
                records.append(
                    {
                        "number": case["number"],
                        "function": case["function"],
                        "query": case["query"],
                        "input_customer_id": case["customer_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                log_progress(args.quiet, f"  -> error={type(exc).__name__}: {exc}")
    finally:
        if not args.keep_demo_data:
            log_progress(args.quiet, "Cleaning up demo data...")
            reset_demo_data()

    print(json.dumps({"thread_id": args.thread_id, "results": records}, ensure_ascii=True, indent=2))
    return 1 if any("error" in record for record in records) else 0


def log_progress(quiet: bool, message: str) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def log_node_trace(quiet: bool, node_trace: list[dict[str, Any]]) -> None:
    if quiet:
        return
    for item in node_trace:
        node = item["node"]
        state = item["state"]
        print(f"  node: {node}", file=sys.stderr, flush=True)
        if node == "planner":
            print(
                f"    intent={state.get('intent')} customer={state.get('active_customer_id')} "
                f"order={state.get('active_order_id')} follow_up={state.get('requires_follow_up')}",
                file=sys.stderr,
                flush=True,
            )
            if state.get("tool_calls"):
                calls = [call.get("name") for call in state["tool_calls"]]
                print(f"    tool_calls={calls}", file=sys.stderr, flush=True)
            if state.get("plan_steps"):
                print(f"    steps={state['plan_steps']}", file=sys.stderr, flush=True)
        elif node in {"read_tools", "actions"}:
            print(
                f"    tools={state.get('tool_result_keys', [])} "
                f"customer={state.get('active_customer_id')} order={state.get('active_order_id')}",
                file=sys.stderr,
                flush=True,
            )
        elif node in {"memory", "memory_update"}:
            print(
                f"    long_term_memory_count={state.get('long_term_memory_count', 0)}",
                file=sys.stderr,
                flush=True,
            )
        elif node == "verifier":
            print(
                f"    errors={state.get('verification_errors', [])} "
                f"tool_keys={state.get('tool_result_keys', [])}",
                file=sys.stderr,
                flush=True,
            )
        elif node == "respond":
            print(f"    final_response={state.get('final_response')}", file=sys.stderr, flush=True)
        else:
            print(f"    {state}", file=sys.stderr, flush=True)


def reset_demo_data() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url, future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE orders
                SET status = CASE order_id
                    WHEN 12345 THEN 'in_transit'
                    WHEN 1001 THEN 'processing'
                    WHEN 5678 THEN 'delivered'
                    WHEN 2222 THEN 'delivered'
                    WHEN 7890 THEN 'delivered'
                    ELSE status
                END
                WHERE order_id IN (12345, 1001, 5678, 2222, 7890)
                """
            )
        )
        connection.execute(
            text(
                """
                DELETE FROM complaints
                WHERE issue NOT IN ('delivery was late last month', 'late delivery again')
                """
            )
        )
        connection.execute(
            text(
                """
                DELETE FROM customer_memory
                WHERE NOT (
                    (customer_id = 1 AND `key` = 'refund_preference' AND `value` = 'Remember I prefer refunds')
                    OR (customer_id = 1 AND `key` = 'issue_history' AND `value` = 'Repeated late delivery complaints')
                    OR (customer_id = 2 AND `key` = 'loyalty_note' AND `value` = 'Frequent buyer of computer accessories')
                )
                """
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
