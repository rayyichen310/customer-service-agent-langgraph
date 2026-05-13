from __future__ import annotations

import argparse
import json
import re
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
                result, node_trace = agent.trace(
                    thread_id=args.thread_id,
                    message=case["query"],
                    customer_id=case["customer_id"],
                )
                records.append(
                    {
                        "number": case["number"],
                        "function": case["function"],
                        "query": case["query"],
                        "input_customer_id": case["customer_id"],
                        "planner_iterations": planner_iterations(node_trace),
                        "tool_calls": planner_tool_call_names(node_trace),
                        "requested_actions": planner_requested_action_names(node_trace),
                        "order_id": scorecard_order_id(result.order_id, node_trace),
                        "customer_id": result.customer_id,
                        "tool_results": result.tool_results,
                        "verified_facts": result.verified_facts,
                        "response_constraints": result.response_constraints,
                        "verification_errors": result.verification_errors,
                        "response": result.response,
                        "node_trace": node_trace,
                    }
                )
                records[-1]["assertion_failures"] = scorecard_record_failures(records[-1])
                if args.show_node_trace:
                    log_node_trace(args.quiet, node_trace)
                log_progress(args.quiet, f"  tool_calls: {planner_tool_call_names(node_trace)}")
                actions = planner_requested_action_names(node_trace)
                if actions:
                    log_progress(args.quiet, f"  requested_actions: {actions}")
                if result.verification_errors:
                    log_progress(args.quiet, f"  verifier: {result.verification_errors[0]}")
                if records[-1]["assertion_failures"]:
                    log_progress(args.quiet, f"  assertions: {records[-1]['assertion_failures']}")
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

    failed_records = failed_scorecard_records(records)
    print(
        json.dumps(
            {
                "thread_id": args.thread_id,
                "results": records,
                "failed": [
                    {
                        "number": record["number"],
                        "function": record["function"],
                        "error": record.get("error"),
                        "assertion_failures": record.get("assertion_failures", []),
                    }
                    for record in failed_records
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return scorecard_exit_code(records)


def log_progress(quiet: bool, message: str) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def planner_iterations(node_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    iterations = []
    for index, item in enumerate(
        [item for item in node_trace if item.get("node") == "planner"],
        start=1,
    ):
        state = item.get("state", {})
        iterations.append(
            {
                "iteration": index,
                "tool_calls": [
                    call.get("name", "")
                    for call in state.get("tool_calls", [])
                ],
                "requested_actions": [
                    action.get("name", "")
                    for action in state.get("requested_actions", [])
                ],
                "requires_follow_up": state.get("requires_follow_up"),
            }
        )
    return iterations


def planner_tool_call_names(node_trace: list[dict[str, Any]]) -> list[str]:
    names = []
    for item in node_trace:
        if item.get("node") == "planner":
            names.extend(
                call.get("name", "")
                for call in item.get("state", {}).get("tool_calls", [])
            )
        if item.get("node") == "verifier" and _verifier_resolved_pending_intent(item):
            names.extend(
                action.get("name", "")
                for action in item.get("state", {}).get("requested_actions", [])
            )
    return names


def planner_requested_action_names(node_trace: list[dict[str, Any]]) -> list[str]:
    names = []
    for item in node_trace:
        if item.get("node") == "planner" or (
            item.get("node") == "verifier" and _verifier_resolved_pending_intent(item)
        ):
            names.extend(
                action.get("name", "")
                for action in item.get("state", {}).get("requested_actions", [])
            )
    return names


def scorecard_order_id(
    fallback_order_id: int | None,
    node_trace: list[dict[str, Any]],
) -> int | None:
    for item in reversed(node_trace):
        state = item.get("state", {})
        for action in reversed(state.get("requested_actions", [])):
            order_id = _int_or_none(action.get("args", {}).get("order_id"))
            if order_id is not None:
                return order_id

    for item in reversed(node_trace):
        order_id = _int_or_none(item.get("state", {}).get("pending_order_id"))
        if order_id is not None:
            return order_id

    return fallback_order_id


def scorecard_record_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    number = record["number"]
    tool_results = record.get("tool_results", {})
    verified_facts = record.get("verified_facts", {})
    tool_calls = record.get("tool_calls", [])
    requested_actions = record.get("requested_actions", [])
    errors = record.get("verification_errors", [])
    response = record.get("response", "")

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    if number == 1:
        require(tool_results.get("order", {}).get("status") == "in_transit", "order 12345 should be in_transit")
    elif number == 2:
        require(tool_results.get("order", {}).get("status") == "processing", "order 1001 should be processing")
    elif number == 3:
        require(tool_results.get("customer", {}).get("customer_id") == 1, "customer profile should be customer 1")
        require("order" not in tool_results, "profile response should not include stale order result")
    elif number == 4:
        require("order_lookup" in tool_calls, "refund should look up order first")
        require("request_refund" in requested_actions, "refund should request refund action")
        require(
            tool_results.get("refund", {}).get("status") == "refund_requested",
            "order 5678 should be refund_requested",
        )
        require(
            verified_facts.get("refund_request", {}).get("order_id") == 5678,
            "refund response should be grounded to order 5678",
        )
        require(
            verified_facts.get("refund_request", {}).get("status") == "refund_requested",
            "refund response should be grounded to refund_requested",
        )
        require(
            verified_facts.get("refund_request", {}).get("created_this_turn") is True,
            "refund response should be grounded as a current-turn request",
        )
        require(
            _response_mentions(response, "refund", "5678"),
            "refund response should mention the grounded refund and order",
        )
        require(_has_successful_action_style(response), "refund response should use 2-3 concise sentences")
        require(
            not _response_makes_unsupported_future_promise(response),
            "refund response should not promise unsupported future handling",
        )
        require(not _response_says_already_requested(response), "refund response should not say already requested")
        require(not errors, "refund should not ask for more info")
    elif number == 5:
        require("request_log_complaint" in requested_actions, "complaint should request complaint action")
        require(
            tool_results.get("complaint", {}).get("order_id") == 2222,
            "complaint should be logged for order 2222",
        )
        require(
            verified_facts.get("complaint_logged", {}).get("order_id") == 2222,
            "complaint response should be grounded to order 2222",
        )
        require(
            _response_mentions(response, "complaint", "2222"),
            "complaint response should mention the grounded complaint and order",
        )
        require(_has_successful_action_style(response), "complaint response should use 2-3 concise sentences")
        require(_has_empathy_phrase(response), "complaint response should include brief empathy")
        require(
            not _response_makes_unsupported_future_promise(response),
            "complaint response should not promise unsupported future handling",
        )
    elif number == 6:
        require(
            _ordered_contains(tool_calls, ["order_lookup", "request_refund"]),
            "conditional refund should look up before refund",
        )
        require(
            tool_results.get("refund", {}).get("status") == "refund_requested",
            "order 7890 should be refund_requested",
        )
        require(
            verified_facts.get("refund_request", {}).get("order_id") == 7890,
            "conditional refund response should be grounded to order 7890",
        )
        require(
            verified_facts.get("refund_request", {}).get("created_this_turn") is True,
            "conditional refund response should be grounded as a current-turn request",
        )
        require(
            _response_mentions(response, "refund", "7890"),
            "conditional refund response should mention the grounded refund and order",
        )
        require(
            _has_successful_action_style(response),
            "conditional refund response should use 2-3 concise sentences",
        )
        require(
            not _response_makes_unsupported_future_promise(response),
            "conditional refund response should not promise unsupported future handling",
        )
        require(
            not _response_says_already_requested(response),
            "conditional refund response should not say already requested",
        )
    elif number == 7:
        require("little more detail" not in response.lower(), "cancel should not ask for more info when active order is known")
        require(errors, "cancel should return a verifier policy error")
    elif number == 8:
        require("memories" in tool_results, "memory read should include memories")
        require("complaints" in tool_results, "memory read should include complaints")
        require("order" not in tool_results, "memory read should not include stale order result")
    elif number == 9:
        require("request_write_memory" in requested_actions, "memory write should request write action")
        require(
            tool_results.get("memory_write", {}).get("key") == "refund_preference",
            "memory write should update refund_preference",
        )
        require(
            verified_facts.get("memory_written", {}).get("key") == "refund_preference",
            "memory write response should be grounded to refund_preference",
        )
        require(_response_mentions(response, "refund"), "memory write response should mention the grounded preference")
        require(
            _has_successful_action_style(response),
            "memory write response should use 2-3 concise sentences",
        )
        require(
            not _response_makes_unsupported_future_promise(response),
            "memory write response should not promise unsupported future handling",
        )
        require(not errors, "memory write should not ask for more info")
    elif number == 10:
        require("request_log_complaint" in requested_actions, "personalization should log complaint")
        require(tool_results.get("complaint", {}).get("issue"), "personalization should create complaint")
        require(
            verified_facts.get("complaint_logged", {}).get("issue"),
            "personalization response should be grounded to the complaint issue",
        )
        if tool_results.get("issue_patterns"):
            require(
                verified_facts.get("issue_patterns", {}).get("repeated_late_delivery") is True,
                "personalization should ground repeated late-delivery pattern",
            )
            require(
                _response_mentions(response, "late"),
                "personalized response should mention the grounded late-delivery issue",
            )
        require(
            _has_successful_action_style(response),
            "personalized complaint response should use 2-3 concise sentences",
        )
        require(_has_empathy_phrase(response), "personalized complaint response should include brief empathy")
        require(
            not _response_makes_unsupported_future_promise(response),
            "personalized complaint response should not promise unsupported future handling",
        )
    elif number == 11:
        require(errors == ["Order 0 does not exist."], "invalid refund should be blocked by verifier")

    return failures


def failed_scorecard_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if "error" in record or record.get("assertion_failures")
    ]


def scorecard_exit_code(records: list[dict[str, Any]]) -> int:
    return 1 if failed_scorecard_records(records) else 0


def _response_mentions(response: str, *terms: str) -> bool:
    normalized = response.lower()
    return all(term.lower() in normalized for term in terms)


def _response_says_already_requested(response: str) -> bool:
    normalized = response.lower()
    return "already requested" in normalized or "already submitted" in normalized


def _has_successful_action_style(response: str) -> bool:
    return 2 <= _sentence_count(response) <= 3


def _sentence_count(response: str) -> int:
    return len([part for part in re.split(r"[.!?]+", response) if part.strip()])


def _has_empathy_phrase(response: str) -> bool:
    normalized = response.lower()
    return any(
        phrase in normalized
        for phrase in {
            "sorry",
            "understand",
            "appreciate",
            "thanks",
            "thank you",
        }
    )


def _response_makes_unsupported_future_promise(response: str) -> bool:
    normalized = response.lower()
    blocked_phrases = {
        "we are working",
        "we're working",
        "working to resolve",
        "will follow up",
        "we'll follow up",
        "will investigate",
        "we'll investigate",
        "looking into",
        "escalated",
        "will be resolved",
    }
    return any(phrase in normalized for phrase in blocked_phrases)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _verifier_resolved_pending_intent(item: dict[str, Any]) -> bool:
    return str(item.get("state", {}).get("reasoning") or "").startswith(
        "Resolved pending intent"
    )


def _ordered_contains(values: list[str], expected: list[str]) -> bool:
    position = 0
    for value in values:
        if position < len(expected) and value == expected[position]:
            position += 1
    return position == len(expected)


def log_node_trace(quiet: bool, node_trace: list[dict[str, Any]]) -> None:
    if quiet:
        return
    for item in node_trace:
        node = item["node"]
        state = item["state"]
        print(f"  node: {node}", file=sys.stderr, flush=True)
        if node == "planner":
            print(
                f"    customer={state.get('active_customer_id')} order={state.get('active_order_id')} "
                f"follow_up={state.get('requires_follow_up')}",
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
