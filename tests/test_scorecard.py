from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_scorecard.py"
SPEC = importlib.util.spec_from_file_location("run_scorecard", SCRIPT_PATH)
assert SPEC and SPEC.loader
run_scorecard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_scorecard)


def test_scorecard_assertions_fail_bad_refund_record() -> None:
    record = {
        "number": 4,
        "tool_calls": ["order_lookup"],
        "requested_actions": [],
        "tool_results": {"order": {"status": "delivered"}},
        "verification_errors": ["I need a little more detail to help with that."],
        "response": "I need a little more detail to help with that.",
    }

    failures = run_scorecard.scorecard_record_failures(record)
    record["assertion_failures"] = failures

    assert failures
    assert "refund should request refund action" in failures
    assert run_scorecard.scorecard_exit_code([record]) == 1
