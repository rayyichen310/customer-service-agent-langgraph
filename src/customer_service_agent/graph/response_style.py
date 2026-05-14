from __future__ import annotations

import re
from typing import Any


RAW_STATUS_TOKENS = {
    "refund_requested",
    "cancel_requested",
    "in_transit",
}


def polish_customer_response(
    response: str,
    verified_facts: dict[str, Any] | None = None,
) -> str:
    verified_facts = verified_facts or {}
    response = soften_raw_status_tokens(response)
    response = remove_current_turn_already_wording(response, verified_facts)
    response = remove_unverified_action_sentences(response, verified_facts)
    return remove_unsupported_future_promise_sentences(response)


def soften_raw_status_tokens(response: str) -> str:
    polished = response
    for token in RAW_STATUS_TOKENS:
        polished = re.sub(
            rf"`?{re.escape(token)}`?",
            token.replace("_", " "),
            polished,
            flags=re.IGNORECASE,
        )
    polished = re.sub(
        r"(?<!currently )`?\bprocessing\b`?",
        "currently processing",
        polished,
        flags=re.IGNORECASE,
    )
    return polished


def remove_unsupported_future_promise_sentences(response: str) -> str:
    promise_patterns = {
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
    sentences = re.findall(r"[^.!?]+[.!?]*", response)
    kept = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and not any(pattern in sentence.lower() for pattern in promise_patterns)
    ]
    return " ".join(kept).strip() or response.strip()


def remove_current_turn_already_wording(
    response: str,
    verified_facts: dict[str, Any],
) -> str:
    current_turn_action = any(
        verified_facts.get(key, {}).get("created_this_turn")
        for key in {"refund_request", "cancellation_request"}
    )
    if not current_turn_action:
        return response
    response = re.sub(r"\balready submitted\b", "submitted", response, flags=re.IGNORECASE)
    return re.sub(r"\balready requested\b", "requested", response, flags=re.IGNORECASE)


def remove_unverified_action_sentences(
    response: str,
    verified_facts: dict[str, Any],
) -> str:
    if "complaint_logged" not in verified_facts:
        return response

    blocked_terms: set[str] = set()
    if "refund_request" not in verified_facts:
        blocked_terms.add("refund request")
    if "cancellation_request" not in verified_facts:
        blocked_terms.add("cancellation request")

    if not blocked_terms:
        return response

    sentences = re.findall(r"[^.!?]+[.!?]*", response)
    kept = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and not any(term in sentence.lower() for term in blocked_terms)
    ]
    return " ".join(kept).strip() or response.strip()
