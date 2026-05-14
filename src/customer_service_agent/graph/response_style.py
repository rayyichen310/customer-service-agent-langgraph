from __future__ import annotations

from typing import Any


def polish_customer_response(
    response: str,
    verified_facts: dict[str, Any] | None = None,
) -> str:
    """Keep response_style as a minimal boundary, not a repair layer."""
    return response.strip()
