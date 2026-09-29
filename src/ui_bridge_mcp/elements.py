"""Shared accessors for UI Bridge element dicts."""

from __future__ import annotations

from typing import Any


def element_state(el: dict[str, Any]) -> dict[str, Any]:
    """An element's ``state`` object, or ``{}`` when absent or not an object.

    ``state: null`` (or any non-object) must not crash a renderer. The returned
    dict IS the element's own state when it is one, so in-place edits (content
    truncation, sanitizing) still land on the element. Callers that must tell an
    absent state from an empty one check ``el.get("state")`` themselves.
    """
    state = el.get("state")
    return state if isinstance(state, dict) else {}
