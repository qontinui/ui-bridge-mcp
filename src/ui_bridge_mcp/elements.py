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


def stated_visibility(state: dict[str, Any]) -> bool | None:
    """What the snapshot says about visibility: ``True``, ``False``, or ``None``.

    ``None`` means the producer did not say — the ``visible`` field is absent
    or ``null``. A renderer must not read that as visible (the old
    ``state.get("visible", True)`` default) nor as hidden (``not None``).
    """
    visible = state.get("visible")
    return None if visible is None else bool(visible)


def stated_enabled(state: dict[str, Any]) -> bool | None:
    """What the snapshot says about ``enabled``: ``True``, ``False``, or ``None``.

    ``None`` means the field is absent or ``null``; the same reading as
    :func:`stated_visibility`.
    """
    enabled = state.get("enabled")
    return None if enabled is None else bool(enabled)
