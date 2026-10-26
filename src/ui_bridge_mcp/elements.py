"""Shared accessors for UI Bridge element dicts."""

from __future__ import annotations

import re
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


# ui-bridge ai/content-comparison.ts getContentRole: the roles an element's
# `type` implies when it states no contentMetadata.contentRole.
_TYPE_ROLES = {
    "heading": "heading",
    "metric-value": "metric",
    "metric": "metric",
    "status-message": "status",
    "status": "status",
    "label": "label",
    "badge": "badge",
    "table-cell": "table-cell",
    "table-header": "table-header",
    "caption": "caption",
}


def stated_content_role(el: dict[str, Any]) -> tuple[bool, str | None]:
    """The element's content role as the SDK reads it.

    Returns ``(True, role)``, or ``(True, None)`` for an element that is not
    content: ``contentMetadata`` is optional, and the SDK reads its absence as
    "no role" after falling back to the element's ``type``. A
    ``contentMetadata`` that is not an object, or a ``contentRole`` that is not
    a string, says nothing either way: ``(False, None)``, the role is UNKNOWN.
    """
    meta = el.get("contentMetadata")
    if meta is not None:
        if not isinstance(meta, dict):
            return False, None
        role = meta.get("contentRole")
        if role is not None and not isinstance(role, str):
            return False, None
        if role:
            return True, role
    el_type = el.get("type")
    el_type = el_type.lower() if isinstance(el_type, str) else ""
    if re.fullmatch(r"h[1-6]", el_type):
        return True, "heading"
    return True, _TYPE_ROLES.get(el_type)


def is_content_element(el: dict[str, Any]) -> bool:
    """Whether the SDK counts the element as content, not interactive.

    ui-bridge ai/content-comparison.ts isContentElement: a ``content``
    category, any ``contentMetadata`` at all, or a content role read by
    :func:`stated_content_role`.
    """
    if el.get("category") == "content":
        return True
    meta = el.get("contentMetadata")
    # JavaScript truthiness: any object, array or non-empty string counts.
    if meta is not None and meta is not False and meta != "" and meta != 0:
        return True
    known, role = stated_content_role(el)
    return known and role is not None
