"""Tests for agent mode features: RefManager, DiffTracker, formatting helpers."""

from __future__ import annotations

from typing import Any

import pytest  # type: ignore[import-not-found]

from ui_bridge_mcp.server import (
    CONTENT_END,
    CONTENT_START,
    DiffTracker,
    RefManager,
    _format_diff,
    format_element_compact,
    sanitize_element_content,
    truncate_field,
)

# =============================================================================
# Sample data helpers
# =============================================================================


def _el(
    id: str,
    type: str = "button",
    label: str = "",
    category: str = "interactive",
    visible: bool = True,
    enabled: bool = True,
    value: str | None = None,
    checked: bool = False,
    focused: bool = False,
    x: float = 0,
    y: float = 0,
    w: float = 100,
    h: float = 40,
    content_role: str | None = None,
    text_content: str | None = None,
) -> dict[str, Any]:
    """Build a mock element dict."""
    state: dict[str, Any] = {
        "visible": visible,
        "enabled": enabled,
        "rect": {"x": x, "y": y, "width": w, "height": h},
    }
    if value is not None:
        state["value"] = value
    if checked:
        state["checked"] = True
    if focused:
        state["focused"] = True
    if text_content is not None:
        state["textContent"] = text_content
    el: dict[str, Any] = {
        "id": id,
        "type": type,
        "label": label,
        "category": category,
        "state": state,
    }
    if content_role:
        el["contentMetadata"] = {"contentRole": content_role}
    return el


# =============================================================================
# RefManager
# =============================================================================


class TestRefManager:
    def test_assign_sequential(self) -> None:
        rm = RefManager()
        assert rm.assign("btn-1") == "@e1"
        assert rm.assign("btn-2") == "@e2"
        assert rm.assign("btn-3") == "@e3"

    def test_assign_idempotent(self) -> None:
        rm = RefManager()
        ref = rm.assign("btn-1")
        assert rm.assign("btn-1") == ref

    def test_resolve_ref(self) -> None:
        rm = RefManager()
        rm.assign("btn-submit")
        assert rm.resolve("@e1") == "btn-submit"

    def test_resolve_passthrough(self) -> None:
        rm = RefManager()
        assert rm.resolve("btn-submit") == "btn-submit"

    def test_resolve_unknown_ref_raises(self) -> None:
        rm = RefManager()
        with pytest.raises(ValueError, match="Unknown ref @e99"):
            rm.resolve("@e99")

    def test_reset(self) -> None:
        rm = RefManager()
        rm.assign("a")
        rm.assign("b")
        rm.reset()
        # After reset, counter starts over
        assert rm.assign("c") == "@e1"
        # Old ref is gone
        with pytest.raises(ValueError):
            rm.resolve("@e2")

    def test_resolve_after_reset_and_reassign(self) -> None:
        rm = RefManager()
        rm.assign("old-id")
        rm.reset()
        rm.assign("new-id")
        assert rm.resolve("@e1") == "new-id"


# =============================================================================
# DiffTracker
# =============================================================================


class TestDiffTracker:
    def test_first_snapshot_returns_none(self) -> None:
        dt = DiffTracker()
        result = dt.update_and_diff([_el("a"), _el("b")])
        assert result is None

    def test_no_changes(self) -> None:
        dt = DiffTracker()
        elements = [_el("a"), _el("b")]
        dt.update_and_diff(elements)
        diff = dt.update_and_diff(elements)
        assert diff is not None
        assert diff["appeared"] == []
        assert diff["disappeared"] == []
        assert diff["modified"] == []

    def test_appeared(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("a")])
        diff = dt.update_and_diff([_el("a"), _el("b")])
        assert diff is not None
        assert diff["appeared"] == ["b"]
        assert diff["disappeared"] == []

    def test_disappeared(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("a"), _el("b")])
        diff = dt.update_and_diff([_el("a")])
        assert diff is not None
        assert diff["disappeared"] == ["b"]
        assert diff["appeared"] == []

    def test_modified_value(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("input-1", value="")])
        diff = dt.update_and_diff([_el("input-1", value="hello")])
        assert diff is not None
        assert len(diff["modified"]) == 1
        mod = diff["modified"][0]
        assert mod["id"] == "input-1"
        assert mod["changes"]["value"] == {"from": "", "to": "hello"}

    def test_modified_visibility(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("msg", visible=True)])
        diff = dt.update_and_diff([_el("msg", visible=False)])
        assert diff is not None
        assert len(diff["modified"]) == 1
        assert diff["modified"][0]["changes"]["visible"] == {
            "from": True,
            "to": False,
        }

    def test_modified_enabled(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("btn", enabled=False)])
        diff = dt.update_and_diff([_el("btn", enabled=True)])
        assert diff is not None
        assert diff["modified"][0]["changes"]["enabled"] == {
            "from": False,
            "to": True,
        }

    def test_complex_diff(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("a"), _el("b"), _el("c", value="old")])
        diff = dt.update_and_diff([_el("b"), _el("c", value="new"), _el("d")])
        assert diff is not None
        assert diff["appeared"] == ["d"]
        assert diff["disappeared"] == ["a"]
        assert len(diff["modified"]) == 1
        assert diff["modified"][0]["id"] == "c"

    def test_tracks_text_content(self) -> None:
        dt = DiffTracker()
        dt.update_and_diff([_el("p", text_content="Hello")])
        diff = dt.update_and_diff([_el("p", text_content="World")])
        assert diff is not None
        assert diff["modified"][0]["changes"]["textContent"] == {
            "from": "Hello",
            "to": "World",
        }


# =============================================================================
# format_element_compact
# =============================================================================


class TestFormatElementCompact:
    def test_basic(self) -> None:
        el = _el("btn-submit", type="button", label="Submit")
        result = format_element_compact(el, "@e1")
        assert result.startswith("@e1 btn-submit (button)")
        assert '"Submit"' in result

    def test_includes_rect(self) -> None:
        el = _el("btn", x=100, y=200, w=40, h=40)
        result = format_element_compact(el, "@e1")
        assert "[100,200 40x40]" in result

    def test_hidden_flag(self) -> None:
        el = _el("btn", visible=False)
        result = format_element_compact(el, "@e1")
        assert "hidden" in result

    def test_disabled_flag(self) -> None:
        el = _el("btn", enabled=False)
        result = format_element_compact(el, "@e1")
        assert "disabled" in result

    def test_has_value_flag(self) -> None:
        el = _el("input", value="some text")
        result = format_element_compact(el, "@e1")
        assert "has-value" in result

    def test_checked_flag(self) -> None:
        el = _el("cb", checked=True)
        result = format_element_compact(el, "@e1")
        assert "checked" in result

    def test_focused_flag(self) -> None:
        el = _el("input", focused=True)
        result = format_element_compact(el, "@e1")
        assert "focused" in result

    def test_content_role(self) -> None:
        el = _el(
            "h1",
            type="heading",
            category="content",
            content_role="heading",
            label="Page Title",
        )
        result = format_element_compact(el, "@e1")
        assert "content:heading" in result

    def test_no_label(self) -> None:
        el = _el("btn", label="")
        result = format_element_compact(el, "@e1")
        # Should not contain empty quotes
        assert '""' not in result

    def test_multiple_flags(self) -> None:
        el = _el("btn", visible=False, enabled=False)
        result = format_element_compact(el, "@e1")
        assert "hidden" in result
        assert "disabled" in result


# =============================================================================
# truncate_field
# =============================================================================


class TestTruncateField:
    def test_none(self) -> None:
        assert truncate_field(None, 50) is None

    def test_empty(self) -> None:
        assert truncate_field("", 50) == ""

    def test_short(self) -> None:
        assert truncate_field("hello", 50) == "hello"

    def test_exact(self) -> None:
        assert truncate_field("12345", 5) == "12345"

    def test_truncated(self) -> None:
        result = truncate_field("abcdefghij", 5)
        assert result is not None
        assert result.startswith("abcde...")
        assert "[10 chars total]" in result

    def test_long_string(self) -> None:
        long = "x" * 1000
        result = truncate_field(long, 100)
        assert result is not None
        assert len(result) < 200  # truncated + metadata
        assert "[1000 chars total]" in result


# =============================================================================
# sanitize_element_content
# =============================================================================


class TestSanitizeElementContent:
    def test_wraps_text_content(self) -> None:
        el = {"state": {"textContent": "Hello World"}}
        sanitize_element_content(el)
        assert el["state"]["textContent"] == f"{CONTENT_START}Hello World{CONTENT_END}"

    def test_wraps_inner_html(self) -> None:
        el = {"state": {"innerHTML": "<b>Bold</b>"}}
        sanitize_element_content(el)
        assert el["state"]["innerHTML"] == f"{CONTENT_START}<b>Bold</b>{CONTENT_END}"

    def test_wraps_value(self) -> None:
        el = {"state": {"value": "user@example.com"}}
        sanitize_element_content(el)
        assert el["state"]["value"] == f"{CONTENT_START}user@example.com{CONTENT_END}"

    def test_skips_empty_fields(self) -> None:
        el = {"state": {"textContent": "", "value": None}}
        sanitize_element_content(el)
        assert el["state"]["textContent"] == ""
        assert el["state"]["value"] is None

    def test_skips_missing_fields(self) -> None:
        el = {"state": {"visible": True}}
        sanitize_element_content(el)
        assert "textContent" not in el["state"]

    def test_no_state_key(self) -> None:
        el = {"id": "test"}
        sanitize_element_content(el)  # Should not raise

    def test_multiple_fields(self) -> None:
        el = {
            "state": {
                "textContent": "text",
                "innerHTML": "html",
                "value": "val",
            }
        }
        sanitize_element_content(el)
        assert el["state"]["textContent"] == f"{CONTENT_START}text{CONTENT_END}"
        assert el["state"]["innerHTML"] == f"{CONTENT_START}html{CONTENT_END}"
        assert el["state"]["value"] == f"{CONTENT_START}val{CONTENT_END}"


# =============================================================================
# _format_diff
# =============================================================================


class TestFormatDiff:
    def test_no_changes(self) -> None:
        diff: dict[str, list[Any]] = {"appeared": [], "disappeared": [], "modified": []}
        result = _format_diff(diff, RefManager())
        assert result == "No changes detected."

    def test_appeared(self) -> None:
        rm = RefManager()
        rm.assign("new-btn")
        diff = {"appeared": ["new-btn"], "disappeared": [], "modified": []}
        result = _format_diff(diff, rm)
        assert "Appeared (1)" in result
        assert "@e1 (new-btn)" in result

    def test_disappeared(self) -> None:
        rm = RefManager()
        rm.assign("old-btn")
        diff = {"appeared": [], "disappeared": ["old-btn"], "modified": []}
        result = _format_diff(diff, rm)
        assert "Disappeared (1)" in result
        assert "@e1 (old-btn)" in result

    def test_disappeared_without_ref(self) -> None:
        rm = RefManager()
        diff = {"appeared": [], "disappeared": ["unknown-id"], "modified": []}
        result = _format_diff(diff, rm)
        assert "Disappeared (1)" in result
        assert "unknown-id" in result
        assert "@e" not in result

    def test_modified(self) -> None:
        rm = RefManager()
        rm.assign("input-email")
        diff = {
            "appeared": [],
            "disappeared": [],
            "modified": [
                {
                    "id": "input-email",
                    "changes": {"value": {"from": "", "to": "test@test.com"}},
                }
            ],
        }
        result = _format_diff(diff, rm)
        assert "Modified (1)" in result
        assert "@e1 (input-email)" in result
        assert "value" in result
        assert "'test@test.com'" in result

    def test_complex_diff(self) -> None:
        rm = RefManager()
        rm.assign("a")
        rm.assign("b")
        diff = {
            "appeared": ["c"],
            "disappeared": ["a"],
            "modified": [
                {
                    "id": "b",
                    "changes": {
                        "enabled": {"from": False, "to": True},
                        "visible": {"from": False, "to": True},
                    },
                }
            ],
        }
        result = _format_diff(diff, rm)
        assert "Appeared (1)" in result
        assert "Disappeared (1)" in result
        assert "Modified (1)" in result
        assert "@e1 (a)" in result  # disappeared
        assert "@e2 (b)" in result  # modified
