"""The element formatters: an absent or null ``enabled`` field is not read as enabled.

Visibility alone is covered in test_observation_unknown.py; these cases also
pin both fields together and an element with no ``state``.
"""

from typing import Any

from ui_bridge_mcp.server import format_element_compact, format_element_summary


def _el(**state: Any) -> dict[str, Any]:
    return {"id": "b1", "type": "button", "state": state}


def test_stated_visible_and_enabled_carries_no_status() -> None:
    out = format_element_summary(_el(visible=True, enabled=True))
    assert "unknown" not in out
    assert "hidden" not in out
    assert "disabled" not in out


def test_one_absent_field_is_reported_alone() -> None:
    out = format_element_summary(_el(visible=True))
    assert "[enabled unknown]" in out


def test_stated_hidden_and_disabled() -> None:
    out = format_element_summary(_el(visible=False, enabled=False))
    assert "[hidden, disabled]" in out


def test_absent_fields_are_unknown_not_visible_or_enabled() -> None:
    out = format_element_summary(_el())
    assert "[visibility unknown, enabled unknown]" in out


def test_null_fields_are_unknown() -> None:
    out = format_element_summary(_el(visible=None, enabled=None))
    assert "[visibility unknown, enabled unknown]" in out


def test_element_without_state_is_unknown() -> None:
    out = format_element_summary({"id": "b1", "type": "button"})
    assert "[visibility unknown, enabled unknown]" in out


def test_compact_form_agrees() -> None:
    assert "visibility-unknown enabled-unknown" in format_element_compact(_el(), "@e1")
    stated = format_element_compact(_el(visible=False, enabled=False), "@e1")
    assert "hidden disabled" in stated
    assert "unknown" not in format_element_compact(
        _el(visible=True, enabled=True), "@e1"
    )
