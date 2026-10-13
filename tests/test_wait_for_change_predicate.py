"""wait_for_change sends a ``ChangePredicate`` the SDK reads.

ui-bridge ``ai/types.ts`` ``ChangePredicate`` has one optional field per
condition, and ``ai/change-tracker.ts`` matchesPredicate reads only those
fields. The tools used to send ``{"type": "anyChange", "elementId": ...}``,
which matches no field, so nothing ever matched and every wait answered by the
SDK or by the runner's own UI timed out. These tests pin the request body
against that shape.
"""

from __future__ import annotations

from typing import Any

import pytest

import ui_bridge_mcp.server as server_mod

from .mcp_session import answering, call, text

WAIT_TOOLS = [
    ("sdk_wait_for_change", "/ui-bridge/sdk/ai/wait-for-change"),
    ("ui_wait_for_change", "/ui-bridge/control/ai/wait-for-change"),
]

# ui-bridge ai/types.ts (d6ff3940): every ChangePredicate field, and the keys of
# each object-valued one. None marks a scalar field.
CHANGE_PREDICATE_SHAPE: dict[str, frozenset[str] | None] = {
    "elementAppeared": frozenset({"text", "type"}),  # or a string
    "elementDisappeared": None,
    "propertyChanged": frozenset({"elementId", "property", "expectedValue"}),
    "textContains": frozenset({"elementId", "text"}),
    "category": None,
    "anySignificantChange": None,
    "elementCount": frozenset({"min", "type", "text"}),
    "urlChanged": None,
    "urlContains": None,
    "formValid": frozenset({"formId"}),
    "statusChanged": frozenset({"elementId", "direction", "newStatus"}),
}
# ai/types.ts WaitForChangeOptions.
WAIT_FOR_CHANGE_OPTIONS = frozenset({"timeout", "interval", "scope"})

DIFF = {"summary": "s", "changes": {"appeared": [], "disappeared": [], "modified": []}}


def assert_change_predicate_shape(predicate: dict[str, Any]) -> None:
    """Every key is a ChangePredicate field, and every nested key one of its keys."""
    assert predicate, "an empty predicate matches nothing"
    for key, value in predicate.items():
        assert key in CHANGE_PREDICATE_SHAPE, key
        nested = CHANGE_PREDICATE_SHAPE[key]
        if isinstance(value, dict):
            assert nested is not None, key
            assert set(value) <= nested, (key, value)


@pytest.mark.parametrize(
    ("arguments", "predicate"),
    [
        ({"predicate_type": "anyChange"}, {"anySignificantChange": True}),
        (
            {"predicate_type": "elementAppeared", "element_id": "toast"},
            {"elementAppeared": "toast"},
        ),
        (
            {"predicate_type": "elementDisappeared", "element_id": "spinner"},
            {"elementDisappeared": "spinner"},
        ),
        (
            {
                "predicate_type": "elementChanged",
                "element_id": "save",
                "property": "enabled",
            },
            {"propertyChanged": {"elementId": "save", "property": "enabled"}},
        ),
        ({"predicate_type": "countChanged"}, {"elementCount": {"min": 1}}),
        (
            {"predicate_type": "countChanged", "min_count": 3},
            {"elementCount": {"min": 3}},
        ),
    ],
    ids=[
        "anyChange",
        "elementAppeared",
        "elementDisappeared",
        "elementChanged",
        "countChanged-default",
        "countChanged",
    ],
)
@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_predicate_is_a_change_predicate(
    arguments: dict[str, Any],
    predicate: dict[str, Any],
    tool: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = answering(monkeypatch, {path: {"success": True, "data": DIFF}})
    result = call(tool, arguments)
    assert result.isError is False
    [body] = runner.bodies(path)
    assert body == {"predicate": predicate, "options": {"timeout": 5000}}
    assert_change_predicate_shape(body["predicate"])
    assert set(body["options"]) <= WAIT_FOR_CHANGE_OPTIONS


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_options_are_wait_for_change_options(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK reads ``interval``; the ``pollInterval`` sent before reached nothing."""
    runner = answering(monkeypatch, {path: {"success": True, "data": DIFF}})
    call(tool, {"predicate_type": "anyChange", "timeout": 60000, "poll_interval": 50})
    assert runner.bodies(path) == [
        {
            "predicate": {"anySignificantChange": True},
            "options": {"timeout": 60000, "interval": 50},
        }
    ]
    # The HTTP request outlasts the wait instead of cutting it off at 30 s.
    [read_timeout] = runner.read_timeouts
    assert read_timeout is not None and read_timeout >= 70.0


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_element_ref_is_resolved(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = server_mod.RefManager()
    ref = refs.assign("dialog-1")
    monkeypatch.setattr(server_mod, "ref_manager", refs)
    runner = answering(monkeypatch, {path: {"success": True, "data": DIFF}})
    call(tool, {"predicate_type": "elementDisappeared", "element_id": ref})
    assert runner.bodies(path)[0]["predicate"] == {"elementDisappeared": "dialog-1"}


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        (
            {"predicate_type": "anyChange", "element_id": "save"},
            "predicate_type anyChange does not take element_id",
        ),
        (
            {"predicate_type": "elementAppeared", "property": "enabled"},
            "predicate_type elementAppeared does not take property",
        ),
        (
            {"predicate_type": "elementAppeared"},
            "predicate_type elementAppeared needs element_id",
        ),
        (
            {"predicate_type": "elementChanged", "element_id": "save"},
            "predicate_type elementChanged needs property",
        ),
        (
            {"predicate_type": "countChanged", "element_id": "row"},
            "predicate_type countChanged does not take element_id",
        ),
    ],
    ids=["stray", "stray-property", "missing-id", "missing-property", "count-stray"],
)
@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_arguments_a_predicate_cannot_express_are_refused_unsent(
    arguments: dict[str, Any],
    reason: str,
    tool: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping an argument would wait on a different condition than was asked."""
    runner = answering(monkeypatch, {path: {"success": True, "data": DIFF}})
    result = call(tool, arguments)
    assert result.isError is True
    assert text(result) == f"Error: {reason}"
    assert runner.requests == []


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_min_count_below_one_is_refused_unsent(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``elementCount.min: 0`` would match before anything changed."""
    runner = answering(monkeypatch, {path: {"success": True, "data": DIFF}})
    result = call(tool, {"predicate_type": "countChanged", "min_count": 0})
    assert result.isError is True
    assert runner.requests == []
