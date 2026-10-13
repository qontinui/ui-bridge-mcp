"""A failure inside a success envelope reaches the client flagged ``isError``.

Some producers answer a failure with ``{success: true, data: {...}}``: the
request worked, but what it asked for did not happen. Each tool here reads the
producer's own verdict field from ``data``. Each response below is the shape a
named producer sends (ui-bridge at d6ff3940, qontinui-runner at e4801677), so
the tests fail if a tool goes back to reading a field no producer sends.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ui_bridge_mcp.client import AI_ASSERT_STATES

from .mcp_session import answering, call, text


def ok(data: Any) -> dict[str, Any]:
    """A success envelope carrying ``data``."""
    return {"success": True, "data": data}


# =============================================================================
# sdk_ai_assert: a failed assertion is passed:false inside a success envelope
# =============================================================================

ASSERT = "/ui-bridge/sdk/ai/assert"

# ui-bridge ai/assertions.ts assert(): the not-found result, as the SDK sends it.
FAILED_ASSERTION = {
    "passed": False,
    "target": "Save button",
    "targetDescription": "Save button",
    "expected": True,
    "actual": "element not found",
    "failureReason": "Element could not be found",
    "failureCode": "UB-ASSERT-NOT-FOUND",
    "suggestion": "Check if the element exists and is properly labeled",
    "durationMs": 2,
    "timestamp": 1,
}


def test_failed_assertion_carries_reason_and_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {ASSERT: ok(FAILED_ASSERTION)})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is True
    assert text(result) == (
        "Assertion failed: [code=UB-ASSERT-NOT-FOUND] expected 'Save button' is visible\n"
        "  Reason: Element could not be found\n"
        "  Actual: element not found\n"
        "  Suggestion: Check if the element exists and is properly labeled"
    )


def test_passed_assertion_states_the_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {ASSERT: ok({"passed": True, "actual": True})})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "hidden"})
    assert result.isError is False
    assert text(result) == "Assertion passed: 'Save button' is hidden"


@pytest.mark.parametrize("passed", [None, "true", 1], ids=["absent", "string", "int"])
def test_unreadable_assertion_verdict_is_unknown_not_a_pass(
    passed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    data: dict[str, Any] = {"target": "Save button"}
    if passed is not None:
        data["passed"] = passed
    answering(monkeypatch, {ASSERT: ok(data)})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is True
    assert text(result).startswith(
        "Assertion result UNKNOWN: expected 'Save button' is visible"
    )
    assert "This is not a pass." in text(result)


def test_assertion_that_could_not_run_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = {"success": False, "error": "boom", "code": "AI_ASSERT_ERROR"}
    answering(monkeypatch, {ASSERT: failed})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is True
    assert text(result).startswith("Assertion could not be evaluated: [code=")


@pytest.mark.parametrize("state", list(AI_ASSERT_STATES))
def test_assertion_request_asks_what_the_caller_asked(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``target``/``type`` reach an SDK app verbatim; ``assertion`` is the sentence.

    The tool used to send ``{text, state}``, neither of which the SDK reads, so
    every call checked ``{target: '', type: 'exists'}``.
    """
    runner = answering(monkeypatch, {ASSERT: ok({"passed": True})})
    call("sdk_ai_assert", {"text": "Save button", "state": state})
    sentence = AI_ASSERT_STATES[state].sentence.format(target="Save button")
    assert runner.bodies(ASSERT) == [
        {"target": "Save button", "type": state, "assertion": sentence}
    ]


def test_assertion_state_defaults_to_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = answering(monkeypatch, {ASSERT: ok({"passed": True})})
    result = call("sdk_ai_assert", {"text": "Save button"})
    assert text(result) == "Assertion passed: 'Save button' exists"
    assert runner.bodies(ASSERT) == [
        {"target": "Save button", "type": "exists", "assertion": "Save button exists"}
    ]


def test_assertion_sentences_are_the_parser_checked_ones() -> None:
    """Pinned: each sentence was run through the SDK parser (see AI_ASSERT_STATES).

    Changing one needs that check again. "{T} does not exist" in particular
    parses to the whole sentence as the target, so notExists would pass
    vacuously.
    """
    assert {state: s.sentence for state, s in AI_ASSERT_STATES.items()} == {
        "visible": "{target} is visible",
        "hidden": "{target} is hidden",
        "enabled": "{target} is enabled",
        "disabled": "{target} is disabled",
        "focused": "{target} is focused",
        "checked": "{target} is checked",
        "unchecked": "{target} is unchecked",
        "exists": "{target} exists",
        "notExists": "there is no {target} on the page",
    }


def test_a_state_the_sdk_cannot_check_is_refused_unsent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hasText needs an expected value this tool does not take: refuse, never send."""
    runner = answering(monkeypatch, {ASSERT: ok({"passed": True})})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "hasText"})
    assert result.isError is True
    assert "hasText" in text(result)
    assert runner.requests == []


# =============================================================================
# sdk_ai_execute: a failed action is success:false inside a success envelope
# =============================================================================

EXECUTE = "/ui-bridge/sdk/ai/execute"

# ui-bridge ai/types.ts NLActionResponse, as server/handlers.ts aiExecute
# returns it for an action that did not happen.
FAILED_ACTION = {
    "success": False,
    "executedAction": "",
    "confidence": 0,
    "durationMs": 4,
    "timestamp": 1,
    "error": "No element matched 'Save button'",
    "errorCode": "ELEMENT_NOT_FOUND",
    "suggestions": ["Take a snapshot", "Use an element id"],
}


def test_failed_action_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {EXECUTE: ok(FAILED_ACTION)})
    result = call("sdk_ai_execute", {"instruction": "click the Save button"})
    assert result.isError is True
    assert text(result) == (
        "Action failed: [code=ELEMENT_NOT_FOUND] click the Save button\n"
        "  Error: No element matched 'Save button'\n"
        "  Suggestions: Take a snapshot; Use an element id"
    )


def test_executed_action_is_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    executed = {"success": True, "executedAction": "click on save", "confidence": 0.9}
    answering(monkeypatch, {EXECUTE: ok(executed)})
    result = call("sdk_ai_execute", {"instruction": "click the Save button"})
    assert result.isError is False
    assert text(result) == "Executed: click the Save button"


def test_action_without_a_verdict_is_unknown_not_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {EXECUTE: ok({"executedAction": "click on save"})})
    result = call("sdk_ai_execute", {"instruction": "click the Save button"})
    assert result.isError is True
    assert text(result).startswith("Action outcome UNKNOWN for 'click the Save button'")


# =============================================================================
# Undo / redo: executed:false over HTTP, success:false over the relay
# =============================================================================

UNDO_TOOLS = [
    ("sdk_undo", "/ui-bridge/sdk/undo", "Undo", "undone"),
    ("sdk_redo", "/ui-bridge/sdk/redo", "Redo", "redone"),
    ("ui_undo", "/ui-bridge/control/undo", "Undo", "undone"),
    ("ui_redo", "/ui-bridge/control/redo", "Redo", "redone"),
]


@pytest.mark.parametrize(("tool", "path", "verb", "past"), UNDO_TOOLS)
def test_nothing_to_undo_over_http_is_an_error(
    tool: str, path: str, verb: str, past: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ui-bridge server/handlers.ts executeUndo: ``success({executed: false})``."""
    answering(monkeypatch, {path: ok({"executed": False})})
    result = call(tool, {})
    assert result.isError is True
    assert text(result) == f"{verb} was not available: nothing was {past}."


@pytest.mark.parametrize(("tool", "path", "verb", "past"), UNDO_TOOLS)
def test_nothing_to_undo_over_the_relay_is_an_error(
    tool: str, path: str, verb: str, past: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's inner ``success: false`` becomes the envelope's (relay-handlers.ts)."""
    relayed = {
        "success": False,
        "error": "inner success:false",
        "data": {"success": False, "executed": False, "method": "tracker"},
    }
    answering(monkeypatch, {path: relayed})
    result = call(tool, {})
    assert result.isError is True
    assert text(result).startswith("Error: [code=")


@pytest.mark.parametrize(
    "data",
    [
        {"executed": True},
        # The runner's own UI: {success, method} and no `executed`
        # (runner useDebugInspectEvents.ts, case "undo").
        {"success": True, "method": "undoTracker"},
        {"success": True, "executed": True, "method": "tracker"},
    ],
    ids=["sdk-http", "runner-own-ui", "relay"],
)
@pytest.mark.parametrize(("tool", "path", "verb", "past"), UNDO_TOOLS)
def test_executed_undo_is_an_answer(
    data: dict[str, Any],
    tool: str,
    path: str,
    verb: str,
    past: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {path: ok(data)})
    result = call(tool, {})
    assert result.isError is False
    assert text(result) == f"{verb} executed."


@pytest.mark.parametrize(("tool", "path", "verb", "past"), UNDO_TOOLS)
def test_undo_without_a_verdict_is_unknown(
    tool: str, path: str, verb: str, past: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok({"method": "keyboard"})})
    result = call(tool, {})
    assert result.isError is True
    assert text(result).startswith(f"{verb} outcome UNKNOWN")


# =============================================================================
# wait_for_change: a success is the change; only a timeout is an error
# =============================================================================

WAIT_TOOLS = [
    ("sdk_wait_for_change", "/ui-bridge/sdk/ai/wait-for-change"),
    ("ui_wait_for_change", "/ui-bridge/control/ai/wait-for-change"),
]

# ui-bridge ai/change-tracker.ts waitForChange resolves the matching SemanticDiff.
SEMANTIC_DIFF = {
    "summary": "1 element appeared",
    "fromSnapshotId": "s1",
    "toSnapshotId": "s2",
    "changes": {
        "appeared": [{"elementId": "toast"}],
        "disappeared": [],
        "modified": [],
    },
    "durationMs": 3,
    "timestamp": 1,
}


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_detected_change_as_a_semantic_diff_is_an_answer(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK and the runner's own UI answer a match with the diff itself."""
    answering(monkeypatch, {path: ok(SEMANTIC_DIFF)})
    result = call(tool, {"predicate_type": "anyChange"})
    assert result.isError is False
    assert text(result) == (
        "Change detected! Appeared: 1, Disappeared: 0, Modified: 0\n1 element appeared"
    )


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_detected_change_from_the_relay_is_an_answer(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """react/commandHandlers.ts waitForChange resolves {changed: true} on any mutation."""
    answering(monkeypatch, {path: ok({"changed": True, "timestamp": 1})})
    result = call(tool, {"predicate_type": "anyChange"})
    assert result.isError is False
    assert text(result).startswith("Change detected: the page's DOM changed.")
    assert "did not evaluate the predicate" in text(result)


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_success_without_a_diff_is_still_the_change(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok(None)})
    result = call(tool, {"predicate_type": "anyChange"})
    assert result.isError is False
    assert text(result) == "Change detected. The response carried no diff."


@pytest.mark.parametrize(
    "data",
    [{"changed": False, "timedOut": True, "timestamp": 1}, {"timedOut": True}],
    ids=["relay", "timedOut-only"],
)
@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_change_wait_timeout_is_an_error(
    data: dict[str, Any], tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok(data)})
    result = call(tool, {"predicate_type": "anyChange"})
    assert result.isError is True
    assert text(result).startswith("Timed out waiting for change")


@pytest.mark.parametrize(("tool", "path"), WAIT_TOOLS)
def test_change_wait_timeout_thrown_by_the_sdk_is_an_error(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ChangeTracker throws on timeout; the SDK answers success:false."""
    failed = {
        "success": False,
        "error": "waitForChange timed out after 5000ms. Changes detected: 0 appeared",
        "code": "WAIT_FOR_CHANGE_ERROR",
    }
    answering(monkeypatch, {path: failed})
    result = call(tool, {"predicate_type": "anyChange"})
    assert result.isError is True
    assert "waitForChange timed out" in text(result)


# =============================================================================
# wait_for_network_request: timedOut:true is an error; the SDK sends `entry`
# =============================================================================

NETWORK_TOOLS = [
    ("sdk_wait_for_network_request", "/ui-bridge/sdk/network-requests/wait"),
    ("ui_wait_for_network_request", "/ui-bridge/control/network-requests/wait"),
]

# ui-bridge network/types.ts TrackedNetworkRequest and NetworkRequestEntry.
REQUEST: dict[str, Any] = {
    "id": "r1",
    "method": "GET",
    "url": "/api/x",
    "startedAt": 1,
    "status": "completed",
}
ENTRY: dict[str, Any] = {
    "request": REQUEST,
    "response": {"statusCode": 200, "statusText": "OK", "durationMs": 12},
    "isFailure": False,
    "completedAt": 2,
}


@pytest.mark.parametrize(("tool", "path"), NETWORK_TOOLS)
def test_network_wait_timeout_from_the_sdk_is_an_error(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """network/tracker.ts: a match still in flight at the timeout -> timedOut:true."""
    in_flight = {"request": {**REQUEST, "status": "in-flight"}}
    answering(monkeypatch, {path: ok({"entry": in_flight, "timedOut": True})})
    result = call(tool, {"url_pattern": "/api/x"})
    assert result.isError is True
    assert text(result) == (
        "Wait timed out: no matching request completed in time.\n"
        "  Still in flight when the wait ended: GET /api/x"
    )


@pytest.mark.parametrize(("tool", "path"), NETWORK_TOOLS)
def test_network_wait_timeout_from_the_relay_is_an_error(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok({"timedOut": True, "timestamp": 1})})
    result = call(tool, {"url_pattern": "/api/x"})
    assert result.isError is True
    assert text(result) == "Wait timed out: no matching request completed in time."


@pytest.mark.parametrize(("tool", "path"), NETWORK_TOOLS)
def test_network_wait_match_from_the_sdk_names_the_request(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK names the match ``entry``; the old formatter read ``request``."""
    answering(monkeypatch, {path: ok({"entry": ENTRY, "timedOut": False})})
    result = call(tool, {"url_pattern": "/api/x"})
    assert result.isError is False
    assert text(result) == (
        "Matched request: [COMPLETED] GET /api/x → 200 (12ms)\n  Request ID: r1"
    )


@pytest.mark.parametrize(("tool", "path"), NETWORK_TOOLS)
def test_network_wait_match_from_the_relay_names_the_request(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's request carries the HTTP status as a number."""
    relayed = {
        "id": "n7",
        "url": "/api/x",
        "method": "GET",
        "status": 201,
        "statusText": "Created",
        "startTime": 1,
        "duration": 5,
        "inFlight": False,
    }
    answering(monkeypatch, {path: ok({"request": relayed, "timestamp": 1})})
    result = call(tool, {"url_pattern": "/api/x"})
    assert result.isError is False
    assert text(result) == (
        "Matched request: GET /api/x → 201 Created (5ms)\n  Request ID: n7"
    )


@pytest.mark.parametrize(
    ("arguments", "body"),
    [
        (
            {"url_pattern": "/api/x", "method": "GET"},
            {
                "timeout": 30000,
                "urlPattern": "/api/x",
                "url": "/api/x",
                "method": "GET",
            },
        ),
        # The SDK route requires a body, and the readers' default timeouts differ.
        ({}, {"timeout": 30000}),
        ({"timeout": 500}, {"timeout": 500}),
    ],
    ids=["pattern-and-method", "no-arguments", "timeout"],
)
@pytest.mark.parametrize(("tool", "path"), NETWORK_TOOLS)
def test_network_wait_sends_the_fields_each_reader_reads(
    arguments: dict[str, Any],
    body: dict[str, Any],
    tool: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK reads ``urlPattern``, the relay ``url``; ``url_pattern`` reached neither."""
    runner = answering(monkeypatch, {path: ok({"entry": ENTRY, "timedOut": False})})
    call(tool, arguments)
    assert runner.bodies(path) == [body]


def test_network_wait_request_failure_is_still_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No match at all makes the SDK tracker throw: success:false."""
    failed = httpx.Response(
        500, json={"success": False, "error": "waitForRequest timed out after 30000ms"}
    )
    answering(monkeypatch, {"/ui-bridge/sdk/network-requests/wait": failed})
    result = call("sdk_wait_for_network_request", {"url_pattern": "/api/x"})
    assert result.isError is True
    assert text(result).startswith("Network request wait failed:")
