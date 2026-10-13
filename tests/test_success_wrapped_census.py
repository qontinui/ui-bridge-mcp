"""The rest of the census: every other failure a producer wraps in success.

A read of every tool against its producers (ui-bridge d6ff3940, qontinui-runner
e4801677) found more tools that reported a failure, or an unknown, as an
answer. Each test feeds the shape a named producer sends and reads the result
over an in-memory MCP client session.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import pytest

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient

from .mcp_session import FakeRunner, answering, call, text


def ok(data: Any) -> dict[str, Any]:
    """A success envelope carrying ``data``."""
    return {"success": True, "data": data}


SDK_CONNECTED = {"/ui-bridge/sdk/status": ok({"connected": True})}
SDK_ABSENT = {"/ui-bridge/sdk/status": ok({"connected": False})}

# =============================================================================
# A nested success:false is lifted by the client, for every tool at once
# =============================================================================


def test_client_lifts_a_nested_failure_with_its_code() -> None:
    """ui-bridge's relay reads an inner ``success: false`` as the failure; so does this."""
    runner = FakeRunner(
        {
            "/ui-bridge/sdk/page/navigate": ok(
                {"success": False, "error": "Navigation failed", "errorCode": "NAV"}
            )
        }
    )
    client = UIBridgeClient(host="127.0.0.1", port=1)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(runner))

    async def go() -> Any:
        try:
            return await client.sdk_page_navigate("http://x")
        finally:
            await client.close()

    response = asyncio.run(go())
    assert response.success is False
    assert response.code == "NAV"
    assert response.error == "Navigation failed"
    assert response.data == {
        "success": False,
        "error": "Navigation failed",
        "errorCode": "NAV",
    }


@pytest.mark.parametrize(
    ("tool", "arguments", "path"),
    [
        # The runner's own-UI fallback when no SDK app is connected.
        ("sdk_page_navigate", {"url": "http://x"}, "/ui-bridge/sdk/page/navigate"),
        # Runner routes that wrap their own UI's answer in {success: true}.
        (
            "ui_query_selector",
            {"selector": "a"},
            "/ui-bridge/sdk/control/query-selector",
        ),
        ("ui_discover", {}, "/ui-bridge/control/discover"),
    ],
)
def test_nested_failure_is_an_error(
    tool: str, arguments: dict[str, Any], path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = {"success": False, "error": "it did not happen", "code": "NOPE"}
    answering(monkeypatch, {path: ok(nested)})
    result = call(tool, arguments)
    assert result.isError is True
    assert "[code=NOPE status=200] it did not happen" in text(result)


EVALUATE = "/ui-bridge/sdk/control/page-evaluate"


def test_page_evaluate_is_not_lifted_by_the_client() -> None:
    """Its data is foreign: the lift is off for this one route."""
    runner = FakeRunner({EVALUATE: ok({"success": False, "error": "page data"})})
    client = UIBridgeClient(host="127.0.0.1", port=1)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(runner))

    async def go() -> Any:
        try:
            return await client.control_page_evaluate("window.result")
        finally:
            await client.close()

    response = asyncio.run(go())
    assert response.success is True
    assert response.data == {"success": False, "error": "page data"}


def test_page_evaluate_bare_value_with_its_own_success_false_is_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner built before 0dbc0bace answers the evaluated value as bare `data`.

    That value can carry its own ``success: false``; it is the page's data, not
    a verdict, so it stays an answer, said to be UNKNOWN.
    """
    value = {"success": False, "error": "from the page", "items": 2}
    answering(monkeypatch, {EVALUATE: ok(value)})
    result = call("ui_page_evaluate", {"expression": "window.result"})
    assert result.isError is False
    assert '"error": "from the page"' in text(result)
    assert "Note: whether this evaluation succeeded is UNKNOWN." in text(result)


def test_page_evaluate_value_and_type_shape_is_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Since 0dbc0bace the value arrives as {value, type} (runner utils.ts)."""
    shaped = {"value": {"success": False, "error": "from the page"}, "type": "object"}
    answering(monkeypatch, {EVALUATE: ok(shaped)})
    result = call("ui_page_evaluate", {"expression": "window.result"})
    assert result.isError is False
    assert '"type": "object"' in text(result)
    assert "UNKNOWN" not in text(result)


def test_nested_success_is_still_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(
        monkeypatch,
        {"/ui-bridge/sdk/page/navigate": ok({"success": True, "url": "http://x"})},
    )
    result = call("sdk_page_navigate", {"url": "http://x"})
    assert result.isError is False
    assert text(result) == "Navigated to: http://x"


# =============================================================================
# ui_snapshot / ui_diff: the runner's window-capture fallback has no elements
# =============================================================================

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
NATIVE_CAPTURE = ok(
    {
        "source": "native_capture",
        "reason": "UI Bridge frontend not ready",
        "screenshot": PNG,
        "width": 10,
        "height": 10,
        "elements": [],
        "note": "SDK was not connected.",
    }
)


@pytest.mark.parametrize("tool", ["ui_snapshot", "ui_diff"])
def test_window_capture_fallback_is_not_zero_elements(
    tool: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """runner ui_bridge/elements.rs: an empty list here means UNKNOWN, not none."""
    monkeypatch.setattr(server_mod, "control_diff_tracker", server_mod.DiffTracker())
    answering(monkeypatch, {"/ui-bridge/control/snapshot": NATIVE_CAPTURE})
    result = call(tool, {})
    assert result.isError is True
    assert text(result).startswith("No element tree:")
    assert "Reason: UI Bridge frontend not ready" in text(result)
    images = [item for item in result.content if item.type == "image"]
    assert [image.data for image in images] == [PNG]


def test_window_capture_does_not_feed_the_diff_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fed the empty list, the next ui_diff would report every element gone."""
    monkeypatch.setattr(server_mod, "control_diff_tracker", server_mod.DiffTracker())
    answering(monkeypatch, {"/ui-bridge/control/snapshot": NATIVE_CAPTURE})
    call("ui_diff", {})
    answering(
        monkeypatch, {"/ui-bridge/control/snapshot": ok({"elements": [{"id": "a"}]})}
    )
    result = call("ui_diff", {})
    assert result.isError is True
    assert text(result).startswith("No previous snapshot to diff against.")


# =============================================================================
# Bookmarks and diffs
# =============================================================================

BOOKMARK_TOOLS = [
    ("sdk", "/ui-bridge/sdk/ai/bookmark/home"),
    ("ui", "/ui-bridge/control/ai/bookmark/home"),
]


@pytest.mark.parametrize(
    ("data", "is_error", "expect"),
    [
        ({"deleted": True}, False, "Bookmark 'home' deleted."),
        (
            {"deleted": False},
            True,
            "Bookmark 'home' was not deleted: no bookmark has that name.",
        ),
        # The browser relay's shape.
        ({"success": True, "name": "home"}, False, "Bookmark 'home' deleted."),
        ({"name": "home"}, True, "Bookmark 'home': whether it was deleted is UNKNOWN"),
    ],
    ids=["deleted", "not-found", "relay", "unknown"],
)
@pytest.mark.parametrize(("prefix", "path"), BOOKMARK_TOOLS)
def test_bookmark_delete_reads_deleted(
    data: dict[str, Any],
    is_error: bool,
    expect: str,
    prefix: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {path: ok(data)})
    result = call(f"{prefix}_delete_bookmark", {"name": "home"})
    assert result.isError is is_error
    assert text(result).startswith(expect)


DIFF = {
    "summary": "s",
    "changes": {
        "appeared": [{"elementId": "toast", "description": "Saved", "type": "status"}],
        "disappeared": [],
        "modified": [],
    },
}


@pytest.mark.parametrize(("prefix", "path"), BOOKMARK_TOOLS)
def test_bookmark_diff_renders_the_semantic_diff(
    prefix: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {f"{path}/diff": ok(DIFF)})
    result = call(f"{prefix}_diff_from_bookmark", {"name": "home"})
    assert result.isError is False
    assert text(result) == (
        "Diff from bookmark 'home':\n"
        "  Appeared: 1 elements\n"
        "  Disappeared: 0 elements\n"
        "  Modified: 0 elements\n"
        "  New elements:\n"
        "    - status 'Saved'"
    )


@pytest.mark.parametrize(
    ("data", "expect"),
    [
        # The runner's own UI: ChangeTracker.diffFromBookmark -> null.
        (None, "Diff from bookmark 'home': no diff. Bookmark 'home' was not found."),
        # The browser relay sends counts, not a diff.
        (
            {
                "bookmarkTimestamp": 1,
                "currentTimestamp": 2,
                "beforeCount": 4,
                "afterCount": 5,
            },
            "Diff from bookmark 'home': what changed is UNKNOWN.",
        ),
    ],
    ids=["null", "relay-counts"],
)
@pytest.mark.parametrize(("prefix", "path"), BOOKMARK_TOOLS)
def test_bookmark_diff_without_a_diff_is_an_error(
    data: Any, expect: str, prefix: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {f"{path}/diff": ok(data)})
    result = call(f"{prefix}_diff_from_bookmark", {"name": "home"})
    assert result.isError is True
    assert text(result).startswith(expect)


SCOPED = [
    ("sdk_scoped_diff", "/ui-bridge/sdk/ai/scoped-diff"),
    ("ui_scoped_diff", "/ui-bridge/control/ai/scoped-diff"),
]


@pytest.mark.parametrize(("tool", "path"), SCOPED)
def test_scoped_diff_renders_the_semantic_diff(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok(DIFF)})
    result = call(tool, {"scope": "#main", "from_bookmark": "home"})
    assert result.isError is False
    assert text(result).startswith("Scoped diff ('#main'):\n  Appeared: 1 elements")


@pytest.mark.parametrize(
    ("tool", "path", "arguments", "expect"),
    [
        (
            "sdk_scoped_diff",
            "/ui-bridge/sdk/ai/scoped-diff",
            {"scope": "#main"},
            "There is no earlier snapshot to diff against",
        ),
        (
            "ui_scoped_diff",
            "/ui-bridge/control/ai/scoped-diff",
            {"scope": "#main"},
            "The runner's own UI diffs a scope only from a bookmark",
        ),
        (
            "sdk_scoped_diff",
            "/ui-bridge/sdk/ai/scoped-diff",
            {"scope": "#main", "from_bookmark": "gone"},
            "Bookmark 'gone' was not found.",
        ),
    ],
    ids=["sdk-no-baseline", "ui-needs-bookmark", "bookmark-missing"],
)
def test_scoped_diff_with_nothing_to_diff_is_an_error(
    tool: str,
    path: str,
    arguments: dict[str, Any],
    expect: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ui-bridge server/handlers.ts getScopedDiff answers success(null)."""
    answering(monkeypatch, {path: ok(None)})
    result = call(tool, arguments)
    assert result.isError is True
    assert expect in text(result)


def test_scoped_diff_relay_stub_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """react/commandHandlers.ts getScopedDiff: ``{changes: [], scope}``, no diff."""
    answering(
        monkeypatch,
        {"/ui-bridge/sdk/ai/scoped-diff": ok({"changes": [], "scope": "#main"})},
    )
    result = call("sdk_scoped_diff", {"scope": "#main"})
    assert result.isError is True
    assert text(result).startswith("Scoped diff ('#main'): UNKNOWN")


CATEGORIZE = [
    ("sdk_categorize_last_diff", "/ui-bridge/sdk/ai/categorize-last-diff"),
    ("ui_categorize_last_diff", "/ui-bridge/control/ai/categorize-last-diff"),
]


@pytest.mark.parametrize(("tool", "path"), CATEGORIZE)
def test_categorized_diff_is_an_answer(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok({"category": "feedback", "confidence": 0.8})})
    result = call(tool, {})
    assert result.isError is False
    assert text(result) == "Category: feedback (confidence: 80%)"


@pytest.mark.parametrize(("tool", "path"), CATEGORIZE)
def test_category_outside_the_sdk_set_is_unknown(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's stub: ``{category: 'unknown'}``, not a ChangeCategory."""
    answering(monkeypatch, {path: ok({"category": "unknown", "timestamp": 1})})
    result = call(tool, {})
    assert result.isError is True
    assert text(result).startswith("Diff category UNKNOWN")


SUMMARIZE = [
    ("sdk_summarize_diff", "/ui-bridge/sdk/ai/summarize-diff"),
    ("ui_summarize_diff", "/ui-bridge/control/ai/summarize-diff"),
]


@pytest.mark.parametrize(("tool", "path"), SUMMARIZE)
def test_diff_summary_is_an_answer(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok({"summary": "Toast appeared"})})
    result = call(tool, {})
    assert result.isError is False
    assert text(result) == "Toast appeared"


@pytest.mark.parametrize(("tool", "path"), SUMMARIZE)
def test_missing_diff_summary_is_unknown_not_no_changes(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok({"timestamp": 1})})
    result = call(tool, {})
    assert result.isError is True
    assert text(result).startswith("Diff summary UNKNOWN")


def test_relay_summary_stub_is_unknown_not_a_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """react/commandHandlers.ts summarizeDiff always answers this fixed text.

    The SDK answers the same request, with no diff, as NO_DIFF.
    """
    answering(
        monkeypatch,
        {
            "/ui-bridge/sdk/ai/summarize-diff": ok(
                {"summary": "No diff data available", "timestamp": 1}
            )
        },
    )
    result = call("sdk_summarize_diff", {})
    assert result.isError is True
    assert text(result).startswith(
        "Diff summary UNKNOWN: 'No diff data available' is the browser relay's "
        "fixed answer"
    )


STRUCTURED = [
    ("sdk_structured_changes", "/ui-bridge/sdk/ai/structured-changes"),
    ("ui_structured_changes", "/ui-bridge/control/ai/structured-changes"),
]


@pytest.mark.parametrize(("tool", "path"), STRUCTURED)
def test_structured_changes_is_an_answer(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = {
        "hasStructuredData": True,
        "tableChanges": [{"tableId": "t", "addedRows": [1], "removedRows": []}],
        "listChanges": [],
    }
    answering(monkeypatch, {path: ok(analysis)})
    result = call(tool, {})
    assert result.isError is False
    assert text(result) == (
        "Structured change analysis:\n"
        "  Has structured data: True\n"
        "  Table changes: 1\n"
        "    - t: +1 -0 rows"
    )


@pytest.mark.parametrize(("tool", "path"), STRUCTURED)
def test_structured_changes_stub_is_unknown(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's stub: ``{changes: []}``, with no ``hasStructuredData``."""
    answering(monkeypatch, {path: ok({"changes": [], "timestamp": 1})})
    result = call(tool, {})
    assert result.isError is True
    assert text(result).startswith("Structured change analysis UNKNOWN")


EXECUTE_WITH_DIFF = [
    ("sdk_execute_with_diff", "/ui-bridge/sdk/ai/execute-with-diff"),
    ("ui_execute_with_diff", "/ui-bridge/control/ai/execute-with-diff"),
]
EWD_ARGS = {"element_id": "save", "action": "click"}


@pytest.mark.parametrize(("tool", "path"), EXECUTE_WITH_DIFF)
def test_execute_with_diff_success_renders_the_diff(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(
        monkeypatch, {path: ok({"actionSuccess": True, "diff": DIFF, "durationMs": 12})}
    )
    result = call(tool, EWD_ARGS)
    assert result.isError is False
    assert text(result) == (
        "Execute with diff: click on save\n"
        "  Action success: True\n"
        "  Appeared: 1\n"
        "  Disappeared: 0\n"
        "  Modified: 0\n"
        "  Duration: 12ms"
    )


@pytest.mark.parametrize(("tool", "path"), EXECUTE_WITH_DIFF)
def test_execute_with_diff_relay_counts_are_not_a_zero_diff(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """react/commandHandlers.ts executeWithDiff: ``diff: {before, after}``."""
    relayed = {"actionSuccess": True, "diff": {"before": 3, "after": 4, "timestamp": 1}}
    answering(monkeypatch, {path: ok(relayed)})
    result = call(tool, EWD_ARGS)
    assert result.isError is False
    assert "Elements: 3 before, 4 after (this transport sends counts" in text(result)


@pytest.mark.parametrize(("tool", "path"), EXECUTE_WITH_DIFF)
def test_execute_with_diff_without_a_verdict_is_unknown(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner's ``verdict: "indeterminate"`` case: actionSuccess absent."""
    answering(monkeypatch, {path: ok({"diff": DIFF})})
    result = call(tool, EWD_ARGS)
    assert result.isError is True
    assert "Action success: UNKNOWN" in text(result)


# =============================================================================
# Fill form: FillResult, and the relay's filledFields
# =============================================================================

FILL = [
    ("sdk_fill_form", "/ui-bridge/sdk/fill"),
    ("ui_fill_form", "/ui-bridge/control/fill"),
]
FIELDS = {"fields": {"email": "a@b.c", "name": "Ann"}}


@pytest.mark.parametrize(("tool", "path"), FILL)
def test_fill_result_reports_the_fields(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """core/types.ts FillResult; the old formatter read `results` and said 0/0."""
    filled = {
        "success": True,
        "filledCount": 2,
        "errorCount": 0,
        "fields": {
            "email": {"success": True},
            # The runner's own UI reports a failed validation as `error` on a
            # field that was set.
            "name": {"success": True, "error": "Too short"},
        },
    }
    answering(monkeypatch, {path: ok(filled)})
    result = call(tool, FIELDS)
    assert result.isError is False
    assert text(result) == (
        "Fill result: 2/2 fields set\nValidation errors:\n  name: Too short"
    )


@pytest.mark.parametrize(("tool", "path"), FILL)
def test_failed_fill_names_the_failed_fields(
    tool: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FillResult success:false arrives inside a success envelope from the SDK."""
    failed = {
        "success": False,
        "filledCount": 1,
        "errorCount": 1,
        "fields": {
            "email": {"success": True},
            "name": {"success": False, "error": "Element not found"},
        },
    }
    answering(monkeypatch, {path: ok(failed)})
    result = call(tool, FIELDS)
    assert result.isError is True
    assert text(result) == (
        "Fill result: 1/2 fields set\nFailed fields:\n  name: Element not found"
    )


def test_relay_fill_names_the_fields_it_did_not_find(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """react/commandHandlers.ts fillForm answers success:true with filledFields."""
    relayed = {"success": True, "filledFields": ["email"], "totalFields": 2}
    answering(monkeypatch, {"/ui-bridge/sdk/fill": ok(relayed)})
    result = call("sdk_fill_form", FIELDS)
    assert result.isError is True
    assert text(result) == "Fill result: 1/2 fields set\n  Not found: name"


def test_relay_fill_of_every_field_is_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relayed = {"success": True, "filledFields": ["email", "name"], "totalFields": 2}
    answering(monkeypatch, {"/ui-bridge/sdk/fill": ok(relayed)})
    result = call("sdk_fill_form", FIELDS)
    assert result.isError is False
    assert text(result) == "Fill result: 2/2 fields set"


def test_fill_without_a_result_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {"/ui-bridge/sdk/fill": ok({"timestamp": 1})})
    result = call("sdk_fill_form", FIELDS)
    assert result.isError is True
    assert text(result).startswith("Fill result UNKNOWN")


# =============================================================================
# Design: the relay's bare {error} and its placeholder state styles
# =============================================================================


@pytest.mark.parametrize(
    ("tool", "arguments", "path"),
    [
        (
            "sdk_design_styles",
            {"element_id": "save"},
            "/ui-bridge/sdk/design/element/save/styles",
        ),
        (
            "sdk_design_state_styles",
            {"element_id": "save"},
            "/ui-bridge/sdk/design/element/save/state-styles",
        ),
        ("sdk_design_responsive", {}, "/ui-bridge/sdk/design/responsive"),
    ],
)
def test_relay_design_error_is_an_error(
    tool: str, arguments: dict[str, Any], path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """react/commandHandlers.ts answers ``{error}`` with no ``success`` field."""
    answering(monkeypatch, {path: ok({"error": "Element save not found"})})
    result = call(tool, arguments)
    assert result.isError is True
    assert text(result) == "Error: Element save not found"


def test_relay_placeholder_state_styles_are_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``{hover: 'Requires interaction'}`` is a placeholder, not a style."""
    placeholder = {
        "id": "save",
        "baseStyles": {},
        "stateStyles": {"hover": "Requires interaction"},
    }
    answering(
        monkeypatch,
        {"/ui-bridge/sdk/design/element/save/state-styles": ok(placeholder)},
    )
    result = call("sdk_design_state_styles", {"element_id": "save"})
    assert result.isError is True
    assert text(result).startswith("State styles for save UNKNOWN")


def test_state_styles_are_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    styles = {
        "stateStyles": [
            {"state": "default", "diffFromDefault": []},
            {
                "state": "hover",
                "diffFromDefault": [
                    {"property": "color", "defaultValue": "red", "stateValue": "blue"}
                ],
            },
        ]
    }
    answering(
        monkeypatch, {"/ui-bridge/sdk/design/element/save/state-styles": ok(styles)}
    )
    result = call("sdk_design_state_styles", {"element_id": "save"})
    assert result.isError is False
    assert "[hover] (1 changes):\n    color: red → blue" in text(result)


# react/commandHandlers.ts runDesignAudit: {issues: [{element, issue, severity,
# fix?}], checkedElements}.
RELAY_AUDIT = ok(
    {
        "issues": [
            {
                "element": "save-btn",
                "issue": "Low contrast: 2.10:1 (#999 on #fff)",
                "severity": "warning",
                "fix": "Increase contrast to at least 4.5:1",
            },
            {"element": "title", "issue": "Text nearly invisible", "severity": "error"},
        ],
        "checkedElements": 3,
    }
)


def test_relay_style_audit_is_unknown_not_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay's own scan, ``{issues, checkedElements}``, ignores the loaded guide.

    Read with defaults it rendered "Passed: 0 | Failed: 0" (coord finding c577a740).
    """
    answering(monkeypatch, {"/ui-bridge/sdk/design/audit": RELAY_AUDIT})
    result = call("sdk_design_audit", {})
    assert result.isError is True
    assert text(result).startswith("Style audit UNKNOWN")
    # Flagged, and the relay's own findings are not dropped.
    assert text(result).endswith(
        "The relay's own contrast and size scan reported 2 issue(s) over 3 "
        "elements:\n"
        "  [warning] save-btn: Low contrast: 2.10:1 (#999 on #fff) (fix: Increase "
        "contrast to at least 4.5:1)\n"
        "  [error] title: Text nearly invisible"
    )


def test_style_audit_report_is_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "guideName": "brand",
        "totalElements": 4,
        "totalRules": 2,
        "passedCount": 3,
        "failedCount": 1,
        "summary": {"errors": [], "warnings": []},
    }
    answering(monkeypatch, {"/ui-bridge/sdk/design/audit": ok(report)})
    result = call("sdk_design_audit", {})
    assert result.isError is False
    assert text(result) == (
        "Style Audit: brand\nElements: 4 | Rules: 2\nPassed: 3 | Failed: 1"
    )


def test_design_review_reports_a_relay_audit_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(
        monkeypatch,
        {
            "/ui-bridge/sdk/design/snapshot": ok({"elements": []}),
            "/ui-bridge/sdk/design/audit": RELAY_AUDIT,
        },
    )
    result = call(
        "sdk_design_review",
        {"include_state_variations": False, "include_quality_evaluation": False},
    )
    assert "Style audit: UNKNOWN" in text(result)
    assert "the relay's own scan reported 2 issue(s)" in text(result)
    assert "0 passed, 0 failed" not in text(result)


def test_cross_app_compare_stub_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """react/commandHandlers.ts crossAppCompare answers a stub with no scores."""
    answering(
        monkeypatch,
        {
            "/ui-bridge/sdk/connect": ok({"connected": True}),
            "/ui-bridge/sdk/ai/snapshot": ok({"elements": []}),
            "/ui-bridge/sdk/ai/analyze/cross-app-compare": ok(
                {"comparison": "Cross-app comparison requires snapshots from both apps"}
            ),
        },
    )
    result = call(
        "sdk_cross_app_compare", {"source_url": "http://a", "target_url": "http://b"}
    )
    assert result.isError is True
    assert text(result).startswith("Cross-app comparison UNKNOWN")


# =============================================================================
# Idle: UNKNOWN is not "Idle: False"; a target not reported is not idle
# =============================================================================


@pytest.mark.parametrize(
    ("routes", "signal_path", "label"),
    [
        (SDK_CONNECTED, "/ui-bridge/sdk/idle-status/network", "SDK app"),
        (SDK_ABSENT, "/ui-bridge/control/idle-status/network", "Runner"),
    ],
    ids=["sdk", "runner"],
)
def test_idle_signal_without_a_verdict_is_unknown(
    routes: dict[str, Any],
    signal_path: str,
    label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay answers nothing for a signal it does not know."""
    answering(monkeypatch, {**routes, signal_path: ok(None)})
    result = call("get_idle_status", {"signal": "network"})
    assert result.isError is True
    assert text(result).startswith(f"Idle status UNKNOWN for {label}, signal network")


def test_composite_idle_is_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    composite = {
        "idle": True,
        "signals": {
            "network": {"idle": True, "stableForMs": 40},
            "dom": {"idle": True},
        },
    }
    answering(
        monkeypatch, {**SDK_CONNECTED, "/ui-bridge/sdk/idle-status": ok(composite)}
    )
    result = call("get_idle_status", {})
    assert result.isError is False
    assert text(result) == (
        "SDK app Idle: True\n\nSignals:\n"
        "  network: idle (stable 40ms)\n"
        "  dom: idle (stable UNKNOWN)"
    )


TARGETS = ["network", {"indicator": ".spinner"}]


@pytest.mark.parametrize(
    ("routes", "path", "label"),
    [
        (SDK_CONNECTED, "/ui-bridge/sdk/wait-for-targets", " (SDK)"),
        (SDK_ABSENT, "/ui-bridge/control/wait-for-targets", ""),
    ],
    ids=["sdk", "runner"],
)
def test_every_target_idle_is_an_answer(
    routes: dict[str, Any], path: str, label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """idle/composite-idle.ts waitFor: a map keyed by target, indicators prefixed."""
    statuses = {"network": {"idle": True}, "indicator:.spinner": {"idle": True}}
    answering(monkeypatch, {**routes, path: ok(statuses)})
    result = call("wait_for_targets", {"targets": TARGETS})
    assert result.isError is False
    assert text(result) == f"All targets idle{label}: network, indicator:.spinner"


@pytest.mark.parametrize(
    ("statuses", "expect"),
    [
        # runner useNetworkIdleEvents.ts records a failed wait under `_error`.
        ({"_error": "Idle timeout after 500ms"}, "Targets wait failed: Idle timeout"),
        # An unknown target is left out of the map.
        ({"network": {"idle": True}}, "UNKNOWN (the response did not report them)"),
        (
            {"network": {"idle": False}, "indicator:.spinner": {"idle": True}},
            "not idle: network",
        ),
    ],
    ids=["error", "unreported", "busy"],
)
def test_targets_not_confirmed_idle_are_an_error(
    statuses: dict[str, Any], expect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(
        monkeypatch, {**SDK_ABSENT, "/ui-bridge/control/wait-for-targets": ok(statuses)}
    )
    result = call("wait_for_targets", {"targets": TARGETS})
    assert result.isError is True
    assert expect in text(result)


# =============================================================================
# Page refresh, undo state, scroll, query selector
# =============================================================================


def test_refresh_that_reloaded_nothing_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner page.rs augment_refresh_response: ``reloaded: false`` and why."""
    no_op = {"reloaded": False, "message": "Refresh is a no-op in the runner"}
    answering(monkeypatch, {"/ui-bridge/sdk/page/refresh": ok(no_op)})
    result = call("sdk_page_refresh", {})
    assert result.isError is True
    assert text(result) == "Page was not refreshed: Refresh is a no-op in the runner"


def test_refresh_is_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {"/ui-bridge/sdk/page/refresh": ok({"refreshed": True})})
    result = call("sdk_page_refresh", {})
    assert result.isError is False


@pytest.mark.parametrize(
    ("tool", "path", "state"),
    [
        # ui-bridge undo/types.ts UndoRedoState.
        (
            "sdk_undo_state",
            "/ui-bridge/sdk/undo-state",
            {"undoAvailable": True, "redoAvailable": False},
        ),
        # runner: SnapshotUndoContext, canUndo/canRedo.
        (
            "ui_undo_state",
            "/ui-bridge/control/undo-state",
            {"canUndo": True, "canRedo": False},
        ),
    ],
)
def test_undo_state_reads_either_name(
    tool: str, path: str, state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, {path: ok(state)})
    result = call(tool, {})
    assert result.isError is False
    assert "  Undo: available" in text(result)
    assert "  Redo: not available" in text(result)


def test_undo_state_without_a_tracker_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner useDebugInspectEvents.ts: ``{note: "UndoTracker not installed"}``."""
    answering(
        monkeypatch,
        {"/ui-bridge/control/undo-state": ok({"note": "UndoTracker not installed"})},
    )
    result = call("ui_undo_state", {})
    assert result.isError is True
    assert text(result).startswith(
        "Undo/redo state UNKNOWN (UndoTracker not installed)"
    )


def test_scroll_that_moved_nothing_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """An answer: scrolling at the limit is a no-op (react/commandHandlers.ts)."""
    scrolled = {
        "success": True,
        "scrollInfo": {"before": {}, "after": {}, "changed": False},
    }
    answering(monkeypatch, {"/ui-bridge/sdk/element/list/action": ok(scrolled)})
    result = call("sdk_scroll", {"element_id": "list"})
    assert result.isError is False
    assert "the scroll position did not change" in text(result)


QUERY = "/ui-bridge/sdk/control/query-selector"


def test_query_selector_sends_the_action_where_it_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner usePageEvents.ts reads ``params.action``; a top-level one never clicked."""
    runner = answering(monkeypatch, {QUERY: ok({"count": 2, "elements": []})})
    result = call("ui_query_selector", {"selector": "a", "action": "click", "index": 1})
    assert result.isError is False
    assert runner.bodies(QUERY) == [
        {"selector": "a", "index": 1, "params": {"action": "click"}}
    ]


def test_click_with_no_element_at_the_index_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {QUERY: ok({"count": 1, "elements": []})})
    result = call("ui_query_selector", {"selector": "a", "action": "click", "index": 3})
    assert result.isError is True
    assert text(result).startswith("Nothing clicked: 'a' matched 1 element(s)")


def test_query_selector_action_other_than_click_is_refused_unsent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner performs only `click`; any other action was silently ignored."""
    runner = answering(monkeypatch, {QUERY: ok({"count": 1, "elements": []})})
    result = call("ui_query_selector", {"selector": "a", "action": "focus"})
    assert result.isError is True
    assert runner.requests == []
