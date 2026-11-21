"""The design renderers read the shapes the SDK actually sends.

Plan 2026-11-19-the-ui-bridge-design-surface-is-blocked-behind-a-stalled-pr-
unpublished-packages-and-a-score, Phase 4.

- ``/design/audit`` (style-guide mode) answers ``summary`` as COUNTS
  (``{errors: int, warnings: int, info: int}``, SDK ``specs/design-request.ts``
  ``StyleGuideDesignAuditReport``). The renderers iterated it as lists, so a page
  with failing rules raised ``TypeError: object of type 'int' has no len()``.
  The failing rows are in ``results[]``, bounded by ``maxResults``.
- ``/design/responsive`` answers ``{snapshots: [...], method,
  mediaQueriesReevaluated: false, ...}``. The renderer wrapped that whole object
  as one viewport and printed it as a single empty snapshot.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from mcp import types

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient

_AUDIT: dict[str, Any] = {
    "mode": "style-guide",
    "guideName": "brand",
    # DesignSelectionReport: totalElements is the SELECTION, checkedElements
    # what was audited (SDK designAuditOf).
    "totalElements": 4,
    "checkedElements": 4,
    "limit": 200,
    "truncated": False,
    "missingElementIds": [],
    "missingElementCount": 0,
    "unreadableElementIds": [],
    "unreadableElementCount": 0,
    "totalRules": 2,
    "passedCount": 1,
    "failedCount": 3,
    "summary": {"errors": 2, "warnings": 1, "info": 0},
    "includePassed": False,
    "maxResults": 2,
    "listableResults": 3,
    "resultsTruncated": True,
    "results": [
        {
            "elementId": "btn-primary",
            "ruleId": "button-color",
            "passed": False,
            "severity": "error",
            "constraintResults": [
                {
                    "passed": False,
                    "constraint": {"property": "color"},
                    "actualValue": "rgb(1, 2, 3)",
                    "expectedValue": "rgb(0, 0, 0)",
                    "message": "color drifts from token",
                },
                {
                    "passed": True,
                    "constraint": {"property": "fontSize"},
                    "actualValue": "14px",
                    "expectedValue": "14px",
                },
            ],
        },
        {
            "elementId": "nav-link",
            "ruleId": "link-size",
            "passed": False,
            "severity": "warning",
            "constraintResults": [
                {
                    "passed": False,
                    "constraint": {"property": "fontSize"},
                    "actualValue": "11px",
                    "expectedValue": "14px",
                    "message": "font too small",
                }
            ],
        },
    ],
    "timestamp": 1,
    "durationMs": 2,
}

_RESPONSIVE: dict[str, Any] = {
    "snapshots": [
        {
            "viewportWidth": 375,
            "viewportLabel": "mobile",
            "elements": [
                {
                    "elementId": "hero",
                    "rect": {"width": 375, "height": 200},
                    "styles": {"display": "block"},
                }
            ],
            "timestamp": 1,
        },
        {
            "viewportWidth": 1280,
            "viewportLabel": "desktop",
            "elements": [
                {
                    "elementId": "hero",
                    "rect": {"width": 1280, "height": 300},
                    "styles": {"display": "flex"},
                },
                {
                    "elementId": "sidebar",
                    "rect": {"width": 240, "height": 800},
                    "styles": {"display": "block"},
                },
            ],
            "timestamp": 2,
        },
    ],
    "method": "document-width-constraint",
    "mediaQueriesReevaluated": False,
    "windowInnerWidth": 1440,
    "requestedViewports": 3,
    "tokenMatchesViewport": "mobile",
    "timedOut": True,
    "framesSkipped": False,
    "timestamp": 3,
}


def _client(monkeypatch: pytest.MonkeyPatch, routes: dict[str, Any]) -> UIBridgeClient:
    """A client whose requests are answered from ``routes`` (path -> data)."""
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in routes:
            return httpx.Response(200, json={"success": True, "data": routes[path]})
        if path == "/ui-bridge/sdk/design/snapshot":
            return httpx.Response(200, json={"success": True, "data": {"elements": []}})
        return httpx.Response(404, json={"success": False, "error": "Not found"})

    c = UIBridgeClient(host="127.0.0.1", port=1)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def _call(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    async def go() -> tuple[str, bool]:
        try:
            result: Any = await server_mod.call_tool(name, arguments)
        finally:
            if server_mod.client is not None:
                await server_mod.client.close()
        is_error = isinstance(result, types.CallToolResult) and bool(result.isError)
        items = result.content if isinstance(result, types.CallToolResult) else result
        return "\n".join(getattr(item, "text", "") for item in items), is_error

    return asyncio.run(go())


AUDIT_ROUTE = "/ui-bridge/sdk/design/audit"
RESPONSIVE_ROUTE = "/ui-bridge/sdk/design/responsive"


def test_design_audit_reads_summary_counts_and_lists_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``summary.errors`` as an int raised ``TypeError``."""
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: _AUDIT})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Passed: 1 | Failed: 3" in text
    assert "errors=2 warnings=1 info=0" in text
    assert (
        "[btn-primary] button-color (error): color drifts from token "
        "(actual rgb(1, 2, 3), expected rgb(0, 0, 0))"
    ) in text
    assert "[nav-link] link-size (warning): font too small" in text
    # The passing constraint of a failing row is not listed.
    assert "14px, expected 14px" not in text
    # The SDK's truncation is disclosed, with the full listable count.
    assert "2 listed of 3 listable" in text
    assert "maxResults=2" in text
    assert "Elements: checked 4 of 4 selected\n" in text


def test_design_audit_discloses_a_selection_the_element_limit_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``totalElements`` is the selection; the limit audited only part of it."""
    report = {
        **_AUDIT,
        "totalElements": 250,
        "checkedElements": 200,
        "limit": 200,
        "truncated": True,
        "missingElementCount": 2,
        "unreadableElementCount": 3,
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert (
        "Elements: checked 200 of 250 selected (the SDK's element limit=200 cut "
        "the selection; the rest were NOT audited); 2 requested ids not "
        "registered; 3 registered elements had no readable node (not in the "
        "selection)"
    ) in text
    # The selection count is never presented as the checked count.
    assert "Elements: 250" not in text

    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call(
        "sdk_design_review",
        {"include_state_variations": False, "include_quality_evaluation": False},
    )
    assert "Elements: checked 200 of 250 selected" in text


def test_design_audit_missing_summary_is_unknown_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {k: v for k, v in _AUDIT.items() if k != "summary"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Failed by severity: unknown" in text
    assert "errors=0" not in text


def test_design_audit_missing_results_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {k: v for k, v in _AUDIT.items() if k != "results"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "Failing rows: UNKNOWN" in text


def test_design_review_audit_block_reads_summary_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: _AUDIT})
    )
    text, is_error = _call(
        "sdk_design_review",
        {"include_state_variations": False, "include_quality_evaluation": False},
    )
    assert not is_error, text
    assert "Style audit: 1 passed, 3 failed" in text
    assert "errors=2 warnings=1 info=0" in text
    assert "[nav-link] link-size (warning): font too small" in text


def test_design_responsive_reads_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {RESPONSIVE_ROUTE: _RESPONSIVE})
    )
    text, is_error = _call("sdk_design_responsive", {})
    assert not is_error, text
    assert "Responsive snapshots (2 of 3 requested viewports)" in text
    assert "method=document-width-constraint" in text
    assert "mediaQueriesReevaluated=False" in text
    assert "@media queries were NOT re-evaluated" in text
    assert "timedOut=True" in text
    assert "framesSkipped=False" in text
    assert "=== mobile (375px) — 1 elements ===" in text
    assert "=== desktop (1280px) — 2 elements ===" in text
    assert "sidebar: 240×800 display=block" in text


def test_design_responsive_unrecognised_shape_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload with no ``snapshots`` list is not one empty viewport."""
    monkeypatch.setattr(
        server_mod,
        "client",
        _client(monkeypatch, {RESPONSIVE_ROUTE: {"viewportWidth": 375}}),
    )
    text, is_error = _call("sdk_design_responsive", {})
    assert is_error
    assert "not a {snapshots: [...]} payload" in text
    assert "1 viewports" not in text


def test_design_review_responsive_block_reads_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {RESPONSIVE_ROUTE: _RESPONSIVE})
    )
    text, _ = _call(
        "sdk_design_review",
        {
            "include_state_variations": False,
            "include_quality_evaluation": False,
            "include_responsive": True,
        },
    )
    assert "Responsive snapshots (2 of 3 requested viewports)" in text
    assert "mobile (375px): 1 elements" in text
    assert "desktop (1280px): 2 elements" in text
    assert "timedOut=True" in text
