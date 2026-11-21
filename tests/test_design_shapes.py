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
        "registered; 3 registered elements had no HTMLElement node (not counted "
        "in the 250 selected)"
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
    text, is_error = _call("sdk_design_audit", {})
    # Flagged like every other unreadable audit; the counts are kept.
    assert is_error
    assert text.startswith("Style Audit: brand\n")
    assert "Passed: 1 | Failed: 3" in text
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


_HEURISTIC_AUDIT: dict[str, Any] = {
    "mode": "heuristic",
    "reason": "No style guide was provided or loaded; ran the built-in heuristics.",
    "totalElements": 5,
    "checkedElements": 5,
    "limit": 200,
    "truncated": False,
    "missingElementIds": [],
    "missingElementCount": 0,
    "unreadableElementIds": [],
    "unreadableElementCount": 0,
    "maxResults": 50,
    "listableResults": 2,
    "resultsTruncated": False,
    "issues": [
        {
            "element": "save-btn",
            "issue": "Low contrast: 2.10:1",
            "severity": "warning",
            "fix": "Increase contrast to at least 4.5:1",
        },
        {"element": "close-btn", "issue": "Touch target 30x30", "severity": "error"},
    ],
    "heuristicCoverage": [
        {"heuristic": "contrast", "status": "measured", "checked": 5, "notMeasured": 0},
        {
            "heuristic": "select-dark-mode",
            "status": "measured",
            "checked": 0,
            "notMeasured": 0,
        },
        {
            "heuristic": "font-size",
            "status": "measured",
            "checked": 5,
            "notMeasured": 0,
        },
        {
            "heuristic": "touch-target",
            "status": "measured",
            "checked": 2,
            "notMeasured": 0,
        },
    ],
    "timestamp": 1,
}


def test_heuristic_audit_with_issues_is_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: every heuristic answer rendered as ``Style audit UNKNOWN``."""
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: _HEURISTIC_AUDIT})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "UNKNOWN" not in text
    assert text.startswith("Heuristic design audit (no style guide")
    assert (
        "Reason: No style guide was provided or loaded; ran the built-in heuristics."
    ) in text
    assert "Elements: checked 5 of 5 selected\n" in text
    assert "Issues (2 listed):" in text
    assert (
        "  [warning] save-btn: Low contrast: 2.10:1 (fix: Increase contrast to at "
        "least 4.5:1)"
    ) in text
    assert "  [error] close-btn: Touch target 30x30" in text
    assert text.endswith("Heuristic coverage: all 4 heuristics measured")
    # Never a style-guide verdict.
    assert "Passed:" not in text


def test_heuristic_audit_clean_lists_coverage_not_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty issue list on a partly measured page says what was not measured."""
    report = {
        **_HEURISTIC_AUDIT,
        "issues": [],
        "listableResults": 0,
        "heuristicCoverage": [
            {
                "heuristic": "contrast",
                "status": "partial",
                "checked": 3,
                "notMeasured": 2,
                "note": "2 Text elements carried no resolvable background",
            },
            {
                "heuristic": "select-dark-mode",
                "status": "not-applicable",
                "checked": 0,
                "notMeasured": 0,
                "note": "React Native has no <select>",
            },
            {
                "heuristic": "font-size",
                "status": "measured",
                "checked": 5,
                "notMeasured": 0,
            },
            {
                "heuristic": "touch-target",
                "status": "measured",
                "checked": 2,
                "notMeasured": 0,
            },
        ],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Issues (0 listed)\n" in text
    assert text.endswith(
        "Heuristic coverage (heuristics not fully measured):\n"
        "  contrast: partial (checked 3, not measured 2) - 2 Text elements carried "
        "no resolvable background\n"
        "  select-dark-mode: not-applicable (checked 0, not measured 0) - React "
        "Native has no <select>"
    )
    # Measured rows are not listed.
    assert "font-size:" not in text
    assert "touch-target:" not in text


def test_heuristic_audit_discloses_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        **_HEURISTIC_AUDIT,
        "maxResults": 2,
        "listableResults": 7,
        "resultsTruncated": True,
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert (
        "Issues (2 listed of 7 listable; the SDK cut the list at maxResults=2):"
    ) in text


def test_heuristic_audit_without_coverage_says_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older SDK sends no ``heuristicCoverage``: never read as all measured."""
    report = {
        k: v
        for k, v in _HEURISTIC_AUDIT.items()
        if k not in ("heuristicCoverage", "resultsTruncated")
    }
    report["issues"] = []
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Issues (0 listed; whether the SDK truncated the list is unknown)" in text
    assert "Heuristic coverage: NOT REPORTED" in text
    assert "all 4 heuristics measured" not in text


def test_heuristic_audit_names_heuristics_with_no_coverage_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        **_HEURISTIC_AUDIT,
        "heuristicCoverage": _HEURISTIC_AUDIT["heuristicCoverage"][:2],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert ("  no coverage row for: font-size, touch-target (coverage unknown)") in text


def test_heuristic_audit_missing_issues_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {k: v for k, v in _HEURISTIC_AUDIT.items() if k != "issues"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    # Flagged like every other unreadable audit; reason and selection kept.
    assert is_error
    assert text.startswith("Heuristic design audit (no style guide")
    assert "Reason: No style guide was provided" in text
    assert "Elements: checked 5 of 5 selected" in text
    assert "Issues: UNKNOWN (the report carried no issues list)" in text


def test_style_guide_audit_still_renders_as_style_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mode branch leaves the style-guide renderer as it was."""
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: _AUDIT})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert text.startswith("Style Audit: brand\n")
    assert "Passed: 1 | Failed: 3" in text
    assert "Heuristic" not in text


def test_audit_with_no_mode_and_no_counts_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"issues": [{"element": "a", "issue": "b", "severity": "info"}]}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert text.startswith(
        "Style audit UNKNOWN: the response is neither a style-guide audit nor a "
        "heuristic audit (it carries no mode and no counts)"
    )
    assert "The response listed 1 issue(s):\n  [info] a: b" in text


def test_audit_with_an_unknown_mode_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: {"mode": "vibes"}})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert "(its mode 'vibes' is not one this server reads)" in text


def test_heuristic_audit_header_names_the_heuristics_this_server_knows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: _HEURISTIC_AUDIT})
    )
    text, _ = _call("sdk_design_audit", {})
    assert text.startswith(
        "Heuristic design audit (no style guide; the built-in heuristics this "
        "server knows: contrast, select-dark-mode, font-size, touch-target)\n"
    )


def test_heuristic_audit_discloses_the_tool_side_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issues = [
        {"element": f"el-{i}", "issue": "Low contrast", "severity": "warning"}
        for i in range(23)
    ]
    report = {**_HEURISTIC_AUDIT, "issues": issues, "listableResults": 23}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Issues (23 listed):" in text
    assert "  [warning] el-19: Low contrast" in text
    assert "el-20" not in text
    assert "  ... and 3 more (this tool lists at most 20)" in text


def test_heuristic_audit_truncated_without_listable_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {k: v for k, v in _HEURISTIC_AUDIT.items() if k != "listableResults"}
    report.update(maxResults=2, resultsTruncated=True)
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert (
        "Issues (2 listed of unknown listable; the SDK cut the list at maxResults=2):"
    ) in text


def _with_rows(*rows: dict[str, Any]) -> dict[str, Any]:
    """The heuristic fixture with ``rows`` replacing the matching heuristics."""
    names = {row["heuristic"] for row in rows}
    kept = [
        r for r in _HEURISTIC_AUDIT["heuristicCoverage"] if r["heuristic"] not in names
    ]
    return {**_HEURISTIC_AUDIT, "heuristicCoverage": [*kept, *rows]}


def test_heuristic_coverage_row_with_no_status_is_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _with_rows({"heuristic": "font-size", "checked": 5, "notMeasured": 0})
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "  font-size: unknown (checked 5, not measured 0)" in text
    assert "all 4 heuristics measured" not in text


def test_heuristic_coverage_measured_row_with_unmeasured_elements_is_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``measured`` with ``notMeasured > 0`` contradicts itself: shown, not hidden."""
    report = _with_rows(
        {"heuristic": "contrast", "status": "measured", "checked": 5, "notMeasured": 2}
    )
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "  contrast: measured (checked 5, not measured 2)" in text
    assert "all 4 heuristics measured" not in text


def test_heuristic_coverage_measured_row_without_a_count_is_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _with_rows({"heuristic": "contrast", "status": "measured", "checked": 5})
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "  contrast: measured (checked 5, not measured unknown)" in text


@pytest.mark.parametrize("malformed", [None, {"contrast": "measured"}, "all"])
def test_heuristic_coverage_malformed_is_unknown_not_an_old_sdk(
    monkeypatch: pytest.MonkeyPatch, malformed: Any
) -> None:
    report = {**_HEURISTIC_AUDIT, "heuristicCoverage": malformed}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Heuristic coverage: UNKNOWN (heuristicCoverage was malformed: " in text
    assert "predates" not in text
    assert "all 4 heuristics measured" not in text


def test_style_guide_mode_missing_counts_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        k: v for k, v in _AUDIT.items() if k not in ("passedCount", "failedCount")
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert (
        "(its mode is 'style-guide' but passedCount/failedCount are missing or not "
        "numbers)"
    ) in text
    assert "is not one this server reads" not in text

    report = {**_AUDIT, "passedCount": "1"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert "missing or not numbers" in text


def test_unknown_mode_with_counts_stays_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counts under a ``mode`` this server does not know are no style-guide verdict."""
    report = {**_AUDIT, "mode": "lighthouse"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert "(its mode 'lighthouse' is not one this server reads)" in text
    assert "Passed:" not in text

    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_review", _REVIEW_ARGS)
    assert "Style audit: UNKNOWN" in text
    assert "1 passed, 3 failed" not in text


def test_audit_with_no_mode_and_counts_is_a_style_guide_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An SDK predating ``mode`` still answers with counts: read as style-guide."""
    report = {k: v for k, v in _AUDIT.items() if k != "mode"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Passed: 1 | Failed: 3" in text


_REVIEW_ARGS = {"include_state_variations": False, "include_quality_evaluation": False}


def test_design_review_renders_a_heuristic_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the review reported every heuristic audit as ``UNKNOWN``."""
    report = {
        **_HEURISTIC_AUDIT,
        "heuristicCoverage": [
            {
                "heuristic": "touch-target",
                "status": "partial",
                "checked": 2,
                "notMeasured": 1,
                "note": "1 element had no layout",
            },
            *_HEURISTIC_AUDIT["heuristicCoverage"][:3],
        ],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_review", _REVIEW_ARGS)
    assert not is_error, text
    assert "Style audit: UNKNOWN" not in text
    assert "\nHeuristic design audit (no style guide" in text
    assert (
        "  Reason: No style guide was provided or loaded; ran the built-in heuristics."
    ) in text
    assert "  Elements: checked 5 of 5 selected\n" in text
    assert "  Issues (2 listed):" in text
    assert "    [error] close-btn: Touch target 30x30" in text
    assert (
        "  Heuristic coverage (heuristics not fully measured):\n"
        "    touch-target: partial (checked 2, not measured 1) - 1 element had no "
        "layout"
    ) in text
    assert "passed," not in text


def test_design_review_heuristic_audit_without_coverage_says_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {k: v for k, v in _HEURISTIC_AUDIT.items() if k != "heuristicCoverage"}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_review", _REVIEW_ARGS)
    assert "  Heuristic coverage: NOT REPORTED" in text


def test_design_review_audit_with_no_mode_and_no_counts_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"issues": [{"element": "a", "issue": "b", "severity": "info"}]}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_review", _REVIEW_ARGS)
    assert (
        "Style audit: UNKNOWN (the response is neither a style-guide audit nor a "
        "heuristic audit: it carries no mode and no counts; it "
        "listed 1 issue(s))"
    ) in text
    assert "Heuristic design audit" not in text


@pytest.mark.parametrize(
    ("report", "shape"),
    [
        (
            {k: v for k, v in _AUDIT.items() if k != "passedCount"},
            "its mode is 'style-guide' but passedCount/failedCount are missing or "
            "not numbers",
        ),
        (
            {**_AUDIT, "mode": "lighthouse"},
            "its mode 'lighthouse' is not one this server reads",
        ),
        ({"checkedElements": 3}, "it carries no mode and no counts"),
    ],
)
def test_audit_and_review_explain_an_unknown_answer_the_same_way(
    monkeypatch: pytest.MonkeyPatch, report: dict[str, Any], shape: str
) -> None:
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    audit_text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert f"heuristic audit ({shape});" in audit_text

    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    review_text, _ = _call("sdk_design_review", _REVIEW_ARGS)
    assert f"heuristic audit: {shape}" in review_text


def test_design_review_cuts_heuristic_issues_at_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issues = [
        {"element": f"el-{i}", "issue": "Low contrast", "severity": "warning"}
        for i in range(13)
    ]
    report = {**_HEURISTIC_AUDIT, "issues": issues, "listableResults": 13}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_review", _REVIEW_ARGS)
    assert "  Issues (13 listed):" in text
    assert "    [warning] el-9: Low contrast" in text
    assert "el-10" not in text
    assert "    ... and 3 more (this tool lists at most 10)" in text


@pytest.mark.parametrize("tool", ["sdk_design_audit", "sdk_design_review"])
def test_heuristic_audit_issues_as_a_dict_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    report = {**_HEURISTIC_AUDIT, "issues": {"save-btn": "Low contrast"}}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call(tool, {} if tool == "sdk_design_audit" else _REVIEW_ARGS)
    assert "Issues: UNKNOWN (the report carried no issues list)" in text
    assert "Elements: checked 5 of 5 selected" in text
    if tool == "sdk_design_audit":
        assert is_error


@pytest.mark.parametrize("tool", ["sdk_design_audit", "sdk_design_review"])
@pytest.mark.parametrize("name", [["contrast"], {"id": "contrast"}, 7, None])
def test_heuristic_coverage_non_string_name_is_listed_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tool: str, name: Any
) -> None:
    """Regression: an unhashable name raised ``TypeError`` and lost the whole answer."""
    report = {
        **_HEURISTIC_AUDIT,
        "heuristicCoverage": [
            *_HEURISTIC_AUDIT["heuristicCoverage"],
            {"heuristic": name, "status": "measured", "checked": 1, "notMeasured": 0},
        ],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call(tool, {} if tool == "sdk_design_audit" else _REVIEW_ARGS)
    assert not is_error, text
    assert "unhashable" not in text
    assert f"? (heuristic={name!r}): measured (checked 1, not measured 0)" in text
    assert "heuristics measured" not in text


def test_heuristic_coverage_nameless_row_is_not_counted_as_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _HEURISTIC_AUDIT["heuristicCoverage"]
    report = {
        **_HEURISTIC_AUDIT,
        "heuristicCoverage": [
            *rows,
            {"status": "measured", "checked": 1, "notMeasured": 0},
        ],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "all 5 heuristics measured" not in text
    assert "? (heuristic=None): measured" in text


@pytest.mark.parametrize("tool", ["sdk_design_audit", "sdk_design_review"])
def test_heuristic_coverage_non_object_rows_are_counted_not_dropped(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    report = {
        **_HEURISTIC_AUDIT,
        "heuristicCoverage": [*_HEURISTIC_AUDIT["heuristicCoverage"], "contrast", 3],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call(tool, {} if tool == "sdk_design_audit" else _REVIEW_ARGS)
    assert "2 malformed coverage row(s) ignored (not an object)" in text
    assert "heuristics measured" not in text


@pytest.mark.parametrize(
    ("data", "shape"),
    [
        (
            {"passedCount": 3},
            "it carries no mode and passedCount/failedCount are missing or not numbers",
        ),
        (
            {"passedCount": "3", "failedCount": "1"},
            "it carries no mode and passedCount/failedCount are missing or not numbers",
        ),
        ({"checkedElements": 3}, "it carries no mode and no counts"),
        ([1, 2], "it is not a JSON object (list)"),
        ("audit", "it is not a JSON object (str)"),
    ],
)
def test_unknown_shape_reason_matches_the_payload(
    monkeypatch: pytest.MonkeyPatch, data: Any, shape: str
) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: data}))
    audit_text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert f"heuristic audit ({shape});" in audit_text

    monkeypatch.setattr(server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: data}))
    review_text, _ = _call("sdk_design_review", _REVIEW_ARGS)
    assert f"heuristic audit: {shape}" in review_text


def test_heuristic_issue_rows_that_are_not_objects_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        **_HEURISTIC_AUDIT,
        "issues": [*_HEURISTIC_AUDIT["issues"], "Low contrast", 7],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "  2 malformed issue row(s) ignored (not an object)" in text
    assert "[?] ?: ?" not in text


def test_heuristic_issue_null_fields_render_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        **_HEURISTIC_AUDIT,
        "issues": [{"element": None, "issue": "Low contrast", "severity": None}],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "  [?] ?: Low contrast" in text
    assert "None" not in text


def test_heuristic_coverage_unrecognised_name_is_named_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        **_HEURISTIC_AUDIT,
        "heuristicCoverage": [
            *_HEURISTIC_AUDIT["heuristicCoverage"],
            {
                "heuristic": "line-height",
                "status": "measured",
                "checked": 4,
                "notMeasured": 0,
            },
        ],
    }
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, _ = _call("sdk_design_audit", {})
    assert "all 5 heuristics measured" not in text
    assert text.endswith(
        "Heuristic coverage: all 4 heuristics measured\n"
        "  unrecognised heuristic(s): line-height (not one of the heuristics this "
        "server knows; not counted)"
    )


def test_heuristic_issues_with_no_object_row_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {**_HEURISTIC_AUDIT, "issues": ["Low contrast", 7]}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert is_error
    assert "Issues (0 listed):" in text
    assert "  2 malformed issue row(s) ignored (not an object)" in text
    assert "Reason: No style guide was provided" in text


def test_heuristic_issues_listed_count_excludes_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {**_HEURISTIC_AUDIT, "issues": [*_HEURISTIC_AUDIT["issues"], "junk"]}
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: report})
    )
    text, is_error = _call("sdk_design_audit", {})
    assert not is_error, text
    assert "Issues (2 listed):" in text
    assert "  1 malformed issue row(s) ignored (not an object)" in text


def test_inline_guide_answered_with_heuristics_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guide the SDK did not apply: its heuristics are no answer to the audit."""
    monkeypatch.setattr(
        server_mod, "client", _client(monkeypatch, {AUDIT_ROUTE: _HEURISTIC_AUDIT})
    )
    text, is_error = _call(
        "sdk_design_audit", {"guide": {"name": "brand", "rules": []}}
    )
    assert is_error
    assert text.startswith(
        "Style audit did not run: a guide was provided but the SDK ran "
        "heuristics; the guide audit did not run.\n"
        "Heuristic design audit (no style guide"
    )
    # The heuristic findings are kept, not dropped.
    assert "  [error] close-btn: Touch target 30x30" in text


def test_design_audit_description_names_the_heuristic_mode() -> None:
    tools = {tool.name: tool for tool in server_mod.TOOLS}
    for name in ("sdk_design_audit", "sdk_design_review"):
        description = tools[name].description or ""
        assert (
            "contrast,\nselect-dark-mode, font-size, touch-target" in description
            or "contrast, select-dark-mode, font-size, touch-target" in description
        ), name
        assert "no pass/fail verdict" in description.replace("NO", "no"), name


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
