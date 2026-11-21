"""The dead media tools are gone, and the quality tools render values, not a grade.

Plan 2026-10-13-image-generation-and-visual-judgment, Phase 6 (ui-bridge-mcp):

- The six ``sdk_media_*`` tools, ``ui_analyze_image`` and ``ui_audit_images``
  called the runner's ``/ui-bridge/sdk/ai/media/*`` proxies, whose SDK routes
  were deleted on 2026-05-14. Measured live in the plan's Phase 1, every one
  answered ``No active SDK app connection``, ``Not found``, or a 30 s relay
  timeout. They are retired with their eight client methods (decision D6).
  ``ui_audit_images`` came back in Phase 8 over the vision surface
  (``test_audit_images.py``).
- ``/design/evaluate`` no longer carries an A-F ``grade`` (decision D7: values,
  not verdicts), so neither renderer prints one. A stale SDK that still sends
  ``grade`` must not get it rendered either: the MCP surface hands out values.

Plan 2026-11-19-the-ui-bridge-design-surface-is-blocked-behind-a-stalled-pr-
unpublished-packages-and-a-score, Phase 4: the aggregates go the same way. An
SDK that still sends ``overallScore`` / ``uxScore`` must not get them rendered,
and the renderers no longer average metric scores per category themselves. Each
enabled metric's own ``score``, ``weight``, findings and ``rawData`` keys are
the values a caller judges. The design audit reads ``summary`` as COUNTS and
lists failing rows from ``results[]``; the responsive renderer reads
``{snapshots: [...]}``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import httpx
import pytest
from mcp import types

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient

RETIRED_TOOLS = (
    "sdk_media_find",
    "sdk_media_audit_accessibility",
    "sdk_media_audit_performance",
    "sdk_media_snapshot",
    "sdk_media_compare",
    "sdk_media_analyze_batch",
    "ui_analyze_image",
)

RETIRED_CLIENT_METHODS = (
    "sdk_media_find",
    "sdk_media_audit_accessibility",
    "sdk_media_audit_performance",
    "sdk_media_snapshot",
    "sdk_media_compare",
    "sdk_media_analyze",
    "sdk_media_analyze_batch",
    "sdk_media_analyze_page",
)

# A /design/evaluate report as an SDK that predates the change would send it:
# values plus the grades AND the aggregates. Neither may reach the rendered text.
_REPORT: dict[str, Any] = {
    "overallScore": 87,
    "grade": "B",
    "uxScore": 91,
    "uxGrade": "A",
    "contextName": "general",
    "totalElements": 3,
    "durationMs": 4,
    "metrics": [
        {
            "metricId": "contentOverflow",
            "score": 91,
            "label": "Content overflow",
            "category": "ux",
            "enabled": True,
            "weight": 0.1,
            "findings": [],
            "rawData": {"overflowingElements": 0},
        },
        # Two metrics in one category: a per-category average would be 70.
        {
            "metricId": "colorHarmony",
            "score": 60,
            "label": "Color harmony",
            "category": "color",
            "enabled": True,
            "weight": 0.2,
            "findings": [
                {"severity": "warning", "message": "Too many hues", "elementIds": []}
            ],
            "rawData": {"hueCount": 9, "paletteSize": 14},
        },
        {
            "metricId": "wcagContrastCompliance",
            "score": 80,
            "label": "WCAG contrast",
            "category": "color",
            "enabled": True,
            "weight": 0.3,
            "findings": [],
        },
        {
            "metricId": "whitespaceRatio",
            "score": 13,
            "label": "Whitespace ratio",
            "category": "density",
            "enabled": False,
            "weight": 0.0,
            "findings": [],
        },
    ],
    "topIssues": [
        {
            "severity": "warning",
            "message": "Too many hues",
            "recommendation": "Use a smaller palette",
        }
    ],
}

# Strings only an aggregate could produce.
_AGGREGATE_MARKERS = (
    "Overall Score",
    "UX Score",
    "87/100",
    "91/100",
    "Category Scores",
    "Categories:",
    "color=70",
    "Color: 70/100",
    "70/100",
)


def _client(monkeypatch: pytest.MonkeyPatch) -> UIBridgeClient:
    """A client whose every request is answered from the routes table below."""
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/ui-bridge/sdk/design/evaluate":
            return httpx.Response(200, json={"success": True, "data": _REPORT})
        if path == "/ui-bridge/sdk/design/snapshot":
            return httpx.Response(200, json={"success": True, "data": {"elements": []}})
        if path == "/ui-bridge/sdk/design/audit":
            return httpx.Response(
                200, json={"success": False, "error": "No style guide loaded"}
            )
        return httpx.Response(404, json={"success": False, "error": "Not found"})

    c = UIBridgeClient(host="127.0.0.1", port=1)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def _call_tool(name: str, arguments: dict[str, Any]) -> str:
    async def go() -> str:
        try:
            result: Any = await server_mod.call_tool(name, arguments)
        finally:
            if server_mod.client is not None:
                await server_mod.client.close()
        # A failure flagged with isError arrives as a CallToolResult rather than
        # a content list (the ui-bridge-mcp isError fix); read either shape.
        items = result.content if isinstance(result, types.CallToolResult) else result
        return "\n".join(getattr(item, "text", "") for item in items)

    return asyncio.run(go())


def _tool_names() -> list[str]:
    return [tool.name for tool in asyncio.run(server_mod.list_tools())]


def test_retired_media_tools_are_not_listed() -> None:
    names = _tool_names()
    # The listing was read, not empty: an empty one would pass vacuously.
    assert len(names) > 100
    assert [n for n in names if n in RETIRED_TOOLS] == []
    assert [n for n in names if n.startswith("sdk_media_")] == []


def test_design_evaluate_description_promises_no_grade() -> None:
    """The description promised a "letter grade" the SDK no longer sends.

    Read case-insensitively: the old wording was lowercase ("Returns 0-100
    score, letter grade, ..."), so a check for "Grade" could not fail against it.
    """
    tool = next(
        t
        for t in asyncio.run(server_mod.list_tools())
        if t.name == "sdk_design_evaluate"
    )
    description = (tool.description or "").lower()
    # The right description was read, not an empty one.
    assert "0-100" in description
    assert "grade" not in description


@pytest.mark.parametrize("name", RETIRED_TOOLS)
def test_retired_tool_is_unknown(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    assert _call_tool(name, {}) == f"Unknown tool: {name}"


def test_client_has_no_media_methods_or_routes() -> None:
    assert [m for m in RETIRED_CLIENT_METHODS if hasattr(UIBridgeClient, m)] == []
    assert "/ai/media" not in inspect.getsource(UIBridgeClient)


def _assert_per_metric_values(text: str) -> None:
    """Each enabled metric's own values are rendered; a disabled one is not."""
    assert "Content overflow [ux]: score=91 weight=0.1 findings=0" in text
    assert "Color harmony [color]: score=60 weight=0.2 findings=1" in text
    assert "WCAG contrast [color]: score=80 weight=0.3 findings=0" in text
    assert "rawData keys: hueCount, paletteSize" in text
    assert "rawData keys: overflowingElements" in text
    assert "Whitespace ratio" not in text
    assert "[WARNING] Too many hues" in text
    assert "Use a smaller palette" in text


def test_design_evaluate_renders_per_metric_values_and_no_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    text = _call_tool("sdk_design_evaluate", {})
    _assert_per_metric_values(text)
    assert "Overall Score" not in text
    assert "Quality: " not in text
    for marker in _AGGREGATE_MARKERS:
        assert marker not in text, marker
    assert "Grade" not in text


def test_design_review_quality_section_renders_per_metric_values_and_no_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    text = _call_tool(
        "sdk_design_review",
        {"include_state_variations": False, "include_quality_evaluation": True},
    )
    _assert_per_metric_values(text)
    assert "Overall Score" not in text
    assert "Quality: " not in text
    assert "/100 (UX" not in text
    for marker in _AGGREGATE_MARKERS:
        assert marker not in text, marker
    assert "Grade" not in text
