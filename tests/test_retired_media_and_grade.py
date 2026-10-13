"""The dead media tools are gone, and the quality tools render values, not a grade.

Plan 2026-10-13-image-generation-and-visual-judgment, Phase 6 (ui-bridge-mcp):

- The six ``sdk_media_*`` tools, ``ui_analyze_image`` and ``ui_audit_images``
  called the runner's ``/ui-bridge/sdk/ai/media/*`` proxies, whose SDK routes
  were deleted on 2026-05-14. Measured live in the plan's Phase 1, every one
  answered ``No active SDK app connection``, ``Not found``, or a 30 s relay
  timeout. They are retired with their eight client methods (decision D6);
  ``ui_audit_images`` returns in Phase 8 over the vision surface.
- ``/design/evaluate`` no longer carries an A-F ``grade`` (decision D7: values,
  not verdicts), so neither renderer prints one. A stale SDK that still sends
  ``grade`` must not get it rendered either: the MCP surface hands out values.
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
    "ui_audit_images",
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
# values plus the grades. The grades must not reach the rendered text.
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
        }
    ],
    "topIssues": [],
}


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


def test_design_evaluate_renders_values_and_no_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    text = _call_tool("sdk_design_evaluate", {})
    assert "Overall Score: 87/100" in text
    assert "UX Score: 91/100" in text
    assert "Grade" not in text


def test_design_review_quality_section_renders_values_and_no_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    text = _call_tool(
        "sdk_design_review",
        {"include_state_variations": False, "include_quality_evaluation": True},
    )
    assert "Quality: 87/100 (UX 91/100)" in text
    assert "Grade" not in text
