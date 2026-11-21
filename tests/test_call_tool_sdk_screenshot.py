"""``sdk_screenshot`` entered through the MCP dispatch entry point.

Plan 2026-11-19-the-ui-bridge-design-surface-is-blocked-behind-a-stalled-pr-
unpublished-packages-and-a-score, Phase 6. This is the dispatch-path test with a
mocked transport, NOT the phase's arming check: the tool is called through
``server.call_tool`` (what the MCP server dispatches every tool call through)
and only the HTTP transport is mocked, answering the runner's
``POST /ui-bridge/vision/capture`` as the runner does. No client or server
helper is replaced, so the real client code builds and sends the request.

The live arming check — ``sdk_screenshot`` and ``sdk_annotated_screenshot``
returning an image against a temp runner's ``/ui-bridge/vision/capture`` — is
still owed; a mocked transport cannot stand in for it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any

import httpx
import pytest
from mcp import types
from PIL import Image

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient


def test_call_tool_sdk_screenshot_captures_through_vision_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), "navy").save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode()
    seen: list[tuple[str, str, dict[str, Any] | None]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/ui-bridge/vision/capture":
            # The runner's ApiResponse<VisionCaptureResp::Single> answer.
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "url": "/ui-bridge/vision/cache/" + "b" * 64,
                        "base64": png_b64,
                        "sha256": "b" * 64,
                        "width": 32,
                        "height": 16,
                        "bytes": len(buf.getvalue()),
                        "format": "png",
                        "contract": "claude_vision_v1",
                    },
                },
            )
        return httpx.Response(404, json={"success": False, "error": "Not found"})

    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    runner_client = UIBridgeClient(host="127.0.0.1", port=1)
    runner_client._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(server_mod, "client", runner_client)

    async def go() -> Any:
        try:
            return await server_mod.call_tool("sdk_screenshot", {"target": "web"})
        finally:
            await runner_client.close()

    result = asyncio.run(go())
    assert not isinstance(result, types.CallToolResult), result
    texts = [item.text for item in result if item.type == "text"]
    images = [item for item in result if item.type == "image"]
    assert texts == [f"Screenshot of 'web': 32x16px png (runner cache key {'b' * 64})"]
    assert len(images) == 1
    assert images[0].data == png_b64
    assert images[0].mimeType == "image/png"
    assert seen == [
        (
            "POST",
            "/ui-bridge/vision/capture",
            {"inline": True, "force": True, "target": "web"},
        )
    ]
