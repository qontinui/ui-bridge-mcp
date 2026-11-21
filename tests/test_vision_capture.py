"""Screenshot tools capture through ``POST /ui-bridge/vision/capture``.

Plan 2026-11-19-the-ui-bridge-design-surface-is-blocked-behind-a-stalled-pr-
unpublished-packages-and-a-score, Phase 6.

``/ui-bridge/sdk/screenshot`` and ``/ui-bridge/control/annotated-screenshot``
were deleted from the runner on 2026-05-13, so every screenshot tool answered
``Not found``. They now capture through the runner's one capture route with
``inline: true`` (runner ``vision_routes.rs`` ``CaptureRequest``); annotation,
diff, crop and delta stay client-side in ``screenshot.py``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
from typing import Any

import httpx
import pytest
from mcp import types
from PIL import Image

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient
from ui_bridge_mcp.screenshot import BaselineStore, DeltaEncoder

CAPTURE = "/ui-bridge/vision/capture"
RETIRED_ROUTES = (
    "/ui-bridge/sdk/screenshot",
    "/ui-bridge/control/annotated-screenshot",
)
RETIRED_METHODS = (
    "sdk_screenshot",
    "sdk_screenshot_raw",
    "control_annotated_screenshot",
)


def _png(width: int = 40, height: int = 20, color: str = "white") -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _frame(b64: str | None = None, width: int = 40, height: int = 20) -> dict[str, Any]:
    return {
        "url": "/ui-bridge/vision/cache/abc",
        "base64": b64 if b64 is not None else _png(width, height),
        "sha256": "abc",
        "width": width,
        "height": height,
        "bytes": 100,
        "format": "png",
        "contract": "png_strict",
    }


_SNAPSHOT: dict[str, Any] = {
    "elements": [
        {
            "id": "btn",
            "type": "button",
            "label": "Go",
            "state": {"rect": {"x": 4, "y": 4, "width": 10, "height": 6}},
        }
    ],
    "viewport": {"viewportWidth": 40, "viewportHeight": 20},
}


class Runner:
    """A mocked runner: answers routes and records every request it saw."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((path, body))
        answer = self.routes.get(path)
        if callable(answer):
            answer = answer(body)
        if isinstance(answer, httpx.Response):
            return answer
        if answer is not None:
            return httpx.Response(200, json={"success": True, "data": answer})
        return httpx.Response(404, json={"success": False, "error": "Not found"})

    def captures(self) -> list[dict[str, Any]]:
        return [body or {} for path, body in self.requests if path == CAPTURE]


def _install(monkeypatch: pytest.MonkeyPatch, runner: Runner) -> UIBridgeClient:
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    c = UIBridgeClient(host="127.0.0.1", port=1)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(runner.handler))
    monkeypatch.setattr(server_mod, "client", c)
    monkeypatch.setattr(server_mod, "baseline_store", BaselineStore())
    monkeypatch.setattr(server_mod, "delta_encoder", DeltaEncoder())
    return c


def _call(
    name: str, arguments: dict[str, Any], runner: Runner | None = None
) -> tuple[list[Any], bool]:
    """Call a tool. ``runner`` re-arms the mocked transport a prior call closed."""
    if runner is not None and server_mod.client is not None:
        server_mod.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(runner.handler)
        )

    async def go() -> tuple[list[Any], bool]:
        try:
            result: Any = await server_mod.call_tool(name, arguments)
        finally:
            if server_mod.client is not None:
                await server_mod.client.close()
        if isinstance(result, types.CallToolResult):
            return list(result.content), bool(result.isError)
        return list(result), False

    return asyncio.run(go())


def _text(items: list[Any]) -> str:
    return "\n".join(getattr(i, "text", "") for i in items if i.type == "text")


def _images(items: list[Any]) -> list[Any]:
    return [i for i in items if i.type == "image"]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def test_client_requests_no_retired_screenshot_route() -> None:
    source = inspect.getsource(UIBridgeClient)
    assert [m for m in RETIRED_METHODS if hasattr(UIBridgeClient, m)] == []
    assert [r for r in RETIRED_ROUTES if r in source] == []
    # The source was read, not nothing: the replacement is in it.
    assert hasattr(UIBridgeClient, "vision_capture")
    import ui_bridge_mcp.client as client_mod

    assert client_mod.VISION_CAPTURE_ROUTE == CAPTURE


def test_vision_capture_sends_inline_request_and_returns_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner({CAPTURE: _frame()})
    c = _install(monkeypatch, runner)

    async def go() -> Any:
        try:
            return await c.vision_capture(
                target="web", region={"x": 1, "y": 2, "w": 3, "h": 4}, element="btn"
            )
        finally:
            await c.close()

    resp = asyncio.run(go())
    assert resp.success, resp.error
    assert set(resp.data) == {"base64", "width", "height", "sha256", "format"}
    assert resp.data["width"] == 40 and resp.data["height"] == 20
    assert resp.data["sha256"] == "abc"
    assert runner.captures() == [
        {
            "inline": True,
            "force": True,
            "target": "web",
            "region": {"x": 1, "y": 2, "w": 3, "h": 4},
            "element": "btn",
        }
    ]


def test_vision_capture_inline_too_large_names_the_url_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_large = httpx.Response(
        413,
        json={
            "success": False,
            "error": "capture is 6000000 bytes, over the 5242880-byte inline cap",
            "code": "INLINE_TOO_LARGE",
            "hint": {
                "url": "/ui-bridge/vision/cache/deadbeef",
                "bytes": 6000000,
                "maxBytes": 5242880,
            },
        },
    )
    runner = Runner({CAPTURE: too_large})
    c = _install(monkeypatch, runner)

    async def go() -> Any:
        try:
            return await c.vision_capture()
        finally:
            await c.close()

    resp = asyncio.run(go())
    assert not resp.success
    assert resp.code == "INLINE_TOO_LARGE"
    assert resp.status == 413
    assert "GET http://127.0.0.1:1/ui-bridge/vision/cache/deadbeef" in (
        resp.error or ""
    )
    assert "not retried" in (resp.error or "")
    assert len(runner.captures()) == 1


def test_vision_capture_without_inline_bytes_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    del frame["base64"]
    runner = Runner({CAPTURE: frame})
    c = _install(monkeypatch, runner)

    async def go() -> Any:
        try:
            return await c.vision_capture()
        finally:
            await c.close()

    resp = asyncio.run(go())
    assert not resp.success
    assert "without base64" in (resp.error or "")
    assert "UNKNOWN" in (resp.error or "")


# ---------------------------------------------------------------------------
# Tools, by family
# ---------------------------------------------------------------------------


def test_sdk_screenshot_returns_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = Runner({CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot", {"target": "web"})
    assert not is_error, _text(items)
    assert "Screenshot of 'web': 40x20px png" in _text(items)
    images = _images(items)
    assert len(images) == 1 and images[0].mimeType == "image/png"
    assert runner.captures()[0]["target"] == "web"


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("sdk_screenshot", {}),
        ("sdk_annotated_screenshot", {}),
        ("sdk_element_screenshot", {"element_id": "btn"}),
        ("sdk_screenshot_baseline_save", {"baseline_key": "home"}),
        ("sdk_screenshot_diff", {"baseline_key": "home"}),
        ("sdk_screenshot_before", {}),
        ("sdk_screenshot_after", {}),
        ("sdk_screenshot_delta", {}),
    ],
)
def test_sdk_capture_tools_require_a_target(
    tool: str, arguments: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No target would capture the runner's own window: refused, not defaulted."""
    runner = Runner({CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call(tool, arguments)
    assert is_error
    assert f"{tool} needs `target`" in _text(items)
    assert runner.captures() == []


def test_screenshot_tool_reports_inline_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_large = httpx.Response(
        413,
        json={
            "success": False,
            "error": "over the inline cap",
            "code": "INLINE_TOO_LARGE",
            "hint": {"url": "/ui-bridge/vision/cache/feed", "bytes": 9, "maxBytes": 1},
        },
    )
    runner = Runner({CAPTURE: too_large})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot", {"target": "web"})
    assert is_error
    assert "/ui-bridge/vision/cache/feed" in _text(items)
    assert len(runner.captures()) == 1


@pytest.mark.parametrize(
    "tool,snapshot_route,arguments,target",
    [
        ("ui_annotated_screenshot", "/ui-bridge/control/snapshot", {}, None),
        (
            "sdk_annotated_screenshot",
            "/ui-bridge/sdk/snapshot",
            {"target": "web"},
            "web",
        ),
    ],
)
def test_annotated_screenshot_annotates_the_captured_frame(
    tool: str,
    snapshot_route: str,
    arguments: dict[str, Any],
    target: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner({snapshot_route: _SNAPSHOT, CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call(tool, {**arguments, "mode": "all"})
    assert not is_error, _text(items)
    images = _images(items)
    assert len(images) == 1
    annotated = Image.open(io.BytesIO(base64.b64decode(images[0].data)))
    assert annotated.size == (40, 20)
    # The overlay was drawn: the frame was solid white.
    colors = annotated.convert("RGB").getcolors(maxcolors=40 * 20)
    assert colors is not None and len(colors) > 1
    capture = runner.captures()[0]
    assert capture.get("target") == target
    assert "target" in capture if target else "target" not in capture


def test_annotation_scales_css_rects_onto_a_downscaled_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot's CSS viewport, not the frame's pixel size, sets the scale."""
    snapshot = {**_SNAPSHOT, "viewport": {"viewportWidth": 80, "viewportHeight": 40}}
    runner = Runner({"/ui-bridge/control/snapshot": snapshot, CAPTURE: _frame()})
    _install(monkeypatch, runner)
    captured: dict[str, Any] = {}

    def fake_annotate(
        b64: str, elements: Any, width: int, height: int, *a: Any, **k: Any
    ) -> str:
        captured["size"] = (width, height)
        return b64

    monkeypatch.setattr(server_mod, "annotate_screenshot", fake_annotate)
    items, is_error = _call("ui_annotated_screenshot", {})
    assert not is_error, _text(items)
    assert captured["size"] == (80, 40)


@pytest.mark.parametrize(
    "tool,snapshot_route,arguments",
    [
        ("ui_element_screenshot", "/ui-bridge/control/snapshot", {}),
        ("sdk_element_screenshot", "/ui-bridge/sdk/snapshot", {"target": "web"}),
    ],
)
def test_element_screenshot_crops_the_captured_frame(
    tool: str,
    snapshot_route: str,
    arguments: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner({snapshot_route: _SNAPSHOT, CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call(tool, {**arguments, "element_id": "btn", "padding": 0})
    assert not is_error, _text(items)
    cropped = Image.open(io.BytesIO(base64.b64decode(_images(items)[0].data)))
    assert cropped.size == (10, 6)
    assert runner.captures()[0]["contract"] == "png_strict"


def test_baseline_save_then_diff_compares_captured_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = iter([_frame(_png(color="white")), _frame(_png(color="black"))])
    runner = Runner({CAPTURE: lambda body: next(frames)})
    _install(monkeypatch, runner)

    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert not is_error, _text(items)
    assert "Baseline saved: 'home'" in _text(items)

    items, is_error = _call(
        "sdk_screenshot_diff", {"baseline_key": "home", "target": "web"}, runner
    )
    assert not is_error, _text(items)
    assert "Visual diff: FAILED" in _text(items)
    assert "Change: 100.0%" in _text(items)
    assert [c["contract"] for c in runner.captures()] == ["png_strict", "png_strict"]
    assert [c["target"] for c in runner.captures()] == ["web", "web"]


def test_before_after_compares_captured_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner({CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_before", {"target": "web"})
    assert not is_error, _text(items)
    items, is_error = _call("sdk_screenshot_after", {"target": "web"}, runner)
    assert not is_error, _text(items)
    assert "Change: 0.0%" in _text(items)
    assert len(_images(items)) == 2
    assert len(runner.captures()) == 2


def test_delta_first_capture_returns_the_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b64 = _png()
    runner = Runner({CAPTURE: _frame(b64)})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_delta", {"target": "web"})
    assert not is_error, _text(items)
    assert "First capture (full image): 40x20px" in _text(items)
    assert _images(items)[0].data == b64
    items, is_error = _call("sdk_screenshot_delta", {"target": "web"}, runner)
    assert not is_error, _text(items)
    assert "No visual changes detected." in _text(items)
