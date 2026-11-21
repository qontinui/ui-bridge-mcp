"""A failed capture of a ``target`` names its cause; a successful one reads nothing more.

Plan 2026-11-21-vision-capture-cannot-frame-a-websocket-transport-app, Phase 2
(decision D5). A runner build that predates the WebSocket frame source answers a
capture of a WebSocket-transport app with a misleading "device screenshot ...
returned HTTP 404". ui-bridge-mcp sends the capture unconditionally and only
when it FAILS for a ``target`` reads ``/health`` ``visionFrameSources`` and the
target's ``/ui-bridge/apps/registered`` row to say why.

Only the HTTP transport is mocked; the tool is entered through
``server.call_tool`` so the real client and server code run.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

import httpx
import pytest
from mcp import types
from PIL import Image

import ui_bridge_mcp.client as client_mod
import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient

# The reader types are reached through the module, not imported by name, so this
# file still COLLECTS against a client that predates them. Reverted to that
# client, the tests that assert the new diagnosis (the named refusal and the
# UNKNOWN notes) go red on their assertions; the pass-through and regression
# tests (a successful capture, an HTTP or unregistered target, a runner that
# has the frame source) stay green by design, since they pin behaviour the
# change must NOT alter.

CAPTURE = "/ui-bridge/vision/capture"
HEALTH = "/health"
REGISTERED = "/ui-bridge/apps/registered"
WS_APP = "ws-app"
HTTP_APP = "http-app"
OLD_BUILD = "a609e88-1763700000000"
NEW_SOURCES = ["runner_window", "device_http", "adb", "websocket_relay"]

# What a runner predating Phase 1 answers for a WebSocket app (blanket 500).
OLD_BUILD_404 = httpx.Response(
    500,
    json={
        "success": False,
        "error": (
            "device screenshot http://localhost:3001/ui-bridge/control/screenshot "
            "returned HTTP 404"
        ),
    },
)
# What a Phase-1 runner answers for a WebSocket client that cannot screenshot.
UNSUPPORTED = httpx.Response(
    422,
    json={
        "success": False,
        "error": "app 'ws-app' does not advertise the screenshot capability",
        "code": "FRAME_SOURCE_UNSUPPORTED",
    },
)


def _png(width: int = 40, height: int = 20) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _frame() -> dict[str, Any]:
    return {
        "url": "/ui-bridge/vision/cache/" + "c" * 64,
        "base64": _png(),
        "sha256": "c" * 64,
        "width": 40,
        "height": 20,
        "bytes": 100,
        "format": "png",
        "contract": "claude_vision_v1",
    }


def _row(app_id: str, transport: str) -> dict[str, Any]:
    """A ``RegisterAppResponse``: the flattened ``DiscoveredApp`` plus transport."""
    return {
        "appId": app_id,
        "appName": app_id,
        "appType": "web",
        "url": "http://localhost:3001",
        "port": 3001,
        "basePath": "/ui-bridge",
        "capabilities": [],
        "discoveredAt": 0,
        "transport": transport,
        "verifiedOrigin": None,
        "principalClass": "browser",
        "keepAliveMs": None,
    }


_REGISTRY = httpx.Response(
    200,
    json={
        "success": True,
        "data": [_row(WS_APP, "websocket"), _row(HTTP_APP, "http")],
    },
)


def _health(
    sources: list[str] | None, *, top_level_only: bool = False
) -> httpx.Response:
    data: dict[str, Any] = {"status": "ok", "buildId": OLD_BUILD}
    body: dict[str, Any] = {"success": True, "data": data, "buildId": OLD_BUILD}
    if sources is not None:
        (body if top_level_only else data)["visionFrameSources"] = sources
    return httpx.Response(200, json=body)


class Runner:
    """A mocked runner: answers routes and records every path it saw."""

    def __init__(
        self, routes: dict[str, httpx.Response | dict[str, Any] | Exception]
    ) -> None:
        self.routes = routes
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        answer = self.routes.get(path)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        if answer is not None:
            return httpx.Response(200, json={"success": True, "data": answer})
        return httpx.Response(404, json={"success": False, "error": "Not found"})


def _client(runner: Runner) -> UIBridgeClient:
    c = UIBridgeClient(host="127.0.0.1", port=1)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(runner.handler))
    return c


def _call(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
    target: str,
    tool: str = "sdk_screenshot",
) -> tuple[str, bool]:
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    c = _client(runner)
    monkeypatch.setattr(server_mod, "client", c)

    async def go() -> Any:
        try:
            return await server_mod.call_tool(tool, {"target": target})
        finally:
            await c.close()

    result = asyncio.run(go())
    if isinstance(result, types.CallToolResult):
        items, is_error = list(result.content), bool(result.isError)
    else:
        items, is_error = list(result), False
    text = "\n".join(getattr(i, "text", "") for i in items if i.type == "text")
    return text, is_error


# ---------------------------------------------------------------------------
# The five Phase 2 cases
# ---------------------------------------------------------------------------


def test_old_build_websocket_target_returns_the_named_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {CAPTURE: OLD_BUILD_404, REGISTERED: _REGISTRY, HEALTH: _health(None)}
    )
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert is_error
    # The minted code comes FIRST, as describe_error's contract requires.
    assert text.startswith(
        "Error getting screenshot: [code=FRAME_SOURCE_MISSING status=500] "
        f"the runner (build {OLD_BUILD}) cannot capture WebSocket-transport app "
        f"'{WS_APP}': its capture route HTTP-GETs the app's page origin instead "
        "of using the app's socket. A runner built from qontinui-runner "
        "origin/main at or after Phase 1 of plan "
        "2026-11-21-vision-capture-cannot-frame-a-websocket-transport-app "
        "captures it over the app's socket. Runner said: [code=producer_failed"
    ), text
    # It claims no 404 of its own; the only 404 is the runner's own words.
    assert text.count("404") == 1
    # The runner's own words stay visible as evidence, after the cause.
    assert "returned HTTP 404" in text
    assert text.index("cannot capture") < text.index("returned HTTP 404")


def test_successful_capture_makes_no_health_or_registry_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner({CAPTURE: _frame(), REGISTERED: _REGISTRY, HEALTH: _health(None)})
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert not is_error, text
    assert runner.paths == [CAPTURE]


def test_new_build_passes_frame_source_unsupported_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {CAPTURE: UNSUPPORTED, REGISTERED: _REGISTRY, HEALTH: _health(NEW_SOURCES)}
    )
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert is_error
    assert text == (
        "Error getting screenshot: [code=FRAME_SOURCE_UNSUPPORTED status=422] "
        "API error 422: app 'ws-app' does not advertise the screenshot capability"
    )


def test_failed_capture_with_unreadable_health_is_labelled_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {
            CAPTURE: OLD_BUILD_404,
            REGISTERED: _REGISTRY,
            HEALTH: httpx.Response(503, text="upstream unavailable"),
        }
    )
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert is_error
    # The original typed code stays first; the UNKNOWN note follows its words.
    assert text.startswith(
        "Error getting screenshot: [code=producer_failed status=500]"
    )
    assert text.endswith(
        "returned HTTP 404 (runner capability UNKNOWN (GET /health failed: "
        "[code=producer_failed status=503] API error 503: upstream unavailable))"
    ), text
    assert "cannot capture WebSocket-transport app" not in text


def test_http_target_error_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    http_404 = httpx.Response(
        500,
        json={
            "success": False,
            "error": "device screenshot http://localhost:3001/... returned HTTP 404",
        },
    )
    runner = Runner({CAPTURE: http_404, REGISTERED: _REGISTRY, HEALTH: _health(None)})
    text, is_error = _call(monkeypatch, runner, HTTP_APP)
    assert is_error
    assert text == (
        "Error getting screenshot: [code=producer_failed status=500] API error 500: "
        "device screenshot http://localhost:3001/... returned HTTP 404"
    )


# ---------------------------------------------------------------------------
# Further diagnosis arms
# ---------------------------------------------------------------------------


def test_unreadable_registry_is_labelled_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {
            CAPTURE: OLD_BUILD_404,
            REGISTERED: httpx.Response(500, json={"success": False, "error": "boom"}),
            HEALTH: _health(None),
        }
    )
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert is_error
    assert text.startswith(
        "Error getting screenshot: [code=producer_failed status=500]"
    )
    assert (
        "returned HTTP 404 (runner capability UNKNOWN (GET /ui-bridge/apps/registered failed"
        in text
    )


def test_sources_listed_without_websocket_relay_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {
            CAPTURE: OLD_BUILD_404,
            REGISTERED: _REGISTRY,
            HEALTH: _health(["runner_window", "device_http", "adb"]),
        }
    )
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert is_error
    assert text.startswith("Error getting screenshot: [code=FRAME_SOURCE_MISSING")
    assert "cannot capture WebSocket-transport app 'ws-app'" in text


_CACHE_URL = "/ui-bridge/vision/cache/" + "d" * 64
_TOO_LARGE = httpx.Response(
    413,
    json={
        "success": False,
        "error": "over the inline cap",
        "code": "INLINE_TOO_LARGE",
        "hint": {"url": _CACHE_URL, "bytes": 9, "maxBytes": 1},
    },
)


@pytest.mark.parametrize(
    "tool,cache",
    [
        # Default contract: the oversize answer is returned, not followed.
        ("sdk_screenshot", None),
        # Lossless contract: the cache follow-up runs and fails.
        ("sdk_screenshot_before", httpx.Response(500, text="cache gone")),
    ],
)
def test_oversize_capture_failure_is_not_diagnosed(
    tool: str, cache: httpx.Response | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner TOOK the frame, so no frame-source diagnosis is run."""
    routes: dict[str, httpx.Response | dict[str, Any] | Exception] = {
        CAPTURE: _TOO_LARGE,
        REGISTERED: _REGISTRY,
        HEALTH: _health(None),
    }
    if cache is not None:
        routes[_CACHE_URL] = cache
    runner = Runner(routes)
    text, is_error = _call(monkeypatch, runner, WS_APP, tool)
    assert is_error
    assert REGISTERED not in runner.paths and HEALTH not in runner.paths
    assert "cannot capture" not in text and "UNKNOWN (" not in text
    if cache is not None:
        assert runner.paths == [CAPTURE, _CACHE_URL]


def test_transport_failure_is_not_diagnosed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No runner answered, so nothing about its frame sources can be read."""
    runner = Runner(
        {
            CAPTURE: httpx.ConnectError("connection refused"),
            REGISTERED: _REGISTRY,
            HEALTH: _health(None),
        }
    )
    text, is_error = _call(monkeypatch, runner, WS_APP)
    assert is_error
    assert runner.paths == [CAPTURE]
    assert text.startswith("Error getting screenshot: [code=app_unreachable]")
    assert "cannot capture WebSocket" not in text and "UNKNOWN (" not in text


def test_unregistered_target_error_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {CAPTURE: OLD_BUILD_404, REGISTERED: _REGISTRY, HEALTH: _health(None)}
    )
    text, is_error = _call(monkeypatch, runner, "emulator-5554")
    assert is_error
    assert text.startswith(
        "Error getting screenshot: [code=producer_failed status=500]"
    )
    assert "UNKNOWN" not in text and "cannot capture" not in text


# ---------------------------------------------------------------------------
# Client readers: absent (None) and failed (Unknown) stay distinct
# ---------------------------------------------------------------------------


def _read(runner: Runner, reader: str, *args: Any) -> Any:
    c = _client(runner)

    async def go() -> Any:
        try:
            return await getattr(c, reader)(*args)
        finally:
            await c.close()

    return asyncio.run(go())


def test_frame_sources_absent_is_none_with_the_build_id() -> None:
    got = _read(Runner({HEALTH: _health(None)}), "vision_frame_sources")
    assert got == client_mod.RunnerFrameSources(sources=None, build_id=OLD_BUILD)


def test_frame_sources_in_data_win_over_the_top_level_mirror() -> None:
    body = {
        "success": True,
        "data": {"buildId": OLD_BUILD, "visionFrameSources": NEW_SOURCES},
        "buildId": "top-level-build",
        "visionFrameSources": ["runner_window"],
    }
    got = _read(
        Runner({HEALTH: httpx.Response(200, json=body)}), "vision_frame_sources"
    )
    assert got == client_mod.RunnerFrameSources(sources=NEW_SOURCES, build_id=OLD_BUILD)


def test_frame_sources_read_from_the_top_level_mirror() -> None:
    got = _read(
        Runner({HEALTH: _health(NEW_SOURCES, top_level_only=True)}),
        "vision_frame_sources",
    )
    assert got == client_mod.RunnerFrameSources(sources=NEW_SOURCES, build_id=OLD_BUILD)


def test_frame_sources_malformed_or_unreachable_is_unknown() -> None:
    bad = httpx.Response(200, json={"success": True, "data": {"visionFrameSources": 3}})
    assert isinstance(
        _read(Runner({HEALTH: bad}), "vision_frame_sources"), client_mod.Unknown
    )
    assert isinstance(_read(Runner({}), "vision_frame_sources"), client_mod.Unknown)


def test_registered_app_absent_row_is_none_and_failed_read_is_unknown() -> None:
    assert _read(Runner({REGISTERED: _REGISTRY}), "registered_app", "nope") is None
    row = _read(Runner({REGISTERED: _REGISTRY}), "registered_app", WS_APP)
    assert row["transport"] == "websocket"
    assert isinstance(_read(Runner({}), "registered_app", WS_APP), client_mod.Unknown)
