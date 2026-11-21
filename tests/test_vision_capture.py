"""Screenshot tools capture through ``POST /ui-bridge/vision/capture``.

Plan 2026-11-19-the-ui-bridge-design-surface-is-blocked-behind-a-stalled-pr-
unpublished-packages-and-a-score, Phase 6.

``/ui-bridge/sdk/screenshot`` and ``/ui-bridge/control/annotated-screenshot``
were deleted from the runner on 2026-05-14 (runner commit 940fd0f40), so
every screenshot tool answered ``Not found``. They now capture through the runner's one capture route with
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
                "url": "/ui-bridge/vision/cache/" + "d" * 64,
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
    assert "GET http://127.0.0.1:1/ui-bridge/vision/cache/" + "d" * 64 in (
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
    # The default (size-bounded) contract does not follow the url.
    assert [p for p, _ in runner.requests if p.startswith(CACHE)] == []


CACHE = "/ui-bridge/vision/cache/"
_CACHE_URL = CACHE + "a" * 64


def _too_large(size: int = 9000000) -> httpx.Response:
    """The runner's 413; ``size`` is the encoded size it reports in hint.bytes."""
    return httpx.Response(
        413,
        json={
            "success": False,
            "error": f"capture is {size} bytes, over the 5242880-byte inline cap",
            "code": "INLINE_TOO_LARGE",
            "hint": {"url": _CACHE_URL, "bytes": size, "maxBytes": 5242880},
        },
    )


@pytest.mark.parametrize(
    "tool,routes,arguments",
    [
        ("sdk_screenshot_baseline_save", {}, {"baseline_key": "home"}),
        ("sdk_screenshot_before", {}, {}),
        ("sdk_screenshot_delta", {}, {}),
        (
            "sdk_element_screenshot",
            {"/ui-bridge/sdk/snapshot": _SNAPSHOT},
            {"element_id": "btn", "padding": 0},
        ),
    ],
)
def test_lossless_capture_over_the_inline_cap_is_read_from_the_cache(
    tool: str,
    routes: dict[str, Any],
    arguments: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A png_strict frame is not resized, so a big window answers 413.

    The capture was taken and cached; the tool reads it from ``hint.url``
    instead of failing, and does not capture again.
    """
    png = base64.b64decode(_png(40, 20))
    runner = Runner(
        {
            **routes,
            CAPTURE: _too_large(),
            _CACHE_URL: httpx.Response(
                200, content=png, headers={"content-type": "image/png"}
            ),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(tool, {**arguments, "target": "web"})
    assert not is_error, _text(items)
    assert [c["contract"] for c in runner.captures()] == ["png_strict"]
    assert [p for p, _ in runner.requests if p.startswith(CACHE)] == [_CACHE_URL]


def test_client_follows_hint_url_from_the_413_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = base64.b64decode(_png(40, 20))
    runner = Runner(
        {
            CAPTURE: _too_large(),
            _CACHE_URL: httpx.Response(
                200, content=png, headers={"content-type": "image/png"}
            ),
        }
    )
    c = _install(monkeypatch, runner)

    async def go() -> Any:
        try:
            return await c.vision_capture(
                target="web", contract="png_strict", follow_oversize=True
            )
        finally:
            await c.close()

    resp = asyncio.run(go())
    assert resp.success, resp.error
    assert (resp.data["width"], resp.data["height"]) == (40, 20)
    assert resp.data["format"] == "png"
    assert resp.data["sha256"] == "a" * 64
    assert base64.b64decode(resp.data["base64"]) == png


def test_annotation_with_no_viewport_returns_the_plain_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CSS viewport: the scale is UNKNOWN, so nothing is drawn."""
    snapshot = {"elements": _SNAPSHOT["elements"]}
    b64 = _png()
    runner = Runner({"/ui-bridge/control/snapshot": snapshot, CAPTURE: _frame(b64)})
    _install(monkeypatch, runner)
    items, is_error = _call("ui_annotated_screenshot", {"mode": "all"})
    assert not is_error, _text(items)
    assert "Annotation scale UNKNOWN" in _text(items)
    images = _images(items)
    assert len(images) == 1 and images[0].data == b64
    assert images[0].mimeType == "image/png"


def _jpeg() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_screenshot_mime_is_sniffed_from_the_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes win over a wrong ``format`` field."""
    frame = {**_frame(_jpeg()), "format": "png"}
    runner = Runner({CAPTURE: frame})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot", {"target": "web"})
    assert not is_error, _text(items)
    assert "jpeg" in _text(items)
    assert _images(items)[0].mimeType == "image/jpeg"


_NOT_AN_IMAGE = base64.b64encode(b"not an image at all").decode()


def test_screenshot_of_unknown_format_returns_no_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither bytes nor field name a format: UNKNOWN, and no ImageContent."""
    frame = {**_frame(_NOT_AN_IMAGE), "format": None}
    runner = Runner({CAPTURE: frame})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot", {"target": "web"})
    assert is_error
    assert "image format is UNKNOWN" in _text(items)
    assert _images(items) == []


@pytest.mark.parametrize("with_viewport", [True, False])
@pytest.mark.parametrize(
    "tool,snapshot_route,arguments",
    [
        ("ui_annotated_screenshot", "/ui-bridge/control/snapshot", {}),
        ("sdk_annotated_screenshot", "/ui-bridge/sdk/snapshot", {"target": "web"}),
    ],
)
def test_annotated_frame_of_unknown_format_is_an_error(
    tool: str,
    snapshot_route: str,
    arguments: dict[str, Any],
    with_viewport: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As for sdk_screenshot: no image of UNKNOWN format, and isError."""
    snapshot = _SNAPSHOT if with_viewport else {"elements": _SNAPSHOT["elements"]}
    frame = {**_frame(_NOT_AN_IMAGE), "format": None}
    runner = Runner({snapshot_route: snapshot, CAPTURE: frame})
    _install(monkeypatch, runner)
    items, is_error = _call(tool, arguments)
    assert is_error
    assert "Annotated screenshot: the frame's image format is UNKNOWN" in _text(items)
    assert _images(items) == []


def _big_png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _oversize_runner(routes: dict[str, Any], png: bytes, size: int = 9000000) -> Runner:
    return Runner(
        {
            **routes,
            CAPTURE: _too_large(size),
            _CACHE_URL: httpx.Response(
                200, content=png, headers={"content-type": "image/png"}
            ),
        }
    )


def test_delta_first_frame_over_the_cap_is_returned_downscaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _oversize_runner({}, _big_png(3200, 1600))
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_delta", {"target": "web"})
    assert not is_error, _text(items)
    assert "First capture (full image): 3200x1600px" in _text(items)
    assert "DOWNSCALED to 1568x784 from 3200x1600" in _text(items)
    # The note is the delta's own: no comparison ran on a first frame.
    assert "no comparison ran on this first frame" in _text(items)
    assert "the comparison ran at full resolution" not in _text(items)
    shown = Image.open(io.BytesIO(base64.b64decode(_images(items)[0].data)))
    assert shown.size == (1568, 784)


def test_after_composite_over_the_cap_is_returned_downscaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _oversize_runner({}, _big_png(2000, 1000))
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_before", {"target": "web"})
    assert not is_error, _text(items)
    items, is_error = _call("sdk_screenshot_after", {"target": "web"}, runner)
    assert not is_error, _text(items)
    assert "Change: 0.0%" in _text(items)
    assert "[comparison] NOTE:" in _text(items)
    assert "[diff] NOTE: this image is DOWNSCALED to 1568x784 from 2000x1000" in _text(
        items
    )
    assert "the comparison ran at full resolution" in _text(items)
    for image in _images(items):
        size = Image.open(io.BytesIO(base64.b64decode(image.data))).size
        assert max(size) <= 1568


def test_image_within_the_bound_is_not_downscaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image whose long edge is within 1568px is returned untouched."""
    b64 = _png(1500, 800)
    runner = Runner({CAPTURE: _frame(b64, 1500, 800)})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_delta", {"target": "web"})
    assert not is_error, _text(items)
    assert "DOWNSCALED" not in _text(items)
    assert _images(items)[0].data == b64


def test_inline_frame_over_the_bound_is_downscaled_by_its_own_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The returned image's own size decides, not whether it came inline."""
    runner = Runner({CAPTURE: _frame(_png(2000, 1000), 2000, 1000)})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_delta", {"target": "web"})
    assert not is_error, _text(items)
    assert "DOWNSCALED to 1568x784 from 2000x1000" in _text(items)


def test_diff_from_an_over_cap_baseline_and_an_inline_frame_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diff is as large as the stored baseline, though the frame is inline."""
    frames = iter(
        [
            # The baseline: over the inline cap, read from the cache.
            _too_large(),
            # The current frame: small enough to come back inline.
            {"success": True, "data": _frame(_png(400, 200), 400, 200)},
        ]
    )

    def capture(body: Any) -> httpx.Response:
        answer = next(frames)
        return (
            answer
            if isinstance(answer, httpx.Response)
            else httpx.Response(200, json=answer)
        )

    runner = Runner(
        {
            CAPTURE: capture,
            _CACHE_URL: httpx.Response(
                200,
                content=_big_png(3200, 1600),
                headers={"content-type": "image/png"},
            ),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert not is_error, _text(items)
    items, is_error = _call(
        "sdk_screenshot_diff", {"baseline_key": "home", "target": "web"}, runner
    )
    assert not is_error, _text(items)
    assert "NOTE: this image is DOWNSCALED" in _text(items)
    assert "the comparison ran at full resolution" in _text(items)
    size = Image.open(io.BytesIO(base64.b64decode(_images(items)[0].data))).size
    assert max(size) <= 1568


def test_cached_capture_read_is_capped_by_declared_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body whose content-length is over the cap is refused before reading."""
    import ui_bridge_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_CACHED_CAPTURE_BYTES", 1024)
    # hint.bytes under-reports (512), so only the transport check can fire.
    runner = _oversize_runner({}, _big_png(400, 400) + b"\0" * 4096, size=512)
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert is_error
    assert "more than MAX_CACHED_CAPTURE_BYTES=1024 bytes" in _text(items)
    assert "(declared content-length " in _text(items)
    assert "(counted while streaming)" not in _text(items)


class _Chunks(httpx.AsyncByteStream):
    """A chunked body with no content-length."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> Any:
        for chunk in self.chunks:
            yield chunk


def test_cached_capture_read_is_capped_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no content-length, the streaming count stops the read."""
    import ui_bridge_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_CACHED_CAPTURE_BYTES", 1024)
    # hint.bytes under-reports (512), so only the streaming count can fire.
    runner = Runner(
        {
            CAPTURE: _too_large(512),
            _CACHE_URL: httpx.Response(
                200,
                stream=_Chunks([b"\x89PNG" + b"\0" * 600, b"\0" * 600, b"\0" * 600]),
                headers={"content-type": "image/png"},
            ),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert is_error
    assert "more than MAX_CACHED_CAPTURE_BYTES=1024 bytes" in _text(items)
    assert "(counted while streaming)" in _text(items)
    assert "(declared content-length " not in _text(items)


def test_cache_read_error_body_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-200 cache answer is reported from at most 64 KiB of its body."""
    import ui_bridge_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_CACHE_ERROR_BODY_BYTES", 16)
    runner = Runner(
        {
            CAPTURE: _too_large(),
            _CACHE_URL: httpx.Response(
                404,
                stream=_Chunks([b"no cache entry for", b" " + b"z" * 5000]),
            ),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert is_error
    assert _text(items).endswith(
        "API error 404: no cache entry f (truncated at 16 bytes)"
    )
    assert "zzzz" not in _text(items)


def test_cache_read_error_body_of_exactly_the_cap_is_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body of exactly MAX_CACHE_ERROR_BODY_BYTES is whole, not cut."""
    import ui_bridge_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_CACHE_ERROR_BODY_BYTES", 16)
    body = b"0123456789abcdef"
    assert len(body) == 16
    runner = Runner(
        {
            CAPTURE: _too_large(),
            _CACHE_URL: httpx.Response(404, stream=_Chunks([body])),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert is_error
    assert _text(items).endswith("API error 404: 0123456789abcdef")
    assert "truncated" not in _text(items)


def test_cache_read_error_body_within_the_bound_is_not_marked_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        {
            CAPTURE: _too_large(),
            _CACHE_URL: httpx.Response(404, stream=_Chunks([b"no cache entry"])),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert is_error
    assert _text(items).endswith("API error 404: no cache entry")
    assert "truncated" not in _text(items)


def test_hint_bytes_over_the_cap_is_refused_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner's own hint.bytes refuses the read; the cache is never asked."""
    import ui_bridge_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_CACHED_CAPTURE_BYTES", 1024)
    runner = _oversize_runner({}, _big_png(40, 20))
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert is_error
    assert (
        "capture is 9000000 bytes (the runner's hint.bytes), over this client's "
        "MAX_CACHED_CAPTURE_BYTES=1024; it was not read"
    ) in _text(items)
    assert [p for p, _ in runner.requests if p.startswith(CACHE)] == []


def _too_large_hint(hint: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        413,
        json={
            "success": False,
            "error": "over the inline cap",
            "code": "INLINE_TOO_LARGE",
            "hint": hint,
        },
    )


@pytest.mark.parametrize(
    "hint",
    [
        {"url": _CACHE_URL, "bytes": None, "maxBytes": 5242880},
        {"url": _CACHE_URL, "maxBytes": 5242880},
        {"url": _CACHE_URL, "bytes": "9000000", "maxBytes": 5242880},
    ],
    ids=["bytes-null", "bytes-missing", "bytes-string"],
)
def test_hint_bytes_not_an_integer_does_not_refuse_the_read(
    hint: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an integer hint.bytes over the cap refuses; else the cache is read."""
    import ui_bridge_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_CACHED_CAPTURE_BYTES", 1024)
    runner = Runner(
        {
            CAPTURE: _too_large_hint(hint),
            _CACHE_URL: httpx.Response(
                200, content=_big_png(8, 4), headers={"content-type": "image/png"}
            ),
        }
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        "sdk_screenshot_baseline_save", {"baseline_key": "home", "target": "web"}
    )
    assert not is_error, _text(items)
    assert [p for p, _ in runner.requests if p.startswith(CACHE)] == [_CACHE_URL]


@pytest.mark.parametrize(
    "tool,snapshot_route,arguments",
    [
        ("ui_element_screenshot", "/ui-bridge/control/snapshot", {}),
        ("sdk_element_screenshot", "/ui-bridge/sdk/snapshot", {"target": "web"}),
    ],
)
@pytest.mark.parametrize(
    "element_size,scale,expected",
    [
        # A large element at scale 1.
        ((3000, 1000), 1.0, (1568, 523)),
        # A modest element pushed over the bound by scale > 1.
        ((1000, 500), 2.0, (1568, 784)),
    ],
)
def test_element_crop_over_the_bound_is_downscaled(
    tool: str,
    snapshot_route: str,
    arguments: dict[str, Any],
    element_size: tuple[int, int],
    scale: float,
    expected: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width, height = element_size
    snapshot = {
        "elements": [
            {
                "id": "big",
                "state": {"rect": {"x": 0, "y": 0, "width": width, "height": height}},
            }
        ],
        "viewport": {"viewportWidth": 3200, "viewportHeight": 1600},
    }
    runner = Runner(
        {snapshot_route: snapshot, CAPTURE: _frame(_png(3200, 1600), 3200, 1600)}
    )
    _install(monkeypatch, runner)
    items, is_error = _call(
        tool, {**arguments, "element_id": "big", "padding": 0, "scale": scale}
    )
    assert not is_error, _text(items)
    shown = Image.open(io.BytesIO(base64.b64decode(_images(items)[0].data)))
    assert shown.size == expected
    assert "the crop was taken from the full-resolution frame" in _text(items)
    assert "DOWNSCALED" in _text(items)


def test_element_crop_within_the_bound_carries_no_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner({"/ui-bridge/control/snapshot": _SNAPSHOT, CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call(
        "ui_element_screenshot", {"element_id": "btn", "padding": 0}
    )
    assert not is_error, _text(items)
    assert _text(items) == ""
    assert len(_images(items)) == 1


def _big_element_runner(width: int, height: int) -> Runner:
    snapshot = {
        "elements": [
            {
                "id": "big",
                "state": {"rect": {"x": 0, "y": 0, "width": width, "height": height}},
            }
        ],
        "viewport": {"viewportWidth": 3200, "viewportHeight": 1600},
    }
    return Runner(
        {
            "/ui-bridge/control/snapshot": snapshot,
            CAPTURE: _frame(_png(3200, 1600), 3200, 1600),
        }
    )


@pytest.mark.parametrize(
    "fmt,pil_format,mime",
    [("jpeg", "JPEG", "image/jpeg"), ("webp", "WEBP", "image/webp")],
)
@pytest.mark.parametrize("width,height", [(3000, 1000), (200, 100)])
def test_element_crop_format_matches_its_mime(
    fmt: str,
    pil_format: str,
    mime: str,
    width: int,
    height: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The returned bytes are the format the MIME type names, bounded or not."""
    runner = _big_element_runner(width, height)
    _install(monkeypatch, runner)
    items, is_error = _call(
        "ui_element_screenshot", {"element_id": "big", "padding": 0, "format": fmt}
    )
    assert not is_error, _text(items)
    image = _images(items)[0]
    assert image.mimeType == mime
    shown = Image.open(io.BytesIO(base64.b64decode(image.data)))
    assert shown.format == pil_format
    assert max(shown.size) <= 1568


@pytest.mark.parametrize("scale,expected", [(5.0, (1568, 784)), (0.01, (250, 125))])
def test_element_crop_scale_is_clamped(
    scale: float, expected: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scale is clamped to 0.25-2.0, as the annotated tools clamp it."""
    runner = _big_element_runner(1000, 500)
    _install(monkeypatch, runner)
    items, is_error = _call(
        "ui_element_screenshot", {"element_id": "big", "padding": 0, "scale": scale}
    )
    assert not is_error, _text(items)
    shown = Image.open(io.BytesIO(base64.b64decode(_images(items)[0].data)))
    # 5.0 clamps to 2.0 (2000x1000, then bounded to 1568x784); 0.01 to 0.25.
    assert shown.size == expected
    if scale > 2.0:
        # The crop the bound saw was the clamped 2000x1000, not 5000x2500:
        # the bound alone would hide a missing clamp.
        assert "from 2000x1000" in _text(items)


@pytest.mark.parametrize(
    "tool,snapshot_route,arguments",
    [
        ("ui_annotated_screenshot", "/ui-bridge/control/snapshot", {}),
        ("sdk_annotated_screenshot", "/ui-bridge/sdk/snapshot", {"target": "web"}),
    ],
)
def test_annotated_output_at_scale_2_is_bounded(
    tool: str,
    snapshot_route: str,
    arguments: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scale=2 on a 1568px frame would return 3136px; the bound holds."""
    snapshot = {**_SNAPSHOT, "viewport": {"viewportWidth": 1568, "viewportHeight": 784}}
    runner = Runner(
        {snapshot_route: snapshot, CAPTURE: _frame(_png(1568, 784), 1568, 784)}
    )
    _install(monkeypatch, runner)
    items, is_error = _call(tool, {**arguments, "scale": 2.0})
    assert not is_error, _text(items)
    # The note comes first.
    assert items[0].type == "text"
    assert items[0].text.startswith(
        "NOTE: this image is DOWNSCALED to 1568x784 from 3136x1568"
    )
    assert "annotation was drawn at full resolution" in items[0].text
    shown = Image.open(io.BytesIO(base64.b64decode(_images(items)[0].data)))
    assert shown.size == (1568, 784)


@pytest.mark.parametrize(
    "bad_url",
    [
        _CACHE_URL + "/../../control/snapshot",
        _CACHE_URL + "?x=1",
        CACHE + "../" + "a" * 61,
        CACHE + "a" * 63,
        "/ui-bridge/control/snapshot",
    ],
)
def test_hint_url_outside_the_cache_route_is_not_followed(
    bad_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    too_large = httpx.Response(
        413,
        json={
            "success": False,
            "error": "over the inline cap",
            "code": "INLINE_TOO_LARGE",
            "hint": {"url": bad_url, "bytes": 9, "maxBytes": 1},
        },
    )
    runner = Runner({CAPTURE: too_large})
    _install(monkeypatch, runner)
    items, is_error = _call("sdk_screenshot_before", {"target": "web"})
    assert is_error
    assert "no followable cache url" in _text(items)
    # Nothing but the one capture was requested.
    assert [p for p, _ in runner.requests] == [CAPTURE]


@pytest.mark.parametrize(
    "tool",
    [
        "ui_annotated_screenshot",
        "sdk_annotated_screenshot",
        "ui_element_screenshot",
        "sdk_element_screenshot",
        "sdk_screenshot_diff",
        "sdk_screenshot_baseline_save",
        "sdk_screenshot_before",
        "sdk_screenshot_after",
        "sdk_screenshot_delta",
        "sdk_visual_description",
    ],
)
def test_dropped_monitor_argument_is_refused(
    tool: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner({CAPTURE: _frame()})
    _install(monkeypatch, runner)
    items, is_error = _call(tool, {"monitor": 1, "target": "web", "element_id": "b"})
    assert is_error
    assert "cannot capture a monitor" in _text(items)
    assert runner.requests == []


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


@pytest.mark.parametrize(
    "inner",
    [
        # No hint key at all.
        {"success": False, "error": "no"},
        # An explicit null: the old `inner.get("hint", data.get("hint"))`
        # returned it instead of falling back.
        {"success": False, "error": "no", "hint": None},
    ],
)
def test_hint_falls_back_to_the_outer_envelope(
    inner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inner failure whose hint is missing or null takes the envelope's."""
    body = {"success": True, "hint": {"url": "outer"}, "data": inner}
    runner = Runner({CAPTURE: httpx.Response(200, json=body)})
    c = _install(monkeypatch, runner)

    async def go() -> Any:
        try:
            return await c.vision_capture()
        finally:
            await c.close()

    resp = asyncio.run(go())
    assert not resp.success
    assert resp.hint == {"url": "outer"}


@pytest.mark.parametrize(
    "body",
    [
        # A 2xx whose envelope says success: false.
        {"success": False, "error": "nope", "hint": {"url": "x"}},
        # A failure inside a success envelope.
        {
            "success": True,
            "data": {"success": False, "error": "no", "hint": {"url": "x"}},
        },
    ],
)
def test_hint_is_carried_on_a_2xx_failure(
    body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner({CAPTURE: httpx.Response(200, json=body)})
    c = _install(monkeypatch, runner)

    async def go() -> Any:
        try:
            return await c.vision_capture()
        finally:
            await c.close()

    resp = asyncio.run(go())
    assert not resp.success
    assert resp.hint == {"url": "x"}
