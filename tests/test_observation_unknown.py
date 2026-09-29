"""Absent is not clean, and failures carry a typed code.

Plan 2026-09-20-ui-bridge-observations-distinguish-cannot-see-from-not-present-
and-carry-provenance, Phase 3 (ui-bridge-mcp):

- ``generate_visual_description`` prints ``<label>: UNKNOWN (field absent from
  snapshot)`` for an absent field instead of the clean statement its default
  used to produce, and keeps the clean rendering for a present-but-empty field.
  Each changed site gets a pair: absent -> UNKNOWN, present -> no UNKNOWN for
  that label, and the two outputs differ.
- ``UIBridgeClient._request`` carries a machine-readable ``code`` and the HTTP
  ``status`` on every failure, and the tool-result text carries the code.
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import (
    CODE_APP_UNREACHABLE,
    CODE_PRODUCER_FAILED,
    UIBridgeClient,
    UIBridgeResponse,
    extract_code,
)
from ui_bridge_mcp.screenshot import ABSENT_FIELD, generate_visual_description

# =============================================================================
# Fixtures / helpers
# =============================================================================


def _element(eid: str, *, with_state: bool = True) -> dict[str, Any]:
    el: dict[str, Any] = {"id": eid, "type": "button", "category": "interactive"}
    if with_state:
        el["state"] = {
            "visible": True,
            "rect": {"x": 100, "y": 300, "width": 80, "height": 30},
        }
    return el


def _full_snapshot() -> dict[str, Any]:
    """A snapshot where every field the description reads is present and clean."""
    return {
        "elements": [_element("a"), _element("b")],
        "viewport": {
            "viewportWidth": 1280,
            "viewportHeight": 720,
            "scrollY": 0,
            "canScrollDown": False,
        },
        "page": {"title": "Home", "pathname": "/"},
        "modalStack": {"activeModals": []},
        "toasts": {"activeToasts": []},
        "errorSummary": {"health": "healthy", "errorCount": 0, "warningCount": 0},
    }


def _describe(snapshot: dict[str, Any]) -> str:
    raw = snapshot.get("elements")
    elements = raw if isinstance(raw, list) else None
    return generate_visual_description(elements, snapshot)


def _unknown(label: str) -> str:
    return f"{label}: {ABSENT_FIELD}"


# =============================================================================
# (a) generate_visual_description — one absent/present pair per changed site
# =============================================================================


def test_full_snapshot_has_no_unknown() -> None:
    desc = _describe(_full_snapshot())
    assert "UNKNOWN" not in desc
    assert "Viewport: 1280x720px" in desc
    assert "Elements: 2 visible" in desc


def test_error_summary_absent_vs_healthy() -> None:
    present = _full_snapshot()
    absent = copy.deepcopy(present)
    del absent["errorSummary"]

    out_absent = _describe(absent)
    out_present = _describe(present)

    assert "UNKNOWN" in out_absent
    assert _unknown("Health") in out_absent
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


def test_error_summary_present_without_health_key_is_unknown() -> None:
    snap = _full_snapshot()
    del snap["errorSummary"]["health"]
    assert _unknown("Health") in _describe(snap)


def test_error_summary_unhealthy_still_renders() -> None:
    snap = _full_snapshot()
    snap["errorSummary"] = {"health": "degraded", "errorCount": 2, "warningCount": 0}
    desc = _describe(snap)
    assert "Health: degraded" in desc
    assert "Errors: 2" in desc
    assert "UNKNOWN" not in desc


def test_unhealthy_counts_absent_vs_present() -> None:
    present = _full_snapshot()
    present["errorSummary"] = {"health": "degraded", "errorCount": 0, "warningCount": 0}
    absent = _full_snapshot()
    absent["errorSummary"] = {"health": "degraded"}

    out_absent = _describe(absent)
    out_present = _describe(present)

    assert _unknown("  Errors") in out_absent
    assert _unknown("  Warnings") in out_absent
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


def test_scroll_fields_absent_vs_present() -> None:
    present = _full_snapshot()
    absent = copy.deepcopy(present)
    del absent["viewport"]["scrollY"]
    del absent["viewport"]["canScrollDown"]

    out_absent = _describe(absent)
    out_present = _describe(present)

    assert _unknown("Scroll") in out_absent
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


def test_scroll_without_document_height_invents_no_percentage() -> None:
    snap = _full_snapshot()
    snap["viewport"] = {"viewportWidth": 800, "viewportHeight": 600, "scrollY": 500}
    desc = _describe(snap)
    assert "Scroll: 500px down" in desc
    assert "%" not in desc
    assert _unknown("  Document height") in desc
    assert _unknown("  More content below") in desc


def test_scroll_with_document_height_keeps_percentage() -> None:
    snap = _full_snapshot()
    snap["viewport"] = {
        "viewportWidth": 800,
        "viewportHeight": 600,
        "scrollY": 500,
        "documentHeight": 1600,
        "canScrollDown": True,
    }
    desc = _describe(snap)
    assert "Scroll: 50% down (500px)" in desc
    assert "More content below" in desc
    assert "UNKNOWN" not in desc


def test_scroll_position_absent_with_can_scroll_present() -> None:
    snap = _full_snapshot()
    snap["viewport"] = {
        "viewportWidth": 800,
        "viewportHeight": 600,
        "canScrollDown": True,
    }
    desc = _describe(snap)
    assert _unknown("Scroll position") in desc
    assert "  More content below" in desc


def test_state_null_is_unmeasured_not_a_crash() -> None:
    snap = _full_snapshot()
    snap["elements"] = [{"id": "a", "state": None}]
    desc = _describe(snap)
    assert "Elements: 0 visible" in desc
    assert _unknown("Geometry of 1 element") in desc


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("modalStack", "Modals"),
        ("toasts", "Toasts"),
    ],
)
def test_collection_container_absent_vs_present_empty(field: str, label: str) -> None:
    present = _full_snapshot()
    absent = copy.deepcopy(present)
    del absent[field]

    out_absent = _describe(absent)
    out_present = _describe(present)

    assert _unknown(label) in out_absent
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


@pytest.mark.parametrize(
    ("field", "key", "label"),
    [
        ("modalStack", "activeModals", "Modals"),
        ("toasts", "activeToasts", "Toasts"),
    ],
)
def test_collection_key_absent_inside_present_container(
    field: str, key: str, label: str
) -> None:
    snap = _full_snapshot()
    del snap[field][key]
    assert _unknown(label) in _describe(snap)


def test_present_modal_and_toasts_render_as_before() -> None:
    snap = _full_snapshot()
    snap["modalStack"] = {"activeModals": [{"id": "dlg", "title": "Confirm"}]}
    snap["toasts"] = {"activeToasts": [{"severity": "error", "text": "Boom"}]}
    desc = _describe(snap)
    assert "Modal open: Confirm" in desc
    assert "Active toasts: 1" in desc
    assert "[error] Boom" in desc
    assert "UNKNOWN" not in desc


def test_viewport_absent_vs_present() -> None:
    present = _full_snapshot()
    absent = copy.deepcopy(present)
    del absent["viewport"]

    out_absent = _describe(absent)
    out_present = _describe(present)

    # The fabricated 1920x1080 default is gone; so are layout and scroll
    # statements derived from it.
    assert _unknown("Viewport") in out_absent
    assert _unknown("Layout") in out_absent
    assert _unknown("Scroll") in out_absent
    assert "1920x1080" not in out_absent
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


def test_viewport_without_dimensions_is_unknown() -> None:
    snap = _full_snapshot()
    snap["viewport"] = {"scrollY": 0}
    desc = _describe(snap)
    assert _unknown("Viewport") in desc
    assert "1920x1080" not in desc


def test_elements_absent_vs_present_empty() -> None:
    present = _full_snapshot()
    present["elements"] = []
    absent = copy.deepcopy(present)
    del absent["elements"]

    out_absent = _describe(absent)
    out_present = _describe(present)

    assert _unknown("Elements") in out_absent
    assert _unknown("Layout") in out_absent
    assert "0 visible" not in out_absent
    assert "Interactive: 0" not in out_absent
    assert "Elements: 0 visible" in out_present
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


def test_elements_without_state_are_counted_not_dropped() -> None:
    present = _full_snapshot()
    absent = copy.deepcopy(present)
    absent["elements"] = [_element("a"), _element("b", with_state=False)]

    out_absent = _describe(absent)
    out_present = _describe(present)

    assert _unknown("Geometry of 1 element") in out_absent
    assert "UNKNOWN" not in out_present
    assert out_absent != out_present


def test_rect_key_absent_counts_but_rect_null_does_not() -> None:
    snap = _full_snapshot()
    snap["elements"] = [
        {"id": "a", "state": {"visible": True}},  # rect key absent -> unmeasured
        {"id": "b", "state": {"visible": True, "rect": None}},  # stated: no rect
        {"id": "c", "state": {"visible": False}},  # stated invisible
    ]
    desc = _describe(snap)
    assert _unknown("Geometry of 1 element") in desc


def test_page_null_does_not_crash() -> None:
    snap = _full_snapshot()
    snap["page"] = None
    assert "Page:" not in _describe(snap)


# =============================================================================
# (b) _request carries a typed code and the HTTP status
# =============================================================================


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


def _client(monkeypatch: pytest.MonkeyPatch, port: int = 1) -> UIBridgeClient:
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    return UIBridgeClient(host="127.0.0.1", port=port)


def _mocked(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> UIBridgeClient:
    c = _client(monkeypatch)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def _run(c: UIBridgeClient, method: str = "GET") -> UIBridgeResponse:
    async def go() -> UIBridgeResponse:
        try:
            return await c._request(method, "/ui-bridge/sdk/snapshot")
        finally:
            await c.close()

    return asyncio.run(go())


def test_unreachable_port_is_app_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _run(_client(monkeypatch, _closed_port()))
    assert resp.success is False
    assert resp.code == CODE_APP_UNREACHABLE == "app_unreachable"
    assert resp.status is None


def test_read_timeout_is_producer_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.success is False
    assert resp.code == CODE_PRODUCER_FAILED == "producer_failed"
    assert resp.error is not None and "timeout" in resp.error
    assert resp.status is None


def test_connect_timeout_is_app_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.code == CODE_APP_UNREACHABLE
    assert resp.error is not None and "timeout" in resp.error


def test_http_500_keeps_body_code(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"success": False, "error": "x", "code": "UB-NET-ERROR"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=body)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.success is False
    assert resp.code == "UB-NET-ERROR"
    assert resp.status == 500
    assert resp.error is not None and "x" in resp.error


def test_http_500_runner_error_detail_code(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "success": False,
        "error": "runner said no",
        "error_detail": {"code": "RUNNER_REQUIRED", "message": "runner said no"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json=body)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.code == "RUNNER_REQUIRED"
    assert resp.status == 503


def test_http_500_observation_envelope_code(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "status": "unknown",
        "unknown": {"code": "input_missing", "detail": "no elements key"},
        "provenance": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=body)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.code == "input_missing"
    assert resp.status == 500


def test_http_500_unparseable_body_is_producer_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.code == CODE_PRODUCER_FAILED
    assert resp.status == 502
    assert resp.error is not None and "bad gateway" in resp.error


def test_200_without_success_key_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"x": 1}})

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.success is False
    assert resp.status == 200


def test_200_non_object_body_is_producer_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.success is False
    assert resp.code == CODE_PRODUCER_FAILED


def test_200_wrapped_unknown_envelope_surfaces_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "success": True,
        "data": {
            "status": "unknown",
            "unknown": {"code": "producer_not_run", "detail": "no components"},
            "provenance": {},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.success is True
    assert resp.code == "producer_not_run"
    assert resp.status == 200


def test_extract_code_precedence_and_absence() -> None:
    assert extract_code(None) is None
    assert extract_code({"success": True, "data": {"status": "measured"}}) is None
    assert extract_code({"code": "A", "error_code": "B"}) == "A"
    assert extract_code({"error_code": "B"}) == "B"
    assert (
        extract_code(
            {
                "error_detail": {"code": "C"},
                "data": {"status": "unknown", "unknown": {"code": "D"}},
            }
        )
        == "C"
    )


def test_describe_error_prefixes_code_and_status() -> None:
    r = UIBridgeResponse(success=False, error="boom", code="UB-NET-ERROR", status=500)
    assert r.describe_error() == "[code=UB-NET-ERROR status=500] boom"
    assert UIBridgeResponse(success=False, error="plain").describe_error() == "plain"


# =============================================================================
# (c) the tool result carries the typed code
# =============================================================================


def _call_tool(name: str, arguments: dict[str, Any]) -> str:
    async def go() -> str:
        try:
            result = await server_mod.call_tool(name, arguments)
        finally:
            if server_mod.client is not None:
                await server_mod.client.close()
        return "\n".join(getattr(item, "text", "") for item in result)

    return asyncio.run(go())


def test_tool_result_carries_app_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch, _closed_port()))
    text = _call_tool("ui_health", {})
    assert "code=app_unreachable" in text


def test_tool_result_carries_body_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"success": False, "error": "x", "code": "UB-NET-ERROR"}
        )

    monkeypatch.setattr(server_mod, "client", _mocked(monkeypatch, handler))
    text = _call_tool("sdk_visual_description", {})
    assert "code=UB-NET-ERROR" in text
    assert "status=500" in text


def test_visual_description_tool_renders_absent_fields_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot with no errorSummary/elements never renders as healthy/empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        snap = {"viewport": {"viewportWidth": 800, "viewportHeight": 600}}
        return httpx.Response(200, content=json.dumps({"success": True, "data": snap}))

    monkeypatch.setattr(server_mod, "client", _mocked(monkeypatch, handler))
    text = _call_tool("sdk_visual_description", {})
    assert _unknown("Health") in text
    assert _unknown("Elements") in text
    assert "0 visible" not in text


def test_tool_error_result_carries_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that fails through ``_error_result`` (ui_snapshot) shows the code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"success": False, "error": "x", "code": "UB-NET-ERROR"}
        )

    monkeypatch.setattr(server_mod, "client", _mocked(monkeypatch, handler))
    text = _call_tool("ui_snapshot", {})
    assert text.startswith("Error: [code=UB-NET-ERROR status=500]")


def test_http_422_runner_body_keeps_recovery_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "success": False,
        "error": "Element 'btn-sav' not found",
        "code": "ELEMENT_NOT_FOUND",
        "hint": {"closestMatch": "btn-save", "distance": 1},
        "suggestions": ["Re-take a snapshot", "Use btn-save"],
        "error_detail": {
            "code": "ELEMENT_NOT_FOUND",
            "message": "Element 'btn-sav' not found",
            "recovery": "RESNAPSHOT",
            "context": {"element_id": "btn-sav"},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.code == "ELEMENT_NOT_FOUND"
    assert resp.status == 422
    assert resp.error is not None
    lines = resp.error.splitlines()
    assert lines[0] == "API error 422: Element 'btn-sav' not found"
    assert 'hint: {"closestMatch":"btn-save","distance":1}' in lines
    assert "suggestions: Re-take a snapshot; Use btn-save" in lines
    assert "recovery: RESNAPSHOT" in lines
    assert 'context: {"element_id":"btn-sav"}' in lines


def test_http_error_unparseable_body_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="E" * 5000)

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.error is not None
    assert len(resp.error) < 700
    assert "(5000 bytes)" in resp.error


def test_200_success_false_without_code_is_producer_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"success": False, "error": "nope", "suggestions": ["retry"]}
        )

    resp = _run(_mocked(monkeypatch, handler))
    assert resp.success is False
    assert resp.code == CODE_PRODUCER_FAILED
    assert resp.error == "nope\nsuggestions: retry"


def test_unsupported_method_carries_code(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _run(_client(monkeypatch), method="PATCH")
    assert resp.success is False
    assert resp.code == CODE_PRODUCER_FAILED
