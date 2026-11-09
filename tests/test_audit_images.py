"""``ui_audit_images`` and the projection it sends to ``/ui-bridge/vision/assert``.

Plan 2026-10-13-image-generation-and-visual-judgment, Phase 8 (ui-bridge-mcp). The
projection tests carry the fixtures of qontinui-claude-config
``scripts/test-uibridge-to-elementsnapshot.py`` as of this copy, with its real SDK
capture copied to ``tests/fixtures/``. Drift between the two copies is caught on the
other side, by qontinui-claude-config's ``scripts/test-media-projection-parity.py``,
which its CI runs against this repo's checkout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import types

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient
from ui_bridge_mcp.media_projection import (
    L0_IMAGE_ASSERTIONS,
    image_assertions,
    project_audit_row,
    project_audit_snapshot,
    src_digest,
    unread_image_ids,
)

FIXTURES = Path(__file__).parent / "fixtures"

SHARED_SRC = "http://localhost:3001/_next/image?url=%2Fimages%2Fhero.png&w=1920&q=75"
SHARED_DIGEST = "a846bcf412cf708ec0a4ac6a972307cbcb250b75509765fd44c49caecbd01f94"
SHARED_SURROGATE_SRC = "http://localhost:3001/img/\ud800.png"
SHARED_SURROGATE_DIGEST = (
    "6f45e1fb51856d9f4f25eea51558783cc2c060af2ba43ea46c8bdab5dbdf23a6"
)

IMAGE_FACTS: dict[str, Any] = {
    "mediaType": "image",
    "isDecorative": False,
    "renderedWidth": 100,
    "renderedHeight": 50,
    "src": SHARED_SRC,
    "altText": "Hero",
    "naturalWidth": 400,
    "naturalHeight": 200,
    "effectiveResolution": 2,
    "objectFit": "cover",
    "objectPosition": "50% 50%",
    "dimsReserved": True,
    "loading": "eager",
    "decoding": "async",
    "fetchPriority": "high",
    "loadingState": "loaded",
    "format": "webp",
    "encodedBodySize": 48213,
    "isLcpCandidate": True,
    "clsSources": [
        {"value": 0.25, "hadRecentInput": False},
        {"value": 0.5, "hadRecentInput": True},
    ],
    "slotId": "hero",
}


def media_row(media_metadata: Any, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "img",
        "tagName": "img",
        "category": "media",
        "accessibleName": "Alt text that is NOT painted",
        "state": {
            "inViewport": True,
            "textContent": "",
            "rect": {"x": 0, "y": 0, "width": 100, "height": 100},
        },
    }
    if media_metadata is not None:
        row["mediaMetadata"] = media_metadata
    for key, value in overrides.items():
        if key == "state":
            row["state"] = {**row["state"], **value}
        else:
            row[key] = value
    return row


def measured(
    value: dict[str, Any], unmeasured: list[str] | None = None
) -> dict[str, Any]:
    return {
        "status": "measured",
        "value": value,
        "provenance": {
            "coverage": {
                "unmeasured": [{"dimension": d} for d in (unmeasured or [])],
            },
        },
    }


def media_of(row: dict[str, Any]) -> dict[str, Any]:
    projected = project_audit_row(row)
    assert projected is not None
    media: dict[str, Any] = projected["media"]
    return media


# ---- the projection, against the converter's fixtures -----------------------


def test_src_digest_matches_the_shared_vectors() -> None:
    assert src_digest(SHARED_SRC) == SHARED_DIGEST
    assert src_digest(SHARED_SURROGATE_SRC) == SHARED_SURROGATE_DIGEST


def test_the_real_sdk_capture_projects_as_the_converter_projects_it() -> None:
    payload = json.loads(
        (FIXTURES / "sdk-find-media-response.json").read_text(encoding="utf-8")
    )
    rows = payload["elements"]
    assert media_of(rows[0]) == {
        "kind": "img",
        "in_viewport": True,
        "natural_px": {"w": 400.0, "h": 400.0},
        "rendered_px": {"w": 100.0, "h": 100.0},
        "src_digest": hashlib.sha256(b"http://localhost:3000/img/hero.png").hexdigest(),
        "effective_resolution": 2.0,
        "loading": "lazy",
        "decoding": "auto",
        "fetch_priority": "auto",
        "load_state": "loaded",
        "dims_reserved": True,
        "format": "png",
        "alt": {"decorative": False, "text": "Hero"},
        "slot_id": "hero",
        "placeholder": False,
    }


def test_a_full_record_projects_every_field() -> None:
    media = media_of(media_row(measured(IMAGE_FACTS)))
    assert media["object_fit"] == "cover"
    assert media["object_position"] == "50% 50%"
    assert media["transfer_bytes"] == 48213
    assert media["is_lcp_candidate"] is True
    assert media["cls_contribution"] == 0.25
    assert media["src_digest"] == SHARED_DIGEST


def test_an_unmeasured_dimension_is_omitted_even_with_a_value_beside_it() -> None:
    media = media_of(
        media_row(measured(IMAGE_FACTS, ["objectFit", "encodedBodySize", "clsSources"]))
    )
    for field in ("object_fit", "transfer_bytes", "cls_contribution"):
        assert field not in media


def test_a_missing_alt_is_measured_and_an_unread_alt_is_omitted() -> None:
    facts = {k: v for k, v in IMAGE_FACTS.items() if k != "altText"}
    assert media_of(media_row(measured(facts)))["alt"] == {"decorative": False}
    assert "alt" not in media_of(media_row(measured(IMAGE_FACTS, ["altText"])))


def test_a_redacted_row_omits_what_the_sentinel_replaced() -> None:
    redacted = media_row(
        measured(
            {
                **IMAGE_FACTS,
                "src": "[REDACTED]",
                "altText": "[REDACTED]",
                "slotId": "[REDACTED]",
            }
        ),
        state={"redaction": {"content": True}},
    )
    media = media_of(redacted)
    for field in ("src_digest", "alt", "slot_id"):
        assert field not in media
    assert media["natural_px"] == {"w": 400.0, "h": 200.0}


def test_an_unknown_envelope_is_kind_only() -> None:
    unknown = {"status": "unknown", "unknown": {"code": "producer_failed"}}
    assert media_of(media_row(unknown)) == {"kind": "img", "in_viewport": True}


def test_a_media_row_whose_read_failed_stays_with_no_media() -> None:
    unknown = {"status": "unknown", "unknown": {"code": "producer_failed"}}
    row = project_audit_row(media_row(unknown, tagName="div", id="bg"))
    assert row is not None
    assert row["id"] == "bg"
    assert "media" not in row
    # Another `category` does not hide it: the failed envelope says what it is.
    assert project_audit_row(media_row(unknown, tagName="div", category="content"))
    nested = {"id": "x", "tagName": "div", "state": {"mediaMetadata": unknown}}
    assert project_audit_row(nested)


def test_the_legacy_shape_projects_only_what_means_the_same() -> None:
    legacy = {
        "mediaType": "image",
        "src": "/img/a.png",
        "isDecorative": False,
        "altText": "A",
        "naturalWidth": 800,
        "naturalHeight": 600,
        "renderedWidth": 200,
        "renderedHeight": 150,
        "oversizeRatio": 4,
        "loadingState": "loaded",
        "lazyLoading": True,
        "format": "png",
        "transferSize": 0,
    }
    assert media_of(media_row(legacy)) == {
        "kind": "img",
        "in_viewport": True,
        "natural_px": {"w": 800.0, "h": 600.0},
        "rendered_px": {"w": 200.0, "h": 150.0},
        "src_digest": hashlib.sha256(b"/img/a.png").hexdigest(),
        "loading": "lazy",
        "load_state": "loaded",
        "format": "png",
        "alt": {"decorative": False, "text": "A"},
    }


@pytest.mark.parametrize(
    ("tag", "media_type", "kind"),
    [
        ("picture", "picture", "picture"),
        ("img", "picture", "picture"),
        ("svg", "svg", "svg"),
        ("video", "video", "video"),
        ("canvas", "canvas", "canvas"),
        ("div", "image", "role_img"),
        ("section", "background-image", "css_background"),
    ],
)
def test_kinds(tag: str, media_type: str, kind: str) -> None:
    facts = {"mediaType": media_type, "isDecorative": False}
    assert media_of(media_row(measured(facts), tagName=tag))["kind"] == kind


# ---- what the snapshot carries ----------------------------------------------


def test_the_snapshot_carries_only_media_rows_with_id_bbox_reason_and_media() -> None:
    hidden = media_row(
        measured(IMAGE_FACTS),
        id="hidden",
        state={
            "rect": {"x": -5.4, "y": 2.6, "width": -3, "height": 10},
            "visibilityReason": "hidden",
        },
    )
    text = {"id": "h1", "tagName": "h1", "category": "content", "state": {}}
    snapshot = project_audit_snapshot({"elements": [text, hidden, {"no": "id"}]})
    assert [row["id"] for row in snapshot["elements"]] == ["hidden"]
    row = snapshot["elements"][0]
    assert set(row) == {"id", "bbox", "visibility_reason", "media"}
    assert row["bbox"] == {"x": -5, "y": 3, "w": 0, "h": 10}
    assert row["visibility_reason"] == "hidden"


def test_a_partial_rect_is_dropped_rather_than_guessed() -> None:
    row = project_audit_row(
        media_row(measured(IMAGE_FACTS), state={"rect": {"x": 1, "y": 2, "width": 3}})
    )
    assert row is not None
    assert "bbox" not in row


@pytest.mark.parametrize("discover", [None, {}, {"elements": "nope"}, []])
def test_an_unreadable_reply_projects_no_rows(discover: Any) -> None:
    assert project_audit_snapshot(discover) == {"elements": []}


# ---- the assertions it asks for ---------------------------------------------


def test_every_check_is_asked_per_image_with_its_threshold() -> None:
    thresholds = {
        "min_effective": 1.0,
        "max_aspect_delta": 0.05,
        "max_effective": 2.0,
        "max_bytes": 1000,
    }
    got = image_assertions(["a", "b"], thresholds, per_image=True)
    assert len(got) == 2 * len(L0_IMAGE_ASSERTIONS) == 20
    assert {"type": "image_loaded", "element": "b"} in got
    assert {"type": "image_weight_within", "element": "a", "max_bytes": 1000} in got
    assert {"type": "image_not_oversized", "element": "b", "max_effective": 2.0} in got


def test_page_wide_asks_each_check_once_without_an_element() -> None:
    got = image_assertions(["a", "b"], {}, per_image=False)
    assert [a["type"] for a in got] == [kind for kind, _ in L0_IMAGE_ASSERTIONS]
    assert all("element" not in a for a in got)


def test_page_wide_also_names_each_image_whose_read_failed() -> None:
    got = image_assertions(["a", "bg"], {}, per_image=False, unread=["bg"])
    assert len(got) == 2 * len(L0_IMAGE_ASSERTIONS)
    assert {"type": "image_loaded"} in got
    assert {"type": "image_loaded", "element": "bg"} in got
    assert not any(a.get("element") == "a" for a in got)


def test_no_images_still_asks_page_wide_so_the_runner_answers_unknown() -> None:
    got = image_assertions([], {}, per_image=True)
    assert len(got) == len(L0_IMAGE_ASSERTIONS)
    assert all("element" not in a for a in got)


def test_an_omitted_threshold_is_not_invented() -> None:
    got = image_assertions(["a"], {"max_bytes": None}, per_image=True)
    weight = next(a for a in got if a["type"] == "image_weight_within")
    assert "max_bytes" not in weight


# ---- the tool ---------------------------------------------------------------

OUTCOMES = {
    "results": [{"passed": False, "outcome": "unknown", "code": "input_missing"}],
    "passed": 0,
    "failed": 0,
    "unknown": 1,
}


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, httpx.Response],
) -> list[tuple[str, dict[str, Any]]]:
    """Point the server at a client answering from ``routes``; returns the requests."""
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return routes[request.url.path]

    c = UIBridgeClient(host="127.0.0.1", port=1)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server_mod, "client", c)
    return seen


def _call(arguments: dict[str, Any]) -> tuple[str, bool]:
    async def go() -> Any:
        try:
            return await server_mod.call_tool("ui_audit_images", arguments)
        finally:
            if server_mod.client is not None:
                await server_mod.client.close()

    result = asyncio.run(go())
    is_error = isinstance(result, types.CallToolResult) and bool(result.isError)
    items = result.content if isinstance(result, types.CallToolResult) else result
    return "\n".join(getattr(item, "text", "") for item in items), is_error


def test_the_tool_is_listed_and_requires_no_threshold() -> None:
    tool = next(
        t for t in asyncio.run(server_mod.list_tools()) if t.name == "ui_audit_images"
    )
    assert tool.inputSchema["required"] == []
    assert set(tool.inputSchema["properties"]) == {
        "min_effective",
        "max_aspect_delta",
        "max_effective",
        "max_bytes",
        "per_image",
    }


def test_the_tool_discovers_media_and_returns_the_outcomes_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _serve(
        monkeypatch,
        {
            "/ui-bridge/control/discover": httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "elements": [
                            media_row(measured(IMAGE_FACTS), id="hero"),
                            {"id": "h1", "category": "content", "state": {}},
                        ]
                    },
                },
            ),
            "/ui-bridge/vision/assert": httpx.Response(
                200, json={"success": True, "data": OUTCOMES}
            ),
        },
    )

    text, is_error = _call({"max_bytes": 1000})

    assert not is_error
    assert json.loads(text) == OUTCOMES
    assert seen[0] == (
        "/ui-bridge/control/discover",
        {"interactive_only": False, "includeHidden": True, "includeMedia": True},
    )
    path, body = seen[1]
    assert path == "/ui-bridge/vision/assert"
    assert [row["id"] for row in body["snapshot"]["elements"]] == ["hero"]
    assert {
        "type": "image_weight_within",
        "element": "hero",
        "max_bytes": 1000,
    } in body["assertions"]
    assert {"type": "image_loaded", "element": "hero"} in body["assertions"]


def test_an_out_of_range_rect_is_dropped_rather_than_failing_the_snapshot() -> None:
    for rect in (
        {"x": 0, "y": 0, "width": 5e9, "height": 1},
        {"x": 3e9, "y": 0, "width": 1, "height": 1},
    ):
        row = project_audit_row(media_row(measured(IMAGE_FACTS), state={"rect": rect}))
        assert row is not None
        assert "bbox" not in row


def test_the_origin_keeps_the_whole_i32_range() -> None:
    # vision-core's Region origin is an i32, whose minimum is -(2**31): that value
    # parses on the runner, so it is kept. One past either end is dropped.
    for x, y, kept in (
        (-(2**31), 0, True),
        (0, -(2**31), True),
        (2**31 - 1, 0, True),
        (-(2**31) - 1, 0, False),
        (0, -(2**31) - 1, False),
        (2**31, 0, False),
    ):
        rect = {"x": x, "y": y, "width": 1, "height": 1}
        row = project_audit_row(media_row(measured(IMAGE_FACTS), state={"rect": rect}))
        assert row is not None
        assert ("bbox" in row) is kept, (x, y)


def test_a_huge_integer_is_no_measurement_and_never_a_crash() -> None:
    # A JSON integer past a float (`10**400`) parses as a Python int; float() raises
    # OverflowError on it, which once escaped the projection and failed the tool.
    huge = 10**400
    row = project_audit_row(
        media_row(
            measured(IMAGE_FACTS),
            state={"rect": {"x": huge, "y": 0, "width": 1, "height": 1}},
        )
    )
    assert row is not None
    assert "bbox" not in row
    for fact in (
        "naturalWidth",
        "naturalHeight",
        "renderedWidth",
        "renderedHeight",
        "encodedBodySize",
        "effectiveResolution",
    ):
        row = project_audit_row(media_row(measured({**IMAGE_FACTS, fact: huge})))
        assert row is not None and "media" in row, fact


def test_a_boolean_coordinate_is_dropped() -> None:
    row = project_audit_row(
        media_row(
            measured(IMAGE_FACTS),
            state={"rect": {"x": True, "y": 0, "width": 1, "height": 1}},
        )
    )
    assert row is not None
    assert "bbox" not in row


def test_unread_ids_are_every_media_row_with_no_readable_facts() -> None:
    unknown = {"status": "unknown", "unknown": {"code": "producer_failed"}}
    discover = {
        "elements": [
            media_row(measured(IMAGE_FACTS), id="hero"),
            media_row(unknown, id="dead"),
            media_row(unknown, id="bg", tagName="div"),
            {"id": "nested", "tagName": "img", "state": {"mediaMetadata": unknown}},
            {
                "id": "nested-ok",
                "tagName": "img",
                "state": {"mediaMetadata": measured(IMAGE_FACTS)},
            },
            {"id": "h1", "category": "content", "state": {}},
            {"no": "id"},
        ]
    }
    assert unread_image_ids(discover) == ["dead", "bg", "nested"]
    # Readable facts, but no kind to name: no `media`, so no check would count it.
    audio = media_row(measured({"mediaType": "audio"}), id="a", tagName="audio")
    assert unread_image_ids({"elements": [audio]}) == ["a"]
    # A junk top-level value falls through to the nested record, as the projection.
    junk = media_row("junk", id="d", state={"mediaMetadata": measured(IMAGE_FACTS)})
    assert unread_image_ids({"elements": [junk]}) == []
    assert unread_image_ids(None) == []


@pytest.mark.parametrize("per_image", [True, False])
def test_an_image_whose_read_failed_is_asked_about_by_name(
    monkeypatch: pytest.MonkeyPatch, per_image: bool
) -> None:
    unknown = {"status": "unknown", "unknown": {"code": "producer_failed"}}
    seen = _serve(
        monkeypatch,
        {
            "/ui-bridge/control/discover": httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "elements": [
                            media_row(measured(IMAGE_FACTS), id="hero"),
                            media_row(unknown, id="bg", tagName="div"),
                            media_row(unknown, id="dead"),
                        ]
                    },
                },
            ),
            "/ui-bridge/vision/assert": httpx.Response(
                200, json={"success": True, "data": OUTCOMES}
            ),
        },
    )

    _call({"per_image": per_image})

    _, body = seen[1]
    assert {"type": "image_loaded", "element": "bg"} in body["assertions"]
    assert {"type": "image_loaded", "element": "dead"} in body["assertions"]
    bg = next(row for row in body["snapshot"]["elements"] if row["id"] == "bg")
    assert "media" not in bg
    named = {a.get("element") for a in body["assertions"]}
    assert named == ({"hero", "bg", "dead"} if per_image else {None, "bg", "dead"})


def test_a_failed_discover_is_a_tool_error_and_asks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _serve(
        monkeypatch,
        {
            "/ui-bridge/control/discover": httpx.Response(
                200, json={"success": False, "error": "no webview"}
            ),
        },
    )
    text, is_error = _call({})
    assert is_error
    assert "no webview" in text
    assert [path for path, _ in seen] == ["/ui-bridge/control/discover"]


def test_a_failed_assert_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(
        monkeypatch,
        {
            "/ui-bridge/control/discover": httpx.Response(
                200, json={"success": True, "data": {"elements": []}}
            ),
            "/ui-bridge/vision/assert": httpx.Response(
                200, json={"success": False, "error": "bad snapshot"}
            ),
        },
    )
    text, is_error = _call({})
    assert is_error
    assert "bad snapshot" in text
