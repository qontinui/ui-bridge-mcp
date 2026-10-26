"""sdk_ai_search reads the runner's UNWRAPPED search response.

qontinui-runner's handle_ai_search (sdk_client.rs) serves the bare search
response with no ``success`` envelope: the SDK's aiSearch object
(``{results, total, scannedCount, timestamp}``) on the SDK path, and
``engine.search()``'s SearchResponse (``{results, bestMatch, scannedCount,
durationMs, criteria, timestamp}``) on the IPC fallback. Each result wraps the
element: ``{element, confidence, matchReasons, scores}``.
"""

from typing import Any

import pytest

from tests.mcp_session import answering, call

PATH = "/ui-bridge/sdk/ai/search"


def _hit(eid: str, role: str | None = None, conf: float = 0.9) -> dict[str, Any]:
    el: dict[str, Any] = {
        "id": eid,
        "type": "button",
        "label": eid.title(),
        "state": {"visible": True, "enabled": True},
    }
    if role:
        el["type"] = "text"
        el["category"] = "content"
        el["contentMetadata"] = {"contentRole": role}
    return {"element": el, "confidence": conf, "matchReasons": ["text"], "scores": {}}


def _sdk_body(results: list[Any]) -> dict[str, Any]:
    return {
        "results": results,
        "total": len(results),
        "scannedCount": 10,
        "timestamp": 1,
    }


def _ipc_body(results: list[Any]) -> dict[str, Any]:
    return {
        "results": results,
        "bestMatch": results[0] if results else None,
        "scannedCount": 10,
        "durationMs": 3,
        "criteria": {"text": "x"},
        "timestamp": 1,
    }


def _text(result: Any) -> str:
    return "".join(c.text for c in result.content if hasattr(c, "text"))


@pytest.mark.parametrize("shape", [_sdk_body, _ipc_body])
def test_bare_results_are_listed_with_confidence(
    monkeypatch: pytest.MonkeyPatch, shape: Any
) -> None:
    answering(monkeypatch, {PATH: shape([_hit("save", conf=0.87)])})
    result = call("sdk_ai_search", {"text": "save"})
    out = _text(result)
    assert not result.isError
    assert "Found 1 element(s) matching 'save'" in out
    assert "- save (button)" in out
    assert "(confidence 0.87)" in out
    assert "unknown" not in out.lower()


def test_empty_results_is_no_elements_found(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {PATH: _sdk_body([])})
    out = _text(call("sdk_ai_search", {"text": "save"}))
    assert "No elements found matching: save" in out


def test_body_without_results_or_success_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Neither the envelope nor the bare list: still a failure, never "none found".
    answering(monkeypatch, {PATH: {"total": 0}})
    result = call("sdk_ai_search", {"text": "save"})
    assert result.isError
    assert "No elements found" not in _text(result)


def test_unreadable_results_are_unknown_not_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {PATH: _sdk_body([{"confidence": 0.5}, "junk"])})
    out = _text(call("sdk_ai_search", {"text": "save"}))
    assert "UNKNOWN" in out
    assert "No elements found" not in out


def test_partly_unreadable_results_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {PATH: _sdk_body([_hit("save"), "junk"])})
    out = _text(call("sdk_ai_search", {"text": "save"}))
    assert "Found 1 element(s)" in out
    assert "1 result(s) carried no readable element." in out


def test_content_role_filter_reads_the_element(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(
        monkeypatch,
        {
            PATH: _sdk_body(
                [_hit("title", role="heading"), _hit("body", role="paragraph")]
            )
        },
    )
    out = _text(call("sdk_ai_search", {"text": "x", "content_role": "heading"}))
    assert "Found 1 element(s)" in out
    assert "title" in out
    assert "body" not in out


def test_content_types_filter_keeps_matching_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tool schema restricts content_types to content-role names.
    answering(
        monkeypatch,
        {PATH: _sdk_body([_hit("save"), _hit("title", role="heading")])},
    )
    out = _text(call("sdk_ai_search", {"text": "x", "content_types": ["heading"]}))
    assert "Found 1 element(s)" in out
    assert "title" in out
    assert "- save" not in out


def test_malformed_content_metadata_is_unjudged_not_a_non_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A contentMetadata that is not an object says nothing about the role, so
    # whether the element matches the filter is UNKNOWN, not "none found".
    hit = _hit("save")
    hit["element"]["contentMetadata"] = "not-a-dict"
    answering(monkeypatch, {PATH: _sdk_body([hit])})
    result = call("sdk_ai_search", {"text": "x", "content_role": "heading"})
    assert result.isError
    assert "UNKNOWN" in _text(result)
    assert "No elements found" not in _text(result)


def test_filtered_out_with_unreadable_results_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {PATH: _sdk_body([_hit("save"), "junk"])})
    result = call("sdk_ai_search", {"text": "x", "content_role": "heading"})
    out = _text(result)
    assert result.isError
    assert "UNKNOWN" in out
    assert "No elements found" not in out
