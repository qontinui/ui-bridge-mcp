"""execute-with-diff forwards the timeline request and prints its summary honestly.

Plan 2026-09-20-ui-bridge-represents-the-users-path-and-the-passage-of-time,
Phase 4 (ui-bridge-mcp). The SDK's ``ChangeTimeline`` summary is a set of
Observation envelopes; ``measured(0)``, ``absent`` and ``unknown`` are three
different statements and must render as three different lines. A missing
timeline is "opted out" only when the caller opted out.
"""

from __future__ import annotations

from typing import Any

import pytest

from ui_bridge_mcp.client import (
    CODE_APP_UNREACHABLE,
    CODE_PRODUCER_FAILED,
    UNKNOWN_CODES,
    extract_code,
)
from ui_bridge_mcp.timeline import (
    SUMMARY_NUMBER_FIELDS,
    TIMELINE_OPTED_OUT,
    format_timeline,
    render_observation,
)

from .mcp_session import answering, call, text

SDK_ROUTE = "/ui-bridge/sdk/ai/execute-with-diff"
CONTROL_ROUTE = "/ui-bridge/control/ai/execute-with-diff"
ROUTES = {"sdk_execute_with_diff": SDK_ROUTE, "ui_execute_with_diff": CONTROL_ROUTE}

# The seven UnknownCode members the SDK added with the timeline (ui-bridge#258).
NEW_CODES = (
    "frontier_open",
    "ledger_schema_absent",
    "ledger_write_failing",
    "no_observations",
    "to_node_unobserved",
    "performance_observer_unsupported",
    "no_event_before_timeout",
)


def _prov(unmeasured: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "producer": {"id": "sdk/change-timeline", "version": "0.34.0"},
        "observedAt": "2026-10-20T10:00:00Z",
        "evaluatedAt": "2026-10-20T10:00:00Z",
        "coverage": {"considered": 1, "measured": 1, "unmeasured": unmeasured or []},
        "confidence": None,
        "cache": None,
        "source": {"window": {"fromMs": 0, "toMs": 800}},
    }


def measured(value: Any, **kw: Any) -> dict[str, Any]:
    return {"status": "measured", "value": value, "provenance": _prov(**kw)}


def absent() -> dict[str, Any]:
    return {"status": "absent", "provenance": _prov()}


def unknown(code: str, detail: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "unknown": {"code": code, "detail": detail},
        "provenance": _prov(),
    }


def _summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "firstResponseMs": measured(42),
        "settleMs": measured(380),
        "layoutShiftTotal": measured(0),
        "layoutShiftCount": measured(0),
        "layoutShiftAfterFirstResponseMs": absent(),
        "animationsUnsettledAtEnd": measured([]),
        "longTaskTotalMs": unknown(
            "performance_observer_unsupported", "longtask entry type missing"
        ),
        "expectedStates": [],
    }
    summary.update(overrides)
    return summary


def _timeline(**summary_overrides: Any) -> dict[str, Any]:
    return {
        "events": [
            {"type": "action", "offsetMs": 0, "summary": "click"},
            {
                "type": "elements-appeared",
                "offsetMs": 42,
                "summary": "1 appeared",
                "elementIds": ["toast"],
                "count": 1,
            },
            {
                "type": "elements-modified",
                "offsetMs": 90,
                "summary": "2 modified",
                "elementIds": ["a", "b"],
                "count": 2,
            },
            {
                "type": "elements-appeared",
                "offsetMs": 120,
                "summary": "1 appeared",
                "elementIds": ["c"],
                "count": 1,
            },
            {"type": "settled", "offsetMs": 380, "summary": "settled"},
        ],
        "recordedMs": 800,
        "settled": True,
        "observedSources": ["snapshot-poll", "idle-detector", "layout-shift"],
        "summary": _summary(**summary_overrides),
    }


def _diff_result(timeline: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "actionSuccess": True,
        "actionResult": {"success": True},
        "diff": {"changes": {"appeared": [{}], "disappeared": [], "modified": []}},
        "budgetSummary": "B" * 300,
        "settleTimedOut": False,
        "durationMs": 812,
        "timestamp": 0,
    }
    if timeline is not None:
        result["timeline"] = timeline
    return result


def _line(out: str, label: str) -> str:
    matches = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(label)]
    assert len(matches) == 1, (label, out)
    return matches[0]


# =============================================================================
# render_observation: measured vs measured(0) vs absent vs unknown
# =============================================================================


def test_measured_zero_and_unknown_render_differently() -> None:
    assert render_observation(measured(312)) == "measured 312"
    assert render_observation(measured(0)) == "measured 0"
    assert render_observation(measured(0.0123)) == "measured 0.0123"
    assert render_observation(absent()) == "absent"
    out = render_observation(
        unknown("performance_observer_unsupported", "layout-shift unsupported")
    )
    assert out == "UNKNOWN (performance_observer_unsupported: layout-shift unsupported)"
    rendered = {
        render_observation(measured(312)),
        render_observation(measured(0)),
        render_observation(absent()),
        out,
    }
    assert len(rendered) == 4
    assert "0" not in out.split("(")[0]


def test_degraded_measurement_says_what_went_unmeasured() -> None:
    obs = measured(
        3,
        unmeasured=[
            {
                "dimension": "timeline-events:layout-shift",
                "count": 17,
                "code": "input_missing",
            }
        ],
    )
    assert render_observation(obs) == (
        "measured 3 [degraded: timeline-events:layout-shift x17 input_missing]"
    )


@pytest.mark.parametrize(
    "obs",
    [
        None,
        {},
        {"status": "measured"},
        {"status": "measured", "value": "fast"},
        {"status": "unknown"},
        {"status": "unknown", "unknown": {"detail": "no code given"}},
        {"status": "bogus"},
    ],
)
def test_malformed_envelopes_are_unknown_never_a_value(obs: Any) -> None:
    out = render_observation(obs)
    assert out.startswith("UNKNOWN (")
    assert "measured" not in out.split("(")[0]


# =============================================================================
# UnknownCode: the closed set carries the seven new members
# =============================================================================


def test_unknown_codes_mirror_the_sdk_closed_set() -> None:
    assert UNKNOWN_CODES == (
        "app_unreachable",
        "capability_unavailable_in_build",
        "producer_failed",
        "producer_not_run",
        "input_missing",
        "below_confidence_floor",
        "model_reply_unparseable",
        "needs_multi_frame_input",
        "stale_input",
        *NEW_CODES,
    )
    assert CODE_APP_UNREACHABLE in UNKNOWN_CODES
    assert CODE_PRODUCER_FAILED in UNKNOWN_CODES


@pytest.mark.parametrize("code", NEW_CODES)
def test_new_codes_render_as_unknown_with_their_code(code: str) -> None:
    out = render_observation(unknown(code, f"detail for {code}"))
    assert out == f"UNKNOWN ({code}: detail for {code})"
    # The body-level code extraction carries it as the typed code, too.
    assert extract_code({"status": "unknown", "unknown": {"code": code}}) == code
    assert extract_code({"success": True, "data": unknown(code, "x")}) == code


def test_code_outside_the_set_is_unknown_and_says_so() -> None:
    out = render_observation(unknown("brand_new_code", "from a newer SDK"))
    assert out == (
        "UNKNOWN (brand_new_code [not an UnknownCode this server knows]: "
        "from a newer SDK)"
    )


@pytest.mark.parametrize("code", NEW_CODES)
def test_new_code_in_a_tool_result_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    timeline = _timeline(firstResponseMs=unknown(code, "nothing responded"))
    answering(
        monkeypatch, {SDK_ROUTE: {"success": True, "data": _diff_result(timeline)}}
    )
    result = call("sdk_execute_with_diff", {"element_id": "save", "action": "click"})
    assert result.isError is False
    assert _line(text(result), "firstResponseMs:") == (
        f"firstResponseMs: UNKNOWN ({code}: nothing responded)"
    )


# =============================================================================
# format_timeline
# =============================================================================


def test_summary_fields_each_get_one_line() -> None:
    out = "\n".join(format_timeline(_timeline()))
    for name in SUMMARY_NUMBER_FIELDS:
        _line(out, f"{name}:")
    assert _line(out, "firstResponseMs:") == "firstResponseMs: measured 42"
    assert _line(out, "layoutShiftTotal:") == "layoutShiftTotal: measured 0"
    assert (
        _line(out, "layoutShiftAfterFirstResponseMs:")
        == "layoutShiftAfterFirstResponseMs: absent"
    )
    assert _line(out, "longTaskTotalMs:") == (
        "longTaskTotalMs: UNKNOWN (performance_observer_unsupported: "
        "longtask entry type missing)"
    )
    assert (
        _line(out, "animationsUnsettledAtEnd:")
        == "animationsUnsettledAtEnd: measured 0 animations"
    )
    assert _line(out, "Timeline:") == "Timeline: recorded 800ms, settled: true"
    assert _line(out, "observedSources:") == (
        "observedSources: snapshot-poll, idle-detector, layout-shift"
    )
    assert _line(out, "events:") == (
        "events: 5 (action 1, elements-appeared 2, elements-modified 1, settled 1)"
    )
    assert _line(out, "expectedStates:") == "expectedStates: none requested"


def test_running_animations_are_named() -> None:
    animations = [
        {"name": "fade-in", "isInfinite": False},
        {"name": "transition:opacity", "isInfinite": False},
    ]
    out = "\n".join(
        format_timeline(_timeline(animationsUnsettledAtEnd=measured(animations)))
    )
    assert _line(out, "animationsUnsettledAtEnd:") == (
        "animationsUnsettledAtEnd: measured 2 animations: fade-in, transition:opacity"
    )


def test_expect_states_miss_renders_not_entered_within_ms() -> None:
    expected = [
        {"stateId": "saved-toast", "outcome": measured({"notEnteredWithinMs": 2000})},
        {"stateId": "editor", "outcome": measured({"enteredAtMs": 140})},
        {"stateId": "shell", "outcome": measured({"activeThroughoutWithinMs": 2000})},
        {
            "stateId": "dialog",
            "outcome": unknown("input_missing", "no state registry"),
        },
    ]
    out = "\n".join(format_timeline(_timeline(expectedStates=expected)))
    assert _line(out, "saved-toast:") == (
        "saved-toast: measured notEnteredWithinMs=2000"
    )
    assert _line(out, "editor:") == "editor: measured enteredAtMs=140"
    assert _line(out, "shell:") == "shell: measured activeThroughoutWithinMs=2000"
    assert _line(out, "dialog:") == "dialog: UNKNOWN (input_missing: no state registry)"


def test_opted_out_timeline_says_so() -> None:
    assert format_timeline(None, requested=False) == [f"  {TIMELINE_OPTED_OUT}"]
    assert TIMELINE_OPTED_OUT == "Timeline: no timeline (opted out)"


def test_requested_but_missing_timeline_is_unknown_not_opted_out() -> None:
    out = "\n".join(format_timeline(None, requested=True))
    assert "opted out" not in out
    assert out.strip().startswith("Timeline: UNKNOWN (requested")


def test_missing_parts_of_a_timeline_are_unknown() -> None:
    out = "\n".join(format_timeline({"events": "x"}))
    assert _line(out, "Timeline:") == "Timeline: recorded UNKNOWN, settled: UNKNOWN"
    assert _line(out, "summary:").startswith("summary: UNKNOWN")
    assert _line(out, "observedSources:").startswith("observedSources: UNKNOWN")
    assert _line(out, "events:").startswith("events: UNKNOWN")


# =============================================================================
# The tools: what they send, and where the summary lands
# =============================================================================


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_timeline_is_requested_by_default(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    result = call(tool, {"element_id": "save", "action": "click"})
    assert result.isError is False
    (body,) = runner.bodies(route)
    assert body["timeline"] is True
    assert "expectStates" not in body and "withinMs" not in body


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_expect_states_and_window_are_forwarded_in_camel_case(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    call(
        tool,
        {
            "element_id": "save",
            "action": "click",
            "expect_states": ["saved-toast"],
            "within_ms": 45000,
            "timeline_interval": 50,
            "settle_timeout": 5000,
        },
    )
    (body,) = runner.bodies(route)
    assert body["expectStates"] == ["saved-toast"]
    assert body["withinMs"] == 45000
    assert body["timelineInterval"] == 50
    assert body["settleTimeout"] == 5000
    # The HTTP timeout outlasts the recording window (5 s settle + 45 s within).
    assert runner.read_timeouts[-1] is not None
    assert runner.read_timeouts[-1] >= 60.0


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_opt_out_is_forwarded_and_rendered(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(None)}}
    )
    result = call(tool, {"element_id": "save", "action": "click", "timeline": False})
    (body,) = runner.bodies(route)
    assert body["timeline"] is False
    assert TIMELINE_OPTED_OUT in text(result)


def test_summary_precedes_the_bulky_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(
        monkeypatch, {SDK_ROUTE: {"success": True, "data": _diff_result(_timeline())}}
    )
    out = text(call("sdk_execute_with_diff", {"element_id": "save", "action": "click"}))
    assert out.index("firstResponseMs:") < out.index("Appeared:")
    assert out.index("events:") < out.index("Summary: BBB")


def test_failed_action_keeps_its_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _diff_result(_timeline())
    data["actionSuccess"] = False
    body = {"success": False, "error": "element not found", "data": data}
    answering(monkeypatch, {SDK_ROUTE: body})
    result = call("sdk_execute_with_diff", {"element_id": "save", "action": "click"})
    assert result.isError is True
    out = text(result)
    assert "element not found" in out
    assert _line(out, "firstResponseMs:") == "firstResponseMs: measured 42"


def test_failed_action_without_timeline_is_the_plain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {SDK_ROUTE: {"success": False, "error": "boom"}})
    result = call("sdk_execute_with_diff", {"element_id": "save", "action": "click"})
    assert result.isError is True
    assert "Timeline" not in text(result)


# =============================================================================
# Review follow-ups on 1ebcd63
# =============================================================================


def test_unknown_codes_comment_names_the_sdk_twin_not_schemas_main() -> None:
    """The comment must not claim schemas main already carries the 16 codes."""
    import inspect

    import ui_bridge_mcp.client as client_module

    source = inspect.getsource(client_module)
    start = source.index("# The closed Observation ``UnknownCode`` set")
    comment = source[start : source.index("UNKNOWN_CODES: tuple", start)]
    assert "packages/ui-bridge/src/observation/observation.ts" in comment
    assert "twin of qontinui-schemas" not in comment
    assert "once it does" in comment


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_failed_action_with_data_but_no_timeline_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    data = _diff_result(None)
    data["actionSuccess"] = False
    answering(
        monkeypatch, {route: {"success": False, "error": "element gone", "data": data}}
    )
    result = call(tool, {"element_id": "save", "action": "click"})
    assert result.isError is True
    out = text(result)
    assert "element gone" in out
    assert _line(out, "Timeline:").startswith("Timeline: UNKNOWN (requested")


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_timeout_covers_two_settles_plus_the_window(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    call(
        tool,
        {
            "element_id": "save",
            "action": "click",
            "expect_states": ["saved-toast"],
            "within_ms": 30000,
            "settle_timeout": 20000,
        },
    )
    # The relay settles before the action and executeWithDiff again after it:
    # 2 * 20 s + 30 s + 10 s slack.
    assert runner.read_timeouts[-1] == pytest.approx(80.0)


def test_tool_descriptions_join_the_timeline_clause_with_a_comma() -> None:
    from ui_bridge_mcp import server

    descriptions = {
        t.name: t.description or ""
        for t in server.TOOLS
        if t.name in ("sdk_execute_with_diff", "ui_execute_with_diff")
    }
    assert len(descriptions) == 2
    for description in descriptions.values():
        assert "categorization, plus the interaction timeline" in description
        assert ". plus" not in description


@pytest.mark.parametrize(
    "outcome",
    [
        {"enteredAtMs": 140, "notEnteredWithinMs": 2000},
        {},
        {"enteredAtMs": "soon"},
    ],
)
def test_expected_state_outcome_not_exactly_one_numeric_key_is_unknown(
    outcome: dict[str, Any],
) -> None:
    expected = [{"stateId": "saved-toast", "outcome": measured(outcome)}]
    out = "\n".join(format_timeline(_timeline(expectedStates=expected)))
    assert _line(out, "saved-toast:").startswith("saved-toast: UNKNOWN (")
    assert "measured" not in _line(out, "saved-toast:").split("(")[0]


def test_count_unknowable_dimension_omits_the_count() -> None:
    obs = measured(
        [],
        unmeasured=[
            {
                "dimension": "animations-running-before-action(count-unknowable)",
                "count": 0,
                "code": "input_missing",
            }
        ],
    )
    out = "\n".join(format_timeline(_timeline(animationsUnsettledAtEnd=obs)))
    assert _line(out, "animationsUnsettledAtEnd:") == (
        "animationsUnsettledAtEnd: measured 0 animations [degraded: "
        "animations-running-before-action(count-unknowable) input_missing]"
    )


def test_non_number_count_renders_as_question_mark() -> None:
    obs = measured(
        3,
        unmeasured=[
            {"dimension": "timeline-events:layout-shift", "count": "many", "code": "x"}
        ],
    )
    assert render_observation(obs) == (
        "measured 3 [degraded: timeline-events:layout-shift x? x]"
    )


def test_requested_states_answered_with_empty_list_are_unknown() -> None:
    out = "\n".join(
        format_timeline(
            _timeline(expectedStates=[]), expected_state_ids=["saved-toast", "editor"]
        )
    )
    line = _line(out, "expectedStates:")
    assert line.startswith("expectedStates: UNKNOWN (requested saved-toast, editor")
    assert "none requested" not in out


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_tool_passes_requested_states_to_the_renderer(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    out = text(
        call(
            tool,
            {
                "element_id": "save",
                "action": "click",
                "expect_states": ["saved-toast"],
                "within_ms": 2000,
            },
        )
    )
    assert _line(out, "expectedStates:").startswith("expectedStates: UNKNOWN (")


def test_request_docstring_names_the_sdk_type() -> None:
    from ui_bridge_mcp import server

    doc = server._execute_with_diff_request.__doc__ or ""
    assert "ActionWithDiffRequest" in doc
    assert "ActionDiffRequest" not in doc


# =============================================================================
# The element action's value travels in elementAction.params
# =============================================================================


@pytest.mark.parametrize("tool", sorted(ROUTES))
@pytest.mark.parametrize(
    ("action", "key"), [("type", "text"), ("setValue", "value"), ("select", "value")]
)
def test_value_is_sent_as_the_param_the_sdk_action_reads(
    monkeypatch: pytest.MonkeyPatch, tool: str, action: str, key: str
) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    call(tool, {"element_id": "name", "action": action, "value": "Ada"})
    (body,) = runner.bodies(route)
    assert body["elementAction"] == {
        "elementId": "name",
        "action": action,
        "params": {key: "Ada"},
    }


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_no_value_sends_no_params(monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    call(tool, {"element_id": "save", "action": "click"})
    (body,) = runner.bodies(route)
    assert body["elementAction"] == {"elementId": "save", "action": "click"}
