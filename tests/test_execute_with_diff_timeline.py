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


def _state_lines(out: str) -> list[str]:
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "expectedStates:")
    block = []
    for ln in lines[start + 1 :]:
        if not ln.startswith("        "):
            break
        block.append(ln.strip())
    return block


ENTERED = measured({"enteredAtMs": 140})


def _states(requested: list[str] | None, answered: list[str]) -> list[str]:
    expected = [
        {"stateId": sid, "outcome": measured({"enteredAtMs": 10 * (i + 1)})}
        for i, sid in enumerate(answered)
    ]
    out = "\n".join(
        format_timeline(
            _timeline(expectedStates=expected), expected_state_ids=requested
        )
    )
    return _state_lines(out)


# =============================================================================
# Observation envelopes and UnknownCode
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
    out = "\n".join(
        format_timeline(
            _timeline(animationsUnsettledAtEnd=obs), expected_state_ids=None
        )
    )
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


# =============================================================================
# The timeline block
# =============================================================================


def test_summary_fields_each_get_one_line() -> None:
    out = "\n".join(format_timeline(_timeline(), expected_state_ids=None))
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
        format_timeline(
            _timeline(animationsUnsettledAtEnd=measured(animations)),
            expected_state_ids=None,
        )
    )
    assert _line(out, "animationsUnsettledAtEnd:") == (
        "animationsUnsettledAtEnd: measured 2 animations: fade-in, transition:opacity"
    )


def test_opted_out_timeline_says_so() -> None:
    assert format_timeline(None, requested=False, expected_state_ids=None) == [
        f"  {TIMELINE_OPTED_OUT}"
    ]
    assert TIMELINE_OPTED_OUT == "Timeline: no timeline (opted out)"


def test_requested_but_missing_timeline_is_unknown_not_opted_out() -> None:
    out = "\n".join(format_timeline(None, requested=True, expected_state_ids=None))
    assert "opted out" not in out
    assert out.strip().startswith("Timeline: UNKNOWN (requested")


def test_missing_parts_of_a_timeline_are_unknown() -> None:
    out = "\n".join(format_timeline({"events": "x"}, expected_state_ids=None))
    assert _line(out, "Timeline:") == "Timeline: recorded UNKNOWN, settled: UNKNOWN"
    assert _line(out, "summary:").startswith("summary: UNKNOWN")
    assert _line(out, "observedSources:").startswith("observedSources: UNKNOWN")
    assert _line(out, "events:").startswith("events: UNKNOWN")


# =============================================================================
# expectedStates: one line per requested id, in request order
# =============================================================================


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
    out = "\n".join(
        format_timeline(_timeline(expectedStates=expected), expected_state_ids=None)
    )
    assert _line(out, "saved-toast:") == (
        "saved-toast: measured notEnteredWithinMs=2000"
    )
    assert _line(out, "editor:") == "editor: measured enteredAtMs=140"
    assert _line(out, "shell:") == "shell: measured activeThroughoutWithinMs=2000"
    assert _line(out, "dialog:") == "dialog: UNKNOWN (input_missing: no state registry)"


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
    out = "\n".join(
        format_timeline(_timeline(expectedStates=expected), expected_state_ids=None)
    )
    assert _line(out, "saved-toast:").startswith("saved-toast: UNKNOWN (")
    assert "measured" not in _line(out, "saved-toast:").split("(")[0]


def test_requested_states_answered_with_empty_list_are_unknown() -> None:
    out = "\n".join(
        format_timeline(
            _timeline(expectedStates=[]), expected_state_ids=["saved-toast", "editor"]
        )
    )
    line = _line(out, "expectedStates:")
    assert line.startswith("expectedStates: UNKNOWN (requested saved-toast, editor")
    assert "none requested" not in out


@pytest.mark.parametrize(
    ("requested", "answered", "expected_lines"),
    [
        (
            ["a", "b", "c"],
            ["b"],
            [
                "a: UNKNOWN (requested, not answered)",
                "b: measured enteredAtMs=140",
                "c: UNKNOWN (requested, not answered)",
            ],
        ),
        (
            ["a", "a"],
            ["a"],
            [
                "a: measured enteredAtMs=140",
                "a: UNKNOWN (requested, not answered)",
            ],
        ),
        (
            ["a", "a"],
            ["x"],
            [
                "a: UNKNOWN (requested, not answered)",
                "a: UNKNOWN (requested, not answered)",
                "x: measured enteredAtMs=140 (not requested)",
            ],
        ),
    ],
)
def test_expected_states_follow_request_order_as_a_multiset(
    requested: list[str], answered: list[str], expected_lines: list[str]
) -> None:
    expected = [{"stateId": sid, "outcome": ENTERED} for sid in answered]
    out = "\n".join(
        format_timeline(
            _timeline(expectedStates=expected), expected_state_ids=requested
        )
    )
    assert _state_lines(out) == expected_lines


def test_unrequested_answers_keep_their_answer_order() -> None:
    assert _states(["a"], ["x", "y", "x"]) == [
        "a: UNKNOWN (requested, not answered)",
        "x: measured enteredAtMs=10 (not requested)",
        "y: measured enteredAtMs=20 (not requested)",
        "x: measured enteredAtMs=30 (not requested)",
    ]


def test_a_second_answer_to_a_single_request_is_labelled_not_requested() -> None:
    once = _states(["a"], ["a", "a"])
    twice = _states(["a", "a"], ["a", "a"])
    assert once == [
        "a: measured enteredAtMs=10",
        "a: measured enteredAtMs=20 (not requested)",
    ]
    assert twice == ["a: measured enteredAtMs=10", "a: measured enteredAtMs=20"]
    assert once != twice


def test_unrequested_answer_between_requested_ones() -> None:
    assert _states(["a", "b"], ["x", "a", "b"]) == [
        "a: measured enteredAtMs=20",
        "b: measured enteredAtMs=30",
        "x: measured enteredAtMs=10 (not requested)",
    ]


def test_answers_with_no_requested_ids_keep_arrival_order() -> None:
    assert _states(None, ["x", "y", "x"]) == [
        "x: measured enteredAtMs=10",
        "y: measured enteredAtMs=20",
        "x: measured enteredAtMs=30",
    ]


def test_answers_to_an_empty_request_are_all_not_requested() -> None:
    assert _states([], ["b", "a"]) == [
        "b: measured enteredAtMs=10 (not requested)",
        "a: measured enteredAtMs=20 (not requested)",
    ]


@pytest.mark.parametrize("requested", [None, [], ["a"]])
@pytest.mark.parametrize("entry", [{}, {"stateId": ""}, {"stateId": 7}, "garbage"])
def test_answer_without_a_usable_state_id_says_so(
    requested: list[str] | None, entry: Any
) -> None:
    if isinstance(entry, dict):
        entry = {**entry, "outcome": measured({"enteredAtMs": 5})}
    out = "\n".join(
        format_timeline(_timeline(expectedStates=[entry]), expected_state_ids=requested)
    )
    line = _line(out, "?:")
    assert line.endswith(" (no stateId)")
    assert "(not requested)" not in line
    if requested == ["a"]:
        assert _line(out, "a:") == "a: UNKNOWN (requested, not answered)"


def test_requested_question_mark_id_is_not_filled_by_an_answer_without_one() -> None:
    out = "\n".join(
        format_timeline(_timeline(expectedStates=[{}]), expected_state_ids=["?"])
    )
    assert _state_lines(out) == [
        "?: UNKNOWN (requested, not answered)",
        "?: UNKNOWN (not an Observation envelope: null) (no stateId)",
    ]


# =============================================================================
# Value routing: elementAction.params, and refusals
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


@pytest.mark.parametrize("tool", sorted(ROUTES))
@pytest.mark.parametrize("action", ["sendKeys", "click", "autocomplete"])
def test_value_on_an_action_that_does_not_read_it_is_refused(
    monkeypatch: pytest.MonkeyPatch, tool: str, action: str
) -> None:
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    result = call(tool, {"element_id": "box", "action": action, "value": "x"})
    assert result.isError is True
    out = text(result)
    assert f"'value' is not read by the built-in {action!r} action" in out
    assert "custom action registered under this name" in out
    assert "type, setValue, select" in out
    assert runner.bodies(route) == []


@pytest.mark.parametrize("tool", sorted(ROUTES))
@pytest.mark.parametrize("action", ["openRow", "terminal.paste", "setRating"])
def test_value_on_a_custom_action_is_passed_through(
    monkeypatch: pytest.MonkeyPatch, tool: str, action: str
) -> None:
    """A registered custom action receives the whole params object."""
    route = ROUTES[tool]
    runner = answering(
        monkeypatch, {route: {"success": True, "data": _diff_result(_timeline())}}
    )
    result = call(tool, {"element_id": "row", "action": action, "value": "42"})
    assert result.isError is False
    (body,) = runner.bodies(route)
    assert body["elementAction"] == {
        "elementId": "row",
        "action": action,
        "params": {"value": "42"},
    }


def test_builtin_refusal_set_is_the_sdk_switch_minus_value_actions() -> None:
    """The built-in switch in ui-bridge ``control/action-executor.ts``
    performAction, checked against ui-bridge origin/main d84bfa6ffe67, minus
    type/setValue/select."""
    from ui_bridge_mcp import server

    assert server.BUILTIN_ACTIONS_WITHOUT_VALUE == {
        "click",
        "hoverClick",
        "doubleClick",
        "rightClick",
        "middleClick",
        "sendKeys",
        "clear",
        "focus",
        "blur",
        "hover",
        "scroll",
        "scrollIntoView",
        "check",
        "uncheck",
        "toggle",
        "drag",
        "autocomplete",
        "submit",
        "reset",
    }
    assert not server.BUILTIN_ACTIONS_WITHOUT_VALUE & set(server.VALUE_ACTIONS)


def test_override_comment_names_every_fleet_override() -> None:
    import inspect

    from ui_bridge_mcp import server

    source = inspect.getsource(server)
    start = source.index("# An element may register a custom action")
    comment = source[start : source.index("BUILTIN_ACTIONS_WITHOUT_VALUE:", start)]
    assert "the one such override" not in comment
    for name in ("sendKeys", "focus", "blur", "TerminalBridgeProxies.tsx"):
        assert name in comment
    # It names what each override does with a value, not a silent no-op.
    assert "silent no-op too" not in comment
    assert 'takes no argument named "value"' in comment
    assert "'keys' is required" in comment
    assert "always refuse" in comment


# =============================================================================
# HTTP timeout
# =============================================================================


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


# =============================================================================
# Tool wiring: what the tools send, and where the summary lands
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


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_tool_without_expect_states_labels_answers_not_requested(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    route = ROUTES[tool]
    expected = [{"stateId": "x", "outcome": measured({"enteredAtMs": 5})}]
    answering(
        monkeypatch,
        {
            route: {
                "success": True,
                "data": _diff_result(_timeline(expectedStates=expected)),
            }
        },
    )
    out = text(call(tool, {"element_id": "save", "action": "click"}))
    assert _line(out, "x:") == "x: measured enteredAtMs=5 (not requested)"


def test_expected_state_ids_is_keyword_required() -> None:
    import inspect

    from ui_bridge_mcp import server

    for fn in (
        format_timeline,
        server._execute_with_diff_failure_text,
        server._execute_with_diff_result,
    ):
        param = inspect.signature(fn).parameters["expected_state_ids"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is inspect.Parameter.empty, fn.__name__


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


def test_request_docstring_names_the_sdk_type() -> None:
    from ui_bridge_mcp import server

    doc = server._execute_with_diff_request.__doc__ or ""
    assert "ActionWithDiffRequest" in doc
    assert "ActionDiffRequest" not in doc


# =============================================================================
# Failure path
# =============================================================================


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


def test_failed_action_without_timeline_is_the_plain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answering(monkeypatch, {SDK_ROUTE: {"success": False, "error": "boom"}})
    result = call("sdk_execute_with_diff", {"element_id": "save", "action": "click"})
    assert result.isError is True
    assert "Timeline" not in text(result)


@pytest.mark.parametrize("tool", sorted(ROUTES))
def test_inner_refusal_prints_no_timeline_line(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """A refusal unwrapped from a success envelope recorded nothing."""
    route = ROUTES[tool]
    refusal = {
        "success": False,
        "error": "expectStates requires withinMs",
        "code": "EXPECT_STATES_REQUIRES_WITHIN_MS",
    }
    answering(monkeypatch, {route: {"success": True, "data": refusal}})
    result = call(
        tool, {"element_id": "save", "action": "click", "expect_states": ["a"]}
    )
    assert result.isError is True
    out = text(result)
    assert "expectStates requires withinMs" in out
    assert "EXPECT_STATES_REQUIRES_WITHIN_MS" in out
    assert "Timeline" not in out
