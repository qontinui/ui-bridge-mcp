"""Render an executeWithDiff interaction timeline and its Observation envelopes.

The SDK's ``ChangeTimeline`` (ui-bridge ``ai/types.ts``) carries the events an
action produced, the sources the recording could observe, and a
``TimelineSummary`` whose every field is an Observation envelope:
``measured`` with a value, ``absent`` (the producer looked over the whole
window and the thing did not occur), or ``unknown`` with a closed
``UnknownCode`` (it could not look). ``measured(0)`` and ``unknown`` are
different shapes on the wire, and they render differently here: an unknown is
never printed as a bare ``0``.

Nothing here is a verdict. No value is called slow, janky or healthy, and no
threshold is applied; thresholds are the caller's.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .client import UNKNOWN_CODES

# ``TimelineSummary`` fields that are ``Observation<number>``, in declaration
# order (ui-bridge ``ai/types.ts``).
SUMMARY_NUMBER_FIELDS: tuple[str, ...] = (
    "firstResponseMs",
    "settleMs",
    "layoutShiftTotal",
    "layoutShiftCount",
    "layoutShiftAfterFirstResponseMs",
)
SUMMARY_ANIMATIONS_FIELD = "animationsUnsettledAtEnd"
SUMMARY_LONG_TASK_FIELD = "longTaskTotalMs"

# Every ``TimelineEventType``, in declaration order (ui-bridge ``ai/types.ts``).
TIMELINE_EVENT_TYPES: tuple[str, ...] = (
    "action",
    "elements-appeared",
    "elements-disappeared",
    "elements-modified",
    "page-changed",
    "settled",
    "layout-shift",
    "animation-started",
    "animation-ended",
    "long-task",
    "network-request-started",
    "network-request-finished",
    "state-entered",
    "state-exited",
)

# ``ExpectedStateOutcome``: exactly one of these keys is present.
EXPECTED_STATE_OUTCOME_KEYS: tuple[str, ...] = (
    "enteredAtMs",
    "notEnteredWithinMs",
    "activeThroughoutWithinMs",
)

TIMELINE_OPTED_OUT = "Timeline: no timeline (opted out)"

# A coverage dimension whose name carries this marker documents that how many
# things went unmeasured is unknowable; its ``count`` carries no meaning
# (ui-bridge ``ai/types.ts`` ``animationsUnsettledAtEnd``, e.g.
# ``animations-running-before-action(count-unknowable)``), so no ``xN`` is shown.
COUNT_UNKNOWABLE_MARKER = "(count-unknowable)"

_COMPACT_LIMIT = 200


def _compact(value: Any) -> str:
    text = json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    return text if len(text) <= _COMPACT_LIMIT else text[: _COMPACT_LIMIT - 1] + "…"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def format_number(value: Any) -> str:
    """A JSON number as written: ``0`` stays ``0``, ``0.0123`` stays ``0.0123``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _number_value(value: Any) -> str | None:
    return format_number(value) if _is_number(value) else None


def _degradation(observation: dict[str, Any]) -> str:
    """`` [degraded: …]`` when ``provenance.coverage.unmeasured`` is non-empty."""
    provenance = observation.get("provenance")
    coverage = provenance.get("coverage") if isinstance(provenance, dict) else None
    unmeasured = coverage.get("unmeasured") if isinstance(coverage, dict) else None
    if not isinstance(unmeasured, list) or not unmeasured:
        return ""
    parts = []
    for entry in unmeasured:
        if not isinstance(entry, dict):
            parts.append(_compact(entry))
            continue
        dimension = entry.get("dimension", "?")
        code = entry.get("code", "?")
        if isinstance(dimension, str) and COUNT_UNKNOWABLE_MARKER in dimension:
            parts.append(f"{dimension} {code}")
            continue
        count = entry.get("count")
        shown = format_number(count) if _is_number(count) else "?"
        parts.append(f"{dimension} x{shown} {code}")
    return f" [degraded: {'; '.join(parts)}]"


def render_unknown(unknown: Any) -> str:
    """``UNKNOWN (<code>: <detail>)``; a code outside ``UnknownCode`` says so."""
    if not isinstance(unknown, dict):
        return "UNKNOWN (no code: the envelope carried no 'unknown' object)"
    code = unknown.get("code")
    detail = unknown.get("detail")
    detail_text = detail if isinstance(detail, str) and detail else "no detail"
    if not isinstance(code, str) or not code:
        return f"UNKNOWN (no code: {detail_text})"
    if code not in UNKNOWN_CODES:
        return f"UNKNOWN ({code} [not an UnknownCode this server knows]: {detail_text})"
    return f"UNKNOWN ({code}: {detail_text})"


def render_observation(
    observation: Any, render_value: Callable[[Any], str | None] = _number_value
) -> str:
    """One Observation envelope, rendered honestly.

    ``measured <v>`` (with ``[degraded: …]`` when coverage is partial),
    ``absent``, or ``UNKNOWN (<code>: <detail>)``. ``render_value`` returns
    ``None`` for a value it cannot read, which renders UNKNOWN rather than a
    guess.
    """
    if not isinstance(observation, dict):
        return f"UNKNOWN (not an Observation envelope: {_compact(observation)})"
    status = observation.get("status")
    if status == "measured":
        if "value" not in observation:
            return "UNKNOWN (status 'measured' but the envelope carried no value)"
        shown = render_value(observation["value"])
        if shown is None:
            return (
                "UNKNOWN (status 'measured' but the value is not the expected "
                f"type: {_compact(observation['value'])})"
            )
        return f"measured {shown}{_degradation(observation)}"
    if status == "absent":
        return "absent"
    if status == "unknown":
        return render_unknown(observation.get("unknown"))
    return f"UNKNOWN (unrecognized Observation status {status!r})"


def _animations_value(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    names = [
        str(a.get("name", "?")) if isinstance(a, dict) else _compact(a) for a in value
    ]
    if not names:
        return "0 animations"
    shown = ", ".join(names[:5])
    more = f", ... and {len(names) - 5} more" if len(names) > 5 else ""
    return f"{len(names)} animation{'s' if len(names) != 1 else ''}: {shown}{more}"


def _expected_outcome_value(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    present = [k for k in EXPECTED_STATE_OUTCOME_KEYS if k in value]
    if len(present) != 1 or not _is_number(value[present[0]]):
        return None
    return f"{present[0]}={format_number(value[present[0]])}"


def _expected_states_lines(
    expected: Any, requested_ids: Sequence[str] | None
) -> list[str]:
    """The expectedStates block; ``requested_ids`` is what the caller asked for.

    An empty list is "none requested" only when nothing was requested. When
    states were requested and the producer answered none of them, that is
    UNKNOWN: it did not say what became of them. The SDK promises one entry
    per requested id (``ai/types.ts`` ``TimelineSummary.expectedStates``), so
    each requested id it did not answer is listed as UNKNOWN, not dropped.
    Answers to ids not requested follow in answer order, labelled
    ``(not requested)``; ``requested_ids=None`` (the caller did not say what
    it asked for) lists every answer in answer order, unlabelled.
    """
    if not isinstance(expected, list):
        return [
            "    expectedStates: UNKNOWN (the summary carried no expectedStates list)"
        ]
    if not expected:
        if requested_ids:
            return [
                "    expectedStates: UNKNOWN (requested "
                f"{', '.join(requested_ids)}, but the producer answered none of them)"
            ]
        return ["    expectedStates: none requested"]
    lines = ["    expectedStates:"]
    # Every answer, in answer order, with its stateId (``None`` when it carried
    # no usable one). A requested id consumes the first answer for it not yet
    # consumed (a multiset: an id requested twice needs two answers); whatever
    # is left keeps its answer order.
    answers: list[tuple[str | None, dict[str, Any]]] = []
    for entry in expected:
        result = entry if isinstance(entry, dict) else {}
        raw_id = result.get("stateId")
        answers.append((raw_id if isinstance(raw_id, str) and raw_id else None, result))
    consumed = [False] * len(answers)

    def _line(state_id: str, result: dict[str, Any], suffix: str = "") -> str:
        outcome = render_observation(result.get("outcome"), _expected_outcome_value)
        return f"      {state_id}: {outcome}{suffix}"

    for state_id in requested_ids or ():
        index = next(
            (
                i
                for i, (answered_id, _) in enumerate(answers)
                if not consumed[i] and answered_id == state_id
            ),
            None,
        )
        if index is None:
            lines.append(f"      {state_id}: UNKNOWN (requested, not answered)")
        else:
            consumed[index] = True
            lines.append(_line(state_id, answers[index][1]))
    # ``requested_ids=None`` means the caller did not say what it asked for, so
    # nothing can be called unrequested; otherwise a leftover answer is one.
    # An answer with no usable stateId cannot be called either: it says so.
    unrequested = "" if requested_ids is None else " (not requested)"
    for i, (answered_id, result) in enumerate(answers):
        if consumed[i]:
            continue
        if answered_id is None:
            lines.append(_line("?", result, " (no stateId)"))
        else:
            lines.append(_line(answered_id, result, unrequested))
    return lines


def _event_counts(events: Any) -> str:
    if not isinstance(events, list):
        return "UNKNOWN (the timeline carried no events list)"
    counts = Counter(
        str(e["type"]) if isinstance(e, dict) and "type" in e else "(untyped)"
        for e in events
    )
    ordered = [t for t in TIMELINE_EVENT_TYPES if counts.get(t)]
    ordered += sorted(t for t in counts if t not in TIMELINE_EVENT_TYPES)
    parts = [f"{t} {counts[t]}" for t in ordered]
    breakdown = f" ({', '.join(parts)})" if parts else ""
    return f"{len(events)}{breakdown}"


def format_timeline(
    timeline: Any,
    *,
    requested: bool = True,
    expected_state_ids: Sequence[str] | None,
) -> list[str]:
    """The timeline block of an executeWithDiff result, one line per fact.

    ``requested=False`` is the caller's ``timeline: false`` opt-out.
    ``expected_state_ids`` are the ``expectStates`` the caller sent. A
    requested timeline the response did not carry is UNKNOWN: the transport
    that answered did not record one, which says nothing about the page.
    """
    if not requested:
        return [f"  {TIMELINE_OPTED_OUT}"]
    if timeline is None:
        return [
            "  Timeline: UNKNOWN (requested, but the response carried none; the "
            "transport that answered did not record one)"
        ]
    if not isinstance(timeline, dict):
        return [f"  Timeline: UNKNOWN (not a timeline object: {_compact(timeline)})"]

    recorded = timeline.get("recordedMs")
    settled = timeline.get("settled")
    recorded_text = (
        f"{format_number(recorded)}ms" if _is_number(recorded) else "UNKNOWN"
    )
    settled_text = str(settled).lower() if isinstance(settled, bool) else "UNKNOWN"
    lines = [f"  Timeline: recorded {recorded_text}, settled: {settled_text}"]

    summary = timeline.get("summary")
    if isinstance(summary, dict):
        lines.append("    summary:")
        for name in SUMMARY_NUMBER_FIELDS:
            lines.append(f"      {name}: {render_observation(summary.get(name))}")
        lines.append(
            f"      {SUMMARY_ANIMATIONS_FIELD}: "
            f"{render_observation(summary.get(SUMMARY_ANIMATIONS_FIELD), _animations_value)}"
        )
        lines.append(
            f"      {SUMMARY_LONG_TASK_FIELD}: "
            f"{render_observation(summary.get(SUMMARY_LONG_TASK_FIELD))}"
        )
        lines.extend(
            "  " + line
            for line in _expected_states_lines(
                summary.get("expectedStates"), expected_state_ids
            )
        )
    else:
        lines.append("    summary: UNKNOWN (the timeline carried no summary)")

    sources = timeline.get("observedSources")
    if isinstance(sources, list):
        shown = ", ".join(str(s) for s in sources) if sources else "(none)"
        lines.append(f"    observedSources: {shown}")
    else:
        lines.append(
            "    observedSources: UNKNOWN (the timeline did not say what it observed)"
        )
    lines.append(f"    events: {_event_counts(timeline.get('events'))}")
    return lines
