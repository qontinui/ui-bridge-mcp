"""Render the runner's journey query answers (``GET /apps/{app_id}/journey/*``).

Plan ``2026-09-20-ui-bridge-represents-the-users-path-and-the-passage-of-time``,
Phase 2 (the queries) and Phase 4 (``edge-timing``). The runner's shapes are
``qontinui-runner`` ``src-tauri/src/journey/routes.rs`` and the shared types in
``qontinui-schemas`` ``rust/src/journey/query.rs`` and
``rust-vision-core/src/edge_timing.rs``.

Every route but ``health`` answers the shared Observation envelope:

- ``measured`` — a POSITIVE answer (a path found, a node listed); its value is
  the query result, and the gaps it was computed over ride in
  ``provenance.coverage.unmeasured`` (a degraded measurement).
- ``absent`` — a SETTLED negative: the searched part was fully explored under
  this filter and the thing was not there (``not_reachable``, no dead end).
- ``unknown`` — the negative could NOT be settled (an open frontier, a node
  never observed, a ledger that is not being written, an empty ledger, a
  failed read). It is never a negative, and it renders unmistakably
  differently from ``absent``. The query result rides in
  ``provenance.source.result``, so the frontier entries and departures that
  would settle it are printed with it.

The shared ``JourneyCoverage`` block rides inside ``provenance.coverage``. The
COVERAGE line is printed FIRST, before any answer, so a truncated tool result
still carries it.

Nothing here is a verdict: no threshold, no "slow", no "healthy". An edge's
timing is a distribution with its sample size, never a single latency.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from .timeline import format_number, format_unmeasured, render_observation

# The filter every journey query route takes (runner ``JourneyParams``).
FILTER_ARGS: tuple[str, ...] = ("since", "app_version", "run_kind")
RUN_KINDS: tuple[str, ...] = ("agent_action", "explorer", "passive_session")

# ``FilterEcho`` wire keys, in the order the filter line prints them.
_ECHO_KEYS: tuple[tuple[str, str], ...] = (
    ("since", "since"),
    ("app_version", "appVersion"),
    ("run_kind", "runKind"),
)

# ``EdgeTiming`` metrics, in declaration order (``edge_timing.rs``).
TIMING_METRICS: tuple[str, ...] = (
    "firstResponseMs",
    "settleMs",
    "layoutShiftTotal",
    "layoutShiftCount",
    "layoutShiftAfterFirstResponseMs",
    "longTaskTotalMs",
)

# ``JourneyUnmeasuredDepartures`` fields, with the dimension each is reported
# under (``JourneySearchFrontier::unmeasured``).
_DEPARTURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("rejectedFrontier", "rejected_frontier"),
    ("settleTimeout", "settle_timeout_departures"),
    ("rejected", "rejected_departures"),
    ("errored", "errored_departures"),
    ("filteredOut", "filtered_departures"),
    ("withdrawn", "withdrawn_departures"),
    ("excludedBackArrivals", "excluded_back_arrivals"),
    ("excludedUrlArrivals", "excluded_url_arrivals"),
)

DEFAULT_LIST_LIMIT = 50
_COMPACT_LIMIT = 200


def _compact(value: Any) -> str:
    text = json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    return text if len(text) <= _COMPACT_LIMIT else text[: _COMPACT_LIMIT - 1] + "…"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _count(value: Any) -> str:
    """A count as served; a missing or non-numeric one is UNKNOWN, never 0."""
    return format_number(value) if _is_number(value) else "UNKNOWN"


def _limited(
    items: Sequence[Any], limit: int, render: Callable[[Any], str], indent: str
) -> list[str]:
    lines = [f"{indent}{render(item)}" for item in items[:limit]]
    if len(items) > limit:
        lines.append(
            f"{indent}... and {len(items) - limit} more (not listed; raise `limit`)"
        )
    return lines


# =============================================================================
# Envelope access
# =============================================================================


class JourneyAnswer:
    """The parts of one journey envelope a renderer reads.

    ``result`` is the query result whatever the status: the ``value`` of a
    measured answer, ``provenance.source.result`` otherwise (absent when the
    read itself failed).
    """

    def __init__(self, body: Any) -> None:
        self.body = _dict(body)
        self.status = self.body.get("status")
        self.provenance = _dict(self.body.get("provenance"))
        self.coverage = _dict(self.provenance.get("coverage"))
        self.source = _dict(self.provenance.get("source"))
        if self.status == "measured":
            self.result = _dict(self.body.get("value"))
        else:
            self.result = _dict(self.source.get("result"))

    @property
    def unmeasured(self) -> list[Any] | None:
        """``coverage.unmeasured``; ``None`` when the envelope did not carry it."""
        raw = self.coverage.get("unmeasured")
        return raw if isinstance(raw, list) else None

    @property
    def settled(self) -> bool:
        """True only when the envelope REPORTED an empty ``unmeasured`` list."""
        return self.unmeasured == []

    def ledger(self) -> dict[str, Any] | None:
        for holder in (self.coverage, self.result, self.source):
            ledger = holder.get("ledger")
            if isinstance(ledger, dict):
                return ledger
        return None

    def filter_echo(self) -> dict[str, Any] | None:
        for holder in (self.result, self.source):
            echo = holder.get("filterEcho")
            if isinstance(echo, dict):
                return echo
        return None


def _ledger_text(ledger: dict[str, Any] | None) -> str:
    if ledger is None:
        return "ledger=UNKNOWN (the answer carried no ledger state)"
    state = ledger.get("state")
    state_text = state if isinstance(state, str) and state else "UNKNOWN"
    if state == "writing":
        return f"ledger={state_text}"
    detail = ledger.get("detail")
    suffix = f" ({detail})" if isinstance(detail, str) and detail else ""
    return f"ledger={state_text}{suffix}"


def _unmeasured_text(answer: JourneyAnswer) -> str:
    unmeasured = answer.unmeasured
    if unmeasured is None:
        return "unmeasured: UNKNOWN (the answer did not report what it left out)"
    if not unmeasured:
        return "unmeasured: none"
    return f"unmeasured: {'; '.join(format_unmeasured(unmeasured))}"


def _same_filter_value(arg: str, asked: str, applied: Any) -> bool:
    """Whether the runner applied what was asked; ``since`` compares as instants.

    The runner echoes ``since`` re-serialized in UTC, so ``+02:00`` comes back
    as ``Z`` two hours earlier: the same filter, a different string.
    """
    if arg == "since" and isinstance(applied, str):
        try:
            return datetime.fromisoformat(asked) == datetime.fromisoformat(applied)
        except ValueError:
            pass
    return bool(applied == asked)


def _filter_text(echo: dict[str, Any] | None, requested: dict[str, str] | None) -> str:
    if echo is None:
        return (
            "filter: UNKNOWN (the answer echoed no filter, so which filter was "
            "applied cannot be confirmed)"
        )
    parts = []
    for arg, key in _ECHO_KEYS:
        value = echo.get(key)
        parts.append(f"{arg}={value if value is not None else 'none'}")
    text = f"filter: {' '.join(parts)}"
    dropped = []
    for arg, key in _ECHO_KEYS:
        asked = (requested or {}).get(arg)
        if asked is not None and not _same_filter_value(arg, asked, echo.get(key)):
            dropped.append(f"{arg} asked {asked!r}, applied {echo.get(key)!r}")
    if dropped:
        text += f" [FILTER NOT APPLIED AS ASKED: {'; '.join(dropped)}]"
    return text


def _window_text(oldest: Any, newest: Any, what: str) -> str:
    if oldest is None and newest is None:
        return f"{what} window: none (no observation)"
    return f"{what} window: {oldest} .. {newest}"


def _app_versions_text(versions: Any) -> str:
    if not isinstance(versions, list):
        return "appVersions=UNKNOWN"
    shown = ", ".join("(not reported)" if v is None else str(v) for v in versions)
    return f"appVersions=[{shown}]"


def _by_reason(counts: Any, *, nonzero_only: bool = True) -> str:
    if not isinstance(counts, dict):
        return "by reason UNKNOWN"
    parts = [
        f"{reason} {format_number(n)}"
        for reason, n in sorted(counts.items())
        if not nonzero_only or (_is_number(n) and n != 0) or not _is_number(n)
    ]
    return ", ".join(parts) if parts else "none"


# ``provenance.coverage.unmeasured`` dimensions the runner adds for a gap in
# its own READ of the ledger (``routes.rs`` ``ReadGaps::dimensions``), as
# opposed to a gap in the graph: a truncated edge or frontier read, or rows
# that did not parse. Any listed item may be wrong when one is present.
_READ_GAP_PREFIXES: tuple[str, ...] = (
    "edges_beyond_read_bound_",
    "frontier_beyond_read_bound_",
    "rows_unparseable",
)


def read_gaps(answer: JourneyAnswer) -> list[Any]:
    """The unmeasured entries that name a gap in the ledger read itself."""
    return [
        d
        for d in answer.unmeasured or []
        if isinstance(d, dict)
        and isinstance(d.get("dimension"), str)
        and d["dimension"].startswith(_READ_GAP_PREFIXES)
    ]


# Graph-wide doubt counters (``JourneyGraph``): rows that may arrive at any
# node. Read from wherever the answer carries them.
_GRAPH_DOUBT_KEYS: tuple[str, ...] = ("untrustedEdgeRows", "unknownDestinationArrivals")


def _graph_doubt(answer: JourneyAnswer) -> list[str]:
    holders = [answer.coverage, answer.result]
    holders += [_dict(u) for u in _list(answer.result.get("unsettled"))[:1]]
    parts = []
    for key in _GRAPH_DOUBT_KEYS:
        value = next((h[key] for h in holders if key in h), None)
        if _is_number(value) and value:
            parts.append(f"{key}={format_number(value)}")
    return parts


def graph_coverage_line(answer: JourneyAnswer, requested: dict[str, str] | None) -> str:
    """The one-line COVERAGE header of a graph-backed answer."""
    cov = answer.coverage
    parts = [_ledger_text(answer.ledger())]
    if "edgesObserved" in cov:
        parts.append(
            f"edgesObserved={_count(cov.get('edgesObserved'))} "
            f"(toNodeUnobserved {_count(cov.get('edgesToNodeUnobserved'))}, "
            f"filteredOut {_count(cov.get('edgesFilteredOut'))}, "
            f"withdrawn {_count(cov.get('edgesInvalidated'))}, "
            f"rowsRejected {_count(cov.get('rowsRejected'))})"
        )
        parts.append(
            f"nodes={_count(cov.get('nodesObserved'))} "
            f"(unmodelled {_count(cov.get('nodesUnmodelled'))})"
        )
        parts.append(
            f"frontierOpen={_count(cov.get('frontierOpen'))} "
            f"at {_count(cov.get('frontierNodes'))} node(s) "
            f"({_by_reason(cov.get('frontierByReason'))})"
        )
        parts.extend(_graph_doubt(answer))
    else:
        parts.append(
            "edgesObserved=UNKNOWN nodes=UNKNOWN frontierOpen=UNKNOWN (no counts: "
            "the ledger could not be read, so nothing was counted - not zero)"
        )
    parts.append(_unmeasured_text(answer))
    parts.append(_filter_text(answer.filter_echo(), requested))
    if "edgesObserved" in cov:
        parts.append(f"runs={_count(cov.get('runs'))}")
        parts.append(
            _window_text(cov.get("oldestEdgeAt"), cov.get("newestEdgeAt"), "edge")
        )
        parts.append(_app_versions_text(cov.get("appVersions")))
    return "COVERAGE: " + " | ".join(parts)


# =============================================================================
# The answer line
# =============================================================================


def _unknown_line(answer: JourneyAnswer, not_this: str) -> str:
    unknown = _dict(answer.body.get("unknown"))
    code = unknown.get("code")
    code_text = code if isinstance(code, str) and code else "no code"
    detail = unknown.get("detail")
    detail_text = detail if isinstance(detail, str) and detail else "no detail"
    return (
        f"ANSWER: UNKNOWN ({code_text}) - NOT a negative: this does NOT mean "
        f"{not_this}. {detail_text}"
    )


_GRAPH_DEGRADED = (
    "computed over a graph with unmeasured parts - see COVERAGE; anything NOT "
    "listed below is unsettled, not ruled out"
)


def answer_line(
    answer: JourneyAnswer,
    *,
    positive: str,
    negative: str,
    not_this: str,
    degraded: str = _GRAPH_DEGRADED,
) -> str:
    """``ANSWER: MEASURED`` / ``ANSWER: ABSENT`` / ``ANSWER: UNKNOWN``.

    ``positive`` describes a measured answer, ``negative`` a settled absent
    one; ``not_this`` names the negative an unknown must not be read as.
    """
    if answer.status == "measured":
        suffix = "" if answer.settled else f" [DEGRADED: {degraded}]"
        return f"ANSWER: MEASURED - {positive}{suffix}"
    if answer.status == "absent":
        return (
            f"ANSWER: ABSENT - {negative} (a SETTLED negative: every part the "
            "question searched was fully explored under this filter)"
        )
    if answer.status == "unknown":
        return _unknown_line(answer, not_this)
    return (
        f"ANSWER: UNKNOWN (not an Observation envelope: status {answer.status!r}) - "
        f"NOT a negative: this does NOT mean {not_this}."
    )


# =============================================================================
# The unsettled part (what would have to be explored)
# =============================================================================


def _frontier_entry(entry: Any) -> str:
    e = _dict(entry)
    role = e.get("affordanceRole") or "role not reported"
    effect = e.get("declaredEffect") or "effect undeclared"
    return (
        f"{e.get('nodeKey', '?')} affordance {e.get('affordanceFingerprint', '?')} "
        f"({role}; {effect}) reason={e.get('reason', '?')} "
        f"lastSeen={e.get('lastSeenAt', '?')}"
    )


def _departures_text(departures: Any) -> str:
    d = _dict(departures)
    parts = [
        f"{dimension} {format_number(d[key])}"
        for key, dimension in _DEPARTURE_FIELDS
        if _is_number(d.get(key)) and d[key] != 0
    ]
    return ", ".join(parts)


def unsettled_lines(search: Any, limit: int, heading: str) -> list[str]:
    """A ``JourneySearchFrontier``: every node, affordance and departure left open."""
    frontier = _dict(search)
    if not frontier:
        return []
    nodes = _list(frontier.get("nodes"))
    lines: list[str] = []
    for raw in nodes[:limit]:
        node = _dict(raw)
        key = node.get("nodeKey", "?")
        facts = []
        unobserved = node.get("unobservedDepartures")
        if _is_number(unobserved) and unobserved:
            facts.append(
                f"{format_number(unobserved)} departure(s) whose destination was "
                "never observed"
            )
        departures = _departures_text(node.get("unmeasuredDepartures"))
        if departures:
            facts.append(f"unmeasured departures: {departures}")
        open_by_reason = _dict(node.get("openByReason"))
        open_count = sum(len(_list(v)) for v in open_by_reason.values())
        if open_count:
            facts.append(f"{open_count} open affordance(s)")
        lines.append(f"    {key}: {'; '.join(facts) if facts else 'listed'}")
        for reason in sorted(open_by_reason):
            for entry in _list(open_by_reason[reason])[:limit]:
                e = _dict(entry)
                role = e.get("affordanceRole") or "role not reported"
                effect = e.get("declaredEffect") or "effect undeclared"
                lines.append(
                    f"      explore: affordance {e.get('affordanceFingerprint', '?')} "
                    f"({role}; {effect}) reason={reason}"
                )
            hidden = len(_list(open_by_reason[reason])) - limit
            if hidden > 0:
                lines.append(f"      ... and {hidden} more {reason} (raise `limit`)")
    if len(nodes) > limit:
        lines.append(f"    ... and {len(nodes) - limit} more node(s) (raise `limit`)")
    for key, label in (
        ("unplacedRejectedRows", "untrusted edge row(s) placeable at no node"),
        ("unplacedRejectedFrontier", "rejected frontier row(s) placeable at no node"),
    ):
        n = frontier.get(key)
        if _is_number(n) and n:
            lines.append(f"    {format_number(n)} {label} (could be anywhere)")
    off_graph = _departures_text(frontier.get("offGraphDepartures"))
    if off_graph:
        lines.append(f"    off-graph unmeasured departures: {off_graph}")
    if not lines:
        return []
    return [heading, *lines]


_SETTLE_HEADING = "Unsettled part (explore these to settle the question):"


# =============================================================================
# Per-query renderers
# =============================================================================


def _render(
    body: Any,
    requested: dict[str, str] | None,
    *,
    positive: Callable[[JourneyAnswer], str],
    negative: str,
    not_this: str,
    details: Callable[[JourneyAnswer], list[str]],
    limit: int,
) -> str:
    answer = JourneyAnswer(body)
    lines = [
        graph_coverage_line(answer, requested),
        answer_line(
            answer, positive=positive(answer), negative=negative, not_this=not_this
        ),
    ]
    lines.extend(_gap_lines(answer))
    lines.extend(details(answer))
    if not answer.settled:
        lines.extend(
            unsettled_lines(answer.result.get("frontier"), limit, _SETTLE_HEADING)
        )
    return "\n".join(lines)


def _gap_lines(answer: JourneyAnswer) -> list[str]:
    """Name a gap in the ledger READ, which no exploration can settle."""
    gaps = read_gaps(answer)
    if not gaps:
        return []
    return [
        f"  Read gap: {'; '.join(format_unmeasured(gaps))} - the ledger was not "
        "read in full, so nothing below is settled (narrow with `since`)."
    ]


def _ledger_unknown_lines(answer: JourneyAnswer) -> list[str]:
    """Why an answer that never reached the graph is unknown."""
    if answer.result:
        return []
    return [
        "  No query result: the ledger was not read (see the ANSWER detail). "
        "Nothing here is an empty graph."
    ]


def format_reachable(
    body: Any, from_key: str, requested: dict[str, str] | None, limit: int
) -> str:
    def details(answer: JourneyAnswer) -> list[str]:
        r = answer.result
        if not r:
            return _ledger_unknown_lines(answer)
        lines = []
        if r.get("fromObserved") is False:
            lines.append(
                f"  Start node {from_key!r} was never observed under this filter "
                "(unknown, not 'reaches nothing')."
            )
        reached = sorted(
            _list(r.get("reachable")),
            key=lambda n: (_dict(n).get("steps", 0), str(_dict(n).get("nodeKey"))),
        )
        if reached:
            lines.append(f"  Reached ({len(reached)}, start included at 0 steps):")
            lines.extend(
                _limited(
                    reached,
                    limit,
                    lambda n: f"[{_count(_dict(n).get('steps'))}] {_dict(n).get('nodeKey', '?')}",
                    "    ",
                )
            )
        return lines

    def positive(answer: JourneyAnswer) -> str:
        reached = _list(answer.result.get("reachable"))
        others = sum(1 for n in reached if _dict(n).get("nodeKey") != from_key)
        return f"{from_key!r} reaches {others} other node(s)"

    return _render(
        body,
        requested,
        positive=positive,
        negative=f"{from_key!r} reaches no other node: NOT_REACHABLE for every other node",
        not_this=f"that {from_key!r} reaches nothing",
        details=details,
        limit=limit,
    )


def _edge_text(edge: Any) -> str:
    e = _dict(edge)
    trigger = _dict(e.get("trigger"))
    target = []
    if trigger.get("targetRole"):
        target.append(f"role={trigger['targetRole']}")
    if trigger.get("targetFingerprint"):
        target.append(f"target={trigger['targetFingerprint']}")
    if trigger.get("declaredEffect"):
        target.append(f"effect={trigger['declaredEffect']}")
    target_text = f" [{', '.join(target)}]" if target else ""
    return (
        f"{e.get('fromKey', '?')} -> {e.get('toKey', '?')} via "
        f"{trigger.get('actionType', '?')}/{trigger.get('navigationTrigger', '?')}"
        f"{target_text} outcome={e.get('outcome', '?')} "
        f"seen {_count(e.get('observations'))}x"
    )


def format_steps(
    body: Any,
    from_key: str,
    to_key: str,
    requested: dict[str, str] | None,
    limit: int,
) -> str:
    def details(answer: JourneyAnswer) -> list[str]:
        r = answer.result
        if not r:
            return _ledger_unknown_lines(answer)
        lines = []
        for flag, key in (("fromObserved", from_key), ("toObserved", to_key)):
            if r.get(flag) is False:
                lines.append(
                    f"  {key!r} was never observed under this filter (so no path "
                    "to or from it can be ruled out)."
                )
        path = _list(r.get("path"))
        if path:
            lines.append("  Path (one shortest):")
            lines.extend(_limited(path, limit, _edge_text, "    "))
        return lines

    def positive(answer: JourneyAnswer) -> str:
        return (
            f"{to_key!r} is reachable from {from_key!r} in "
            f"{_count(answer.result.get('steps'))} step(s)"
        )

    return _render(
        body,
        requested,
        positive=positive,
        negative=f"NOT_REACHABLE: no observed path from {from_key!r} to {to_key!r}",
        not_this=f"that {to_key!r} is not reachable from {from_key!r}",
        details=details,
        limit=limit,
    )


def _back_only(value: Any) -> str:
    if value is True:
        return "back_only: true (the only way out is the browser back button)"
    if value is False:
        return "back_only: false"
    return (
        "back_only: UNKNOWN (a back-button departure was filtered, withdrawn, or "
        "ended error/settle_timeout)"
    )


def _unexplored_text(node: Any) -> str:
    n = _dict(node)
    facts = []
    open_by_reason = _dict(n.get("openByReason"))
    if open_by_reason:
        facts.append(f"open frontier {_by_reason(open_by_reason)}")
    unobserved = n.get("unobservedDepartures")
    if _is_number(unobserved) and unobserved:
        facts.append(f"{format_number(unobserved)} unobserved departure(s)")
    departures = _departures_text(n.get("unmeasuredDepartures"))
    if departures:
        facts.append(f"unmeasured departures: {departures}")
    if n.get("unsettledByGraph") is True:
        facts.append(
            "the graph as a whole cannot settle it (an unplaced untrusted row, or "
            "the ledger is not writing)"
        )
    return f"{n.get('nodeKey', '?')}: {'; '.join(facts) if facts else 'unsettled'}"


def format_dead_ends(body: Any, requested: dict[str, str] | None, limit: int) -> str:
    def details(answer: JourneyAnswer) -> list[str]:
        r = answer.result
        if not r:
            return _ledger_unknown_lines(answer)
        dead = _list(r.get("deadEnds"))
        unexplored = _list(r.get("unexplored"))
        heading = (
            "Dead ends among the rows read - NOT settled, the read had a gap"
            if read_gaps(answer)
            else "Settled dead ends"
        )
        lines = [f"  {heading} ({len(dead)}):"]
        if dead:
            lines.extend(
                _limited(
                    dead,
                    limit,
                    lambda d: f"{_dict(d).get('nodeKey', '?')} - "
                    f"{_back_only(_dict(d).get('backOnly'))}",
                    "    ",
                )
            )
        else:
            lines.append("    (none)")
        lines.append(
            f"  Unexplored or unsettled - NOT dead ends, NOT exits ({len(unexplored)}):"
        )
        if unexplored:
            lines.extend(_limited(unexplored, limit, _unexplored_text, "    "))
        else:
            lines.append("    (none)")
        return lines

    def positive(answer: JourneyAnswer) -> str:
        return f"{len(_list(answer.result.get('deadEnds')))} settled dead end(s)"

    return _render(
        body,
        requested,
        positive=positive,
        negative="no dead end",
        not_this="that there are no dead ends",
        details=details,
        limit=limit,
    )


def _prior_unsettled_text(node: Any) -> str:
    n = _dict(node)
    causes = []
    doubt = _dict(n.get("arrivalDoubt"))
    for key, label in (("filtered", "filtered out"), ("withdrawn", "withdrawn")):
        value = doubt.get(key)
        if _is_number(value) and value:
            causes.append(
                f"{format_number(value)} {label} click/back arrival(s) into it"
            )
    untrusted = n.get("untrustedEdgeRows")
    if _is_number(untrusted) and untrusted:
        causes.append(
            f"{format_number(untrusted)} untrusted edge row(s) in the graph (any may "
            "arrive here)"
        )
    unknown_dest = n.get("unknownDestinationArrivals")
    if _is_number(unknown_dest) and unknown_dest:
        causes.append(
            f"{format_number(unknown_dest)} click/back row(s) with an unknown "
            "destination (any may arrive here)"
        )
    return (
        f"{n.get('nodeKey', '?')} - arrivals {_by_reason(n.get('arrivals'))}; "
        f"could still be reached by: {'; '.join(causes) if causes else 'UNKNOWN (no cause reported)'}"
    )


def format_prior_knowledge(
    body: Any, requested: dict[str, str] | None, limit: int
) -> str:
    def details(answer: JourneyAnswer) -> list[str]:
        r = answer.result
        if not r:
            return _ledger_unknown_lines(answer)
        nodes = _list(r.get("nodes"))
        unsettled = _list(r.get("unsettled"))
        settled_heading = (
            "Reached only by URL among the rows read - NOT settled, the read had a gap"
            if read_gaps(answer)
            else "Settled: reached only by URL (initial/push/replace/hash), never "
            "by an affordance or back from another node"
        )
        lines = [f"  {settled_heading} ({len(nodes)}):"]
        if nodes:
            lines.extend(
                _limited(
                    nodes,
                    limit,
                    lambda n: f"{_dict(n).get('nodeKey', '?')} - arrivals "
                    f"{_by_reason(_dict(n).get('arrivals'))}",
                    "    ",
                )
            )
        else:
            lines.append("    (none)")
        if "unsettled" not in r:
            lines.append(
                "  Unsettled: UNKNOWN (the answer carried no `unsettled` list, so "
                "the listed nodes cannot be told apart from doubtful ones)"
            )
            return lines
        lines.append(
            "  Unsettled - NOT prior-knowledge findings, NOT ruled out "
            f"({len(unsettled)}):"
        )
        if unsettled:
            lines.extend(_limited(unsettled, limit, _prior_unsettled_text, "    "))
        else:
            lines.append("    (none)")
        return lines

    def positive(answer: JourneyAnswer) -> str:
        return (
            f"{len(_list(answer.result.get('nodes')))} node(s) settled as reachable "
            "only with prior knowledge (a URL)"
        )

    return _render(
        body,
        requested,
        positive=positive,
        negative="no node requires prior knowledge",
        not_this="that every node is reachable without prior knowledge",
        details=details,
        limit=limit,
    )


def format_frontier(
    body: Any, node: str | None, requested: dict[str, str] | None, limit: int
) -> str:
    where = f"at {node!r}" if node else "in the whole graph"

    def details(answer: JourneyAnswer) -> list[str]:
        r = answer.result
        if not r:
            return _ledger_unknown_lines(answer)
        lines = []
        if r.get("nodeObserved") is False:
            lines.append(
                f"  Node {node!r} was never observed under this filter (unknown, "
                "not 'nothing open there')."
            )
        entries = _list(r.get("entries"))
        if entries:
            lines.append(f"  Open frontier entries {where} ({len(entries)}):")
            lines.extend(_limited(entries, limit, _frontier_entry, "    "))
        return lines

    def positive(answer: JourneyAnswer) -> str:
        return (
            f"{len(_list(answer.result.get('entries')))} open frontier entr"
            f"{'y' if len(_list(answer.result.get('entries'))) == 1 else 'ies'} {where}"
        )

    return _render(
        body,
        requested,
        positive=positive,
        negative=f"nothing left to explore {where}",
        not_this=f"that nothing is left to explore {where}",
        details=details,
        limit=limit,
    )


# =============================================================================
# health (not an envelope)
# =============================================================================


def format_health(body: Any) -> str:
    """``JourneyHealthResponse``: the ledger line first, then the counters."""
    h = _dict(body)
    lines = [
        f"LEDGER: {_ledger_text(_dict(h['ledger']) if isinstance(h.get('ledger'), dict) else None)}"
    ]
    probe = h.get("schemaProbe")
    lines.append(
        f"schemaProbe: {probe if isinstance(probe, str) and probe else 'UNKNOWN (not reported)'}"
    )
    counters = h.get("counters")
    if not isinstance(counters, dict):
        lines.append("counters: UNKNOWN (the health answer carried none)")
        return "\n".join(lines)
    lines.append("counters (this runner process only):")
    for key in sorted(counters):
        value = counters[key]
        shown = "none recorded" if value is None else format_number(value)
        lines.append(f"  {key}: {shown}")
    return "\n".join(lines)


# =============================================================================
# edge-timing
# =============================================================================


def timing_coverage_line(
    answer: JourneyAnswer, requested: dict[str, str] | None
) -> str:
    """The one-line COVERAGE header of an edge-timing answer."""
    r = answer.result
    timing = _dict(r.get("timing"))
    parts = [_ledger_text(answer.ledger())]
    if timing:
        parts.append(
            f"n={_count(timing.get('n'))} "
            f"(withoutTimeline {_count(timing.get('withoutTimeline'))}, "
            f"timelinesUnparseable {_count(r.get('timelinesUnparseable'))}; "
            f"excluded: filteredOut {_count(r.get('edgesFilteredOut'))}, "
            f"withdrawn {_count(r.get('edgesInvalidated'))}, "
            f"rowsRejected {_count(r.get('rowsRejected'))})"
        )
        parts.append(
            f"pooled: by outcome {_by_reason(r.get('samplesByOutcome'))}; "
            f"by trigger {_by_reason(r.get('samplesByTrigger'))}"
        )
    else:
        parts.append(
            "n=UNKNOWN (no counts: the ledger could not be read, so nothing was "
            "counted - not zero)"
        )
    parts.append(_unmeasured_text(answer))
    parts.append(_filter_text(answer.filter_echo(), requested))
    if timing:
        parts.append(
            _window_text(
                timing.get("oldestObservedAt"),
                timing.get("newestObservedAt"),
                "observation",
            )
        )
        parts.append(_app_versions_text(timing.get("appVersions")))
    return "COVERAGE: " + " | ".join(parts)


def _distribution_line(name: str, dist: Any) -> str:
    d = _dict(dist)
    if not d:
        return f"  {name}: UNKNOWN (no distribution in the answer)"
    unknown = _dict(d.get("unknownByCode"))
    unknown_total = sum(v for v in unknown.values() if _is_number(v))
    unknown_text = f"unknown {format_number(unknown_total)}"
    if unknown:
        unknown_text += f" {{{_by_reason(unknown, nonzero_only=False)}}}"
    counts = (
        f"measured {_count(d.get('measured'))} (degraded {_count(d.get('degraded'))}), "
        f"absent {_count(d.get('absent'))}, {unknown_text}, "
        f"without timeline {_count(d.get('withoutTimeline'))}"
    )
    if not _is_number(d.get("measured")) or d.get("measured") == 0:
        return f"  {name}: no measured value, so no min/median/p90/max - {counts}"
    stats = " | ".join(
        f"{stat} {format_number(d[stat]) if _is_number(d.get(stat)) else 'UNKNOWN'}"
        for stat in ("min", "median", "p90", "max")
    )
    return f"  {name}: {stats} - {counts}"


def _timing_row(row: Any) -> str:
    r = _dict(row)
    head = (
        f"{r.get('observationId', '?')} at {r.get('observedAt', '?')} "
        f"app {r.get('appVersion') or '(not reported)'}"
    )
    if r.get("hasTimeline") is not True:
        return f"{head}: no timeline recorded"
    values = "; ".join(
        f"{metric} {render_observation(r.get(metric))}" for metric in TIMING_METRICS
    )
    return f"{head}: {values}"


def format_edge_timing(
    body: Any,
    from_key: str,
    to_key: str,
    requested: dict[str, str] | None,
    limit: int,
) -> str:
    """The distribution of one edge's timelines, with its sample size.

    Per metric: min / median / p90 / max over the MEASURED values only, beside
    the measured, absent and unknown counts. Never a single latency.
    """
    answer = JourneyAnswer(body)
    timing = _dict(answer.result.get("timing"))
    n = timing.get("n")
    lines = [
        timing_coverage_line(answer, requested),
        answer_line(
            answer,
            positive=(
                f"{_count(n)} observation(s) of {from_key!r} -> {to_key!r} "
                "(a distribution over them, not a single latency)"
            ),
            negative=f"no observation of {from_key!r} -> {to_key!r}",
            not_this=(
                f"that {from_key!r} -> {to_key!r} is fast, slow or never taken; "
                "nothing about its timing was measured"
            ),
            degraded=(
                "not every observation yielded every value - see COVERAGE and "
                "each metric's counts below"
            ),
        ),
    ]
    lines.extend(_gap_lines(answer))
    if not timing:
        lines.extend(_ledger_unknown_lines(answer))
        return "\n".join(lines)
    lines.append(
        "Per metric (nearest-rank over measured values only; no threshold applied):"
    )
    for metric in TIMING_METRICS:
        lines.append(_distribution_line(metric, timing.get(metric)))
    rows = _list(timing.get("observations"))
    if rows:
        lines.append(f"Observations ({len(rows)}, by observedAt):")
        lines.extend(_limited(rows, limit, _timing_row, "  "))
    return "\n".join(lines)
