"""Runner journey answers, shaped exactly as the runner serves them.

Built from the Rust types the routes serialize (qontinui-runner
``src-tauri/src/journey/routes.rs`` ``answer`` / ``answer_unavailable`` /
``answer_edge_timing`` / ``finish``; qontinui-schemas ``rust/src/journey/
query.rs`` and ``rust-vision-core/src/edge_timing.rs``), field for field:

- a ``measured`` envelope carries the query result as ``value`` and
  ``provenance.source = {query, appId}``;
- a non-measured one carries it as ``provenance.source.result``;
- the ``JourneyCoverage`` block is merged INTO ``provenance.coverage`` beside
  ``considered / measured / unmeasured`` (not for edge-timing);
- a read that could not be made carries ``filterEcho`` and ``ledger`` in
  ``provenance.source`` and, for a failed read, no coverage counts at all.
"""

from __future__ import annotations

import copy
import math
from typing import Any

APP = "qontinui-web"
EVALUATED_AT = "2026-10-24T12:00:00.000Z"
FRONTIER_REASONS = (
    "not_yet_activated",
    "effect_undeclared",
    "effect_write",
    "effect_destructive",
    "budget_exhausted",
    "activation_failed",
)

HOME = "console#home"
RUNS = "console#runs"
RUN_DETAIL = "console#run-detail"
SETTINGS = "unmodelled:/admin/settings"

WRITING = {
    "state": "writing",
    "detail": "journey tables present; 12 edge(s) written by this process",
}
WRITE_FAILING = {
    "state": "write_failing",
    "detail": "3 of the last 50 journey writes failed; last error: pool timed out",
}
SCHEMA_ABSENT = {
    "state": "schema_absent",
    "detail": "embedded DB predates the journey tables",
}


def no_filter() -> dict[str, Any]:
    return {"since": None, "appVersion": None, "runKind": None}


def departures(**over: int) -> dict[str, Any]:
    d = {
        "rejectedFrontier": 0,
        "filteredOut": 0,
        "withdrawn": 0,
        "settleTimeout": 0,
        "errored": 0,
        "excludedBackArrivals": 0,
        "excludedUrlArrivals": 0,
    }
    d.update(over)
    return d


def node(key: str) -> dict[str, Any]:
    spec, _, states = key.partition("#")
    if key.startswith("unmodelled:"):
        return {
            "specId": None,
            "stateIds": [],
            "pageLabel": None,
            "pathnameTemplate": key.removeprefix("unmodelled:"),
            "modelled": False,
        }
    return {
        "specId": spec,
        "stateIds": states.split(","),
        "pageLabel": spec,
        "pathnameTemplate": None,
        "modelled": True,
    }


def frontier_entry(
    node_key: str,
    fingerprint: str,
    reason: str = "not_yet_activated",
    role: str | None = "button",
    effect: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "appId": APP,
        "nodeKey": node_key,
        "node": node(node_key),
        "affordanceFingerprint": fingerprint,
        "reason": reason,
        "firstSeenAt": "2026-10-20T09:00:00Z",
        "lastSeenAt": "2026-10-23T17:30:00Z",
    }
    # skip_serializing_if = "Option::is_none"
    if role is not None:
        entry["affordanceRole"] = role
    if effect is not None:
        entry["declaredEffect"] = effect
    return entry


def coverage(**over: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "nodesObserved": 4,
        "nodesUnmodelled": 1,
        "edgesObserved": 12,
        "edgesToNodeUnobserved": 0,
        "edgesFilteredOut": 0,
        "edgesInvalidated": 0,
        "rowsRejected": 0,
        "frontierOpen": 0,
        "frontierNodes": 0,
        "frontierByReason": {r: 0 for r in FRONTIER_REASONS},
        "runs": 3,
        "edgesWithoutRunId": 0,
        "oldestEdgeAt": "2026-10-20T09:00:00Z",
        "newestEdgeAt": "2026-10-23T17:30:00Z",
        "appVersions": [None, "1.4.0"],
        "ledger": copy.deepcopy(WRITING),
    }
    c.update(over)
    return c


def search_frontier(
    nodes: list[dict[str, Any]] | None = None, **over: Any
) -> dict[str, Any]:
    nodes = nodes or []
    by_reason = {r: 0 for r in FRONTIER_REASONS}
    open_entries = 0
    unobserved = 0
    for n in nodes:
        unobserved += n["unobservedDepartures"]
        for reason, entries in n["openByReason"].items():
            by_reason[reason] += len(entries)
            open_entries += len(entries)
    f: dict[str, Any] = {
        "nodes": nodes,
        "openEntries": open_entries,
        "byReason": by_reason,
        "unobservedDepartures": unobserved,
        "unmeasuredDepartures": departures(),
        "offGraphDepartures": departures(),
        "unplacedRejectedRows": 0,
        "unplacedRejectedFrontier": 0,
    }
    f.update(over)
    return f


def search_node(
    key: str, open_entries: list[dict[str, Any]] | None = None, **over: Any
) -> dict[str, Any]:
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for e in open_entries or []:
        by_reason.setdefault(e["reason"], []).append(e)
    n: dict[str, Any] = {
        "nodeKey": key,
        "openByReason": by_reason,
        "unobservedDepartures": 0,
        "unmeasuredDepartures": departures(),
    }
    n.update(over)
    return n


def dim(dimension: str, count: int, code: str) -> dict[str, Any]:
    return {"dimension": dimension, "count": count, "code": code}


def envelope(
    query: str,
    status: str,
    result: dict[str, Any],
    *,
    unmeasured: list[dict[str, Any]] | None = None,
    code: str | None = None,
    detail: str = "",
    journey_coverage: dict[str, Any] | None = None,
    considered: int | None = None,
    measured: int | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """``routes.rs`` ``answer`` + ``finish``."""
    unmeasured = unmeasured or []
    jc = journey_coverage if journey_coverage is not None else result.get("coverage")
    n = considered if considered is not None else (jc or {}).get("nodesObserved", 0)
    obs_cov: dict[str, Any] = {
        "considered": n,
        "measured": measured
        if measured is not None
        else (n if not unmeasured else max(n - 1, 0)),
        "unmeasured": unmeasured,
    }
    if jc is not None:
        for k, v in jc.items():
            obs_cov.setdefault(k, copy.deepcopy(v))
    source: dict[str, Any] = {"query": query, "appId": APP}
    prov = {
        "producer": {"id": f"runner/journey/{query}", "version": "0.1.0"},
        "observedAt": observed_at or (jc or {}).get("newestEdgeAt"),
        "evaluatedAt": EVALUATED_AT,
        "coverage": obs_cov,
        "confidence": None,
        "cache": None,
        "source": source,
    }
    body: dict[str, Any] = {"status": status, "provenance": prov}
    if status == "measured":
        body["value"] = result
    else:
        source["result"] = result
    if status == "unknown":
        body["unknown"] = {"code": code, "detail": detail}
    return body


def schema_absent(query: str) -> dict[str, Any]:
    """``answer_unavailable`` with ``LedgerUnavailable::SchemaAbsent``."""
    empty = coverage(
        nodesObserved=0,
        nodesUnmodelled=0,
        edgesObserved=0,
        runs=0,
        oldestEdgeAt=None,
        newestEdgeAt=None,
        appVersions=[],
        ledger=copy.deepcopy(SCHEMA_ABSENT),
    )
    obs_cov: dict[str, Any] = {"considered": 0, "measured": 0, "unmeasured": []}
    obs_cov.update(empty)
    return {
        "status": "unknown",
        "unknown": {
            "code": "ledger_schema_absent",
            "detail": f"{query}: {SCHEMA_ABSENT['detail']}",
        },
        "provenance": {
            "producer": {"id": f"runner/journey/{query}", "version": "0.1.0"},
            "observedAt": None,
            "evaluatedAt": EVALUATED_AT,
            "coverage": obs_cov,
            "confidence": None,
            "cache": None,
            "source": {
                "query": query,
                "appId": APP,
                "filterEcho": no_filter(),
                "ledger": copy.deepcopy(SCHEMA_ABSENT),
            },
        },
    }


def read_failed(query: str) -> dict[str, Any]:
    """``answer_unavailable`` with ``LedgerUnavailable::Failed``: no counts."""
    return {
        "status": "unknown",
        "unknown": {
            "code": "producer_failed",
            "detail": f"{query}: the journey ledger could not be read: pool timed out",
        },
        "provenance": {
            "producer": {"id": f"runner/journey/{query}", "version": "0.1.0"},
            "observedAt": None,
            "evaluatedAt": EVALUATED_AT,
            "coverage": {"considered": 0, "measured": 0, "unmeasured": []},
            "confidence": None,
            "cache": None,
            "source": {
                "query": query,
                "appId": APP,
                "filterEcho": no_filter(),
                "ledger": copy.deepcopy(WRITING),
            },
        },
    }


# =============================================================================
# steps: C reachable only through frontier-open B (plan acceptance 2)
# =============================================================================

B_FRONTIER = [
    frontier_entry(RUNS, "fp-open-run", role="link"),
    frontier_entry(
        RUNS, "fp-delete", reason="effect_destructive", effect="destructive"
    ),
]


def steps_unknown() -> dict[str, Any]:
    cov = coverage(frontierOpen=2, frontierNodes=1)
    cov["frontierByReason"].update(not_yet_activated=1, effect_destructive=1)
    sf = search_frontier([search_node(RUNS, B_FRONTIER)])
    result = {
        "fromKey": HOME,
        "toKey": RUN_DETAIL,
        "fromObserved": True,
        "toObserved": True,
        "steps": None,
        "path": [],
        "frontier": sf,
        "coverage": cov,
        "filterEcho": no_filter(),
    }
    return envelope(
        "steps",
        "unknown",
        result,
        unmeasured=[dim("frontier", 2, "frontier_open")],
        code="frontier_open",
        detail=(
            "steps: the searched part of the graph has 2 open frontier entries "
            "(affordances seen and never activated); the negative cannot be settled "
            "until they are explored — they are listed in "
            "provenance.source.result.frontier"
        ),
    )


def steps_absent() -> dict[str, Any]:
    result = {
        "fromKey": HOME,
        "toKey": SETTINGS,
        "fromObserved": True,
        "toObserved": True,
        "steps": None,
        "path": [],
        "frontier": search_frontier(),
        "coverage": coverage(),
        "filterEcho": no_filter(),
    }
    return envelope("steps", "absent", result)


def graph_edge(frm: str, to: str, action: str = "click") -> dict[str, Any]:
    return {
        "fromKey": frm,
        "toKey": to,
        "trigger": {
            "chokePoint": "sdk_element_action",
            "actionType": action,
            "navigationTrigger": "affordance",
            "targetRole": "link",
            "targetFingerprint": f"fp-{to}",
        },
        "outcome": "changed",
        "observations": 3,
        "firstObservedAt": "2026-10-20T09:00:00Z",
        "lastObservedAt": "2026-10-23T17:30:00Z",
    }


def steps_measured() -> dict[str, Any]:
    result = {
        "fromKey": HOME,
        "toKey": RUN_DETAIL,
        "fromObserved": True,
        "toObserved": True,
        "steps": 2,
        "path": [graph_edge(HOME, RUNS), graph_edge(RUNS, RUN_DETAIL)],
        "frontier": search_frontier(),
        "coverage": coverage(),
        "filterEcho": no_filter(),
    }
    return envelope("steps", "measured", result)


# =============================================================================
# reachable
# =============================================================================


def reachable_measured(extra_nodes: int = 0) -> dict[str, Any]:
    reached = [
        {"nodeKey": HOME, "steps": 0},
        {"nodeKey": RUNS, "steps": 1},
    ] + [{"nodeKey": f"console#page-{i:03d}", "steps": 2} for i in range(extra_nodes)]
    cov = coverage(
        nodesObserved=len(reached),
        edgesToNodeUnobserved=1,
        frontierOpen=1,
        frontierNodes=1,
    )
    cov["frontierByReason"]["not_yet_activated"] = 1
    sf = search_frontier(
        [
            search_node(
                RUNS, [frontier_entry(RUNS, "fp-open-run")], unobservedDepartures=1
            )
        ]
    )
    result = {
        "fromKey": HOME,
        "fromObserved": True,
        "reachable": reached,
        "frontier": sf,
        "coverage": cov,
        "filterEcho": no_filter(),
    }
    return envelope(
        "reachable",
        "measured",
        result,
        unmeasured=[
            dim("frontier", 1, "frontier_open"),
            dim("unobserved_departures", 1, "to_node_unobserved"),
        ],
    )


def reachable_unobserved_start() -> dict[str, Any]:
    result = {
        "fromKey": "console#nowhere",
        "fromObserved": False,
        "reachable": [],
        "frontier": search_frontier(),
        "coverage": coverage(),
        "filterEcho": {
            "since": "2026-10-22T00:00:00Z",
            "appVersion": None,
            "runKind": None,
        },
    }
    return envelope(
        "reachable",
        "unknown",
        result,
        unmeasured=[dim("subject_unobserved", 1, "no_observations")],
        code="no_observations",
        detail=(
            "reachable: the ledger holds no observation this question can be answered "
            "from (an empty ledger under this filter, or a node it has never seen)"
        ),
    )


# =============================================================================
# dead-ends
# =============================================================================


def dead_ends_measured() -> dict[str, Any]:
    cov = coverage(frontierOpen=1, frontierNodes=1)
    cov["frontierByReason"]["effect_write"] = 1
    unexplored = [
        {
            "nodeKey": RUNS,
            "openByReason": {"effect_write": 1},
            "unobservedDepartures": 0,
            "unmeasuredDepartures": departures(),
            "unsettledByGraph": False,
            "unconfirmedDestination": False,
        },
        {
            "nodeKey": SETTINGS,
            "openByReason": {},
            "unobservedDepartures": 0,
            "unmeasuredDepartures": departures(settleTimeout=1),
            "unsettledByGraph": False,
            "unconfirmedDestination": True,
        },
    ]
    sf = search_frontier(
        [
            search_node(
                RUNS, [frontier_entry(RUNS, "fp-save", "effect_write", effect="write")]
            ),
            search_node(SETTINGS, unmeasuredDepartures=departures(settleTimeout=1)),
        ],
        unmeasuredDepartures=departures(settleTimeout=1),
    )
    result = {
        "deadEnds": [
            {"nodeKey": RUN_DETAIL, "backOnly": True},
            {"nodeKey": "console#about", "backOnly": False},
            {"nodeKey": "console#export-done", "backOnly": None},
        ],
        "unexplored": unexplored,
        "frontier": sf,
        "coverage": cov,
        "filterEcho": no_filter(),
    }
    return envelope(
        "dead-ends",
        "measured",
        result,
        unmeasured=[
            dim("frontier", 1, "frontier_open"),
            dim("settle_timeout_departures", 1, "no_event_before_timeout"),
            # routes.rs answer_dead_ends: a non-empty `unexplored` degrades
            # the listing.
            dim("unexplored_nodes", 2, "input_missing"),
        ],
    )


def dead_ends_unknown_unexplored() -> dict[str, Any]:
    """No settled dead end, one unexplored node: unknown, not 'no dead ends'."""
    cov = coverage(frontierOpen=1, frontierNodes=1)
    cov["frontierByReason"]["not_yet_activated"] = 1
    result = {
        "deadEnds": [],
        "unexplored": [
            {
                "nodeKey": RUN_DETAIL,
                "openByReason": {"not_yet_activated": 1},
                "unobservedDepartures": 0,
                "unmeasuredDepartures": departures(),
                "unsettledByGraph": False,
                "unconfirmedDestination": False,
            }
        ],
        "frontier": search_frontier(
            [
                search_node(
                    RUN_DETAIL,
                    [frontier_entry(RUN_DETAIL, "fp-back-link", role="link")],
                )
            ]
        ),
        "coverage": cov,
        "filterEcho": no_filter(),
    }
    return envelope(
        "dead-ends",
        "unknown",
        result,
        unmeasured=[
            dim("frontier", 1, "frontier_open"),
            dim("unexplored_nodes", 1, "input_missing"),
        ],
        code="frontier_open",
        detail=(
            "dead-ends: 1 open frontier entry (affordances seen and never "
            "activated) in the searched part; they are listed in "
            "provenance.source.result"
        ),
    )


def dead_ends_absent() -> dict[str, Any]:
    result = {
        "deadEnds": [],
        "unexplored": [],
        "frontier": search_frontier(),
        "coverage": coverage(),
        "filterEcho": no_filter(),
    }
    return envelope("dead-ends", "absent", result)


# =============================================================================
# prior-knowledge, frontier
# =============================================================================


def prior_knowledge_measured() -> dict[str, Any]:
    result = {
        "nodes": [{"nodeKey": SETTINGS, "arrivals": {"initial": 2, "push": 1}}],
        "unsettled": [
            {
                "nodeKey": "console#invite",
                "arrivals": {"initial": 1},
                "arrivalDoubt": {"filtered": 2, "withdrawn": 0},
                "unplacedRejectedRows": 0,
                "unknownDestinationArrivals": 0,
                "openFrontierEntries": 0,
                "ledgerNotWriting": False,
                "unconfirmedDestination": False,
            }
        ],
        "frontier": search_frontier(),
        "coverage": coverage(),
        "filterEcho": {"since": None, "appVersion": "1.4.0", "runKind": "agent_action"},
    }
    # routes.rs answer_prior_knowledge: settled nodes exist -> measured,
    # degraded by the unsettled ones.
    return envelope(
        "prior-knowledge",
        "measured",
        result,
        unmeasured=[dim("unsettled_prior_knowledge_nodes", 1, "input_missing")],
    )


def prior_knowledge_graph_doubt() -> dict[str, Any]:
    """Graph-wide doubt: no node settles, so the negative about the rest is unknown."""
    result = {
        "nodes": [],
        "unsettled": [
            {
                "nodeKey": SETTINGS,
                "arrivals": {"push": 1},
                "arrivalDoubt": {"filtered": 0, "withdrawn": 0},
                "unplacedRejectedRows": 2,
                "unknownDestinationArrivals": 1,
                "openFrontierEntries": 3,
                "ledgerNotWriting": True,
                "unconfirmedDestination": True,
            }
        ],
        "frontier": search_frontier(
            [
                search_node(
                    RUNS,
                    [
                        frontier_entry(RUNS, "fp-a"),
                        frontier_entry(RUNS, "fp-b"),
                        frontier_entry(RUNS, "fp-c"),
                    ],
                )
            ],
            unplacedRejectedRows=2,
        ),
        "coverage": coverage(
            rowsRejected=1,
            frontierOpen=3,
            frontierNodes=1,
            ledger=copy.deepcopy(WRITE_FAILING),
        ),
        "filterEcho": no_filter(),
    }
    result["coverage"]["frontierByReason"]["not_yet_activated"] = 3
    # JourneySearchFrontier::unmeasured order (ledger, frontier, ...,
    # unplaced_rejected_rows), then the route's extra dimension.
    return envelope(
        "prior-knowledge",
        "unknown",
        result,
        unmeasured=[
            dim("ledger", 1, "ledger_write_failing"),
            dim("frontier", 3, "frontier_open"),
            dim("unplaced_rejected_rows", 2, "producer_failed"),
            dim("unsettled_prior_knowledge_nodes", 1, "input_missing"),
        ],
        code="ledger_write_failing",
        detail=(
            "prior-knowledge: the journey ledger is not being written, so a negative "
            "over it could describe a stale graph as a finished one (see "
            "provenance.coverage.ledger)"
        ),
    )


_ROUTE_EXTRA_DIMENSIONS = frozenset(
    {"unexplored_nodes", "unsettled_prior_knowledge_nodes"}
)


def _read_gap(body: dict[str, Any], query: str) -> dict[str, Any]:
    """The runner's answer when its read had a gap: unknown, the gap named."""
    cov = body["provenance"]["coverage"]
    # routes.rs `answer`: the shared search-frontier dimensions, then the
    # read's gaps, then the route's extra dimensions; the code is the first.
    base = cov["unmeasured"]
    extra = [d for d in base if d["dimension"] in _ROUTE_EXTRA_DIMENSIONS]
    shared = [d for d in base if d["dimension"] not in _ROUTE_EXTRA_DIMENSIONS]
    gaps = [
        dim("edges_beyond_read_bound_50000", 1, "input_missing"),
        dim("rows_unparseable", 3, "producer_failed"),
    ]
    cov["unmeasured"] = shared + gaps + extra
    first = cov["unmeasured"][0]
    source = body["provenance"]["source"]
    if body["status"] == "measured":
        source["result"] = body.pop("value")
    body["status"] = "unknown"
    details = {
        "frontier": f"{query}: {first['count']} open frontier "
        f"entr{'y' if first['count'] == 1 else 'ies'} (affordances "
        "seen and never activated) in the searched part; they are listed in "
        "provenance.source.result",
        "edges_beyond_read_bound_50000": f"{query}: the read bound was reached "
        "(50000 edges, newest first); older edges were not read",
    }
    body["unknown"] = {
        "code": first["code"],
        "detail": details.get(
            first["dimension"],
            f"{query}: {first['count']} {first['dimension']} not measured "
            f"({first['code']})",
        ),
    }
    return body


def dead_ends_read_gap() -> dict[str, Any]:
    return _read_gap(dead_ends_measured(), "dead-ends")


def prior_knowledge_read_gap() -> dict[str, Any]:
    return _read_gap(prior_knowledge_measured(), "prior-knowledge")


def frontier_measured() -> dict[str, Any]:
    cov = coverage(frontierOpen=2, frontierNodes=1)
    cov["frontierByReason"].update(not_yet_activated=1, effect_destructive=1)
    sf = search_frontier([search_node(RUNS, B_FRONTIER)])
    result = {
        "nodeKey": None,
        "nodeObserved": True,
        "entries": B_FRONTIER,
        "frontier": sf,
        "coverage": cov,
        "filterEcho": no_filter(),
    }
    return envelope(
        "frontier",
        "measured",
        result,
        unmeasured=[dim("frontier", 2, "frontier_open")],
    )


def empty_ledger(query: str) -> dict[str, Any]:
    """An empty ledger answers ``unknown: no_observations`` for every query."""
    cov = coverage(
        nodesObserved=0,
        nodesUnmodelled=0,
        edgesObserved=0,
        runs=0,
        oldestEdgeAt=None,
        newestEdgeAt=None,
        appVersions=[],
    )
    result = {
        "deadEnds": [],
        "unexplored": [],
        "frontier": search_frontier(),
        "coverage": cov,
        "filterEcho": no_filter(),
    }
    return envelope(
        query,
        "unknown",
        result,
        unmeasured=[dim("no_observations", 1, "no_observations")],
        code="no_observations",
        detail=f"{query}: the ledger holds no observation this question can be answered from",
    )


# =============================================================================
# health
# =============================================================================


def health() -> dict[str, Any]:
    return {
        "appId": APP,
        "ledger": copy.deepcopy(WRITING),
        "schemaProbe": "present",
        "counters": {
            "edgesWritten": 12,
            "pendingEdgesOpen": 0,
            "frontierUpserts": 30,
            "frontierRowsCleared": 9,
            "writesNotAttempted": 0,
            "writeFailuresTotal": 0,
            "timelinesWritten": 10,
            "timelinesUnparseable": 0,
            "lastTimelineError": None,
            "recentWriteFailures": 0,
            "recentWindow": 50,
            "lastWriteAt": "2026-10-23T17:30:00Z",
            "lastError": None,
            "lastPruneAt": None,
            "lastPruneDeleted": None,
            "lastPruneSkipped": None,
        },
    }


# =============================================================================
# edge-timing
# =============================================================================


def _obs_measured(value: float) -> dict[str, Any]:
    return {
        "status": "measured",
        "value": value,
        "provenance": {
            "producer": {"id": "ui-bridge/change-tracker", "version": "0.4.0"},
            "observedAt": None,
            "evaluatedAt": "2026-10-23T17:30:00Z",
            "coverage": {"considered": 1, "measured": 1, "unmeasured": []},
            "confidence": None,
            "cache": None,
            "source": None,
        },
    }


def _obs_unknown(code: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "unknown": {"code": code, "detail": "PerformanceObserver has no layout-shift"},
        "provenance": {
            "producer": {"id": "ui-bridge/change-tracker", "version": "0.4.0"},
            "observedAt": None,
            "evaluatedAt": "2026-10-23T17:30:00Z",
            "coverage": {"considered": 0, "measured": 0, "unmeasured": []},
            "confidence": None,
            "cache": None,
            "source": None,
        },
    }


def _dist(
    values: list[float],
    *,
    unknown: dict[str, int] | None = None,
    absent: int = 0,
    without: int = 0,
) -> dict[str, Any]:
    s = sorted(values)
    if not s:
        stats: dict[str, Any] = {"min": None, "median": None, "p90": None, "max": None}
    else:

        def rank(p: float) -> float:
            return s[min(max(math.ceil(p * len(s)), 1), len(s)) - 1]

        stats = {"min": s[0], "median": rank(0.5), "p90": rank(0.9), "max": s[-1]}
    return {
        "measured": len(s),
        "degraded": 0,
        "absent": absent,
        "unknownByCode": unknown or {},
        "withoutTimeline": without,
        **stats,
    }


def _row(i: int, settle: float | None, first: float | None) -> dict[str, Any]:
    has = settle is not None
    row: dict[str, Any] = {
        "observationId": f"obs-{i}",
        "observedAt": f"2026-10-2{i}T10:00:00Z",
        "appVersion": "1.4.0",
        "hasTimeline": has,
        "firstResponseMs": _obs_measured(first) if has and first is not None else None,
        "settleMs": _obs_measured(settle) if has else None,
        "layoutShiftTotal": _obs_unknown("performance_observer_unsupported")
        if has
        else None,
        "layoutShiftCount": _obs_unknown("performance_observer_unsupported")
        if has
        else None,
        "layoutShiftAfterFirstResponseMs": (
            _obs_unknown("performance_observer_unsupported") if has else None
        ),
        "longTaskTotalMs": _obs_measured(0) if has else None,
    }
    return row


def edge_timing_measured() -> dict[str, Any]:
    rows = [
        _row(1, 100.0, 40.0),
        _row(2, 300.0, 60.0),
        _row(3, 200.0, 50.0),
        _row(4, None, None),
    ]
    unsupported = {"performance_observer_unsupported": 3}
    timing = {
        "n": 4,
        "withoutTimeline": 1,
        "oldestObservedAt": "2026-10-21T10:00:00Z",
        "newestObservedAt": "2026-10-24T10:00:00Z",
        "appVersions": ["1.4.0"],
        "firstResponseMs": _dist([40.0, 60.0, 50.0], without=1),
        "settleMs": _dist([100.0, 300.0, 200.0], without=1),
        "layoutShiftTotal": _dist([], unknown=unsupported, without=1),
        "layoutShiftCount": _dist([], unknown=unsupported, without=1),
        "layoutShiftAfterFirstResponseMs": _dist([], unknown=unsupported, without=1),
        "longTaskTotalMs": _dist([0.0, 0.0, 0.0], without=1),
        "observations": rows,
    }
    value = {
        "fromKey": HOME,
        "toKey": RUNS,
        "filterEcho": no_filter(),
        "edgesFilteredOut": 1,
        "edgesInvalidated": 0,
        "timelinesUnparseable": 0,
        "rowsRejected": 0,
        "samplesByOutcome": {"changed": 3, "settle_timeout": 1},
        "samplesByTrigger": {"affordance": 4},
        "timing": timing,
        "ledger": copy.deepcopy(WRITING),
    }
    return envelope(
        "edge-timing",
        "measured",
        value,
        unmeasured=[dim("timeline", 1, "input_missing")],
        journey_coverage=None,
        considered=4,
        measured=3,
        observed_at="2026-10-24T10:00:00Z",
    )


def edge_timing_empty() -> dict[str, Any]:
    empty = _dist([])
    timing = {
        "n": 0,
        "withoutTimeline": 0,
        "oldestObservedAt": None,
        "newestObservedAt": None,
        "appVersions": [],
        **{
            m: copy.deepcopy(empty)
            for m in (
                "firstResponseMs",
                "settleMs",
                "layoutShiftTotal",
                "layoutShiftCount",
                "layoutShiftAfterFirstResponseMs",
                "longTaskTotalMs",
            )
        },
        "observations": [],
    }
    value = {
        "fromKey": HOME,
        "toKey": SETTINGS,
        "filterEcho": {
            "since": "2026-10-23T00:00:00Z",
            "appVersion": None,
            "runKind": None,
        },
        "edgesFilteredOut": 2,
        "edgesInvalidated": 1,
        "timelinesUnparseable": 0,
        "rowsRejected": 0,
        "samplesByOutcome": {},
        "samplesByTrigger": {},
        "timing": timing,
        "ledger": copy.deepcopy(WRITING),
    }
    return envelope(
        "edge-timing",
        "unknown",
        value,
        code="no_observations",
        detail=(
            f'edge-timing: no live observation of "{HOME}" -> "{SETTINGS}" under this '
            "filter (2 filtered out, 1 withdrawn)"
        ),
        journey_coverage=None,
        considered=0,
    )


def graph_with_rejected_rows() -> dict[str, Any]:
    """A ``JourneyGraph`` (the graph route's value) with rejected rows.

    Two rejected entries standing for three distinct rows: ``rowsRejected``
    counts entries, Σ``rows`` counts rows.
    """
    result = {
        "nodes": [],
        "edges": [],
        "rejected": [
            {
                "kind": "edge",
                "row": "edge-7",
                "reason": "fromNode: pageLabel contains '/'",
                "rows": 2,
            },
            {
                "kind": "frontier",
                "row": f"{RUNS} fp-x",
                "reason": "nodeKey mismatch",
                "rows": 1,
            },
        ],
        "offGraphDepartures": departures(),
        "rejectedFrontierDoublePlaced": 0,
        "unplacedRejectedRows": 2,
        "unplacedRejectedFrontier": 0,
        "unknownDestinationArrivals": 0,
        "coverage": coverage(rowsRejected=2),
        "filterEcho": no_filter(),
    }
    return envelope(
        "graph",
        "measured",
        result,
        unmeasured=[dim("unplaced_rejected_rows", 2, "producer_failed")],
    )


def edge_timing_rejected() -> dict[str, Any]:
    """n=0 with a rejected row of the pair: the code comes from `rows_rejected`.

    ``answer_edge_timing``: ``rows_rejected`` (rows of this pair that fail the
    shared trust rule), then every untrusted row in the read as
    ``unplaced_rejected_rows``; the code is the first dimension's.
    """
    body = edge_timing_empty()
    value = body["provenance"]["source"]["result"]
    value["rowsRejected"] = 1
    body["provenance"]["coverage"]["unmeasured"] = [
        dim("rows_rejected", 1, "producer_failed"),
        dim("unplaced_rejected_rows", 4, "producer_failed"),
    ]
    body["unknown"] = {
        "code": "producer_failed",
        "detail": "edge-timing: 1 rows_rejected not measured (producer_failed)",
    }
    return body
