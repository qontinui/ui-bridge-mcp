"""Journey query tools: unknown is never a negative, and coverage comes first.

Plan ``2026-09-20-ui-bridge-represents-the-users-path-and-the-passage-of-time``,
Phase 2 (ui-bridge-mcp tools) and Phase 4 (``journey_edge_timing``). Each tool
is called over an MCP client session against a fake runner answering the
shapes in ``journey_fixtures`` (built from the runner's Rust types).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.journey import _graph_node_text
from ui_bridge_mcp.timeline import format_timeline

from . import journey_fixtures as F
from .mcp_session import answering, call, text

ROOT = f"/apps/{F.APP}/journey"


def run(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    route: str,
    body: Any,
    **arguments: Any,
) -> tuple[str, bool, Any]:
    runner = answering(monkeypatch, {f"{ROOT}/{route}": body})
    result = call(tool, {"app_id": F.APP, **arguments})
    return text(result), bool(result.isError), runner


def first_line(output: str) -> str:
    return output.splitlines()[0]


# =============================================================================
# unknown vs absent vs measured
# =============================================================================


def test_unknown_steps_is_not_rendered_as_not_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, is_error, _ = run(
        monkeypatch,
        "journey_steps_to",
        "steps",
        F.steps_unknown(),
        **{"from": F.HOME, "to": F.RUN_DETAIL},
    )
    assert not is_error, "an unknown answer is an answer, not a tool failure"
    answer = out.splitlines()[1]
    assert answer.startswith("ANSWER: UNKNOWN (frontier_open) - NOT a negative")
    assert "does NOT mean" in answer
    assert "NOT_REACHABLE" not in out
    assert "ABSENT" not in out
    # The frontier entries that would settle it are named.
    assert "Unsettled part (explore these to settle the question):" in out
    assert "explore: affordance fp-open-run (link; effect undeclared)" in out
    assert (
        "explore: affordance fp-delete (button; destructive) reason=effect_destructive"
    ) in out


def test_absent_steps_is_a_settled_not_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, is_error, _ = run(
        monkeypatch,
        "journey_steps_to",
        "steps",
        F.steps_absent(),
        **{"from": F.HOME, "to": F.SETTINGS},
    )
    assert not is_error
    answer = out.splitlines()[1]
    assert answer.startswith("ANSWER: ABSENT - NOT_REACHABLE")
    assert "SETTLED negative" in answer
    assert "UNKNOWN" not in out
    assert "Unsettled part" not in out


def test_unknown_absent_and_measured_render_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = {"from": F.HOME, "to": F.RUN_DETAIL}
    outs = []
    for body in (F.steps_unknown(), F.steps_absent(), F.steps_measured()):
        out, _, _ = run(monkeypatch, "journey_steps_to", "steps", body, **args)
        outs.append(out.splitlines()[1].split(" - ")[0])
    assert outs == [
        "ANSWER: UNKNOWN (frontier_open)",
        "ANSWER: ABSENT",
        "ANSWER: MEASURED",
    ]


def test_measured_steps_prints_the_path(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_steps_to",
        "steps",
        F.steps_measured(),
        **{"from": F.HOME, "to": F.RUN_DETAIL},
    )
    assert "is reachable from 'console#home' in 2 step(s)" in out
    assert (
        "console#home -> console#runs via click/affordance [role=link, "
        "target=fp-console#runs] outcome=changed seen 3x"
    ) in out
    assert "DEGRADED" not in out


def test_unobserved_start_is_unknown_not_reaches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_reachable_from",
        "reachable",
        F.reachable_unobserved_start(),
        **{"from": "console#nowhere"},
    )
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (no_observations)")
    assert "never observed under this filter" in out
    assert "subject_unobserved x1 no_observations" in first_line(out)


def test_degraded_measured_answer_says_unlisted_is_not_ruled_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_reachable_from",
        "reachable",
        F.reachable_measured(),
        **{"from": F.HOME},
    )
    answer = out.splitlines()[1]
    assert answer.startswith(
        "ANSWER: MEASURED - 'console#home' reaches 1 other node(s)"
    )
    assert "DEGRADED" in answer and "not ruled out" in answer
    assert "[0] console#home" in out and "[1] console#runs" in out
    assert "1 departure(s) whose destination was never observed" in out


# =============================================================================
# the coverage line comes first
# =============================================================================


@pytest.mark.parametrize(
    ("tool", "route", "body", "args"),
    [
        (
            "journey_steps_to",
            "steps",
            F.steps_unknown(),
            {"from": F.HOME, "to": F.RUNS},
        ),
        (
            "journey_reachable_from",
            "reachable",
            F.reachable_measured(),
            {"from": F.HOME},
        ),
        ("journey_graph", "graph", F.graph_measured(), {}),
        ("journey_dead_ends", "dead-ends", F.dead_ends_measured(), {}),
        (
            "journey_requires_prior_knowledge",
            "prior-knowledge",
            F.prior_knowledge_measured(),
            {},
        ),
        ("journey_frontier", "frontier", F.frontier_measured(), {}),
        (
            "journey_edge_timing",
            "edge-timing",
            F.edge_timing_measured(),
            {"from": F.HOME, "to": F.RUNS},
        ),
    ],
)
def test_coverage_line_is_first(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    route: str,
    body: Any,
    args: dict[str, Any],
) -> None:
    out, _, _ = run(monkeypatch, tool, route, body, **args)
    line = first_line(out)
    assert line.startswith("COVERAGE: ledger=writing | ")
    assert "unmeasured: " in line
    assert "filter: since=" in line
    assert out.splitlines()[1].startswith("ANSWER: ")


def test_graph_coverage_line_carries_every_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_steps_to",
        "steps",
        F.steps_unknown(),
        **{"from": F.HOME, "to": F.RUN_DETAIL},
    )
    line = first_line(out)
    assert line == (
        "COVERAGE: ledger=writing | edgesObserved=12 (toNodeUnobserved 0, "
        "filteredOut 0, withdrawn 0, rowsRejected 0 entries) | nodes=4 (unmodelled 1) | "
        "frontierOpen=2 at 1 node(s) (effect_destructive 1, not_yet_activated 1) | "
        "unmeasured: frontier x2 frontier_open | "
        "filter: since=none app_version=none run_kind=none | runs=3 | "
        "edge window: 2026-10-20T09:00:00Z .. 2026-10-23T17:30:00Z | "
        "appVersions=[(not reported), 1.4.0]"
    )


def test_truncated_output_still_carries_the_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_reachable_from",
        "reachable",
        F.reachable_measured(extra_nodes=200),
        **{"from": F.HOME, "limit": 5},
    )
    # The tool's own listing is bounded, and says how much it left out.
    assert "... and 197 more (not listed; raise `limit`)" in out
    assert "console#page-004" not in out
    # A client that cuts the result after its first line still has the
    # coverage: ledger state, counts, unmeasured dimensions and filter.
    cut = out[: len(first_line(out))]
    for fact in (
        "ledger=writing",
        "edgesObserved=12",
        "nodes=202",
        "frontierOpen=1",
        "frontier x1 frontier_open",
        "unobserved_departures x1 to_node_unobserved",
        "filter: since=none",
    ):
        assert fact in cut


def test_filter_is_sent_and_a_dropped_one_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, runner = run(
        monkeypatch,
        "journey_reachable_from",
        "reachable",
        F.reachable_unobserved_start(),
        **{
            "from": "console#nowhere",
            "since": "2026-10-22T00:00:00Z",
            "run_kind": "explorer",
        },
    )
    assert runner.requests[0][1] == f"{ROOT}/reachable"
    line = first_line(out)
    assert "filter: since=2026-10-22T00:00:00Z app_version=none run_kind=none" in line
    assert (
        "FILTER NOT APPLIED AS ASKED: run_kind asked 'explorer', applied None" in line
    )


def test_only_set_parameters_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.URL] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=F.dead_ends_absent())

    answering(monkeypatch, {f"{ROOT}/dead-ends": None})
    assert server_mod.client is not None
    server_mod.client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(capture)
    )
    call("journey_dead_ends", {"app_id": F.APP, "app_version": "1.4.0"})
    assert dict(seen[0].params) == {"app_version": "1.4.0"}


# =============================================================================
# ledger unavailable
# =============================================================================


@pytest.mark.parametrize(
    ("tool", "route", "args"),
    [
        ("journey_steps_to", "steps", {"from": F.HOME, "to": F.RUNS}),
        ("journey_graph", "graph", {}),
        ("journey_dead_ends", "dead-ends", {}),
        ("journey_edge_timing", "edge-timing", {"from": F.HOME, "to": F.RUNS}),
    ],
)
def test_schema_absent_is_unknown_never_an_empty_graph(
    monkeypatch: pytest.MonkeyPatch, tool: str, route: str, args: dict[str, Any]
) -> None:
    out, is_error, _ = run(monkeypatch, tool, route, F.schema_absent(route), **args)
    assert not is_error
    line = first_line(out)
    assert line.startswith(
        "COVERAGE: ledger=schema_absent (embedded DB predates the journey tables)"
    )
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (ledger_schema_absent)")
    assert "Nothing here is an empty graph." in out
    assert "ABSENT" not in out.replace("ledger_schema_absent", "")
    assert "Settled dead ends" not in out


def test_failed_read_carries_no_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", F.read_failed("dead-ends")
    )
    line = first_line(out)
    assert "edgesObserved=UNKNOWN nodes=UNKNOWN frontierOpen=UNKNOWN" in line
    assert "not zero" in line
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (producer_failed)")


def test_empty_ledger_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", F.empty_ledger("dead-ends")
    )
    assert "edgesObserved=0 " in first_line(out)
    assert "no_observations x1 no_observations" in first_line(out)
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (no_observations)")


def test_runner_refusal_is_a_tool_error_with_its_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = httpx.Response(
        400,
        json={
            "error": "bad_request",
            "detail": "`since` is not RFC 3339 (premature end of input): yesterday",
        },
    )
    out, is_error, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", refusal, since="yesterday"
    )
    assert is_error
    assert "status=400" in out
    assert "`since` is not RFC 3339" in out


def test_missing_arguments_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = answering(monkeypatch)
    # The input schema refuses an absent argument; the tool refuses an empty one.
    result = call("journey_steps_to", {"app_id": F.APP, "from": F.HOME})
    assert result.isError and "'to' is a required property" in text(result)
    result = call("journey_steps_to", {"app_id": F.APP, "from": F.HOME, "to": ""})
    assert result.isError and "`to` required" in text(result)
    result = call("journey_dead_ends", {"app_id": ""})
    assert result.isError and "`app_id` is required" in text(result)
    result = call("journey_dead_ends", {"app_id": F.APP, "limit": 0})
    assert result.isError and "0 is less than the minimum of 1" in text(result)
    assert runner.requests == []


# =============================================================================
# dead ends
# =============================================================================


def test_dead_ends_separate_settled_from_unexplored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", F.dead_ends_measured()
    )
    lines = out.splitlines()
    settled = lines.index("  Settled dead ends (3):")
    unexplored = lines.index(
        "  Unexplored or unsettled - NOT dead ends, NOT exits (2):"
    )
    assert settled < unexplored
    settled_block = lines[settled + 1 : unexplored]
    unexplored_block = lines[unexplored + 1 : unexplored + 3]
    assert all("console#runs" not in line for line in settled_block)
    assert unexplored_block == [
        "    console#runs: open frontier effect_write 1",
        "    unmodelled:/admin/settings: unmeasured departures: "
        "settle_timeout_departures 1; unconfirmed destination (every observed "
        "arrival ended settle_timeout or error - it may be an intermediate state, "
        "not a page)",
    ]


def test_dead_end_back_only_null_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", F.dead_ends_measured()
    )
    assert (
        "    console#run-detail - back_only: true (the only way out is the browser "
        "back button)"
    ) in out
    assert "    console#about - back_only: false" in out
    assert "    console#export-done - back_only: UNKNOWN (" in out


def test_no_settled_dead_end_with_unexplored_nodes_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", F.dead_ends_unknown_unexplored()
    )
    assert out.splitlines()[1].startswith(
        "ANSWER: UNKNOWN (frontier_open) - NOT a negative: this does NOT mean that "
        "there are no dead ends."
    )
    assert "  Settled dead ends (0):\n    (none)" in out
    assert "    console#run-detail: open frontier not_yet_activated 1" in out


def test_settled_no_dead_end_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(monkeypatch, "journey_dead_ends", "dead-ends", F.dead_ends_absent())
    assert out.splitlines()[1].startswith("ANSWER: ABSENT - no dead end (a SETTLED")


# =============================================================================
# prior knowledge, frontier, health
# =============================================================================


def test_prior_knowledge_lists_arrivals(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_requires_prior_knowledge",
        "prior-knowledge",
        F.prior_knowledge_measured(),
        app_version="1.4.0",
        run_kind="agent_action",
    )
    assert "filter: since=none app_version=1.4.0 run_kind=agent_action" in first_line(
        out
    )
    assert "    unmodelled:/admin/settings - arrivals initial 2, push 1" in out


def test_prior_knowledge_unsettled_is_its_own_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_requires_prior_knowledge",
        "prior-knowledge",
        F.prior_knowledge_measured(),
    )
    lines = out.splitlines()
    settled = next(i for i, x in enumerate(lines) if x.startswith("  Settled: reached"))
    unsettled = lines.index(
        "  Unsettled - NOT prior-knowledge findings, NOT ruled out (1):"
    )
    assert settled < unsettled
    assert all("console#invite" not in x for x in lines[settled:unsettled])
    assert lines[unsettled + 1] == (
        "    console#invite - arrivals initial 1; could still be reached by: "
        "2 filtered out click/back arrival(s) into it"
    )
    assert "1 node(s) settled as reachable only with prior knowledge" in lines[1]


def test_prior_knowledge_graph_doubt_in_coverage_and_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_requires_prior_knowledge",
        "prior-knowledge",
        F.prior_knowledge_graph_doubt(),
    )
    line = first_line(out)
    assert "unplacedRejectedRows=2 | unknownDestinationArrivals=1" in line
    assert "rowsRejected 1 entry (distinct row count not in this answer)" in line
    # The coverage agrees with the unsettled node's ledgerNotWriting and
    # openFrontierEntries, and the leading `ledger` dimension sets the code.
    assert line.startswith("COVERAGE: ledger=write_failing (")
    assert "frontierOpen=3 at 1 node(s)" in line
    assert (
        "unmeasured: ledger x1 ledger_write_failing; frontier x3 frontier_open; "
        "unplaced_rejected_rows x2 producer_failed; unsettled_prior_knowledge_nodes "
        "x1 input_missing"
    ) in line
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (ledger_write_failing)")
    assert "  Settled: reached only by URL" in out and "(0):\n    (none)" in out
    assert (
        "    unmodelled:/admin/settings - arrivals push 1; could still be reached "
        "by: 2 rejected edge row(s) placeable at no node (any may arrive here); 3 "
        "open frontier entries in the graph (any may be the link into it); the "
        "ledger is not writing (a click into it may not have been recorded); 1 "
        "click/back row(s) with an unknown destination (any may arrive here); "
        "unconfirmed destination (no arrival ended changed or no_change - it may "
        "be an intermediate state, not a page a URL leads to)"
    ) in out
    assert "    2 rejected edge row(s) placeable at no node (could be anywhere)" in out


def test_prior_knowledge_settled_and_unsettled_is_degraded_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_requires_prior_knowledge",
        "prior-knowledge",
        F.prior_knowledge_measured(),
    )
    assert "unmeasured: unsettled_prior_knowledge_nodes x1 input_missing" in first_line(
        out
    )
    assert out.splitlines()[1].startswith("ANSWER: MEASURED - 1 node(s) settled")
    assert "[DEGRADED:" in out.splitlines()[1]


def test_rejected_rows_sum_distinct_rows() -> None:
    from ui_bridge_mcp.journey import JourneyAnswer, graph_coverage_line

    line = graph_coverage_line(JourneyAnswer(F.graph_with_rejected_rows()), None)
    assert "rowsRejected 2 entries = 3 distinct row(s)" in line
    assert "unplacedRejectedRows=2" in line


def test_graph_doubt_is_absent_from_coverage_when_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_requires_prior_knowledge",
        "prior-knowledge",
        F.prior_knowledge_measured(),
    )
    assert "unplacedRejectedRows" not in first_line(out)
    assert "unknownDestinationArrivals" not in first_line(out)


def test_prior_knowledge_without_unsettled_list_says_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.prior_knowledge_measured()
    del body["value"]["unsettled"]
    out, _, _ = run(
        monkeypatch, "journey_requires_prior_knowledge", "prior-knowledge", body
    )
    assert "  Unsettled: UNKNOWN (the answer carried no `unsettled` list" in out


@pytest.mark.parametrize(
    ("tool", "route", "body", "heading", "code", "before", "after"),
    [
        (
            "journey_dead_ends",
            "dead-ends",
            F.dead_ends_read_gap(),
            "  Dead ends among the rows read - NOT settled, the read had a gap (3):",
            "frontier_open",
            "frontier x1 frontier_open; settle_timeout_departures x1 "
            "no_event_before_timeout; ",
            "; unexplored_nodes x2 input_missing",
        ),
        (
            "journey_requires_prior_knowledge",
            "prior-knowledge",
            F.prior_knowledge_read_gap(),
            "  Reached only by URL among the rows read - NOT settled, the read had "
            "a gap (1):",
            "input_missing",
            "",
            "; unsettled_prior_knowledge_nodes x1 input_missing",
        ),
        (
            "journey_frontier",
            "frontier",
            F._read_gap(F.frontier_measured(), "frontier"),
            "  Open frontier entries in the whole graph among the rows read - NOT "
            "complete, the read had a gap (2):",
            "frontier_open",
            "frontier x2 frontier_open; ",
            " | filter:",
        ),
    ],
)
def test_read_gap_is_unknown_with_the_gap_named(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    route: str,
    body: Any,
    heading: str,
    code: str,
    before: str,
    after: str,
) -> None:
    """The base answer's own dimensions are kept, the read's gaps follow them
    and the route's extra dimension comes last (routes.rs ``answer``); the
    code is the first dimension's."""
    out, is_error, _ = run(monkeypatch, tool, route, body)
    assert not is_error
    lines = out.splitlines()
    gap = (
        "edges_beyond_read_bound_20000 x1 input_missing; "
        "rows_unparseable x3 producer_failed"
    )
    assert f"unmeasured: {before}{gap}{after}" in lines[0]
    assert lines[1].startswith(f"ANSWER: UNKNOWN ({code}) - NOT a negative")
    assert lines[2] == (
        f"  Read gap: {gap} - the ledger was not read in full, so nothing below "
        "is settled."
    )
    assert "since" not in lines[1] + lines[2]
    assert heading in lines
    assert "Settled dead ends" not in out
    assert "MEASURED" not in out


def test_timeline_network_event_with_null_url_renders() -> None:
    timeline = {
        "recordedMs": 900,
        "settled": True,
        "summary": None,
        "observedSources": ["network"],
        "events": [
            {"type": "action", "offsetMs": 0},
            {
                "type": "network-request-started",
                "offsetMs": 5,
                "requestId": None,
                "method": "GET",
                "url": None,
                "hasQuery": False,
            },
        ],
    }
    lines = format_timeline(timeline, expected_state_ids=None)
    assert lines[-1] == "    events: 2 (action 1, network-request-started 1)"


def test_frontier_lists_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, runner = run(
        monkeypatch, "journey_frontier", "frontier", F.frontier_measured(), node=F.RUNS
    )
    assert runner.requests[0][1] == f"{ROOT}/frontier"
    assert "Open frontier entries at 'console#runs' (2):" in out
    assert (
        "console#runs affordance fp-delete (button; destructive) "
        "reason=effect_destructive lastSeen=2026-10-23T17:30:00Z"
    ) in out


def test_health_prints_ledger_first(monkeypatch: pytest.MonkeyPatch) -> None:
    out, is_error, _ = run(monkeypatch, "journey_health", "health", F.health())
    assert not is_error
    assert first_line(out) == "LEDGER: ledger=writing"
    assert "schemaProbe: present" in out
    assert "  edgesWritten: 12" in out
    assert "  lastError: none recorded" in out


def test_health_prints_every_field_it_has_no_place_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block a newer runner adds is shown whole and uninterpreted, never
    dropped: its ``null`` may mean UNKNOWN, so it is not read as zero or "none"."""
    out, is_error, _ = run(
        monkeypatch, "journey_health", "health", F.health_with_passive_capture()
    )
    assert not is_error
    assert first_line(out) == "LEDGER: ledger=writing"
    lines = out.splitlines()
    block = lines[lines.index("passiveCapture (as served, uninterpreted):") :]
    assert block == [
        "passiveCapture (as served, uninterpreted):",
        "  closingSnapshotsRefusedTotal: null",
        "  countsComplete: false",
        f"  detail: {F.PASSIVE_NEVER_RECEIVED_EVICTED}",
        "  lastClosingSnapshotRefusal: null",
        "  lastReceivedAt: null",
        "  lastRefusal: null",
        "  receivedTotal: 0",
        "  refusedTotal: null",
        "  scope: this app, since this runner process started",
        "  state: never_received",
        "  unattributedRefusals: 0",
    ]
    assert "appId" not in out


def test_health_without_counters_still_prints_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.health_with_passive_capture()
    del body["counters"]
    out, _, _ = run(monkeypatch, "journey_health", "health", body)
    assert "counters: UNKNOWN (the health answer carried none)" in out
    assert "  state: never_received" in out


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (None, "null"),
        (0, "0"),
        (False, "false"),
        ("text", "text"),
        ({}, "{}"),
        (["é" * 300], '["' + "é" * 300 + '"]'),
    ],
)
def test_health_prints_a_bare_field_whole(
    monkeypatch: pytest.MonkeyPatch, value: Any, shown: str
) -> None:
    out, _, _ = run(
        monkeypatch, "journey_health", "health", {**F.health(), "newField": value}
    )
    assert f"newField (as served, uninterpreted): {shown}" in out.splitlines()


def test_health_takes_no_filter() -> None:
    schema = next(t for t in server_mod.TOOLS if t.name == "journey_health").inputSchema
    assert set(schema["properties"]) == {"app_id"}


# =============================================================================
# the graph
# =============================================================================


NODES_TITLE = (
    "every node of a row the filter kept, plus every node with a valid open "
    "frontier row, which no filter narrows"
)
EDGES_TITLE = (
    "every trigger and outcome; a departure with no observed destination is not an "
    "edge (toNodeUnobserved in COVERAGE); only affordance + changed is a navigation "
    "step"
)


def test_graph_lists_every_node_key_with_what_is_open_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, is_error, runner = run(
        monkeypatch, "journey_graph", "graph", F.graph_measured()
    )
    assert not is_error
    assert runner.requests[0][1] == f"{ROOT}/graph"
    lines = out.splitlines()
    assert lines[0].startswith("COVERAGE: ledger=writing | edgesObserved=7 ")
    assert lines[1].startswith(
        "ANSWER: MEASURED - 3 node(s) and 3 aggregated edge(s) listed [DEGRADED:"
    )
    assert f"  Nodes, {NODES_TITLE} (3):" in lines
    assert (
        f"    {F.HOME} (console, modelled): unmeasured departures: "
        "filtered_departures 1"
    ) in lines
    assert (
        f"    {F.RUNS} (console, modelled): open frontier effect_write 1, "
        "not_yet_activated 1; 1 unobserved departure(s); unmeasured departures: "
        "settle_timeout_departures 2; 1 filtered out click/back arrival(s)"
    ) in lines
    assert (
        f"    {F.SETTINGS} (/admin/settings, unmodelled): unconfirmed destination "
        "(no arrival ended changed or no_change - it may be an intermediate state, "
        "not a page)"
    ) in lines


def test_a_node_with_nothing_open_is_a_bare_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_settled())
    assert f"    {F.HOME} (console, modelled)" in out.splitlines()


def test_graph_lists_edges_of_every_trigger_with_their_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_measured())
    lines = out.splitlines()
    assert f"  Edges, {EDGES_TITLE} (3):" in lines
    assert (
        f"    {F.HOME} -> {F.RUNS} via click/affordance [role=link, target=fp-"
        f"{F.RUNS}] outcome=changed seen 3x between 2026-10-20T09:00:00Z and "
        "2026-10-23T17:30:00Z"
    ) in lines
    assert any(
        f"{F.RUNS} -> {F.HOME} via navigate/pop outcome=changed seen 1x" in line
        for line in lines
    )
    assert any(
        f"{F.RUNS} -> {F.SETTINGS} via click/affordance" in line
        and "outcome=settle_timeout seen 2x" in line
        for line in lines
    )


def test_settled_graph_is_measured_and_not_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_settled())
    assert out.splitlines()[1] == (
        "ANSWER: MEASURED - 2 node(s) and 1 aggregated edge(s) listed"
    )
    assert "Left out" not in out
    assert "Rejected rows" not in out
    assert "UNKNOWN" not in out


def test_graph_names_what_it_placed_at_no_listed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_off_node_doubt())
    lines = out.splitlines()
    assert "unplacedRejectedRows=2" in lines[0]
    assert "rowsRejected 3 entries = 4 distinct row(s)" in lines[0]
    assert "  Rejected rows, which failed their contract (3):" in lines
    assert any(
        line.startswith('    edge edge-9 x2: observedAt "yesterday" is not RFC 3339')
        for line in lines
    )
    assert (
        "    frontier bad#node#key fp-y x1: node is not canonical: specId must not "
        "contain '#' (the key's spec/state separator)"
    ) in lines
    assert "  Left out at no listed node:" in lines
    assert (
        "    2 rejected edge row(s) placeable at no node (could be anywhere)" in lines
    )
    assert (
        "    1 rejected frontier row(s) placeable at no node (could be anywhere)"
        in lines
    )
    assert (
        "    off-graph unmeasured departures: rejected_frontier 1, "
        "filtered_departures 4"
    ) in lines
    assert (
        "  1 rejected frontier row(s) are counted at two node keys, listed or "
        "off-graph (their key column and their node disagree)."
    ) in lines


def test_graph_with_no_node_or_edge_still_names_the_rows_it_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_only_rejected_rows())
    lines = out.splitlines()
    assert lines[1].startswith("ANSWER: UNKNOWN (no_observations)")
    assert "Nodes" not in out
    assert "Edges" not in out
    assert any(line.startswith("    edge edge-9 x2: ") for line in lines)
    assert (
        "    off-graph unmeasured departures: rejected_frontier 2, "
        "filtered_departures 4"
    ) in lines


def test_graph_with_no_edge_is_unknown_and_still_names_its_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontier is never filtered, so a node can outlive every edge."""
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_frontier_only())
    lines = out.splitlines()
    assert lines[1].startswith("ANSWER: UNKNOWN (no_observations) - NOT a negative")
    assert f"    {F.HOME} (console, modelled): open frontier not_yet_activated 1" in (
        lines
    )
    assert "    (none)" in lines


def test_empty_graph_lists_nothing_as_if_it_were_a_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_empty())
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (no_observations)")
    assert "Nodes" not in out
    assert "Edges" not in out
    assert "UNKNOWN, not zero" not in out


def test_failed_graph_read_carries_no_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    out, is_error, _ = run(
        monkeypatch, "journey_graph", "graph", F.read_failed("graph")
    )
    assert not is_error
    assert "edgesObserved=UNKNOWN nodes=UNKNOWN frontierOpen=UNKNOWN" in first_line(out)
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (producer_failed)")
    assert "Nodes" not in out


def test_graph_read_gap_says_the_listing_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_read_gap())
    lines = out.splitlines()
    assert lines[1].startswith("ANSWER: MEASURED")
    assert "[DEGRADED:" in lines[1]
    assert lines[2].startswith("  Read gap: edges_beyond_read_bound_20000 x1")
    assert (
        "  Nodes among the rows read - NOT complete, the read had a gap, "
        f"{NODES_TITLE} (2):"
    ) in lines
    assert (
        "  Edges among the rows read - NOT complete, the read had a gap, "
        f"{EDGES_TITLE} (1):"
    ) in lines


def test_graph_limit_cuts_the_lists_and_never_the_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_measured(), limit=1)
    lines = out.splitlines()
    assert f"  Nodes, {NODES_TITLE} (3):" in lines
    assert f"  Edges, {EDGES_TITLE} (3):" in lines
    assert lines.count("    ... and 2 more (not listed; raise `limit`)") == 2
    assert lines[1].startswith("ANSWER: MEASURED - 3 node(s) and 3 aggregated edge(s)")


def test_graph_sends_only_the_filter_and_flags_a_dropped_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``limit`` is the tool's own; the runner refuses a parameter it does not read."""
    answering(monkeypatch)
    seen: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=F.graph_measured())

    server_mod.client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(capture)
    )
    result = call(
        "journey_graph", {"app_id": F.APP, "run_kind": "explorer", "limit": 5}
    )
    assert dict(seen[0].url.params) == {"run_kind": "explorer"}
    assert "FILTER NOT APPLIED AS ASKED: run_kind asked 'explorer'" in first_line(
        text(result)
    )


@pytest.mark.parametrize("missing", ["nodes", "edges"])
def test_a_graph_without_a_list_is_unknown_not_zero(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    body = F.graph_settled()
    del body["value"][missing]
    out, _, _ = run(monkeypatch, "journey_graph", "graph", body)
    lines = out.splitlines()
    assert (
        "UNKNOWN " + ("node(s)" if missing == "nodes" else "aggregated edge(s)")
        in (lines[1])
    )
    title = f"Nodes, {NODES_TITLE}" if missing == "nodes" else f"Edges, {EDGES_TITLE}"
    assert f"  {title}: UNKNOWN (the answer carried no list)" in lines


def test_a_graph_without_its_left_out_fields_is_unknown_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.graph_settled()
    for key in (
        "rejected",
        "offGraphDepartures",
        "unplacedRejectedRows",
        "unplacedRejectedFrontier",
        "rejectedFrontierDoublePlaced",
        "unknownDestinationArrivals",
    ):
        del body["value"][key]
    out, _, _ = run(monkeypatch, "journey_graph", "graph", body)
    lines = out.splitlines()
    assert "  Rejected rows: UNKNOWN (the answer carried no list)" in lines
    assert (
        "  UNKNOWN, not zero (the answer did not carry them): unplacedRejectedRows, "
        "unplacedRejectedFrontier, rejectedFrontierDoublePlaced, "
        "unknownDestinationArrivals, offGraphDepartures"
    ) in lines


def test_a_partial_departures_block_names_the_missing_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.graph_settled()
    del body["value"]["offGraphDepartures"]["withdrawn"]
    del body["value"]["nodes"][0]["unmeasuredDepartures"]["errored"]
    out, _, _ = run(monkeypatch, "journey_graph", "graph", body)
    lines = out.splitlines()
    assert (
        "  UNKNOWN, not zero (the answer did not carry them): "
        "offGraphDepartures.withdrawn_departures"
    ) in lines
    assert (
        f"    {F.HOME} (console, modelled): UNKNOWN, not clear: errored_departures"
    ) in lines


def test_a_node_missing_its_doubt_fields_never_reads_as_explored() -> None:
    assert _graph_node_text({"key": "a#b", "node": {"modelled": True}}) == (
        "a#b (modelled): UNKNOWN, not clear: frontier, unobservedDepartures, "
        "unmeasuredDepartures, arrivalDoubt, unconfirmedDestination"
    )
    assert _graph_node_text({"key": "a#b"}).startswith("a#b (modelled UNKNOWN): ")


@pytest.mark.parametrize(
    ("field", "served"),
    [
        ("frontier", None),
        ("frontier", "3"),
        ("unobservedDepartures", None),
        ("unobservedDepartures", "3"),
        ("unobservedDepartures", True),
        ("unmeasuredDepartures", None),
        ("unmeasuredDepartures", []),
        ("arrivalDoubt", None),
        ("arrivalDoubt", []),
        ("unconfirmedDestination", None),
        ("unconfirmedDestination", "false"),
    ],
)
def test_a_served_null_or_wrong_type_is_unknown_not_nothing_open(
    field: str, served: Any
) -> None:
    text = _graph_node_text({**F.graph_node(F.HOME), field: served})
    assert text.endswith(f"UNKNOWN, not clear: {field}")


@pytest.mark.parametrize("served", [None, "3", True])
@pytest.mark.parametrize("field", ["filtered", "withdrawn"])
def test_a_null_arrival_doubt_counter_is_unknown(field: str, served: Any) -> None:
    node = F.graph_node(
        F.HOME, arrivalDoubt={"filtered": 0, "withdrawn": 0, field: served}
    )
    assert _graph_node_text(node).endswith(f"UNKNOWN, not clear: arrivalDoubt.{field}")


@pytest.mark.parametrize("served", [None, "3", True])
def test_a_null_departure_counter_is_unknown(
    monkeypatch: pytest.MonkeyPatch, served: Any
) -> None:
    body = F.graph_settled()
    body["value"]["offGraphDepartures"]["withdrawn"] = served
    body["value"]["nodes"][0]["unmeasuredDepartures"]["errored"] = served
    out, _, _ = run(monkeypatch, "journey_graph", "graph", body)
    lines = out.splitlines()
    assert (
        "  UNKNOWN, not zero (the answer did not carry them): "
        "offGraphDepartures.withdrawn_departures"
    ) in lines
    assert (
        f"    {F.HOME} (console, modelled): UNKNOWN, not clear: errored_departures"
    ) in lines


def test_arrival_doubt_labels_say_which_exclusion_it_was() -> None:
    node = F.graph_node(F.HOME, arrivalDoubt={"filtered": 1, "withdrawn": 2})
    assert _graph_node_text(node).endswith(
        "1 filtered out click/back arrival(s); 2 withdrawn click/back arrival(s)"
    )


def test_a_rejected_row_without_a_count_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.graph_off_node_doubt()
    body["value"]["rejected"][0]["rows"] = None
    out, _, _ = run(monkeypatch, "journey_graph", "graph", body)
    assert any(
        line.startswith("    edge edge-9 xUNKNOWN: ") for line in out.splitlines()
    )


def test_the_answer_does_not_call_a_frontier_node_observed_under_the_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(monkeypatch, "journey_graph", "graph", F.graph_measured())
    assert "observed under" not in out
    assert "which no filter narrows" in out


def test_every_journey_tool_has_a_route_and_every_route_a_tool() -> None:
    """The runner's ``journey/routes.rs`` ``router()``, as of its landed routes."""
    tools = {t.name for t in server_mod.TOOLS if t.name.startswith("journey_")}
    assert tools == set(server_mod._JOURNEY_ROUTES)
    assert {route for route, _, _ in server_mod._JOURNEY_ROUTES.values()} == {
        "health",
        "graph",
        "reachable",
        "steps",
        "dead-ends",
        "prior-knowledge",
        "frontier",
        "edge-timing",
    }


def test_node_key_description_names_the_tool_that_lists_every_key() -> None:
    schema = next(
        t for t in server_mod.TOOLS if t.name == "journey_reachable_from"
    ).inputSchema
    assert "journey_graph" in schema["properties"]["from"]["description"]


# =============================================================================
# edge timing
# =============================================================================


def test_edge_timing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    out, is_error, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        F.edge_timing_measured(),
        **{"from": F.HOME, "to": F.RUNS},
    )
    assert not is_error
    assert first_line(out) == (
        "COVERAGE: ledger=writing | n=4 (withoutTimeline 1, timelinesUnparseable 0; "
        "excluded: filteredOut 1, withdrawn 0; untrusted edge rows anywhere in the read: rowsRejected 0) | pooled: by outcome "
        "changed 3, settle_timeout 1; by trigger affordance 4 | unmeasured: timeline x1 "
        "input_missing | filter: since=none app_version=none run_kind=none | "
        "observation window: 2026-10-21T10:00:00Z .. 2026-10-24T10:00:00Z | "
        "appVersions=[1.4.0]"
    )
    assert out.splitlines()[1].startswith(
        "ANSWER: MEASURED - 4 observation(s) of 'console#home' -> 'console#runs' "
        "(a distribution over them, not a single latency)"
    )
    assert (
        "  settleMs: min 100 | median 200 | p90 300 | max 300 - measured 3 "
        "(degraded 0), absent 0, unknown 0, without timeline 1"
    ) in out
    assert (
        "  layoutShiftTotal: no measured value, so no min/median/p90/max - measured 0 "
        "(degraded 0), absent 0, unknown 3 {performance_observer_unsupported 3}, "
        "without timeline 1"
    ) in out
    assert "  longTaskTotalMs: min 0 | median 0 | p90 0 | max 0" in out
    assert "obs-4 at 2026-10-24T10:00:00Z app 1.4.0: no timeline recorded" in out
    assert "settleMs measured 300" in out
    assert "layoutShiftTotal UNKNOWN (performance_observer_unsupported:" in out
    assert "latency:" not in out.lower()


def test_edge_timing_n_zero_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    out, is_error, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        F.edge_timing_empty(),
        **{"from": F.HOME, "to": F.SETTINGS, "since": "2026-10-23T00:00:00Z"},
    )
    assert not is_error
    line = first_line(out)
    assert (
        "n=0 (withoutTimeline 0, timelinesUnparseable 0; excluded: filteredOut 2, "
        "withdrawn 1; untrusted edge rows anywhere in the read: rowsRejected 0) | pooled: by outcome none; by trigger none"
        in line
    )
    assert "unmeasured: none |" in line
    assert "observation window: none (no observation)" in line
    answer = out.splitlines()[1]
    assert answer.startswith("ANSWER: UNKNOWN (no_observations) - NOT a negative")
    assert "nothing about its timing was measured" in answer
    assert "  settleMs: no measured value, so no min/median/p90/max - measured 0" in out
    assert "Observations (" not in out


def test_since_echoed_in_utc_is_the_same_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.reachable_unobserved_start()
    body["provenance"]["source"]["result"]["filterEcho"]["since"] = (
        "2026-10-21T22:00:00Z"
    )
    out, _, _ = run(
        monkeypatch,
        "journey_reachable_from",
        "reachable",
        body,
        **{"from": "console#nowhere", "since": "2026-10-22T00:00:00+02:00"},
    )
    assert "FILTER NOT APPLIED AS ASKED" not in first_line(out)


def test_edge_timing_untrusted_rows_set_the_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        F.edge_timing_rejected(),
        **{"from": F.HOME, "to": F.SETTINGS, "since": "2026-10-23T00:00:00Z"},
    )
    line = first_line(out)
    assert (
        "withdrawn 1; untrusted edge rows anywhere in the read: rowsRejected 4)" in line
    )
    assert "unmeasured: unplaced_rejected_rows x4 producer_failed" in line
    assert "rows_rejected x" not in line.replace("unplaced_rejected_rows x", "")
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (producer_failed)")


def test_missing_unmeasured_is_unknown_not_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.steps_measured()
    del body["provenance"]["coverage"]["unmeasured"]
    out, _, _ = run(
        monkeypatch,
        "journey_steps_to",
        "steps",
        body,
        **{"from": F.HOME, "to": F.RUN_DETAIL},
    )
    line = first_line(out)
    assert "unmeasured: UNKNOWN (the answer did not report what it left out)" in line
    assert "unmeasured: none" not in line
    # Not reported is not settled: the measured answer is marked degraded.
    assert "[DEGRADED:" in out.splitlines()[1]


@pytest.mark.parametrize(
    "body", [F.read_failed("dead-ends"), F.schema_absent("dead-ends")]
)
def test_unread_ledger_reports_no_gaps_as_not_read(
    monkeypatch: pytest.MonkeyPatch, body: Any
) -> None:
    out, _, _ = run(monkeypatch, "journey_dead_ends", "dead-ends", body)
    assert "unmeasured: none reported (the ledger was not read)" in first_line(out)


def test_empty_ledger_hides_empty_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _, _ = run(
        monkeypatch, "journey_dead_ends", "dead-ends", F.empty_ledger("dead-ends")
    )
    assert "Settled dead ends" not in out
    assert "Unexplored or unsettled" not in out
    assert "(none)" not in out.split("\n", 1)[1]


def test_measured_edge_timing_with_untrusted_rows_says_n_is_a_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = F.edge_timing_measured()
    body["value"]["rowsRejected"] = 4
    body["provenance"]["coverage"]["unmeasured"].append(
        F.dim("unplaced_rejected_rows", 4, "producer_failed")
    )
    out, _, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        body,
        **{"from": F.HOME, "to": F.RUNS},
    )
    answer = out.splitlines()[1]
    assert answer.startswith("ANSWER: MEASURED")
    assert "n is a LOWER BOUND" in answer
    assert "not every observation yielded every value" not in answer


def test_measured_edge_timing_missing_only_values_keeps_the_value_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        F.edge_timing_measured(),
        **{"from": F.HOME, "to": F.RUNS},
    )
    answer = out.splitlines()[1]
    assert "not every observation yielded every value" in answer
    assert "LOWER BOUND" not in answer


@pytest.mark.parametrize(
    "extra",
    [
        ("unplaced_rejected_rows", 4, "producer_failed"),
        ("ledger", 1, "ledger_write_failing"),
        ("edges_beyond_read_bound_20000", 1, "input_missing"),
        ("rows_unparseable", 2, "producer_failed"),
    ],
)
def test_every_uncounted_row_dimension_makes_n_a_lower_bound(
    monkeypatch: pytest.MonkeyPatch, extra: tuple[str, int, str]
) -> None:
    body = F.edge_timing_measured()
    body["provenance"]["coverage"]["unmeasured"].append(F.dim(*extra))
    out, _, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        body,
        **{"from": F.HOME, "to": F.RUNS},
    )
    answer = out.splitlines()[1]
    assert "n is a LOWER BOUND" in answer
    assert "not every observation yielded every value" not in answer


@pytest.mark.parametrize("unmeasured", ["delete", None, "not-a-list"])
def test_a_missing_unmeasured_list_never_earns_the_value_only_reason(
    monkeypatch: pytest.MonkeyPatch, unmeasured: object
) -> None:
    body = F.edge_timing_measured()
    if unmeasured == "delete":
        del body["provenance"]["coverage"]["unmeasured"]
    else:
        body["provenance"]["coverage"]["unmeasured"] = unmeasured
    out, _, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        body,
        **{"from": F.HOME, "to": F.RUNS},
    )
    answer = out.splitlines()[1]
    assert "n may be a LOWER BOUND" in answer
    assert "UNKNOWN" in answer
    assert "not every observation yielded every value" not in answer


@pytest.mark.parametrize("junk", ["timeline", ["timeline"], {"count": 1}])
def test_an_unreadable_unmeasured_entry_never_earns_the_value_only_reason(
    monkeypatch: pytest.MonkeyPatch, junk: object
) -> None:
    body = F.edge_timing_measured()
    body["provenance"]["coverage"]["unmeasured"].append(junk)
    out, _, _ = run(
        monkeypatch,
        "journey_edge_timing",
        "edge-timing",
        body,
        **{"from": F.HOME, "to": F.RUNS},
    )
    answer = out.splitlines()[1]
    assert "n is a LOWER BOUND" in answer
    assert "not every observation yielded every value" not in answer
