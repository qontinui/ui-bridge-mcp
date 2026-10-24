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
        "filteredOut 0, withdrawn 0, rowsRejected 0) | nodes=4 (unmodelled 1) | "
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
        "settle_timeout_departures 1",
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
    assert "untrustedEdgeRows=2 | unknownDestinationArrivals=1" in line
    assert out.splitlines()[1].startswith("ANSWER: UNKNOWN (producer_failed)")
    assert "  Settled: reached only by URL" in out and "(0):\n    (none)" in out
    assert (
        "    unmodelled:/admin/settings - arrivals push 1; could still be reached "
        "by: 2 untrusted edge row(s) in the graph (any may arrive here); 1 "
        "click/back row(s) with an unknown destination (any may arrive here)"
    ) in out


def test_graph_doubt_is_absent_from_coverage_when_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _, _ = run(
        monkeypatch,
        "journey_requires_prior_knowledge",
        "prior-knowledge",
        F.prior_knowledge_measured(),
    )
    assert "untrustedEdgeRows" not in first_line(out)
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
    ("tool", "route", "body", "heading"),
    [
        (
            "journey_dead_ends",
            "dead-ends",
            F.dead_ends_read_gap(),
            "  Dead ends among the rows read - NOT settled, the read had a gap (3):",
        ),
        (
            "journey_requires_prior_knowledge",
            "prior-knowledge",
            F.prior_knowledge_read_gap(),
            "  Reached only by URL among the rows read - NOT settled, the read had "
            "a gap (1):",
        ),
    ],
)
def test_read_gap_is_unknown_with_the_gap_named(
    monkeypatch: pytest.MonkeyPatch, tool: str, route: str, body: Any, heading: str
) -> None:
    out, is_error, _ = run(monkeypatch, tool, route, body)
    assert not is_error
    lines = out.splitlines()
    assert (
        "unmeasured: edges_beyond_read_bound_50000 x1 input_missing; "
        "rows_unparseable x3 producer_failed"
    ) in lines[0]
    assert lines[1].startswith("ANSWER: UNKNOWN (input_missing) - NOT a negative")
    assert lines[2] == (
        "  Read gap: edges_beyond_read_bound_50000 x1 input_missing; "
        "rows_unparseable x3 producer_failed - the ledger was not read in full, so "
        "nothing below is settled (narrow with `since`)."
    )
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


def test_health_takes_no_filter() -> None:
    schema = next(t for t in server_mod.TOOLS if t.name == "journey_health").inputSchema
    assert set(schema["properties"]) == {"app_id"}


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
        "excluded: filteredOut 1, withdrawn 0, rowsRejected 0) | pooled: by outcome "
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
        "withdrawn 1, rowsRejected 1) | pooled: by outcome none; by trigger none"
        in line
    )
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
