"""A failed tool call reaches the client with isError: true, an answer with false.

The MCP spec reports a tool's own failure inside the result, with
``isError: true``. Until 2026-10-13 every ui-bridge-mcp tool returned a plain
content list, which the SDK always wraps as ``isError: false``. A client that
branched on the flag read every failure as a success: an unreachable runner, a
failed request, an unknown tool, a caught exception (coord finding 1b13b2de).

These tests drive the server through a real in-memory MCP client session, so
they observe the flag exactly as a client reads it.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeResponse
from ui_bridge_mcp.screenshot import BaselineStore

from .mcp_session import answering, call, install, text


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


def ok(data: Any) -> dict[str, Any]:
    """A success envelope carrying ``data``."""
    return {"success": True, "data": data}


def _call_over_session(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """``mcp_session.call``, spelled out in this file for the arming proof.

    The arming prover reads only a test's same-file closure, so the test that
    arms this branch (test_failed_request_is_an_error) has to reach the MCP
    client session's ``call_tool`` in this file, where the population enters.
    """

    async def go() -> types.CallToolResult:
        try:
            async with create_connected_server_and_client_session(
                server_mod.server
            ) as session:
                return await session.call_tool(name, arguments)
        finally:
            if server_mod.client is not None:
                await server_mod.client.close()

    return asyncio.run(go())


# =============================================================================
# Failures carry isError: true
# =============================================================================


def test_unreachable_runner_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, port=_closed_port())
    result = call("ui_health", {})
    assert result.isError is True
    assert "code=app_unreachable" in text(result)


def test_failed_request_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = httpx.Response(
        500, json={"success": False, "error": "x", "code": "UB-NET-ERROR"}
    )
    answering(monkeypatch, {"/ui-bridge/control/snapshot": failed})
    result = _call_over_session("ui_snapshot", {})
    assert result.isError is True
    assert text(result).startswith("Error: [code=UB-NET-ERROR status=500]")


def test_unknown_tool_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch)
    result = call("no_such_tool", {})
    assert result.isError is True
    assert text(result) == "Unknown tool: no_such_tool"


def test_caught_exception_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown ref raises before any request; the catch-all reports it."""
    answering(monkeypatch)
    result = call("ui_click", {"element_id": "@e999"})
    assert result.isError is True
    assert text(result).startswith("Error: Unknown ref @e999")


def test_unmet_precondition_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first sdk_diff has nothing to diff against: no diff was produced."""
    monkeypatch.setattr(server_mod, "sdk_diff_tracker", server_mod.DiffTracker())
    answering(monkeypatch, {"/ui-bridge/sdk/snapshot": ok({"elements": [{"id": "s"}]})})
    result = call("sdk_diff", {})
    assert result.isError is True
    assert text(result).startswith("No previous snapshot to diff against.")


def test_failed_assertion_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed assertion is ``passed: false`` inside a SUCCESS envelope.

    ``success: false`` would be the request failing, not the assertion
    (ui-bridge ai/assertions.ts returns ``passed: false`` instead of throwing).
    """
    failed = {
        "passed": False,
        "failureReason": "Element is hidden",
        "failureCode": "UB-ASSERT-NOT-VISIBLE",
    }
    answering(monkeypatch, {"/ui-bridge/sdk/ai/assert": ok(failed)})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is True
    assert text(result).startswith(
        "Assertion failed: [code=UB-ASSERT-NOT-VISIBLE] expected 'Save button' is visible"
    )


# =============================================================================
# Answers carry isError: false, including empty ones
# =============================================================================


def test_answer_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, {"/health": ok({"status": "ok"})})
    result = call("ui_health", {})
    assert result.isError is False
    assert text(result) == "Runner is healthy and accessible."


def test_empty_answer_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """sdk_status reached the runner and it said: not connected. That is an answer."""
    answering(monkeypatch, {"/ui-bridge/sdk/status": ok({"connected": False})})
    result = call("sdk_status", {})
    assert result.isError is False
    assert text(result) == "SDK not connected"


def test_passed_assertion_is_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The success path used to read an unbound ``state`` and report an error."""
    answering(monkeypatch, {"/ui-bridge/sdk/ai/assert": ok({"passed": True})})
    result = call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is False
    assert text(result) == "Assertion passed: 'Save button' is visible"


# =============================================================================
# The helpers, and a ratchet over every failure branch
# =============================================================================


def test_error_helpers_flag_the_result() -> None:
    failed = UIBridgeResponse(success=False, error="boom", code="producer_failed")
    result = server_mod._error_result(failed)
    assert isinstance(result, types.CallToolResult)
    assert result.isError is True
    assert text(result) == "Error: [code=producer_failed] boom"
    assert server_mod._tool_error("nope").isError is True


def _module_ast() -> ast.Module:
    return ast.parse(inspect.getsource(server_mod))


def _call_tool_ast() -> ast.AsyncFunctionDef:
    return next(
        node
        for node in ast.walk(_module_ast())
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "call_tool"
    )


def _is_failed_response_test(test: ast.expr) -> bool:
    """``not <resp>.success``."""
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Attribute)
        and test.operand.attr == "success"
    )


def _is_flagging_call(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("_error_result", "_tool_error")
    )


def test_every_failed_response_branch_returns_a_flagged_error() -> None:
    """``if not <resp>.success:`` must return ``_error_result`` or ``_tool_error``.

    A branch that returned a plain content list would reach the client as
    ``isError: false``: the defect this file exists to keep fixed.
    """
    fn = _call_tool_ast()
    branches = 0
    unflagged: list[int] = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.If) and _is_failed_response_test(node.test)):
            continue
        branches += 1
        returns = [n for n in node.body if isinstance(n, ast.Return)]
        if not (returns and all(_is_flagging_call(r.value) for r in returns)):
            unflagged.append(node.lineno)
    # The scan found the branches, not nothing: a vacuous pass is not a pass.
    assert branches > 100
    assert unflagged == []


def _leading_text(value: ast.expr) -> str | None:
    """The constant text a string or f-string starts with."""
    head = (
        value.values[0] if isinstance(value, ast.JoinedStr) and value.values else value
    )
    if isinstance(head, ast.Constant) and isinstance(head.value, str):
        return head.value
    return None


def test_no_error_text_is_returned_unflagged() -> None:
    """No unflagged answer whose text starts with "Error" survives in the module.

    Covers both answer spellings: an inline ``[TextContent(text=...)]`` and the
    ``_text(...)`` helper.
    """
    offenders: list[int] = []
    for node in ast.walk(_module_ast()):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        texts: list[ast.expr] = []
        if isinstance(node.value, ast.List):
            texts = [
                keyword.value
                for element in node.value.elts
                if isinstance(element, ast.Call)
                for keyword in element.keywords
                if keyword.arg == "text"
            ]
        elif (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_text"
            and node.value.args
        ):
            texts = [node.value.args[0]]
        if any((_leading_text(t) or "").startswith("Error") for t in texts):
            offenders.append(node.lineno)
    assert offenders == []


# =============================================================================
# Every failure path the ratchets cannot see has a behavioural test
# =============================================================================
#
# The two ratchets reach a failure return only when it is a direct return of an
# ``if not <resp>.success:`` branch in call_tool, or when its text starts with
# "Error". Anywhere else -- an unmet precondition, a missing referent, a
# timeout, a verdict read out of a success envelope -- a failure could lose its
# flag with both ratchets green. Each such call site is listed below with a
# scenario that reaches it through an MCP client session, and
# test_failure_path_census pins that list against the source.


def _site_template(arg: ast.expr) -> str:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        return "".join(
            str(v.value) if isinstance(v, ast.Constant) else "{}" for v in arg.values
        )
    return ast.unparse(arg)


def _unreached_failure_sites() -> list[str]:
    """``"<tool or helper>: <message template>"`` for every site no ratchet reaches."""
    tree = _module_ast()
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }

    def enclosing_function(
        node: ast.AST,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return node
        return None

    def tool_branch(node: ast.AST) -> str:
        while node in parents:
            parent = parents[node]
            if isinstance(parent, ast.If) and node in parent.body:
                test = parent.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "name"
                    and len(test.ops) == 1
                ):
                    target = test.comparators[0]
                    if isinstance(test.ops[0], ast.Eq) and isinstance(
                        target, ast.Constant
                    ):
                        return str(target.value)
                    if isinstance(test.ops[0], ast.In) and isinstance(
                        target, ast.Tuple
                    ):
                        return "|".join(
                            str(e.value)
                            for e in target.elts
                            if isinstance(e, ast.Constant)
                        )
            node = parent
        return "call_tool"

    def in_failed_response_branch(node: ast.Call) -> bool:
        ret = parents.get(node)
        if not (isinstance(ret, ast.Return) and ret.value is node):
            return False
        branch = parents.get(ret)
        return (
            isinstance(branch, ast.If)
            and ret in branch.body
            and _is_failed_response_test(branch.test)
        )

    sites: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_flagging_call(node)):
            continue
        assert isinstance(node.func, ast.Name)
        function = enclosing_function(node)
        if function is None or function.name == "_error_result":
            continue
        message = _site_template(node.args[0]) if node.args else ""
        if node.func.id == "_tool_error" and message.startswith("Error"):
            continue  # test_no_error_text_is_returned_unflagged reaches it
        if function.name == "call_tool":
            if in_failed_response_branch(node):
                continue  # test_every_failed_response_branch_... reaches it
            scope = tool_branch(node)
        else:
            scope = function.name
        if node.func.id == "_error_result":
            message = f"_error_result({', '.join(ast.unparse(a) for a in node.args)})"
        sites.append(f"{scope}: {message}")
    return sites


@dataclass(frozen=True)
class FailurePath:
    """A scenario that reaches one failure call site through a client session."""

    site: str
    tool: str
    arguments: dict[str, Any]
    routes: dict[str, Any]
    expect: str
    setup: Callable[[pytest.MonkeyPatch], None] | None = None


def _fresh(
    attribute: str, factory: Callable[[], Any]
) -> Callable[[pytest.MonkeyPatch], None]:
    def setup(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server_mod, attribute, factory())

    return setup


SDK_ABSENT = {"/ui-bridge/sdk/status": ok({"connected": False})}
SDK_CONNECTED = {"/ui-bridge/sdk/status": ok({"connected": True})}
DIFF_TOOLS = {
    "sdk": ("/ui-bridge/sdk/ai", "sdk"),
    "ui": ("/ui-bridge/control/ai", "ui"),
}
FILL_REQUEST = {"fields": {"email": "a@b.c", "name": "Ann"}}
FAILED_FILL = {
    "success": False,
    "filledCount": 1,
    "errorCount": 1,
    "fields": {"email": {"success": True}, "name": {"success": False, "error": "x"}},
}
SEMANTIC_DIFF_NO_CHANGE: dict[str, Any] = {
    "changes": {"appeared": [], "disappeared": [], "modified": []}
}

ASSERT_SITE = "_assertion_result: "
UNDO_SITE = "_undo_redo_result: "
DIFF_SITE = "_diff_result: "
CATEGORY_SITE = "_categorized_diff_result: "

FAILURE_PATHS: list[FailurePath] = [
    FailurePath(
        ASSERT_SITE + "Assertion result UNKNOWN: expected {}, but the response carried "
        "no boolean 'passed' (got {}). This is not a pass.",
        "sdk_ai_assert",
        {"text": "Save"},
        {"/ui-bridge/sdk/ai/assert": ok({})},
        "Assertion result UNKNOWN",
    ),
    FailurePath(
        ASSERT_SITE + "'\\n'.join(lines)",
        "sdk_ai_assert",
        {"text": "Save"},
        {"/ui-bridge/sdk/ai/assert": ok({"passed": False})},
        "Assertion failed:",
    ),
    FailurePath(
        "_nl_action_result: Action outcome UNKNOWN for '{}': the response carried no "
        "boolean 'success' (got {}). This is not a success.",
        "sdk_ai_execute",
        {"instruction": "click Save"},
        {"/ui-bridge/sdk/ai/execute": ok({})},
        "Action outcome UNKNOWN",
    ),
    *[
        FailurePath(
            UNDO_SITE + "{} was not available: nothing was {}.",
            tool,
            {},
            {path: ok({"executed": False})},
            "Undo was not available" if "undo" in tool else "Redo was not available",
        )
        for tool, path in (
            ("sdk_undo", "/ui-bridge/sdk/undo"),
            ("sdk_redo", "/ui-bridge/sdk/redo"),
            ("ui_undo", "/ui-bridge/control/undo"),
            ("ui_redo", "/ui-bridge/control/redo"),
        )
    ],
    *[
        FailurePath(
            UNDO_SITE + "{} outcome UNKNOWN: the response carried no boolean "
            "'executed' (got {}). Whether anything was {} is not known.",
            tool,
            {},
            {path: ok({})},
            f"{verb} outcome UNKNOWN",
        )
        for tool, path, verb in (
            ("sdk_undo", "/ui-bridge/sdk/undo", "Undo"),
            ("ui_redo", "/ui-bridge/control/redo", "Redo"),
        )
    ],
    *[
        FailurePath(
            "_wait_for_change_result: Timed out waiting for change: the condition "
            "was not met before the timeout.",
            tool,
            {"predicate_type": "anyChange"},
            {path: ok({"changed": False, "timedOut": True})},
            "Timed out waiting for change",
        )
        for tool, path in (
            ("sdk_wait_for_change", "/ui-bridge/sdk/ai/wait-for-change"),
            ("ui_wait_for_change", "/ui-bridge/control/ai/wait-for-change"),
        )
    ],
    *[
        FailurePath(
            "_native_capture_result: text",
            tool,
            {},
            {
                "/ui-bridge/control/snapshot": ok(
                    {"source": "native_capture", "elements": []}
                )
            },
            "No element tree:",
            setup=_fresh("control_diff_tracker", server_mod.DiffTracker),
        )
        for tool in ("ui_snapshot", "ui_diff")
    ],
    *[
        FailurePath(
            "_bookmark_deleted_result: Bookmark '{}' was not deleted: no bookmark has "
            "that name.",
            f"{prefix}_delete_bookmark",
            {"name": "home"},
            {f"{base}/bookmark/home": ok({"deleted": False})},
            "Bookmark 'home' was not deleted",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            "_bookmark_deleted_result: Bookmark '{}': whether it was deleted is UNKNOWN; "
            "the response carried no boolean 'deleted' (got {}).",
            f"{prefix}_delete_bookmark",
            {"name": "home"},
            {f"{base}/bookmark/home": ok({"name": "home"})},
            "Bookmark 'home': whether it was deleted is UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            DIFF_SITE + "{}: no diff. {}",
            f"{prefix}_diff_from_bookmark",
            {"name": "home"},
            {f"{base}/bookmark/home/diff": ok(None)},
            "Diff from bookmark 'home': no diff.",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            DIFF_SITE + "{}: no diff. {}",
            f"{prefix}_scoped_diff",
            {"scope": "#main"},
            {f"{base}/scoped-diff": ok(None)},
            "Scoped diff ('#main'): no diff.",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            DIFF_SITE + "{}: what changed is UNKNOWN. This transport sends element "
            "counts, not a diff: {} elements then, {} now.",
            f"{prefix}_diff_from_bookmark",
            {"name": "home"},
            {f"{base}/bookmark/home/diff": ok({"beforeCount": 1, "afterCount": 2})},
            "Diff from bookmark 'home': what changed is UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            DIFF_SITE + "{}: UNKNOWN; the response carried no diff, so this transport "
            "does not compute one.",
            f"{prefix}_scoped_diff",
            {"scope": "#main"},
            {f"{base}/scoped-diff": ok({"changes": [], "scope": "#main"})},
            "Scoped diff ('#main'): UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            CATEGORY_SITE + "No previous diff to categorize.",
            f"{prefix}_categorize_last_diff",
            {},
            {f"{base}/categorize-last-diff": ok(None)},
            "No previous diff to categorize.",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            CATEGORY_SITE + "Diff category UNKNOWN: the response named no category "
            "the SDK defines (got {}).",
            f"{prefix}_categorize_last_diff",
            {},
            {f"{base}/categorize-last-diff": ok({"category": "unknown"})},
            "Diff category UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    FailurePath(
        "_diff_summary_result: Diff summary UNKNOWN: '{}' is the browser relay's "
        "fixed answer, not a summary of any diff (the SDK answers a missing diff "
        "with NO_DIFF).",
        "sdk_summarize_diff",
        {},
        {"/ui-bridge/sdk/ai/summarize-diff": ok({"summary": "No diff data available"})},
        "Diff summary UNKNOWN: 'No diff data available'",
    ),
    *[
        FailurePath(
            "_diff_summary_result: Diff summary UNKNOWN: the response carried no "
            "summary (got {}).",
            f"{prefix}_summarize_diff",
            {},
            {f"{base}/summarize-diff": ok({})},
            "Diff summary UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            "_structured_changes_result: Structured change analysis UNKNOWN: the "
            "response carried no 'hasStructuredData'; this transport does not "
            "analyze structured changes.",
            f"{prefix}_structured_changes",
            {},
            {f"{base}/structured-changes": ok({"changes": []})},
            "Structured change analysis UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    *[
        FailurePath(
            "_execute_with_diff_result: '\\n'.join(lines)",
            f"{prefix}_execute_with_diff",
            {"element_id": "save", "action": "click"},
            {f"{base}/execute-with-diff": ok({"diff": SEMANTIC_DIFF_NO_CHANGE})},
            "Execute with diff: click on save\n  Action success: UNKNOWN",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    FailurePath(
        "_fill_form_result: {}\n  Not found: {}",
        "sdk_fill_form",
        FILL_REQUEST,
        {"/ui-bridge/sdk/fill": ok({"filledFields": ["email"], "totalFields": 2})},
        "Fill result: 1/2 fields set\n  Not found: name",
    ),
    *[
        FailurePath(
            "_fill_form_result: Fill result UNKNOWN: the response said neither which "
            "fields were set nor whether the fill succeeded.",
            tool,
            FILL_REQUEST,
            {path: ok({})},
            "Fill result UNKNOWN",
        )
        for tool, path in (
            ("sdk_fill_form", "/ui-bridge/sdk/fill"),
            ("ui_fill_form", "/ui-bridge/control/fill"),
        )
    ],
    *[
        FailurePath(
            f"{tool}: _format_fill_result(failed_fill)",
            tool,
            FILL_REQUEST,
            {path: ok(FAILED_FILL)},
            "Fill result: 1/2 fields set\nFailed fields:",
        )
        for tool, path in (
            ("sdk_fill_form", "/ui-bridge/sdk/fill"),
            ("ui_fill_form", "/ui-bridge/control/fill"),
        )
    ],
    *[
        FailurePath(
            "_undo_state_result: Undo/redo state UNKNOWN{}: the response said neither "
            "whether undo nor whether redo is available.",
            tool,
            {},
            {path: ok({"note": "UndoTracker not installed"})},
            "Undo/redo state UNKNOWN",
        )
        for tool, path in (
            ("sdk_undo_state", "/ui-bridge/sdk/undo-state"),
            ("ui_undo_state", "/ui-bridge/control/undo-state"),
        )
    ],
    *[
        FailurePath(
            "_idle_result: Idle status UNKNOWN for {}: the response carried no boolean "
            "'idle' (got {}).",
            "get_idle_status",
            arguments,
            {**status, path: ok({})},
            "Idle status UNKNOWN",
        )
        for arguments, status, path in (
            ({"signal": "dom"}, SDK_CONNECTED, "/ui-bridge/sdk/idle-status/dom"),
            ({}, SDK_ABSENT, "/ui-bridge/control/idle-status"),
        )
    ],
    FailurePath(
        "_wait_for_targets_result: Targets wait failed: {}",
        "wait_for_targets",
        {"targets": ["network"]},
        {**SDK_ABSENT, "/ui-bridge/control/wait-for-targets": ok({"_error": "late"})},
        "Targets wait failed: late",
    ),
    *[
        FailurePath(
            "_wait_for_targets_result: f'Targets wait{mode_label} did not confirm "
            "every target idle; ' + '; '.join(problems)",
            "wait_for_targets",
            {"targets": ["network"]},
            {**status, path: ok({"network": {"idle": False}})},
            "Targets wait",
        )
        for status, path in (
            (SDK_CONNECTED, "/ui-bridge/sdk/wait-for-targets"),
            (SDK_ABSENT, "/ui-bridge/control/wait-for-targets"),
        )
    ],
    *[
        FailurePath(
            "_network_wait_result: '\\n'.join(lines)",
            tool,
            {},
            {path: ok({"timedOut": True})},
            "Wait timed out",
        )
        for tool, path in (
            ("sdk_wait_for_network_request", "/ui-bridge/sdk/network-requests/wait"),
            ("ui_wait_for_network_request", "/ui-bridge/control/network-requests/wait"),
        )
    ],
    FailurePath(
        "ui_health: Runner not accessible: {}",
        "ui_health",
        {},
        {"/health": {"success": False, "error": "down"}},
        "Runner not accessible:",
    ),
    FailurePath(
        "sdk_page_refresh: Page was not refreshed: {}",
        "sdk_page_refresh",
        {},
        {"/ui-bridge/sdk/page/refresh": ok({"reloaded": False, "message": "no-op"})},
        "Page was not refreshed: no-op",
    ),
    FailurePath(
        "sdk_elements: Element list UNKNOWN: the response carried no element list "
        "(got {}). This is not 'no elements'.",
        "sdk_elements",
        {},
        {"/ui-bridge/sdk/elements": ok({"total": 0})},
        "Element list UNKNOWN: the response carried no element list (got NoneType)",
    ),
    FailurePath(
        "sdk_elements: Element list UNKNOWN{}: none proven present. {}",
        "sdk_elements",
        {},
        {"/ui-bridge/sdk/elements": ok(["junk"])},
        "Element list UNKNOWN: none proven present. 1 element(s) were not objects.",
    ),
    FailurePath(
        "sdk_ai_search: Search result UNKNOWN for '{}': the response carried no "
        "`results` list (got {}). This is not 'no elements found'.",
        "sdk_ai_search",
        {"text": "Save"},
        {"/ui-bridge/sdk/ai/search": ok({"matches": []})},
        "Search result UNKNOWN for 'Save':",
    ),
    FailurePath(
        "sdk_ai_search: Search result UNKNOWN for '{}': {} result(s) carried no "
        "readable element. This is not 'no elements found'.",
        "sdk_ai_search",
        {"text": "Save"},
        {"/ui-bridge/sdk/ai/search": ok({"results": [{"confidence": 0.5}]})},
        "Search result UNKNOWN for 'Save': 1 result(s) carried no readable",
    ),
    FailurePath(
        "sdk_ai_search: Search result UNKNOWN for '{}'{}: none proven to match. {}",
        "sdk_ai_search",
        {"text": "Save", "content_role": "heading"},
        {
            "/ui-bridge/sdk/ai/search": ok(
                {"results": [{"element": {"id": "a", "contentMetadata": "x"}}]}
            )
        },
        "Search result UNKNOWN for 'Save' (role: heading): none proven",
    ),
    FailurePath(
        "sdk_cross_app_compare: Cross-app comparison UNKNOWN: the response carried "
        "no scores; this transport does not compare apps.",
        "sdk_cross_app_compare",
        {"source_url": "http://a", "target_url": "http://b"},
        {
            "/ui-bridge/sdk/connect": ok({"connected": True}),
            "/ui-bridge/sdk/ai/snapshot": ok({"elements": []}),
            "/ui-bridge/sdk/ai/analyze/cross-app-compare": ok({"comparison": "stub"}),
        },
        "Cross-app comparison UNKNOWN",
    ),
    FailurePath(
        "ui_diff: No previous snapshot to diff against. Call ui_snapshot first.",
        "ui_diff",
        {},
        {"/ui-bridge/control/snapshot": ok({"elements": [{"id": "s"}]})},
        "No previous snapshot to diff against.",
        setup=_fresh("control_diff_tracker", server_mod.DiffTracker),
    ),
    FailurePath(
        "sdk_diff: No previous snapshot to diff against. Call sdk_snapshot first.",
        "sdk_diff",
        {},
        {"/ui-bridge/sdk/snapshot": ok({"elements": [{"id": "s"}]})},
        "No previous snapshot to diff against.",
        setup=_fresh("sdk_diff_tracker", server_mod.DiffTracker),
    ),
    *[
        FailurePath(
            "ui_element_screenshot|sdk_element_screenshot: Element '{}' not found in "
            "snapshot.",
            tool,
            {"element_id": "gone"},
            {snapshot: ok({"elements": [{"id": "other"}]})},
            "Element 'gone' not found in snapshot.",
        )
        for tool, snapshot in (
            ("ui_element_screenshot", "/ui-bridge/control/snapshot"),
            ("sdk_element_screenshot", "/ui-bridge/sdk/snapshot"),
        )
    ],
    *[
        FailurePath(
            "ui_element_screenshot|sdk_element_screenshot: Element '{}' has no "
            "bounding rect.",
            tool,
            {"element_id": "box"},
            {
                snapshot: ok({"elements": [{"id": "box"}]}),
                shot: ok({"screenshot": "aGk=", "width": 10, "height": 10}),
            },
            "Element 'box' has no bounding rect.",
        )
        for tool, snapshot, shot in (
            (
                "ui_element_screenshot",
                "/ui-bridge/control/snapshot",
                "/ui-bridge/control/annotated-screenshot",
            ),
            (
                "sdk_element_screenshot",
                "/ui-bridge/sdk/snapshot",
                "/ui-bridge/sdk/screenshot",
            ),
        )
    ],
    FailurePath(
        "sdk_screenshot_diff: No baseline found for '{}'. Save one first with "
        "sdk_screenshot_baseline_save.",
        "sdk_screenshot_diff",
        {"baseline_key": "home"},
        {},
        "No baseline found for 'home'.",
        setup=_fresh("baseline_store", BaselineStore),
    ),
    FailurePath(
        "sdk_screenshot_after: No 'before' screenshot saved. Call "
        "sdk_screenshot_before first.",
        "sdk_screenshot_after",
        {},
        {},
        "No 'before' screenshot saved.",
        setup=_fresh("baseline_store", BaselineStore),
    ),
    FailurePath(
        "sdk_design_audit: _style_audit_unknown(report)",
        "sdk_design_audit",
        {},
        {
            "/ui-bridge/sdk/design/audit": ok(
                {"issues": [{"type": "contrast"}], "checkedElements": 3}
            )
        },
        "Style audit UNKNOWN",
    ),
    FailurePath(
        "sdk_design_state_styles: State styles for {} UNKNOWN: the response carried "
        "no per-state list; this transport cannot compute state styles.",
        "sdk_design_state_styles",
        {"element_id": "save"},
        {
            "/ui-bridge/sdk/design/element/save/state-styles": ok(
                {"stateStyles": {"hover": "Requires interaction"}}
            )
        },
        "State styles for save UNKNOWN",
    ),
    *[
        FailurePath(
            f"{prefix}_get_bookmark: Bookmark '{{}}' not found.",
            f"{prefix}_get_bookmark",
            {"name": "home"},
            {f"{base}/bookmark/home": ok(None)},
            "Bookmark 'home' not found.",
        )
        for base, prefix in DIFF_TOOLS.values()
    ],
    FailurePath(
        "ui_query_selector: Nothing clicked: '{}' matched {} element(s), so there is "
        "no element at index {}.",
        "ui_query_selector",
        {"selector": "a", "action": "click", "index": 2},
        {"/ui-bridge/sdk/control/query-selector": ok({"count": 1, "elements": []})},
        "Nothing clicked:",
    ),
    FailurePath(
        "sdk_status: SDK connection UNKNOWN: the response carried no boolean "
        "'connected' (got {}). This is not 'not connected'.",
        "sdk_status",
        {},
        {"/ui-bridge/sdk/status": ok({"url": "http://x"})},
        "SDK connection UNKNOWN:",
    ),
    FailurePath(
        "sdk_find: Find result UNKNOWN for '{}': the response did not say "
        "whether, or how unambiguously, it matched (got found={}, ambiguous={}). "
        "This is not 'not found'.",
        "sdk_find",
        {"query": "Save button"},
        {"/ui-bridge/sdk/ai/find": ok({"reason": "x"})},
        "Find result UNKNOWN for 'Save button':",
    ),
    FailurePath(
        "sdk_design_diff: Snapshot diff verdict UNKNOWN: whether the changes are "
        "significant is not known. This is not 'No'.\n{}",
        "sdk_design_diff",
        {},
        {
            "/ui-bridge/sdk/design/evaluate/diff": ok(
                {"added": [], "removed": [], "modified": []}
            )
        },
        "Snapshot diff verdict UNKNOWN:",
    ),
    FailurePath(
        "_journey_tool: _error_result(response)",
        "journey_dead_ends",
        {"app_id": "web"},
        {},
        "journey_dead_ends (web): [code=producer_failed status=404]",
    ),
    FailurePath(
        "_journey_tool: {}: `app_id` is required",
        "journey_dead_ends",
        {"app_id": ""},
        {},
        "journey_dead_ends: `app_id` is required",
    ),
    FailurePath(
        "_journey_tool: {}: {} required",
        "journey_steps_to",
        {"app_id": "web", "from": "a#x", "to": ""},
        {},
        "journey_steps_to: `to` required",
    ),
    FailurePath(
        "call_tool: Unknown tool: {}",
        "no_such_tool",
        {},
        {},
        "Unknown tool: no_such_tool",
    ),
]


def test_failure_path_census() -> None:
    """Every failure site no ratchet reaches has a scenario below, and no more.

    A new site fails this until it gets a scenario; a deleted one fails it
    until its scenario goes.
    """
    sites = _unreached_failure_sites()
    assert len(sites) == len(set(sites)), "two sites share a key: reword one"
    # The census read the source, not nothing.
    assert len(sites) > 30
    assert sorted(sites) == sorted({path.site for path in FAILURE_PATHS})


@pytest.mark.parametrize(
    "path",
    FAILURE_PATHS,
    ids=[
        f"{p.tool}:{p.site.split(': ', 1)[0]}:{i}" for i, p in enumerate(FAILURE_PATHS)
    ],
)
def test_failure_path_is_flagged(
    path: FailurePath, monkeypatch: pytest.MonkeyPatch
) -> None:
    if path.setup is not None:
        path.setup(monkeypatch)
    answering(monkeypatch, path.routes)
    result = call(path.tool, path.arguments)
    assert text(result).startswith(path.expect), text(result)
    assert result.isError is True
