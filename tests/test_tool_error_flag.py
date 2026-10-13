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
from typing import Any

import httpx
import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient, UIBridgeResponse

Handler = Callable[[httpx.Request], httpx.Response]


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


def _client(
    monkeypatch: pytest.MonkeyPatch, *, port: int = 1, handler: Handler | None = None
) -> UIBridgeClient:
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    c = UIBridgeClient(host="127.0.0.1", port=port)
    if handler is not None:
        c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def _answering(routes: dict[str, dict[str, Any]]) -> Handler:
    """A runner that answers each listed path with its body, and 404s the rest."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"success": False, "error": "Not found"})
        return httpx.Response(200, json=body)

    return handler


def _call(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """Call a tool over an MCP client session, as a client would."""

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


def _text(result: types.CallToolResult) -> str:
    return "\n".join(getattr(item, "text", "") for item in result.content)


# =============================================================================
# Failures carry isError: true
# =============================================================================


def test_unreachable_runner_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch, port=_closed_port()))
    result = _call("ui_health", {})
    assert result.isError is True
    assert "code=app_unreachable" in _text(result)


def test_failed_request_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"success": False, "error": "x", "code": "UB-NET-ERROR"}
        )

    monkeypatch.setattr(server_mod, "client", _client(monkeypatch, handler=handler))
    result = _call("ui_snapshot", {})
    assert result.isError is True
    assert _text(result).startswith("Error: [code=UB-NET-ERROR status=500]")


def test_unknown_tool_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    result = _call("no_such_tool", {})
    assert result.isError is True
    assert _text(result) == "Unknown tool: no_such_tool"


def test_caught_exception_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown ref raises before any request; the catch-all reports it."""
    monkeypatch.setattr(server_mod, "client", _client(monkeypatch))
    result = _call("ui_click", {"element_id": "@e999"})
    assert result.isError is True
    assert _text(result).startswith("Error: Unknown ref @e999")


def test_unmet_precondition_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first sdk_diff has nothing to diff against: no diff was produced."""
    monkeypatch.setattr(server_mod, "sdk_diff_tracker", server_mod.DiffTracker())
    snapshot = {"success": True, "data": {"elements": [{"id": "save"}]}}
    monkeypatch.setattr(
        server_mod,
        "client",
        _client(monkeypatch, handler=_answering({"/ui-bridge/sdk/snapshot": snapshot})),
    )
    result = _call("sdk_diff", {})
    assert result.isError is True
    assert _text(result).startswith("No previous snapshot to diff against.")


def test_failed_assertion_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = {"success": False, "error": "'Save button' is not visible"}
    monkeypatch.setattr(
        server_mod,
        "client",
        _client(monkeypatch, handler=_answering({"/ui-bridge/sdk/ai/assert": failed})),
    )
    result = _call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is True
    assert _text(result).startswith("Assertion failed:")


# =============================================================================
# Answers carry isError: false, including empty ones
# =============================================================================


def test_answer_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = {"success": True, "data": {"status": "ok"}}
    monkeypatch.setattr(
        server_mod,
        "client",
        _client(monkeypatch, handler=_answering({"/health": healthy})),
    )
    result = _call("ui_health", {})
    assert result.isError is False
    assert _text(result) == "Runner is healthy and accessible."


def test_empty_answer_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """sdk_status reached the runner and it said: not connected. That is an answer."""
    status = {"success": True, "data": {"connected": False}}
    monkeypatch.setattr(
        server_mod,
        "client",
        _client(monkeypatch, handler=_answering({"/ui-bridge/sdk/status": status})),
    )
    result = _call("sdk_status", {})
    assert result.isError is False
    assert _text(result) == "SDK not connected"


def test_passed_assertion_is_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The success path used to read an unbound ``state`` and report an error."""
    passed = {"success": True, "data": {"passed": True}}
    monkeypatch.setattr(
        server_mod,
        "client",
        _client(monkeypatch, handler=_answering({"/ui-bridge/sdk/ai/assert": passed})),
    )
    result = _call("sdk_ai_assert", {"text": "Save button", "state": "visible"})
    assert result.isError is False
    assert _text(result) == "Assertion passed: 'Save button' is visible"


# =============================================================================
# The helpers, and a ratchet over every failure branch
# =============================================================================


def test_error_helpers_flag_the_result() -> None:
    failed = UIBridgeResponse(success=False, error="boom", code="producer_failed")
    result = server_mod._error_result(failed)
    assert isinstance(result, types.CallToolResult)
    assert result.isError is True
    assert _text(result) == "Error: [code=producer_failed] boom"
    assert server_mod._tool_error("nope").isError is True


def _call_tool_ast() -> ast.AsyncFunctionDef:
    source = inspect.getsource(server_mod)
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "call_tool"
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
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Attribute)
            and test.operand.attr == "success"
        ):
            continue
        branches += 1
        returns = [n for n in node.body if isinstance(n, ast.Return)]
        flagged = bool(returns) and all(
            isinstance(r.value, ast.Call)
            and isinstance(r.value.func, ast.Name)
            and r.value.func.id in ("_error_result", "_tool_error")
            for r in returns
        )
        if not flagged:
            unflagged.append(node.lineno)
    # The scan found the branches, not nothing: a vacuous pass is not a pass.
    assert branches > 100
    assert unflagged == []


def test_no_inline_error_text_is_returned_unflagged() -> None:
    """No ``return [TextContent(text="Error...")]`` survives in call_tool."""
    fn = _call_tool_ast()
    offenders: list[int] = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.List)):
            continue
        for element in node.value.elts:
            if not isinstance(element, ast.Call):
                continue
            for keyword in element.keywords:
                if keyword.arg != "text":
                    continue
                value = keyword.value
                head = (
                    value.values[0]
                    if isinstance(value, ast.JoinedStr) and value.values
                    else value
                )
                if (
                    isinstance(head, ast.Constant)
                    and isinstance(head.value, str)
                    and head.value.startswith("Error")
                ):
                    offenders.append(node.lineno)
    assert offenders == []
