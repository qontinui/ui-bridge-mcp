"""Drive the server as an MCP client does: over an in-memory client session.

A tool result is read the way a client reads it, ``isError`` included, and the
fake runner records every request, so a test can pin what a tool SENDS as well
as how it reads the answer.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

import ui_bridge_mcp.server as server_mod
from ui_bridge_mcp.client import UIBridgeClient


@dataclass
class FakeRunner:
    """Answers each listed path, 404s the rest, and records every request.

    A route's answer is a JSON body served with status 200, or an
    ``httpx.Response`` served as it is.
    """

    routes: dict[str, Any] = field(default_factory=dict)
    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    read_timeouts: list[float | None] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        self.read_timeouts.append(request.extensions.get("timeout", {}).get("read"))
        answer = self.routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={"success": False, "error": "Not found"})
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def bodies(self, path: str) -> list[Any]:
        """The JSON bodies sent to ``path``, in order."""
        return [body for (_method, p, body) in self.requests if p == path]


def install(
    monkeypatch: pytest.MonkeyPatch,
    runner: FakeRunner | None = None,
    *,
    port: int = 1,
) -> FakeRunner | None:
    """Point the server's client at ``runner`` (or at a bare ``port``)."""
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    client = UIBridgeClient(host="127.0.0.1", port=port)
    if runner is not None:
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(runner))
    monkeypatch.setattr(server_mod, "client", client)
    return runner


def answering(
    monkeypatch: pytest.MonkeyPatch, routes: dict[str, Any] | None = None
) -> FakeRunner:
    """Install a fake runner answering ``routes``, and return it."""
    runner = FakeRunner(dict(routes or {}))
    install(monkeypatch, runner)
    return runner


def call(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
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


def text(result: types.CallToolResult) -> str:
    """The result's text content, joined."""
    return "\n".join(getattr(item, "text", "") for item in result.content)
