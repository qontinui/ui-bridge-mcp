"""MCP Server for UI Bridge - enables AI to inspect and interact with UI elements.

This server provides tools for:
- Inspecting UI element positions, bounds, and state
- Interacting with elements (click, type, focus)
- Working with both the runner's own UI (Control mode) and external browser tabs (External mode)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .client import UIBridgeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MCP Server instance
server = Server("ui-bridge-mcp")
client: UIBridgeClient | None = None


def get_client() -> UIBridgeClient:
    """Get or create the UI Bridge client."""
    global client
    if client is None:
        client = UIBridgeClient()
    return client


def format_element_summary(element: dict[str, Any]) -> str:
    """Format an element for display."""
    elem_id = element.get("id", "unknown")
    elem_type = element.get("type", "unknown")
    label = element.get("label", "")
    state = element.get("state", {})
    rect = state.get("rect", {})
    visible = state.get("visible", True)
    enabled = state.get("enabled", True)

    bounds = ""
    if rect:
        bounds = f" @ ({rect.get('x', 0):.0f}, {rect.get('y', 0):.0f}, {rect.get('width', 0):.0f}x{rect.get('height', 0):.0f})"

    status = []
    if not visible:
        status.append("hidden")
    if not enabled:
        status.append("disabled")
    status_str = f" [{', '.join(status)}]" if status else ""

    return f"- {elem_id} ({elem_type}): {label}{bounds}{status_str}"


# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------

TOOLS = [
    # Health check
    types.Tool(
        name="ui_health",
        description="Check if the qontinui-runner is running and accessible.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    # Control Mode Tools
    types.Tool(
        name="ui_snapshot",
        description="""Get a complete snapshot of the runner's UI (Control mode).

Returns all registered elements with their current state including:
- Element ID, type, and label
- Bounding box (x, y, width, height)
- Visibility and enabled state
- Available actions (click, type, focus, etc.)

Use this to understand the current UI state before interacting with elements.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ui_discover",
        description="""Trigger element discovery in the runner's UI.

Call this if elements aren't showing up in ui_snapshot - it forces
a fresh registration of all interactive elements.""",
        inputSchema={
            "type": "object",
            "properties": {
                "interactive_only": {
                    "type": "boolean",
                    "description": "Only discover interactive elements (buttons, inputs, etc.)",
                    "default": False,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="ui_get_element",
        description="""Get detailed information about a specific UI element.

Returns the element's full state including bounds, visibility,
enabled state, text content, and available actions.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id (e.g., 'sidebar-nav-item-settings')",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_click",
        description="""Click an element in the runner's UI.

Use ui_snapshot first to find the element_id you want to click.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to click",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_type",
        description="""Type text into an input element in the runner's UI.

Use ui_snapshot first to find the element_id of the input field.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to type into",
                },
                "text": {
                    "type": "string",
                    "description": "The text to type",
                },
            },
            "required": ["element_id", "text"],
        },
    ),
    types.Tool(
        name="ui_focus",
        description="Focus an element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to focus",
                },
            },
            "required": ["element_id"],
        },
    ),
    # External Mode Tools
    types.Tool(
        name="extension_status",
        description="""Check if the Chrome extension is connected.

The extension enables interaction with external browser tabs.
Returns connection status and WebSocket URL.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="extension_list_tabs",
        description="""List all browser tabs accessible via the Chrome extension.

Returns tab IDs, titles, URLs, and which tab is active.
Use the tab_id with extension_select_tab to target a specific tab.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="extension_select_tab",
        description="""Select a browser tab for subsequent operations.

After selecting a tab, extension_get_elements will read from that tab.""",
        inputSchema={
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "integer",
                    "description": "The tab ID from extension_list_tabs",
                },
            },
            "required": ["tab_id"],
        },
    ),
    types.Tool(
        name="extension_get_elements",
        description="""Get all elements from the selected browser tab.

Returns a list of elements with their:
- ID, type, and accessible name
- Role and ARIA attributes
- Visibility and enabled state
- Available actions
- Bounding box positions

Use extension_select_tab first to choose which tab to inspect.""",
        inputSchema={
            "type": "object",
            "properties": {
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds for element discovery",
                    "default": 30,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="extension_click",
        description="""Click an element in the browser tab by CSS selector.

Use extension_get_elements first to understand the page structure.""",
        inputSchema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to click",
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 10,
                },
            },
            "required": ["selector"],
        },
    ),
    types.Tool(
        name="extension_type",
        description="""Type text into an element in the browser tab.

Use extension_get_elements first to find the input field's selector.""",
        inputSchema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the input element",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the element",
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 10,
                },
            },
            "required": ["selector", "text"],
        },
    ),
    types.Tool(
        name="extension_screenshot",
        description="""Take a screenshot of the current browser tab.

Returns a base64-encoded PNG image.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


@server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
async def list_tools() -> list[types.Tool]:
    """List available UI Bridge tools."""
    return TOOLS


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    ui_client = get_client()

    try:
        # Health check
        if name == "ui_health":
            response = await ui_client.health()
            if response.success:
                return [
                    types.TextContent(
                        type="text", text="Runner is healthy and accessible."
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text", text=f"Runner not accessible: {response.error}"
                    )
                ]

        # Control Mode Tools
        elif name == "ui_snapshot":
            response = await ui_client.control_snapshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]

            data = response.data or {}
            elements = data.get("elements", [])

            lines = [f"UI Snapshot ({len(elements)} elements):", ""]

            # Group by type
            by_type: dict[str, list[dict[str, Any]]] = {}
            for el in elements:
                el_type = el.get("type", "unknown")
                if el_type not in by_type:
                    by_type[el_type] = []
                by_type[el_type].append(el)

            for el_type, els in sorted(by_type.items()):
                lines.append(f"## {el_type} ({len(els)})")
                for el in els:
                    lines.append(format_element_summary(el))
                lines.append("")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "ui_discover":
            interactive_only = arguments.get("interactive_only", False)
            response = await ui_client.control_discover(interactive_only)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text="Element discovery completed. Use ui_snapshot to see results.",
                )
            ]

        elif name == "ui_get_element":
            element_id = arguments["element_id"]
            response = await ui_client.control_get_element(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=json.dumps(response.data, indent=2))
            ]

        elif name == "ui_click":
            element_id = arguments["element_id"]
            response = await ui_client.control_click(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Clicked element: {element_id}")
            ]

        elif name == "ui_type":
            element_id = arguments["element_id"]
            text = arguments["text"]
            response = await ui_client.control_type(element_id, text)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Typed '{text}' into element: {element_id}"
                )
            ]

        elif name == "ui_focus":
            element_id = arguments["element_id"]
            response = await ui_client.control_focus(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Focused element: {element_id}")
            ]

        # External Mode Tools
        elif name == "extension_status":
            response = await ui_client.extension_status()
            if not response.success:
                return [
                    types.TextContent(
                        type="text", text=f"Extension not connected: {response.error}"
                    )
                ]
            data = response.data or {}
            connected = data.get("connected", False)
            ws_url = data.get("websocket_url", "")
            if connected:
                return [
                    types.TextContent(
                        type="text", text=f"Extension connected via {ws_url}"
                    )
                ]
            else:
                return [types.TextContent(type="text", text="Extension not connected")]

        elif name == "extension_list_tabs":
            response = await ui_client.extension_list_tabs()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            tabs = data.get("tabs", [])

            lines = [f"Browser Tabs ({len(tabs)}):", ""]
            for tab in tabs:
                active = " [ACTIVE]" if tab.get("active") else ""
                lines.append(
                    f"- ID {tab.get('id')}: {tab.get('title', 'Untitled')}{active}"
                )
                lines.append(f"  URL: {tab.get('url', 'unknown')}")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "extension_select_tab":
            tab_id = arguments["tab_id"]
            response = await ui_client.extension_select_tab(tab_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            title = data.get("title", "unknown")
            return [
                types.TextContent(
                    type="text", text=f"Selected tab: {title} (ID: {tab_id})"
                )
            ]

        elif name == "extension_get_elements":
            timeout_secs = arguments.get("timeout_secs", 30)
            response = await ui_client.extension_get_elements(timeout_secs)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]

            data = response.data or {}
            elements = data.get("elements", [])

            lines = [f"Page Elements ({len(elements)}):", ""]

            # Group by type
            elements_by_type: dict[str, list[dict[str, Any]]] = {}
            for el in elements:
                el_type = el.get("type") or el.get("tagName") or "unknown"
                if el_type not in elements_by_type:
                    elements_by_type[el_type] = []
                elements_by_type[el_type].append(el)

            for el_type, els in sorted(elements_by_type.items()):
                lines.append(f"## {el_type} ({len(els)})")
                for el in els[:10]:  # Limit to first 10 per type
                    elem_id = el.get("id", "no-id")
                    label = (
                        el.get("label")
                        or el.get("text")
                        or el.get("accessibility", {}).get("accessibleName")
                        or ""
                    )
                    visible = el.get("visible", True)
                    enabled = el.get("enabled", True)

                    status = []
                    if not visible:
                        status.append("hidden")
                    if not enabled:
                        status.append("disabled")
                    status_str = f" [{', '.join(status)}]" if status else ""

                    lines.append(f"- {elem_id}: {label[:50]}{status_str}")

                if len(els) > 10:
                    lines.append(f"  ... and {len(els) - 10} more")
                lines.append("")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "extension_click":
            selector = arguments["selector"]
            timeout_secs = arguments.get("timeout_secs", 10)
            response = await ui_client.extension_click(selector, timeout_secs)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text=f"Clicked element: {selector}")]

        elif name == "extension_type":
            selector = arguments["selector"]
            text = arguments["text"]
            timeout_secs = arguments.get("timeout_secs", 10)
            response = await ui_client.extension_type(selector, text, timeout_secs)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Typed '{text}' into: {selector}")
            ]

        elif name == "extension_screenshot":
            response = await ui_client.extension_screenshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            # Return base64 image as text (MCP will handle it)
            return [
                types.TextContent(
                    type="text",
                    text=f"Screenshot captured (base64 PNG, {len(data.get('data', ''))} bytes)",
                )
            ]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"Error calling tool {name}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def main() -> None:
    """Run the MCP server."""
    logger.info("Starting UI Bridge MCP server")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run() -> None:
    """Entry point for the MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
