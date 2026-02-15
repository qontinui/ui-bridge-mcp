"""MCP Server for UI Bridge - enables AI to inspect and interact with UI elements.

This server provides tools for:
- Inspecting UI element positions, bounds, and state
- Interacting with elements (click, type, focus)
- Working with both the runner's own UI (Control mode) and SDK-integrated apps (SDK mode)
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
    category = element.get("category", "")
    content_meta = element.get("contentMetadata", {})
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

    # Include content role for content elements
    content_str = ""
    if category == "content" and content_meta:
        content_role = content_meta.get("contentRole", "")
        if content_role:
            content_str = f" [content:{content_role}]"

    return f"- {elem_id} ({elem_type}): {label}{bounds}{status_str}{content_str}"


def _normalize_components(raw: Any) -> list[dict[str, Any]]:
    """Normalize component data to ComponentInfo shape.

    ControlSnapshot components have {id, name, actions}.
    ComponentInfo also needs type and stateKeys.
    """
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for comp in raw:
        if not isinstance(comp, dict):
            continue
        result.append(
            {
                "id": comp.get("id", ""),
                "name": comp.get("name", comp.get("id", "")),
                "type": comp.get("type", "component"),
                "stateKeys": comp.get(
                    "stateKeys",
                    (
                        list(comp.get("state", {}).keys())
                        if isinstance(comp.get("state"), dict)
                        else []
                    ),
                ),
                "actions": comp.get("actions", []),
            }
        )
    return result


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
    # SDK Mode Tools - External SDK-Integrated Apps
    types.Tool(
        name="sdk_connect",
        description="""Connect to an SDK-integrated web app.

Provide the app's URL to establish a connection. The runner will discover
the SDK endpoints and begin tracking UI elements.

Example: Connect to qontinui-web at http://localhost:3001""",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The app URL (e.g., 'http://localhost:3001')",
                },
            },
            "required": ["url"],
        },
    ),
    types.Tool(
        name="sdk_disconnect",
        description="Disconnect from the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_status",
        description="""Check SDK app connection status.

Returns whether connected, the app URL, and available capabilities.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_snapshot",
        description="""Get a complete snapshot of the SDK app's UI.

Returns all registered elements with their current state including:
- Element ID, type, and label
- Bounding box (x, y, width, height)
- Visibility and enabled state
- Available actions
- Content metadata (for content elements like headings, paragraphs, badges, etc.)

Use this to understand the current UI state before interacting with elements.""",
        inputSchema={
            "type": "object",
            "properties": {
                "include_content": {
                    "type": "boolean",
                    "description": (
                        "Include content (non-interactive) elements like headings, "
                        "paragraphs, badges, metrics, etc. Defaults to true. "
                        "Set to false to only get interactive elements."
                    ),
                    "default": True,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_elements",
        description="""List all registered UI elements in the SDK app.

Returns element IDs, types, labels, and current state.
Supports filtering by content type to find specific kinds of elements.""",
        inputSchema={
            "type": "object",
            "properties": {
                "content_only": {
                    "type": "boolean",
                    "description": (
                        "If true, only return content (non-interactive) elements "
                        "like headings, paragraphs, badges, metrics, etc. "
                        "Defaults to false (returns all elements)."
                    ),
                    "default": False,
                },
                "content_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "heading",
                            "paragraph",
                            "list-item",
                            "table-cell",
                            "table-header",
                            "label",
                            "caption",
                            "blockquote",
                            "code-block",
                            "badge",
                            "status-message",
                            "metric-value",
                            "description-text",
                            "nav-text",
                            "content-generic",
                        ],
                    },
                    "description": (
                        "Filter to elements matching specific content types. "
                        "Example: ['heading', 'badge', 'metric-value'] to find "
                        "headings, badges, and metric values on the page."
                    ),
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_discover",
        description="""Trigger element discovery in the SDK app.

Forces a fresh scan of the page for all UI elements.
Supports filtering to find only interactive or content elements.
Call this if elements aren't showing up in sdk_snapshot or sdk_elements.""",
        inputSchema={
            "type": "object",
            "properties": {
                "interactive_only": {
                    "type": "boolean",
                    "description": (
                        "Only discover interactive elements (buttons, inputs, etc.). "
                        "Defaults to false."
                    ),
                    "default": False,
                },
                "include_content": {
                    "type": "boolean",
                    "description": (
                        "Include content (non-interactive) elements like headings, "
                        "paragraphs, badges, metrics, etc. Defaults to true. "
                        "Ignored if interactive_only is true."
                    ),
                    "default": True,
                },
                "content_roles": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "heading",
                            "body-text",
                            "list-item",
                            "table-cell",
                            "table-header",
                            "label",
                            "caption",
                            "quote",
                            "code",
                            "badge",
                            "status",
                            "metric",
                            "description",
                            "navigation",
                            "generic",
                        ],
                    },
                    "description": (
                        "Filter content elements to these roles. "
                        "Only applies when content elements are included. "
                        "Example: ['heading', 'metric'] to only discover headings and metrics."
                    ),
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_get_element",
        description="""Get detailed information about a specific element.

Returns the element's full state including bounds, visibility,
enabled state, text content, and available actions.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_click",
        description="""Click an element in the SDK app by its data-ui-id.

Use sdk_snapshot or sdk_elements first to find the element_id.""",
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
        name="sdk_type",
        description="""Type text into an input element in the SDK app.

Use sdk_snapshot first to find the element_id of the input field.""",
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
        name="sdk_ai_search",
        description="""Search for elements by natural language description.

Finds elements matching a text description using AI.
Example: 'the login button' or 'email input field'

Supports optional content filters to narrow results to specific content types.""",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Natural language description of the element to find",
                },
                "content_role": {
                    "type": "string",
                    "enum": [
                        "heading",
                        "body-text",
                        "list-item",
                        "table-cell",
                        "table-header",
                        "label",
                        "caption",
                        "quote",
                        "code",
                        "badge",
                        "status",
                        "metric",
                        "description",
                        "navigation",
                        "generic",
                    ],
                    "description": (
                        "Filter results to elements with this content role. "
                        "Example: 'metric' to find only metric/statistic values."
                    ),
                },
                "content_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "heading",
                            "paragraph",
                            "list-item",
                            "table-cell",
                            "table-header",
                            "label",
                            "caption",
                            "blockquote",
                            "code-block",
                            "badge",
                            "status-message",
                            "metric-value",
                            "description-text",
                            "nav-text",
                            "content-generic",
                        ],
                    },
                    "description": (
                        "Filter results to elements matching these content types. "
                        "Example: ['heading', 'badge'] to only search headings and badges."
                    ),
                },
            },
            "required": ["text"],
        },
    ),
    types.Tool(
        name="sdk_ai_execute",
        description="""Execute an action by natural language instruction.

Interprets the instruction and performs the appropriate action.
Example: 'click the Submit button' or 'type hello into the search field'""",
        inputSchema={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language instruction to execute",
                },
            },
            "required": ["instruction"],
        },
    ),
    types.Tool(
        name="sdk_ai_assert",
        description="""Assert element state using natural language.

Verifies that an element matches the expected state.
Example: assert 'error message' is 'hidden'""",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Element description or text to find",
                },
                "state": {
                    "type": "string",
                    "description": "Expected state (e.g., 'visible', 'hidden', 'enabled', 'disabled')",
                },
            },
            "required": ["text"],
        },
    ),
    types.Tool(
        name="sdk_page_summary",
        description="""Get an AI-friendly summary of the current page.

Returns a structured summary of the page layout, navigation,
key elements, and overall state.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_screenshot",
        description="""Capture a screenshot of the monitor where the SDK app is running.

Returns screenshot metadata.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    # Page Navigation Tools
    types.Tool(
        name="sdk_page_refresh",
        description="""Refresh the current page in the connected SDK app.

Triggers a full page reload. The UI Bridge connection will
re-establish automatically after the page reloads.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_page_navigate",
        description="""Navigate the connected SDK app to a specific URL.

Changes the page location. Useful for navigating to a different
route or page within the app.""",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (e.g., 'http://localhost:3001/dashboard')",
                },
            },
            "required": ["url"],
        },
    ),
    types.Tool(
        name="sdk_page_go_back",
        description="""Go back in browser history in the connected SDK app.

Equivalent to clicking the browser's back button.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_page_go_forward",
        description="""Go forward in browser history in the connected SDK app.

Equivalent to clicking the browser's forward button.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    # Cross-App Analysis Tools
    types.Tool(
        name="sdk_analyze_data",
        description="""Extract labeled data values from the connected SDK app's page.

Returns each data-bearing element with its label, raw value, normalized value,
and classified data type (text, number, currency, date, email, etc.).
Useful for understanding what data is displayed on the page.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_analyze_regions",
        description="""Segment the connected SDK app's page into semantic regions.

Returns detected regions (header, navigation, sidebar, main-content, footer,
form, table, card, modal, toolbar) with their bounding boxes and element IDs.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_analyze_structured_data",
        description="""Extract tables and lists from the connected SDK app's page.

Detects grid-like spatial arrangements as tables (with column headers and rows)
and repeating element patterns as lists (with field schemas and items).""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_cross_app_compare",
        description="""Compare two SDK-integrated apps side by side.

Connects to source and target apps sequentially, captures semantic snapshots
from both, then runs a full cross-app comparison analysis.

Returns a report with scores (0-1) for:
- Data completeness: how many source fields exist in target
- Format alignment: whether matching fields use the same display format
- Presentation alignment: layout similarity (grid, hierarchy, density)
- Navigation parity: how many nav items are matched
- Action parity: whether matched elements have the same interactions
- Overall score: weighted combination

Also compares content elements between apps:
- Headings: matched, changed, source-only, target-only
- Metrics: matched values, changed values, missing metrics
- Statuses/badges: matched, changed indicators
- Labels: matched, source-only, target-only
- Tables: column structure, row counts, cell value differences
- Heading hierarchy: heading level distribution differences

Returns a prioritized list of issues (errors, warnings, info) including
content differences.

Set include_components=true to also fetch and compare registered components
(state keys, actions) between the two apps.

Example: Compare Runner (localhost:1420) with qontinui-web (localhost:3001)""",
        inputSchema={
            "type": "object",
            "properties": {
                "source_url": {
                    "type": "string",
                    "description": "URL of the source app (e.g., 'http://localhost:1420')",
                },
                "target_url": {
                    "type": "string",
                    "description": "URL of the target app (e.g., 'http://localhost:3001')",
                },
                "include_components": {
                    "type": "boolean",
                    "description": "Also fetch and compare registered components between apps",
                    "default": False,
                },
            },
            "required": ["source_url", "target_url"],
        },
    ),
]


@server.list_tools()  # type: ignore
async def list_tools() -> list[types.Tool]:
    """List available UI Bridge tools."""
    return TOOLS


@server.call_tool()  # type: ignore
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

        # SDK Mode Tools
        elif name == "sdk_connect":
            url = arguments["url"]
            response = await ui_client.sdk_connect(url)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Connected to SDK app at {url}")
            ]

        elif name == "sdk_disconnect":
            response = await ui_client.sdk_disconnect()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Disconnected from SDK app")]

        elif name == "sdk_status":
            response = await ui_client.sdk_status()
            if not response.success:
                return [
                    types.TextContent(
                        type="text", text=f"SDK not connected: {response.error}"
                    )
                ]
            data = response.data or {}
            connected = data.get("connected", False)
            app_url = data.get("app_url", "unknown")
            if connected:
                return [
                    types.TextContent(type="text", text=f"SDK connected to {app_url}")
                ]
            else:
                return [types.TextContent(type="text", text="SDK not connected")]

        elif name == "sdk_snapshot":
            include_content = arguments.get("include_content", True)
            response = await ui_client.sdk_snapshot(
                include_content=include_content,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            elements = data.get("elements", [])

            # Client-side content filtering as fallback until SDK handlers
            # support the includeContent parameter natively
            if not include_content:
                elements = [el for el in elements if el.get("category") != "content"]

            lines = [f"SDK Snapshot ({len(elements)} elements):", ""]
            sdk_by_type: dict[str, list[dict[str, Any]]] = {}
            for el in elements:
                el_type = el.get("type", "unknown")
                if el_type not in sdk_by_type:
                    sdk_by_type[el_type] = []
                sdk_by_type[el_type].append(el)
            for el_type, els in sorted(sdk_by_type.items()):
                lines.append(f"## {el_type} ({len(els)})")
                for el in els:
                    lines.append(format_element_summary(el))
                lines.append("")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_elements":
            content_only = arguments.get("content_only", False)
            content_types = arguments.get("content_types")
            response = await ui_client.sdk_elements(
                content_only=content_only,
                content_types=content_types,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            elements = data.get("elements", [])

            # Client-side content filtering as fallback until SDK handlers
            # support the contentOnly/contentTypes parameters natively
            if content_only:
                elements = [el for el in elements if el.get("category") == "content"]
            if content_types:
                ct_set = set(content_types)
                elements = [
                    el
                    for el in elements
                    if el.get("contentMetadata", {}).get("contentRole") in ct_set
                    or el.get("type") in ct_set
                ]

            filter_desc = ""
            if content_only:
                filter_desc = " (content only)"
            elif content_types:
                filter_desc = f" (filtered: {', '.join(content_types)})"
            lines = [f"SDK Elements ({len(elements)}){filter_desc}:", ""]
            for el in elements:
                lines.append(format_element_summary(el))
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_discover":
            interactive_only = arguments.get("interactive_only", False)
            include_content = arguments.get("include_content", True)
            content_roles = arguments.get("content_roles")
            response = await ui_client.sdk_discover(
                interactive_only=interactive_only,
                include_content=include_content,
                content_roles=content_roles,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            elements = data.get("elements", [])
            total = data.get("total", len(elements))
            desc_parts = []
            if interactive_only:
                desc_parts.append("interactive only")
            elif not include_content:
                desc_parts.append("excluding content")
            if content_roles:
                desc_parts.append(f"roles: {', '.join(content_roles)}")
            desc = f" ({', '.join(desc_parts)})" if desc_parts else ""
            return [
                types.TextContent(
                    type="text",
                    text=f"Element discovery completed{desc}. Found {total} elements. "
                    "Use sdk_snapshot or sdk_elements to see results.",
                )
            ]

        elif name == "sdk_get_element":
            element_id = arguments["element_id"]
            response = await ui_client.sdk_element(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=json.dumps(response.data, indent=2))
            ]

        elif name == "sdk_click":
            element_id = arguments["element_id"]
            response = await ui_client.sdk_element_action(element_id, "click")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Clicked element: {element_id}")
            ]

        elif name == "sdk_type":
            element_id = arguments["element_id"]
            text = arguments["text"]
            response = await ui_client.sdk_element_action(
                element_id, "type", {"text": text}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Typed '{text}' into element: {element_id}"
                )
            ]

        elif name == "sdk_ai_search":
            text = arguments["text"]
            content_role = arguments.get("content_role")
            content_types = arguments.get("content_types")
            response = await ui_client.sdk_ai_search(
                text,
                content_role=content_role,
                content_types=content_types,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            matches = data.get("matches", [])

            # Client-side content filtering as fallback until SDK handlers
            # support the contentRole/contentTypes parameters natively
            if content_role:
                matches = [
                    m
                    for m in matches
                    if m.get("contentMetadata", {}).get("contentRole") == content_role
                ]
            if content_types:
                ct_set = set(content_types)
                matches = [
                    m
                    for m in matches
                    if m.get("contentMetadata", {}).get("contentRole") in ct_set
                    or m.get("type") in ct_set
                ]

            filter_desc = ""
            if content_role:
                filter_desc = f" (role: {content_role})"
            elif content_types:
                filter_desc = f" (types: {', '.join(content_types)})"

            if not matches:
                return [
                    types.TextContent(
                        type="text",
                        text=f"No elements found matching: {text}{filter_desc}",
                    )
                ]
            lines = [
                f"Found {len(matches)} element(s) matching '{text}'{filter_desc}:",
                "",
            ]
            for m in matches:
                lines.append(format_element_summary(m))
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_ai_execute":
            instruction = arguments["instruction"]
            response = await ui_client.sdk_ai_execute(instruction)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text=f"Executed: {instruction}")]

        elif name == "sdk_ai_assert":
            text = arguments["text"]
            state = arguments.get("state")
            response = await ui_client.sdk_ai_assert(text, state)
            if not response.success:
                return [
                    types.TextContent(
                        type="text", text=f"Assertion failed: {response.error}"
                    )
                ]
            return [
                types.TextContent(
                    type="text",
                    text=f"Assertion passed: '{text}' is {state or 'as expected'}",
                )
            ]

        elif name == "sdk_page_summary":
            response = await ui_client.sdk_ai_summary()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            summary = data.get("summary", json.dumps(data, indent=2))
            return [types.TextContent(type="text", text=summary)]

        elif name == "sdk_page_refresh":
            response = await ui_client.sdk_page_refresh()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Page refreshed successfully")]

        elif name == "sdk_page_navigate":
            url = arguments.get("url", "")
            if not url:
                return [types.TextContent(type="text", text="Error: url is required")]
            response = await ui_client.sdk_page_navigate(url)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text=f"Navigated to: {url}")]

        elif name == "sdk_page_go_back":
            response = await ui_client.sdk_page_go_back()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Navigated back")]

        elif name == "sdk_page_go_forward":
            response = await ui_client.sdk_page_go_forward()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Navigated forward")]

        elif name == "sdk_screenshot":
            response = await ui_client.sdk_screenshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            path = data.get("screenshot_path", data.get("path", "unknown"))
            return [types.TextContent(type="text", text=f"Screenshot captured: {path}")]

        # Cross-App Analysis Tools
        elif name == "sdk_analyze_data":
            response = await ui_client.sdk_ai_analyze_data()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            values = data.get("values", {})
            lines = [f"Page Data ({len(values)} values extracted):", ""]
            for label, info in values.items():
                raw = info.get("rawValue", "")
                dtype = info.get("dataType", "unknown")
                lines.append(f"- {label}: {raw} ({dtype})")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_analyze_regions":
            response = await ui_client.sdk_ai_analyze_regions()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            regions = data.get("regions", [])
            lines = [f"Page Regions ({len(regions)} detected):", ""]
            for r in regions:
                rtype = r.get("type", "unknown")
                label = r.get("label", "")
                elem_count = len(r.get("elementIds", []))
                conf = r.get("confidence", 0)
                lines.append(
                    f"- {label} ({rtype}): {elem_count} elements, confidence={conf:.2f}"
                )
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_analyze_structured_data":
            response = await ui_client.sdk_ai_analyze_structured_data()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            tables = data.get("tables", [])
            lists = data.get("lists", [])
            lines = [f"Structured Data ({len(tables)} tables, {len(lists)} lists):", ""]
            for t in tables:
                cols = t.get("columns", [])
                rows = t.get("rows", [])
                headers = [c.get("header", "") for c in cols]
                lines.append(
                    f"Table: {t.get('label', 'untitled')} ({len(cols)} cols, {len(rows)} rows)"
                )
                lines.append(f"  Columns: {', '.join(headers)}")
            for lst in lists:
                items = lst.get("items", [])
                lines.append(
                    f"List: {lst.get('label', 'untitled')} ({len(items)} items)"
                )
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_cross_app_compare":
            source_url = arguments["source_url"]
            target_url = arguments["target_url"]
            include_components = arguments.get("include_components", False)

            # Step 1: Connect to source and get snapshot
            connect_resp = await ui_client.sdk_connect(source_url)
            if not connect_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error connecting to source {source_url}: {connect_resp.error}",
                    )
                ]

            source_snap_resp = await ui_client.sdk_ai_snapshot()
            if not source_snap_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error getting source snapshot: {source_snap_resp.error}",
                    )
                ]
            source_snapshot = source_snap_resp.data

            # Optionally fetch source components
            source_components = None
            if include_components:
                comp_resp = await ui_client.sdk_components()
                if comp_resp.success and comp_resp.data:
                    raw = (
                        comp_resp.data
                        if isinstance(comp_resp.data, list)
                        else comp_resp.data.get("components", comp_resp.data)
                    )
                    source_components = _normalize_components(raw)

            # Step 2: Connect to target and get snapshot
            connect_resp = await ui_client.sdk_connect(target_url)
            if not connect_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error connecting to target {target_url}: {connect_resp.error}",
                    )
                ]

            target_snap_resp = await ui_client.sdk_ai_snapshot()
            if not target_snap_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error getting target snapshot: {target_snap_resp.error}",
                    )
                ]
            target_snapshot = target_snap_resp.data

            # Optionally fetch target components
            target_components = None
            if include_components:
                comp_resp = await ui_client.sdk_components()
                if comp_resp.success and comp_resp.data:
                    raw = (
                        comp_resp.data
                        if isinstance(comp_resp.data, list)
                        else comp_resp.data.get("components", comp_resp.data)
                    )
                    target_components = _normalize_components(raw)

            # Step 3: Build comparison request body and run comparison
            compare_body: dict[str, Any] = {
                "sourceSnapshot": source_snapshot,
                "targetSnapshot": target_snapshot,
            }
            if source_components is not None and target_components is not None:
                compare_body["sourceComponents"] = source_components
                compare_body["targetComponents"] = target_components

            compare_resp = await ui_client._request(
                "POST",
                "/ui-bridge/sdk/ai/analyze/cross-app-compare",
                compare_body,
            )
            if not compare_resp.success:
                return [
                    types.TextContent(
                        type="text", text=f"Error comparing: {compare_resp.error}"
                    )
                ]

            data = compare_resp.data or {}
            scores = data.get("scores", {})
            issues = data.get("issues", [])
            summary = data.get("summary", "")
            components = data.get("components")
            content_comparison = data.get("contentComparison")

            lines = [
                "Cross-App Comparison Report",
                f"Source: {source_url}",
                f"Target: {target_url}",
                "",
                "Scores:",
                f"  Data completeness:      {scores.get('dataCompleteness', 0):.0%}",
                f"  Format alignment:       {scores.get('formatAlignment', 0):.0%}",
                f"  Presentation alignment: {scores.get('presentationAlignment', 0):.0%}",
                f"  Navigation parity:      {scores.get('navigationParity', 0):.0%}",
                f"  Action parity:          {scores.get('actionParity', 0):.0%}",
                f"  Overall score:          {scores.get('overallScore', 0):.0%}",
            ]

            if components:
                matches = components.get("matches", [])
                src_only = components.get("sourceOnly", [])
                tgt_only = components.get("targetOnly", [])
                lines.append("")
                lines.append(
                    f"Components ({len(matches)} matched, {len(src_only)} source-only, {len(tgt_only)} target-only):"
                )
                for m in matches[:10]:
                    src_name = m.get("source", {}).get("name", "?")
                    tgt_name = m.get("target", {}).get("name", "?")
                    conf = m.get("confidence", 0)
                    missing_keys = m.get("stateKeyDiff", {}).get("missing", [])
                    missing_actions = m.get("actionDiff", {}).get("missing", [])
                    notes = []
                    if missing_keys:
                        notes.append(f"missing keys: {', '.join(missing_keys)}")
                    if missing_actions:
                        notes.append(f"missing actions: {', '.join(missing_actions)}")
                    note_str = f" ({'; '.join(notes)})" if notes else ""
                    lines.append(f"  {src_name} <-> {tgt_name} ({conf:.0%}){note_str}")

            # Content comparison section
            if content_comparison:
                lines.append("")
                lines.append("Content Comparison:")

                # Headings
                headings = content_comparison.get("headings", {})
                h_matched = headings.get("matched", [])
                h_src_only = headings.get("sourceOnly", [])
                h_tgt_only = headings.get("targetOnly", [])
                h_changed = headings.get("changed", [])

                if h_matched or h_src_only or h_tgt_only or h_changed:
                    lines.append(
                        f"  Headings ({len(h_matched)} matched, "
                        f"{len(h_changed)} changed, "
                        f"{len(h_src_only)} source-only, "
                        f"{len(h_tgt_only)} target-only):"
                    )
                    for h in h_matched[:5]:
                        level_str = (
                            f" (h{h.get('level', '?')})" if h.get("level") else ""
                        )
                        lines.append(f'    = "{h.get("source", "")}"{level_str}')
                    for h in h_changed[:5]:
                        lines.append(
                            f'    ~ "{h.get("source", "")}" -> "{h.get("target", "")}"'
                        )
                    for h in h_src_only[:5]:
                        lines.append(f'    - "{h}" (source only)')
                    for h in h_tgt_only[:5]:
                        lines.append(f'    + "{h}" (target only)')

                # Metrics
                metrics = content_comparison.get("metrics", {})
                m_matched = metrics.get("matched", [])
                m_changed = metrics.get("changed", [])
                m_src_only = metrics.get("sourceOnly", [])
                m_tgt_only = metrics.get("targetOnly", [])

                if m_matched or m_changed or m_src_only or m_tgt_only:
                    lines.append(
                        f"  Metrics ({len(m_matched)} matched, "
                        f"{len(m_changed)} changed, "
                        f"{len(m_src_only)} source-only, "
                        f"{len(m_tgt_only)} target-only):"
                    )
                    for m in m_matched[:5]:
                        lines.append(
                            f'    = "{m.get("label", "")}": {m.get("sourceValue", "")}'
                        )
                    for m in m_changed[:10]:
                        lines.append(
                            f'    ~ "{m.get("label", "")}": '
                            f'"{m.get("sourceValue", "")}" -> "{m.get("targetValue", "")}"'
                        )
                    for label in m_src_only[:5]:
                        lines.append(f'    - "{label}" (source only)')
                    for label in m_tgt_only[:5]:
                        lines.append(f'    + "{label}" (target only)')

                # Statuses
                statuses = content_comparison.get("statuses", {})
                s_matched = statuses.get("matched", [])
                s_changed = statuses.get("changed", [])

                if s_matched or s_changed:
                    lines.append(
                        f"  Statuses ({len(s_matched)} matched, {len(s_changed)} changed):"
                    )
                    for s in s_matched[:5]:
                        lines.append(
                            f'    = "{s.get("label", "")}": {s.get("sourceStatus", "")}'
                        )
                    for s in s_changed[:10]:
                        lines.append(
                            f'    ~ "{s.get("label", "")}": '
                            f'"{s.get("sourceStatus", "")}" -> "{s.get("targetStatus", "")}"'
                        )

                # Labels
                labels = content_comparison.get("labels", {})
                l_matched = labels.get("matched", [])
                l_src_only = labels.get("sourceOnly", [])
                l_tgt_only = labels.get("targetOnly", [])

                if l_src_only or l_tgt_only:
                    lines.append(
                        f"  Labels ({len(l_matched)} matched, "
                        f"{len(l_src_only)} source-only, "
                        f"{len(l_tgt_only)} target-only):"
                    )
                    for label in l_src_only[:5]:
                        lines.append(f'    - "{label}" (source only)')
                    for label in l_tgt_only[:5]:
                        lines.append(f'    + "{label}" (target only)')

                # Tables
                tables = content_comparison.get("tables", [])
                if tables:
                    lines.append(f"  Tables ({len(tables)} compared):")
                    for t in tables[:5]:
                        src_label = t.get("sourceLabel", "?")
                        col_match = (
                            "columns match"
                            if t.get("columnsMatch", False)
                            else "columns differ"
                        )
                        src_rows = t.get("sourceRowCount", 0)
                        tgt_rows = t.get("targetRowCount", 0)
                        cell_diffs = len(t.get("cellDifferences", []))
                        lines.append(
                            f'    "{src_label}": {col_match}, '
                            f"{src_rows} vs {tgt_rows} rows, "
                            f"{cell_diffs} cell diff(s)"
                        )
                        src_only_cols = t.get("sourceOnlyColumns", [])
                        tgt_only_cols = t.get("targetOnlyColumns", [])
                        if src_only_cols:
                            lines.append(
                                f"      Source-only columns: {', '.join(src_only_cols)}"
                            )
                        if tgt_only_cols:
                            lines.append(
                                f"      Target-only columns: {', '.join(tgt_only_cols)}"
                            )

                # Heading hierarchy
                hierarchy = content_comparison.get("headingHierarchy", [])
                if hierarchy:
                    diffs = [
                        h
                        for h in hierarchy
                        if h.get("sourceCount", 0) != h.get("targetCount", 0)
                    ]
                    if diffs:
                        lines.append("  Heading Hierarchy Differences:")
                        for h in diffs:
                            lines.append(
                                f"    h{h.get('level', '?')}: "
                                f"{h.get('sourceCount', 0)} (source) vs "
                                f"{h.get('targetCount', 0)} (target)"
                            )

                # Content parity score
                content_parity = content_comparison.get("contentParity", 0)
                lines.append(f"  Content parity: {content_parity:.0%}")

            lines.append("")
            lines.append(f"Issues ({len(issues)}):")
            for issue in issues[:20]:
                severity = issue.get("severity", "info").upper()
                desc = issue.get("description", "")
                lines.append(f"  [{severity}] {desc}")

            if len(issues) > 20:
                lines.append(f"  ... and {len(issues) - 20} more issues")

            lines.append("")
            lines.append(summary)

            return [types.TextContent(type="text", text="\n".join(lines))]

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
