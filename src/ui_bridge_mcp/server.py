"""MCP Server for UI Bridge - enables AI to inspect and interact with UI elements.

This server provides tools for:
- Inspecting UI element positions, bounds, and state
- Interacting with elements (click, type, focus)
- Working with both the runner's own UI (Control mode) and SDK-integrated apps (SDK mode)
"""

from __future__ import annotations

import asyncio
import base64
import io
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


# =============================================================================
# Agent Mode: Compact Refs
# =============================================================================


class RefManager:
    """Assigns compact refs (@e1, @e2, ...) to element IDs for agent mode."""

    def __init__(self) -> None:
        self._ref_counter = 0
        self._ref_to_id: dict[str, str] = {}
        self._id_to_ref: dict[str, str] = {}

    def reset(self) -> None:
        """Reset refs. Call at start of each snapshot."""
        self._ref_counter = 0
        self._ref_to_id.clear()
        self._id_to_ref.clear()

    def assign(self, element_id: str) -> str:
        """Assign a compact ref to an element ID."""
        if element_id in self._id_to_ref:
            return self._id_to_ref[element_id]
        self._ref_counter += 1
        ref = f"@e{self._ref_counter}"
        self._ref_to_id[ref] = element_id
        self._id_to_ref[element_id] = ref
        return ref

    def resolve(self, ref_or_id: str) -> str:
        """Resolve @eN to real ID, or pass through if already an ID."""
        if ref_or_id.startswith("@e"):
            resolved = self._ref_to_id.get(ref_or_id)
            if resolved is None:
                raise ValueError(
                    f"Unknown ref {ref_or_id}. Take a new snapshot to refresh refs."
                )
            return resolved
        return ref_or_id


# =============================================================================
# Agent Mode: Snapshot Diffing
# =============================================================================


class DiffTracker:
    """Tracks element state between snapshots for diffing."""

    TRACKED_PROPS = ("visible", "enabled", "focused", "checked", "value", "textContent")

    def __init__(self) -> None:
        self._last_elements: dict[str, dict[str, Any]] | None = None

    def update_and_diff(self, elements: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Store snapshot, return diff against previous (or None if first)."""
        new_map = {el["id"]: el for el in elements if "id" in el}
        diff = None
        if self._last_elements is not None:
            diff = self._compute(self._last_elements, new_map)
        self._last_elements = new_map
        return diff

    def _compute(
        self, old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        appeared = [eid for eid in new if eid not in old]
        disappeared = [eid for eid in old if eid not in new]
        modified: list[dict[str, Any]] = []
        for eid in new:
            if eid in old:
                changes = self._prop_changes(old[eid], new[eid])
                if changes:
                    modified.append({"id": eid, "changes": changes})
        return {
            "appeared": appeared,
            "disappeared": disappeared,
            "modified": modified,
        }

    def _prop_changes(
        self, old_el: dict[str, Any], new_el: dict[str, Any]
    ) -> dict[str, Any]:
        old_state = old_el.get("state", {})
        new_state = new_el.get("state", {})
        changes: dict[str, Any] = {}
        for prop in self.TRACKED_PROPS:
            old_val = old_state.get(prop)
            new_val = new_state.get(prop)
            if old_val != new_val:
                changes[prop] = {"from": old_val, "to": new_val}
        return changes


# Module-level singletons
ref_manager = RefManager()
control_diff_tracker = DiffTracker()
sdk_diff_tracker = DiffTracker()


# =============================================================================
# Agent Mode: Content Boundary Markers
# =============================================================================

CONTENT_START = "<<CONTENT>>"
CONTENT_END = "<</CONTENT>>"


def sanitize_element_content(data: dict[str, Any]) -> dict[str, Any]:
    """Wrap user-generated content fields in boundary markers."""
    state = data.get("state", {})
    for field in ("textContent", "innerHTML", "value"):
        if field in state and state[field]:
            state[field] = f"{CONTENT_START}{state[field]}{CONTENT_END}"
    return data


# =============================================================================
# Agent Mode: Output Size Helpers
# =============================================================================


def truncate_field(text: str | None, max_len: int) -> str | None:
    """Truncate a text field to max_len chars."""
    if not text or len(text) <= max_len:
        return text
    return f"{text[:max_len]}... [{len(text)} chars total]"


# =============================================================================
# Formatting
# =============================================================================


def format_element_compact(element: dict[str, Any], ref: str) -> str:
    """Single-line compact format for agent mode."""
    elem_id = element.get("id", "?")
    elem_type = element.get("type", "?")
    label = element.get("label", "")
    category = element.get("category", "")
    content_meta = element.get("contentMetadata", {})
    state = element.get("state", {})
    rect = state.get("rect", {})

    parts = [ref, elem_id, f"({elem_type})"]
    if label:
        parts.append(f'"{label}"')
    if rect:
        parts.append(
            f'[{rect.get("x", 0):.0f},{rect.get("y", 0):.0f} '
            f'{rect.get("width", 0):.0f}x{rect.get("height", 0):.0f}]'
        )

    # Content role for content elements
    if category == "content" and content_meta:
        content_role = content_meta.get("contentRole", "")
        if content_role:
            parts.append(f"content:{content_role}")

    flags: list[str] = []
    if not state.get("visible", True):
        flags.append("hidden")
    if not state.get("enabled", True):
        flags.append("disabled")
    if state.get("value"):
        flags.append("has-value")
    if state.get("checked"):
        flags.append("checked")
    if state.get("focused"):
        flags.append("focused")
    if flags:
        parts.append(" ".join(flags))

    return " ".join(parts)


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

Use agent_mode=true for compact output with short refs (@e1, @e2).
Use interactive_only=true to exclude content elements.
Use max_elements to limit output size.""",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_mode": {
                    "type": "boolean",
                    "description": (
                        "Compact output with short refs (@e1, @e2). "
                        "Use refs in subsequent actions. "
                        "Full details via ui_get_element."
                    ),
                    "default": False,
                },
                "interactive_only": {
                    "type": "boolean",
                    "description": (
                        "Only return interactive elements (buttons, inputs, links). "
                        "Excludes static content."
                    ),
                    "default": False,
                },
                "max_elements": {
                    "type": "integer",
                    "description": "Max elements to return. Remaining summarized as count.",
                },
                "max_content_length": {
                    "type": "integer",
                    "description": (
                        "Max chars per text field (label, value). "
                        "Longer values truncated."
                    ),
                },
            },
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
enabled state, text content, and available actions.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id or agent ref (e.g., '@e1', 'sidebar-nav-item-settings')",
                },
                "max_content_length": {
                    "type": "integer",
                    "description": "Max chars per text field. Longer values truncated.",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_click",
        description="""Click an element in the runner's UI.

Use ui_snapshot first to find the element_id you want to click.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id or agent ref (@e1)",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_type",
        description="""Type text into an input element in the runner's UI.

Use ui_snapshot first to find the element_id of the input field.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id or agent ref (@e1)",
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
    types.Tool(
        name="ui_blur",
        description="Remove focus from an element in the runner's UI.",
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
        name="ui_hover",
        description="Hover over an element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to hover over",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_double_click",
        description="Double-click an element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to double-click",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_right_click",
        description="Right-click an element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to right-click",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_clear",
        description="Clear the value of an input element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to clear",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_select",
        description="Select an option in a dropdown/select element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id",
                },
                "value": {
                    "type": "string",
                    "description": "The value to select",
                },
                "by_label": {
                    "type": "boolean",
                    "description": "Select by label text instead of value",
                    "default": False,
                },
            },
            "required": ["element_id", "value"],
        },
    ),
    types.Tool(
        name="ui_scroll",
        description="Scroll within an element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to scroll",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction",
                },
                "amount": {
                    "type": "number",
                    "description": "Scroll amount in pixels",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_check",
        description="Check a checkbox element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The checkbox element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_uncheck",
        description="Uncheck a checkbox element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The checkbox element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_toggle",
        description="Toggle a checkbox element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The checkbox element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_set_value",
        description="Set the value of an input element directly in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id",
                },
                "value": {
                    "type": "string",
                    "description": "The value to set",
                },
            },
            "required": ["element_id", "value"],
        },
    ),
    types.Tool(
        name="ui_drag",
        description="""Drag an element to a target in the runner's UI.

Drag from source element to target element or position.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The source element's data-ui-id to drag",
                },
                "target_element_id": {
                    "type": "string",
                    "description": "The target element's data-ui-id to drop on",
                },
                "steps": {
                    "type": "number",
                    "description": "Number of intermediate mousemove steps (default: 10)",
                },
                "hold_delay": {
                    "type": "number",
                    "description": "Delay in ms before first move (default: 100)",
                },
            },
            "required": ["element_id", "target_element_id"],
        },
    ),
    types.Tool(
        name="ui_submit",
        description="Submit the form containing the element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id (element or its parent form)",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="ui_reset",
        description="Reset the form containing the element in the runner's UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id (element or its parent form)",
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

Use agent_mode=true for compact output with short refs (@e1, @e2).
Use interactive_only=true to exclude content elements.
Use max_elements to limit output size.""",
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
                "agent_mode": {
                    "type": "boolean",
                    "description": (
                        "Compact output with short refs (@e1, @e2). "
                        "Use refs in subsequent actions. "
                        "Full details via sdk_get_element."
                    ),
                    "default": False,
                },
                "interactive_only": {
                    "type": "boolean",
                    "description": (
                        "Only return interactive elements (buttons, inputs, links). "
                        "Excludes static content. Overrides include_content."
                    ),
                    "default": False,
                },
                "max_elements": {
                    "type": "integer",
                    "description": "Max elements to return. Remaining summarized as count.",
                },
                "max_content_length": {
                    "type": "integer",
                    "description": (
                        "Max chars per text field (label, value). "
                        "Longer values truncated."
                    ),
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_elements",
        description="""List all registered UI elements in the SDK app.

Returns element IDs, types, labels, and current state.
Supports filtering by content type to find specific kinds of elements.
Use agent_mode=true for compact output with short refs (@e1, @e2).""",
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
                "agent_mode": {
                    "type": "boolean",
                    "description": (
                        "Compact output with short refs (@e1, @e2). "
                        "Use refs in subsequent actions."
                    ),
                    "default": False,
                },
                "max_elements": {
                    "type": "integer",
                    "description": "Max elements to return. Remaining summarized as count.",
                },
                "max_content_length": {
                    "type": "integer",
                    "description": (
                        "Max chars per text field (label, value). "
                        "Longer values truncated."
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
enabled state, text content, and available actions.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id or agent ref (e.g., '@e1')",
                },
                "max_content_length": {
                    "type": "integer",
                    "description": "Max chars per text field. Longer values truncated.",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_click",
        description="""Click an element in the SDK app by its data-ui-id.

Use sdk_snapshot or sdk_elements first to find the element_id.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id or agent ref (@e1)",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_type",
        description="""Type text into an input element in the SDK app.

Use sdk_snapshot first to find the element_id of the input field.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id or agent ref (@e1)",
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
        name="sdk_clear",
        description="Clear an input element in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id to clear",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_select",
        description="Select an option in a dropdown in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id",
                },
                "value": {
                    "type": "string",
                    "description": "The value to select",
                },
            },
            "required": ["element_id", "value"],
        },
    ),
    types.Tool(
        name="sdk_focus",
        description="Focus an element in the SDK app.",
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
        name="sdk_blur",
        description="Remove focus from an element in the SDK app.",
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
        name="sdk_hover",
        description="Hover over an element in the SDK app.",
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
        name="sdk_double_click",
        description="Double-click an element in the SDK app.",
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
        name="sdk_right_click",
        description="Right-click an element in the SDK app.",
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
        name="sdk_scroll",
        description="Scroll within an element in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction",
                },
                "amount": {
                    "type": "number",
                    "description": "Scroll amount in pixels",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_check",
        description="Check a checkbox in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The checkbox element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_uncheck",
        description="Uncheck a checkbox in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The checkbox element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_toggle",
        description="Toggle a checkbox in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The checkbox element's data-ui-id",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_set_value",
        description="Set the value of an input element directly in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's data-ui-id",
                },
                "value": {
                    "type": "string",
                    "description": "The value to set",
                },
            },
            "required": ["element_id", "value"],
        },
    ),
    types.Tool(
        name="sdk_drag",
        description="Drag an element to a target in the SDK app.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The source element's data-ui-id",
                },
                "target_element_id": {
                    "type": "string",
                    "description": "The target element's data-ui-id",
                },
                "steps": {
                    "type": "number",
                    "description": "Number of intermediate mousemove steps (default: 10)",
                },
            },
            "required": ["element_id", "target_element_id"],
        },
    ),
    types.Tool(
        name="sdk_submit",
        description="Submit the form containing the element in the SDK app.",
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
        name="sdk_reset",
        description="Reset the form containing the element in the SDK app.",
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
    # Agent Mode Tools
    types.Tool(
        name="ui_diff",
        description="""Show what changed since the last ui_snapshot.

Returns appeared, disappeared, and modified elements.
Must call ui_snapshot at least once before using this.
If agent_mode was used, includes refs in the output.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_diff",
        description="""Show what changed since the last sdk_snapshot.

Returns appeared, disappeared, and modified elements.
Must call sdk_snapshot at least once before using this.
If agent_mode was used, includes refs in the output.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ui_annotated_screenshot",
        description="""Capture a screenshot of the runner's UI with element labels overlaid.

Each visible element gets a numbered overlay (@e1, @e2) matching agent mode refs.
Returns an annotated image. Useful for understanding element positions visually.""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based). Defaults to primary monitor.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_annotated_screenshot",
        description="""Capture a screenshot of the SDK app's monitor with element labels overlaid.

Each visible element gets a numbered overlay (@e1, @e2) matching agent mode refs.
Returns an annotated image. Useful for understanding element positions visually.""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based). Defaults to primary monitor.",
                },
            },
            "required": [],
        },
    ),
    # =========================================================================
    # SDK Design Review Tools
    # =========================================================================
    types.Tool(
        name="sdk_design_styles",
        description="""Get extended computed styles (~40 CSS properties) for element(s) in the connected SDK app.

Returns layout, typography, visual, and effect properties. Optionally includes
interaction state variations (hover, focus, active, disabled) showing style diffs.

Use this to inspect how an element is actually styled.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID to inspect. If omitted, returns styles for all elements.",
                },
                "include_state_variations": {
                    "type": "boolean",
                    "description": "Also capture hover/focus/active/disabled style variations.",
                    "default": False,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_design_state_styles",
        description="""Get styles across interaction states for an element.

On web: dispatches synthetic events to trigger hover, focus, active, disabled states.
On native (React Native): returns pressed, focused, disabled state variations from
declarative style overrides. Hover and active are not applicable on mobile.

Returns a diff showing which properties change in each state.
Useful for verifying hover effects, focus rings, pressed feedback, etc.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID to inspect.",
                },
                "states": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["hover", "focus", "active", "disabled", "pressed"],
                    },
                    "description": "Which states to capture. Defaults to all.",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_design_responsive",
        description="""Capture design snapshots at multiple viewport widths.

On web: constrains the document width to simulate responsive breakpoints.
On native (React Native): returns a single snapshot at the current device
screen dimensions (RN cannot constrain screen width at runtime).

Preset viewports (web only): mobile (375px), tablet (768px), desktop (1280px), wide (1920px).
Or provide custom viewports as a label→width mapping.""",
        inputSchema={
            "type": "object",
            "properties": {
                "viewports": {
                    "type": "object",
                    "description": 'Custom viewports as {"label": width_px}. Defaults to mobile/tablet/desktop/wide.',
                    "additionalProperties": {"type": "integer"},
                },
                "element_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include these elements. Defaults to all.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_design_audit",
        description="""Run a style audit against a loaded or provided style guide.

Validates element computed styles against design tokens and rules defined
in a StyleGuideConfig. Returns pass/fail results grouped by severity.

Load a guide first with sdk_design_load_guide, or provide one inline.""",
        inputSchema={
            "type": "object",
            "properties": {
                "guide": {
                    "type": "object",
                    "description": "Inline StyleGuideConfig. Uses the loaded guide if omitted.",
                },
                "element_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only audit these elements. Defaults to all.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_design_load_guide",
        description="""Load a style guide for subsequent design audits.

The guide defines design tokens (colors, typography, spacing, etc.) and
validation rules that constrain how elements should be styled.

The guide persists in memory until cleared or replaced.""",
        inputSchema={
            "type": "object",
            "properties": {
                "guide": {
                    "type": "object",
                    "description": "StyleGuideConfig JSON with version, name, tokens, and rules.",
                },
            },
            "required": ["guide"],
        },
    ),
    types.Tool(
        name="sdk_design_review",
        description="""Compound design review: snapshot + state variations + audit + quality evaluation in one call.

Works with both web SDK and React Native SDK apps. On native, state variations
use pressed/focused/disabled instead of hover/focus/active/disabled, responsive
snapshots return only the current device dimensions, and pseudo-elements are empty.

Captures a full design snapshot, optionally captures state variations for
interactive elements, runs a style audit if a guide is loaded, and evaluates
overall UI quality with scores and actionable recommendations.

This is the primary tool for design review — use it instead of calling
individual design tools separately.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only review these elements. Defaults to all.",
                },
                "include_responsive": {
                    "type": "boolean",
                    "description": "Also capture responsive snapshots at standard breakpoints.",
                    "default": False,
                },
                "include_state_variations": {
                    "type": "boolean",
                    "description": "Capture hover/focus/active/disabled variations for interactive elements.",
                    "default": True,
                },
                "quality_context": {
                    "type": "string",
                    "description": "Quality evaluation context (general, minimal, data-dense, mobile, accessibility, or a custom name from loaded style guide). Defaults to 'general'.",
                },
                "include_quality_evaluation": {
                    "type": "boolean",
                    "description": "Run holistic quality evaluation and include score/findings.",
                    "default": True,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_design_evaluate",
        description="""Run holistic UI quality evaluation. Returns 0-100 score, letter grade,
per-metric scores across density/spacing/color/typography/consistency,
and actionable recommendations.

Contexts adjust what's measured and how strictly:
- general: Balanced evaluation for most web apps
- minimal: Emphasizes whitespace and simplicity
- data-dense: Lenient on density, strict on alignment and consistency
- mobile: Prioritizes touch targets and readability
- accessibility: Focused on WCAG compliance (contrast, heading hierarchy, touch targets)

Use this as the primary tool for assessing overall UI quality.""",
        inputSchema={
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "enum": [
                        "general",
                        "minimal",
                        "data-dense",
                        "mobile",
                        "accessibility",
                    ],
                    "description": "Evaluation context. Defaults to 'general'.",
                },
                "custom_context": {
                    "type": "object",
                    "description": "Custom context with metric weights/thresholds (overrides context).",
                },
                "element_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only evaluate these elements. Defaults to all.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_design_diff",
        description="""Save a UI baseline or diff against a saved baseline for regression detection.

Two modes:
1. save_baseline=true: Save current element state as baseline
2. save_baseline=false (default): Diff current state against saved baseline

Returns added/removed/modified elements and cumulative layout shift score.""",
        inputSchema={
            "type": "object",
            "properties": {
                "save_baseline": {
                    "type": "boolean",
                    "description": "If true, save current state as baseline instead of diffing.",
                    "default": False,
                },
                "label": {
                    "type": "string",
                    "description": "Label for the baseline (only used when save_baseline=true).",
                },
                "element_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include these elements. Defaults to all.",
                },
            },
            "required": [],
        },
    ),
]


@server.list_tools()  # type: ignore
async def list_tools() -> list[types.Tool]:
    """List available UI Bridge tools."""
    return TOOLS


@server.call_tool()  # type: ignore
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent]:
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
            agent_mode = arguments.get("agent_mode", False)
            interactive_only = arguments.get("interactive_only", False)
            max_elements = arguments.get("max_elements")
            max_content_length = arguments.get("max_content_length")

            response = await ui_client.control_snapshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]

            data = response.data or {}
            elements = data.get("elements", [])

            # Feature 2: Interactive-only filtering
            if interactive_only:
                elements = [el for el in elements if el.get("category") != "content"]

            # Update diff tracker (control mode)
            control_diff_tracker.update_and_diff(elements)

            # Feature 3: Truncate content fields
            if max_content_length:
                for el in elements:
                    el["label"] = truncate_field(el.get("label"), max_content_length)
                    state = el.get("state", {})
                    for field in ("textContent", "value"):
                        if field in state:
                            state[field] = truncate_field(
                                state.get(field), max_content_length
                            )

            # Feature 3: Limit element count
            overflow = 0
            if max_elements and len(elements) > max_elements:
                overflow = len(elements) - max_elements
                elements = elements[:max_elements]

            total_count = len(elements) + overflow

            if agent_mode:
                # Feature 1: Compact refs
                ref_manager.reset()
                mode_label = "agent mode"
                if interactive_only:
                    mode_label += ", interactive only"
                lines = [f"UI Snapshot ({total_count} elements, {mode_label})", ""]

                by_type: dict[str, list[dict[str, Any]]] = {}
                for el in elements:
                    el_type = el.get("type", "unknown")
                    if el_type not in by_type:
                        by_type[el_type] = []
                    by_type[el_type].append(el)

                for el_type, els in sorted(by_type.items()):
                    lines.append(f"## {el_type} ({len(els)})")
                    for el in els:
                        ref = ref_manager.assign(el.get("id", "?"))
                        lines.append(format_element_compact(el, ref))
                    lines.append("")
            else:
                lines = [f"UI Snapshot ({total_count} elements):", ""]
                by_type = {}
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

            if overflow:
                lines.append(f"+{overflow} more elements not shown")

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
            element_id = ref_manager.resolve(arguments["element_id"])
            max_content_length = arguments.get("max_content_length")
            response = await ui_client.control_get_element(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            result_data = response.data or {}
            # Feature 5: Content boundary markers
            sanitize_element_content(result_data)
            # Feature 3: Truncate content fields
            if max_content_length:
                state = result_data.get("state", {})
                for field in ("textContent", "innerHTML", "value"):
                    if field in state:
                        state[field] = truncate_field(
                            state.get(field), max_content_length
                        )
            return [
                types.TextContent(type="text", text=json.dumps(result_data, indent=2))
            ]

        elif name == "ui_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_click(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Clicked element: {element_id}")
            ]

        elif name == "ui_type":
            element_id = ref_manager.resolve(arguments["element_id"])
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
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_focus(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Focused element: {element_id}")
            ]

        elif name == "ui_blur":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "blur")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Blurred element: {element_id}")
            ]

        elif name == "ui_hover":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_hover(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Hovered element: {element_id}")
            ]

        elif name == "ui_double_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "doubleClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Double-clicked element: {element_id}"
                )
            ]

        elif name == "ui_right_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "rightClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Right-clicked element: {element_id}"
                )
            ]

        elif name == "ui_clear":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "clear")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Cleared element: {element_id}")
            ]

        elif name == "ui_select":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            params = {"value": value}
            if arguments.get("by_label"):
                params["byLabel"] = True
            response = await ui_client.control_action(element_id, "select", params)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Selected '{value}' in element: {element_id}"
                )
            ]

        elif name == "ui_scroll":
            element_id = ref_manager.resolve(arguments["element_id"])
            scroll_params: dict[str, Any] = {}
            if "direction" in arguments:
                scroll_params["direction"] = arguments["direction"]
            if "amount" in arguments:
                scroll_params["amount"] = arguments["amount"]
            response = await ui_client.control_action(
                element_id, "scroll", scroll_params
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Scrolled element: {element_id}")
            ]

        elif name == "ui_check":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "check")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Checked element: {element_id}")
            ]

        elif name == "ui_uncheck":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "uncheck")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Unchecked element: {element_id}")
            ]

        elif name == "ui_toggle":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "toggle")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Toggled element: {element_id}")
            ]

        elif name == "ui_set_value":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            response = await ui_client.control_action(
                element_id, "setValue", {"value": value}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Set value '{value}' on element: {element_id}"
                )
            ]

        elif name == "ui_drag":
            element_id = ref_manager.resolve(arguments["element_id"])
            target_id = ref_manager.resolve(arguments["target_element_id"])
            params = {"target": {"elementId": target_id}}
            if "steps" in arguments:
                params["steps"] = arguments["steps"]
            if "hold_delay" in arguments:
                params["holdDelay"] = arguments["hold_delay"]
            response = await ui_client.control_action(element_id, "drag", params)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Dragged {element_id} to {target_id}"
                )
            ]

        elif name == "ui_submit":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "submit")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Submitted form for element: {element_id}"
                )
            ]

        elif name == "ui_reset":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "reset")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Reset form for element: {element_id}"
                )
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
            agent_mode = arguments.get("agent_mode", False)
            interactive_only = arguments.get("interactive_only", False)
            max_elements = arguments.get("max_elements")
            max_content_length = arguments.get("max_content_length")

            response = await ui_client.sdk_snapshot(
                include_content=include_content,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            elements = data.get("elements", [])

            # Feature 2: Interactive-only filtering (overrides include_content)
            if interactive_only:
                elements = [el for el in elements if el.get("category") != "content"]
            elif not include_content:
                elements = [el for el in elements if el.get("category") != "content"]

            # Update diff tracker (SDK mode)
            sdk_diff_tracker.update_and_diff(elements)

            # Feature 3: Truncate content fields
            if max_content_length:
                for el in elements:
                    el["label"] = truncate_field(el.get("label"), max_content_length)
                    state = el.get("state", {})
                    for field in ("textContent", "value"):
                        if field in state:
                            state[field] = truncate_field(
                                state.get(field), max_content_length
                            )

            # Feature 3: Limit element count
            overflow = 0
            if max_elements and len(elements) > max_elements:
                overflow = len(elements) - max_elements
                elements = elements[:max_elements]

            total_count = len(elements) + overflow

            if agent_mode:
                # Feature 1: Compact refs
                ref_manager.reset()
                mode_label = "agent mode"
                if interactive_only:
                    mode_label += ", interactive only"
                lines = [
                    f"SDK Snapshot ({total_count} elements, {mode_label})",
                    "",
                ]

                sdk_by_type: dict[str, list[dict[str, Any]]] = {}
                for el in elements:
                    el_type = el.get("type", "unknown")
                    if el_type not in sdk_by_type:
                        sdk_by_type[el_type] = []
                    sdk_by_type[el_type].append(el)

                for el_type, els in sorted(sdk_by_type.items()):
                    lines.append(f"## {el_type} ({len(els)})")
                    for el in els:
                        ref = ref_manager.assign(el.get("id", "?"))
                        lines.append(format_element_compact(el, ref))
                    lines.append("")
            else:
                lines = [f"SDK Snapshot ({total_count} elements):", ""]
                sdk_by_type = {}
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

            if overflow:
                lines.append(f"+{overflow} more elements not shown")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_elements":
            content_only = arguments.get("content_only", False)
            content_types = arguments.get("content_types")
            agent_mode = arguments.get("agent_mode", False)
            max_elements = arguments.get("max_elements")
            max_content_length = arguments.get("max_content_length")

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

            # Truncate content fields
            if max_content_length:
                for el in elements:
                    el["label"] = truncate_field(el.get("label"), max_content_length)
                    state = el.get("state", {})
                    for field in ("textContent", "value"):
                        if field in state:
                            state[field] = truncate_field(
                                state.get(field), max_content_length
                            )

            # Limit element count
            overflow = 0
            if max_elements and len(elements) > max_elements:
                overflow = len(elements) - max_elements
                elements = elements[:max_elements]

            total_count = len(elements) + overflow
            filter_desc = ""
            if content_only:
                filter_desc = " (content only)"
            elif content_types:
                filter_desc = f" (filtered: {', '.join(content_types)})"

            if agent_mode:
                ref_manager.reset()
                lines = [
                    f"SDK Elements ({total_count}){filter_desc} [agent mode]:",
                    "",
                ]
                for el in elements:
                    ref = ref_manager.assign(el.get("id", "?"))
                    lines.append(format_element_compact(el, ref))
            else:
                lines = [f"SDK Elements ({total_count}){filter_desc}:", ""]
                for el in elements:
                    lines.append(format_element_summary(el))

            if overflow:
                lines.append(f"\n+{overflow} more elements not shown")
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
            element_id = ref_manager.resolve(arguments["element_id"])
            max_content_length = arguments.get("max_content_length")
            response = await ui_client.sdk_element(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            result_data = response.data or {}
            # Feature 5: Content boundary markers
            sanitize_element_content(result_data)
            # Feature 3: Truncate content fields
            if max_content_length:
                state = result_data.get("state", {})
                for field in ("textContent", "innerHTML", "value"):
                    if field in state:
                        state[field] = truncate_field(
                            state.get(field), max_content_length
                        )
            return [
                types.TextContent(type="text", text=json.dumps(result_data, indent=2))
            ]

        elif name == "sdk_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "click")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Clicked element: {element_id}")
            ]

        elif name == "sdk_type":
            element_id = ref_manager.resolve(arguments["element_id"])
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

        elif name == "sdk_clear":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "clear")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Cleared element: {element_id}")
            ]

        elif name == "sdk_select":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            response = await ui_client.sdk_element_action(
                element_id, "select", {"value": value}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Selected '{value}' in element: {element_id}"
                )
            ]

        elif name == "sdk_focus":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "focus")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Focused element: {element_id}")
            ]

        elif name == "sdk_blur":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "blur")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Blurred element: {element_id}")
            ]

        elif name == "sdk_hover":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "hover")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Hovered element: {element_id}")
            ]

        elif name == "sdk_double_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "doubleClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Double-clicked element: {element_id}"
                )
            ]

        elif name == "sdk_right_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "rightClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Right-clicked element: {element_id}"
                )
            ]

        elif name == "sdk_scroll":
            element_id = ref_manager.resolve(arguments["element_id"])
            sdk_scroll_params: dict[str, Any] = {}
            if "direction" in arguments:
                sdk_scroll_params["direction"] = arguments["direction"]
            if "amount" in arguments:
                sdk_scroll_params["amount"] = arguments["amount"]
            response = await ui_client.sdk_element_action(
                element_id, "scroll", sdk_scroll_params or None
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Scrolled element: {element_id}")
            ]

        elif name == "sdk_check":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "check")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Checked element: {element_id}")
            ]

        elif name == "sdk_uncheck":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "uncheck")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Unchecked element: {element_id}")
            ]

        elif name == "sdk_toggle":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "toggle")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=f"Toggled element: {element_id}")
            ]

        elif name == "sdk_set_value":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            response = await ui_client.sdk_element_action(
                element_id, "setValue", {"value": value}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Set value '{value}' on element: {element_id}"
                )
            ]

        elif name == "sdk_drag":
            element_id = ref_manager.resolve(arguments["element_id"])
            target_id = ref_manager.resolve(arguments["target_element_id"])
            params = {"target": {"elementId": target_id}}
            if "steps" in arguments:
                params["steps"] = arguments["steps"]
            response = await ui_client.sdk_element_action(element_id, "drag", params)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Dragged {element_id} to {target_id}"
                )
            ]

        elif name == "sdk_submit":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "submit")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Submitted form for element: {element_id}"
                )
            ]

        elif name == "sdk_reset":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "reset")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Reset form for element: {element_id}"
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

        # Agent Mode: Diff Tools
        elif name == "ui_diff":
            response = await ui_client.control_snapshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            elements = data.get("elements", [])
            diff = control_diff_tracker.update_and_diff(elements)
            if diff is None:
                return [
                    types.TextContent(
                        type="text",
                        text="No previous snapshot to diff against. Call ui_snapshot first.",
                    )
                ]
            return [
                types.TextContent(type="text", text=_format_diff(diff, ref_manager))
            ]

        elif name == "sdk_diff":
            response = await ui_client.sdk_snapshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            elements = data.get("elements", [])
            diff = sdk_diff_tracker.update_and_diff(elements)
            if diff is None:
                return [
                    types.TextContent(
                        type="text",
                        text="No previous snapshot to diff against. Call sdk_snapshot first.",
                    )
                ]
            return [
                types.TextContent(type="text", text=_format_diff(diff, ref_manager))
            ]

        # Agent Mode: Annotated Screenshots
        elif name == "ui_annotated_screenshot":
            monitor = arguments.get("monitor")
            # Get snapshot for element positions
            snap_resp = await ui_client.control_snapshot()
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text", text=f"Error getting snapshot: {snap_resp.error}"
                    )
                ]
            snap_elements = (snap_resp.data or {}).get("elements", [])
            # Get screenshot
            screenshot_resp = await ui_client.control_annotated_screenshot(
                monitor=monitor
            )
            if not screenshot_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error getting screenshot: {screenshot_resp.error}",
                    )
                ]
            ss_data = screenshot_resp.data or {}
            screenshot_b64 = ss_data.get("screenshot", "")
            ss_width = ss_data.get("width", 0)
            ss_height = ss_data.get("height", 0)
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]
            annotated_b64 = _annotate_screenshot(
                screenshot_b64, snap_elements, ss_width, ss_height, ref_manager
            )
            return [
                types.ImageContent(
                    type="image", data=annotated_b64, mimeType="image/png"
                )
            ]

        elif name == "sdk_annotated_screenshot":
            monitor = arguments.get("monitor")
            # Get snapshot for element positions
            snap_resp = await ui_client.sdk_snapshot()
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text", text=f"Error getting snapshot: {snap_resp.error}"
                    )
                ]
            snap_elements = (snap_resp.data or {}).get("elements", [])
            # Get screenshot
            screenshot_resp = await ui_client.sdk_screenshot_raw(monitor=monitor)
            if not screenshot_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error getting screenshot: {screenshot_resp.error}",
                    )
                ]
            ss_data = screenshot_resp.data or {}
            screenshot_b64 = ss_data.get("screenshot", "")
            ss_width = ss_data.get("width", 0)
            ss_height = ss_data.get("height", 0)
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]
            annotated_b64 = _annotate_screenshot(
                screenshot_b64, snap_elements, ss_width, ss_height, ref_manager
            )
            return [
                types.ImageContent(
                    type="image", data=annotated_b64, mimeType="image/png"
                )
            ]

        # =====================================================================
        # SDK Design Review Tools
        # =====================================================================
        elif name == "sdk_design_styles":
            element_id = arguments.get("element_id", "")
            include_state_variations = arguments.get("include_state_variations", False)

            if element_id:
                # Resolve ref if needed
                element_id = ref_manager.resolve(element_id)
                response = await ui_client.sdk_design_element_styles(element_id)
                if not response.success:
                    return [
                        types.TextContent(type="text", text=f"Error: {response.error}")
                    ]
                result_lines = [f"Design styles for {element_id}:"]
                data = response.data or {}
                styles = data.get("styles", {})
                for prop, val in styles.items():
                    if val and val != "none" and val != "normal" and val != "0px":
                        result_lines.append(f"  {prop}: {val}")

                if include_state_variations:
                    sv_resp = await ui_client.sdk_design_state_styles(element_id)
                    if sv_resp.success:
                        sv_data = sv_resp.data or {}
                        for state_info in sv_data.get("stateStyles", []):
                            state_name = state_info.get("state", "?")
                            diffs = state_info.get("diffFromDefault", [])
                            if diffs:
                                result_lines.append(f"\n  [{state_name}] changes:")
                                for d in diffs:
                                    result_lines.append(
                                        f"    {d['property']}: {d['defaultValue']} → {d['stateValue']}"
                                    )

                return [types.TextContent(type="text", text="\n".join(result_lines))]
            else:
                # Get snapshot of all elements
                response = await ui_client.sdk_design_snapshot()
                if not response.success:
                    return [
                        types.TextContent(type="text", text=f"Error: {response.error}")
                    ]
                data = response.data or {}
                elements = data.get("elements", [])
                result_lines = [f"Design snapshot ({len(elements)} elements):"]
                for el in elements[:50]:  # Limit output
                    eid = el.get("elementId", "?")
                    etype = el.get("type", "?")
                    styles = el.get("styles", {})
                    font_size = styles.get("fontSize", "?")
                    color = styles.get("color", "?")
                    bg = styles.get("backgroundColor", "?")
                    result_lines.append(
                        f"  {eid} ({etype}): font={font_size} color={color} bg={bg}"
                    )
                if len(elements) > 50:
                    result_lines.append(f"  ... and {len(elements) - 50} more")
                return [types.TextContent(type="text", text="\n".join(result_lines))]

        elif name == "sdk_design_state_styles":
            element_id = ref_manager.resolve(arguments["element_id"])
            states = arguments.get("states")
            response = await ui_client.sdk_design_state_styles(element_id, states)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            result_lines = [f"State styles for {element_id}:"]
            for state_info in data.get("stateStyles", []):
                state_name = state_info.get("state", "?")
                diffs = state_info.get("diffFromDefault", [])
                if state_name == "default":
                    result_lines.append("\n  [default] (base styles)")
                elif diffs:
                    result_lines.append(f"\n  [{state_name}] ({len(diffs)} changes):")
                    for d in diffs:
                        result_lines.append(
                            f"    {d['property']}: {d['defaultValue']} → {d['stateValue']}"
                        )
                else:
                    result_lines.append(f"\n  [{state_name}] no changes")
            return [types.TextContent(type="text", text="\n".join(result_lines))]

        elif name == "sdk_design_responsive":
            viewports = arguments.get("viewports")
            element_ids = arguments.get("element_ids")
            response = await ui_client.sdk_design_responsive(viewports, element_ids)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            snapshots: list[dict[str, object]] = (
                response.data if isinstance(response.data, list) else []
            )
            if isinstance(response.data, dict):
                snapshots = (
                    response.data.get("data", [])
                    if "data" in response.data
                    else [response.data]
                )
            result_lines = [f"Responsive snapshots ({len(snapshots)} viewports):"]
            for snap in snapshots:
                vw = snap.get("viewportWidth", "?")
                label = snap.get("viewportLabel", "")
                elements = snap.get("elements", [])
                result_lines.append(
                    f"\n  === {label} ({vw}px) — {len(elements)} elements ==="
                )
                for el in elements[:20]:
                    eid = el.get("elementId", "?")
                    rect = el.get("rect", {})
                    w = rect.get("width", "?")
                    h = rect.get("height", "?")
                    display = el.get("styles", {}).get("display", "?")
                    result_lines.append(f"    {eid}: {w}×{h} display={display}")
                if len(elements) > 20:
                    result_lines.append(f"    ... and {len(elements) - 20} more")
            return [types.TextContent(type="text", text="\n".join(result_lines))]

        elif name == "sdk_design_audit":
            guide = arguments.get("guide")
            element_ids = arguments.get("element_ids")
            response = await ui_client.sdk_design_audit(guide, element_ids)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            report = response.data or {}
            result_lines = [
                f"Style Audit: {report.get('guideName', '?')}",
                f"Elements: {report.get('totalElements', 0)} | Rules: {report.get('totalRules', 0)}",
                f"Passed: {report.get('passedCount', 0)} | Failed: {report.get('failedCount', 0)}",
            ]
            summary = report.get("summary", {})
            errors = summary.get("errors", [])
            warnings = summary.get("warnings", [])
            if errors:
                result_lines.append(f"\nErrors ({len(errors)}):")
                for r in errors[:20]:
                    eid = r.get("elementId", "?")
                    rule_id = r.get("ruleId", "?")
                    for cr in r.get("constraintResults", []):
                        if not cr.get("passed"):
                            result_lines.append(
                                f"  [{eid}] {rule_id}: {cr.get('message', '?')}"
                            )
            if warnings:
                result_lines.append(f"\nWarnings ({len(warnings)}):")
                for r in warnings[:20]:
                    eid = r.get("elementId", "?")
                    rule_id = r.get("ruleId", "?")
                    for cr in r.get("constraintResults", []):
                        if not cr.get("passed"):
                            result_lines.append(
                                f"  [{eid}] {rule_id}: {cr.get('message', '?')}"
                            )
            return [types.TextContent(type="text", text="\n".join(result_lines))]

        elif name == "sdk_design_load_guide":
            guide = arguments["guide"]
            response = await ui_client.sdk_design_load_guide(guide)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=f"Style guide loaded: {guide.get('name', '?')} ({len(guide.get('rules', []))} rules)",
                )
            ]

        elif name == "sdk_design_review":
            element_ids = arguments.get("element_ids")
            include_responsive = arguments.get("include_responsive", False)
            include_state_variations = arguments.get("include_state_variations", True)
            quality_context = arguments.get("quality_context", "general")
            include_quality_evaluation = arguments.get(
                "include_quality_evaluation", True
            )
            result_lines = ["=== Design Review ==="]

            # 1. Get design snapshot
            snap_resp = await ui_client.sdk_design_snapshot(element_ids)
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error getting design snapshot: {snap_resp.error}",
                    )
                ]
            snap_data = snap_resp.data or {}
            elements = snap_data.get("elements", [])
            result_lines.append(f"\nSnapshot: {len(elements)} elements")
            for el in elements[:30]:
                eid = el.get("elementId", "?")
                etype = el.get("type", "?")
                styles = el.get("styles", {})
                result_lines.append(
                    f"  {eid} ({etype}): font={styles.get('fontSize', '?')} "
                    f"color={styles.get('color', '?')} bg={styles.get('backgroundColor', '?')}"
                )
            if len(elements) > 30:
                result_lines.append(f"  ... and {len(elements) - 30} more")

            # 2. State variations for interactive elements
            if include_state_variations:
                interactive_ids = [
                    el.get("elementId")
                    for el in elements
                    if el.get("type")
                    in (
                        "button",
                        "input",
                        "select",
                        "link",
                        "checkbox",
                        "radio",
                        "textarea",
                        "pressable",
                        "touchable",
                        "switch",
                    )
                ]
                if interactive_ids:
                    result_lines.append(
                        f"\nState variations ({len(interactive_ids)} interactive elements):"
                    )
                    for eid in interactive_ids[:10]:
                        sv_resp = await ui_client.sdk_design_state_styles(eid)
                        if sv_resp.success:
                            sv_data = sv_resp.data or {}
                            for state_info in sv_data.get("stateStyles", []):
                                diffs = state_info.get("diffFromDefault", [])
                                if diffs:
                                    state_name = state_info.get("state", "?")
                                    result_lines.append(
                                        f"  {eid} [{state_name}]: {len(diffs)} changes"
                                    )
                                    for d in diffs[:5]:
                                        result_lines.append(
                                            f"    {d['property']}: {d['defaultValue']} → {d['stateValue']}"
                                        )
                                    if len(diffs) > 5:
                                        result_lines.append(
                                            f"    ... and {len(diffs) - 5} more"
                                        )

            # 3. Responsive snapshots
            if include_responsive:
                resp_resp = await ui_client.sdk_design_responsive(
                    element_ids=element_ids
                )
                if resp_resp.success:
                    resp_snaps: list[dict[str, object]] = (
                        resp_resp.data if isinstance(resp_resp.data, list) else []
                    )
                    if isinstance(resp_resp.data, dict):
                        resp_snaps = (
                            resp_resp.data.get("data", [])
                            if "data" in resp_resp.data
                            else [resp_resp.data]
                        )
                    result_lines.append(f"\nResponsive ({len(resp_snaps)} viewports):")
                    for snap in resp_snaps:
                        label = snap.get("viewportLabel", "?")
                        vw = snap.get("viewportWidth", "?")
                        elems = snap.get("elements", [])
                        count = len(elems) if isinstance(elems, list) else 0
                        result_lines.append(f"  {label} ({vw}px): {count} elements")

            # 4. Style audit (if guide loaded)
            audit_resp = await ui_client.sdk_design_audit(element_ids=element_ids)
            if audit_resp.success:
                report = audit_resp.data or {}
                failed = report.get("failedCount", 0)
                passed = report.get("passedCount", 0)
                result_lines.append(f"\nStyle audit: {passed} passed, {failed} failed")
                summary = report.get("summary", {})
                for sev in ("errors", "warnings"):
                    items = summary.get(sev, [])
                    if items:
                        result_lines.append(f"  {sev.title()} ({len(items)}):")
                        for r in items[:10]:
                            eid = r.get("elementId", "?")
                            for cr in r.get("constraintResults", []):
                                if not cr.get("passed"):
                                    result_lines.append(
                                        f"    [{eid}] {cr.get('message', '?')}"
                                    )
            elif "NO_STYLE_GUIDE" not in (audit_resp.error or ""):
                result_lines.append(f"\nStyle audit: {audit_resp.error}")

            # 5. Quality evaluation
            if include_quality_evaluation:
                try:
                    eval_resp = await ui_client.sdk_design_evaluate(
                        context=quality_context,
                        element_ids=element_ids,
                    )
                    if eval_resp.success:
                        report = eval_resp.data or {}
                        score = report.get("overallScore", "?")
                        grade = report.get("grade", "?")
                        result_lines.append(f"\nQuality: {score}/100 (Grade {grade})")

                        # Category averages
                        metrics = report.get("metrics", [])
                        categories: dict[str, list[int]] = {}
                        for m in metrics:
                            if m.get("enabled"):
                                cat = m.get("category", "?")
                                if cat not in categories:
                                    categories[cat] = []
                                categories[cat].append(m.get("score", 0))
                        if categories:
                            cat_parts = []
                            for cat, scores in categories.items():
                                avg = sum(scores) / len(scores) if scores else 0
                                cat_parts.append(f"{cat}={avg:.0f}")
                            result_lines.append(f"  Categories: {', '.join(cat_parts)}")

                        # Top 5 issues
                        top_issues = report.get("topIssues", [])
                        if top_issues:
                            result_lines.append("  Top issues:")
                            for issue in top_issues[:5]:
                                severity = issue.get("severity", "info").upper()
                                message = issue.get("message", "?")
                                result_lines.append(f"    [{severity}] {message}")
                                rec = issue.get("recommendation")
                                if rec:
                                    result_lines.append(f"      → {rec}")
                    else:
                        result_lines.append(f"\nQuality evaluation: {eval_resp.error}")
                except Exception as e:
                    result_lines.append(f"\nQuality evaluation error: {e}")

            return [types.TextContent(type="text", text="\n".join(result_lines))]

        # =====================================================================
        # Quality Evaluation
        # =====================================================================

        elif name == "sdk_design_evaluate":
            context = arguments.get("context")
            custom_context = arguments.get("custom_context")
            element_ids = arguments.get("element_ids")

            response = await ui_client.sdk_design_evaluate(
                context=context,
                custom_context=custom_context,
                element_ids=element_ids,
            )

            if not response.success:
                return [
                    types.TextContent(
                        type="text", text=f"Quality evaluation error: {response.error}"
                    )
                ]

            report = response.data or {}
            lines = [
                f"=== UI Quality Evaluation ({report.get('contextName', '?')}) ===",
                f"Overall Score: {report.get('overallScore', '?')}/100  Grade: {report.get('grade', '?')}",
                f"Elements: {report.get('totalElements', '?')}  Duration: {report.get('durationMs', '?')}ms",
            ]

            # Category averages
            metrics = report.get("metrics", [])
            eval_categories: dict[str, list[int]] = {}
            for m in metrics:
                if m.get("enabled"):
                    cat = m.get("category", "?")
                    if cat not in eval_categories:
                        eval_categories[cat] = []
                    eval_categories[cat].append(m.get("score", 0))

            if eval_categories:
                lines.append("\nCategory Scores:")
                for cat, scores in eval_categories.items():
                    avg = sum(scores) / len(scores) if scores else 0
                    lines.append(f"  {cat.title()}: {avg:.0f}/100")

            # Per-metric breakdown
            lines.append("\nMetric Details:")
            for m in metrics:
                if not m.get("enabled"):
                    continue
                score = m.get("score", 0)
                label = m.get("label", m.get("metricId", "?"))
                weight = m.get("weight", 0)
                indicator = "✓" if score >= 80 else "⚠" if score >= 50 else "✗"
                lines.append(
                    f"  {indicator} {label}: {score}/100 (weight: {weight:.2f})"
                )

            # Top issues
            top_issues = report.get("topIssues", [])
            if top_issues:
                lines.append(f"\nTop Issues ({len(top_issues)}):")
                for issue in top_issues[:10]:
                    severity = issue.get("severity", "info").upper()
                    message = issue.get("message", "?")
                    lines.append(f"  [{severity}] {message}")
                    rec = issue.get("recommendation")
                    if rec:
                        lines.append(f"    → {rec}")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_design_diff":
            save_baseline = arguments.get("save_baseline", False)
            label = arguments.get("label")
            element_ids = arguments.get("element_ids")

            if save_baseline:
                response = await ui_client.sdk_design_save_baseline(
                    label=label, element_ids=element_ids
                )
                if not response.success:
                    return [
                        types.TextContent(
                            type="text", text=f"Save baseline error: {response.error}"
                        )
                    ]
                data = response.data or {}
                return [
                    types.TextContent(
                        type="text",
                        text=f"Baseline saved: {data.get('elementCount', '?')} elements"
                        + (f" (label: {label})" if label else ""),
                    )
                ]
            else:
                response = await ui_client.sdk_design_diff_baseline(
                    element_ids=element_ids
                )
                if not response.success:
                    return [
                        types.TextContent(
                            type="text", text=f"Diff baseline error: {response.error}"
                        )
                    ]

                diff_report = response.data or {}
                added = diff_report.get("added", [])
                removed = diff_report.get("removed", [])
                modified = diff_report.get("modified", [])
                cls = diff_report.get("cumulativeLayoutShift", 0)
                significant = diff_report.get("hasSignificantChanges", False)

                lines = ["=== Snapshot Diff ==="]
                lines.append(
                    f"Changes: {len(added)} added, {len(removed)} removed, {len(modified)} modified"
                )
                lines.append(f"Cumulative Layout Shift: {cls}")
                lines.append(f"Significant Changes: {'Yes' if significant else 'No'}")

                if added:
                    lines.append(f"\nAdded ({len(added)}):")
                    for d in added[:10]:
                        lines.append(f"  + {d.get('elementId', '?')}")
                    if len(added) > 10:
                        lines.append(f"  ... and {len(added) - 10} more")

                if removed:
                    lines.append(f"\nRemoved ({len(removed)}):")
                    for d in removed[:10]:
                        lines.append(f"  - {d.get('elementId', '?')}")
                    if len(removed) > 10:
                        lines.append(f"  ... and {len(removed) - 10} more")

                if modified:
                    lines.append(f"\nModified ({len(modified)}):")
                    for d in modified[:15]:
                        eid = d.get("elementId", "?")
                        style_changes = d.get("styleChanges", [])
                        layout_shift = d.get("layoutShift")
                        parts = []
                        if style_changes:
                            parts.append(f"{len(style_changes)} style changes")
                        if layout_shift:
                            parts.append(
                                f"layout: dx={layout_shift.get('dx', 0):.0f} "
                                f"dy={layout_shift.get('dy', 0):.0f}"
                            )
                        lines.append(
                            f"  ~ {eid}: {', '.join(parts) if parts else 'modified'}"
                        )
                    if len(modified) > 15:
                        lines.append(f"  ... and {len(modified) - 15} more")

                return [types.TextContent(type="text", text="\n".join(lines))]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"Error calling tool {name}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


# =============================================================================
# Agent Mode Helper Functions
# =============================================================================


def _format_diff(diff: dict[str, Any], rm: RefManager) -> str:
    """Format a snapshot diff for display."""
    appeared = diff.get("appeared", [])
    disappeared = diff.get("disappeared", [])
    modified = diff.get("modified", [])

    if not appeared and not disappeared and not modified:
        return "No changes detected."

    lines = ["UI Diff:"]

    if appeared:
        refs = []
        for eid in appeared:
            ref = rm._id_to_ref.get(eid)
            refs.append(f"{ref} ({eid})" if ref else eid)
        lines.append(f"Appeared ({len(appeared)}): {', '.join(refs)}")

    if disappeared:
        refs = []
        for eid in disappeared:
            ref = rm._id_to_ref.get(eid)
            refs.append(f"{ref} ({eid})" if ref else eid)
        lines.append(f"Disappeared ({len(disappeared)}): {', '.join(refs)}")

    if modified:
        lines.append(f"Modified ({len(modified)}):")
        for m in modified:
            eid = m["id"]
            ref = rm._id_to_ref.get(eid)
            label = f"  {ref} ({eid})" if ref else f"  {eid}"
            changes = m["changes"]
            change_parts = []
            for prop, vals in changes.items():
                from_val = repr(vals["from"])
                to_val = repr(vals["to"])
                change_parts.append(f"{prop} {from_val} -> {to_val}")
            lines.append(f"{label}: {', '.join(change_parts)}")

    return "\n".join(lines)


def _annotate_screenshot(
    screenshot_b64: str,
    elements: list[dict[str, Any]],
    width: int,
    height: int,
    rm: RefManager,
) -> str:
    """Annotate a screenshot with element ref labels. Returns base64 PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed. Returning unannotated screenshot.")
        return screenshot_b64

    img = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
    draw = ImageDraw.Draw(img)

    # Account for DPI scaling: screenshot is physical pixels, rects are CSS pixels
    scale_x = img.width / width if width else 1
    scale_y = img.height / height if height else 1

    # Try to load a small font; fall back to default
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for el in elements:
        state = el.get("state", {})
        rect = state.get("rect", {})
        if not rect or not state.get("visible", True):
            continue

        elem_id = el.get("id", "?")
        ref = rm.assign(elem_id)

        x = rect.get("x", 0) * scale_x
        y = rect.get("y", 0) * scale_y
        w = rect.get("width", 0) * scale_x
        h = rect.get("height", 0) * scale_y

        # Draw rectangle outline
        draw.rectangle((x, y, x + w, y + h), outline="red", width=2)

        # Draw ref label background + text
        text_bbox = draw.textbbox((0, 0), ref, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        label_y = max(y - th - 4, 0)
        draw.rectangle((x, label_y, x + tw + 4, label_y + th + 2), fill="red")
        draw.text((x + 2, label_y), ref, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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
