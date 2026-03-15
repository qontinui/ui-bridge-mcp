"""MCP Server for UI Bridge - enables AI to inspect and interact with UI elements.

This server provides tools for:
- Inspecting UI element positions, bounds, and state
- Interacting with elements (click, type, focus)
- Working with both the runner's own UI (Control mode) and SDK-integrated apps (SDK mode)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .client import UIBridgeClient
from .screenshot import (
    AnnotationOptions,
    BaselineStore,
    DeltaEncoder,
    annotate_screenshot,
    create_before_after,
    crop_to_element,
    diff_screenshots,
    generate_visual_description,
    mime_type_for_format,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MCP Server instance
server = Server("ui-bridge-mcp")
client: UIBridgeClient | None = None
baseline_store = BaselineStore()
delta_encoder = DeltaEncoder()


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


def format_action_error_info(response_data: dict[str, Any] | None) -> str:
    """Extract errorDiff and errorImpact from an action response and format for AI.

    Returns an empty string if no error information is present, or a formatted
    summary of new/resolved errors and UI consequences.
    """
    if not response_data:
        return ""

    parts: list[str] = []

    # Error diff: new vs resolved errors
    error_diff = response_data.get("errorDiff")
    if error_diff:
        new_errors = error_diff.get("newErrors", [])
        resolved = error_diff.get("resolvedErrors", [])
        delta = error_diff.get("errorDelta", 0)

        if new_errors or resolved:
            diff_lines: list[str] = []
            if new_errors:
                diff_lines.append(f"  NEW ({len(new_errors)}):")
                for err in new_errors[:5]:  # Cap at 5 to avoid flooding
                    severity = err.get("severity", "unknown")
                    reason = err.get("reason", "")
                    event = err.get("event", {})
                    msg = event.get(
                        "message",
                        event.get("args", [""])[0]
                        if isinstance(event.get("args"), list)
                        else "",
                    )
                    if isinstance(msg, str) and len(msg) > 200:
                        msg = msg[:200] + "..."
                    diff_lines.append(f"    [{severity}] {msg}")
                    if reason:
                        diff_lines.append(f"           reason: {reason}")
                if len(new_errors) > 5:
                    diff_lines.append(f"    ... and {len(new_errors) - 5} more")
            if resolved:
                diff_lines.append(f"  RESOLVED ({len(resolved)}):")
                for err in resolved[:3]:
                    event = err.get("event", {})
                    msg = event.get(
                        "message",
                        event.get("args", [""])[0]
                        if isinstance(event.get("args"), list)
                        else "",
                    )
                    if isinstance(msg, str) and len(msg) > 200:
                        msg = msg[:200] + "..."
                    diff_lines.append(f"    {msg}")
            if delta != 0:
                diff_lines.append(f"  errorDelta: {'+' if delta > 0 else ''}{delta}")
            parts.append("Error Diff:\n" + "\n".join(diff_lines))

    # Error impact: UI consequences assessment
    error_impact = response_data.get("errorImpact")
    if error_impact:
        impact_lines: list[str] = []
        error_info = error_impact.get("error", {})
        if error_info:
            impact_lines.append(
                f"  error: [{error_info.get('severity', '?')}] {error_info.get('message', '?')}"
            )
        recovery = error_impact.get("recoveryStatus")
        if recovery:
            impact_lines.append(f"  recoveryStatus: {recovery}")
        consequences = error_impact.get("uiConsequences", {})
        if consequences:
            if consequences.get("renderBlocked"):
                impact_lines.append("  RENDER BLOCKED (component tree crashed)")
            if consequences.get("errorBoundaryTriggered"):
                impact_lines.append("  Error boundary triggered")
            removed = consequences.get("elementsRemoved", [])
            if removed:
                impact_lines.append(f"  elementsRemoved: {removed[:10]}")
            added = consequences.get("elementsAdded", [])
            if added:
                impact_lines.append(f"  elementsAdded: {added[:10]}")
            disabled = consequences.get("elementsDisabled", [])
            if disabled:
                impact_lines.append(f"  elementsDisabled: {disabled[:10]}")
            nav = consequences.get("navigationTriggered")
            if nav:
                impact_lines.append(f"  navigationTriggered: {nav}")
        if impact_lines:
            parts.append("Error Impact:\n" + "\n".join(impact_lines))

    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


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
            f"[{rect.get('x', 0):.0f},{rect.get('y', 0):.0f} "
            f"{rect.get('width', 0):.0f}x{rect.get('height', 0):.0f}]"
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


def _format_undo_state(data: dict[str, Any] | None) -> str:
    """Format UndoRedoState data into LLM-readable text."""
    if not data:
        return "Undo/redo state not available."
    lines: list[str] = ["Undo/Redo State:"]
    undo_available = data.get("undoAvailable", False)
    redo_available = data.get("redoAvailable", False)
    if undo_available:
        desc = data.get("undoDescription")
        depth = data.get("undoDepth")
        line = "  Undo: available"
        if desc:
            line += f" — {desc}"
        if depth is not None:
            line += f" ({depth} steps)"
        lines.append(line)
        undo_stack = data.get("undoStack", [])
        if undo_stack:
            lines.append("  Undo stack (most recent first):")
            for entry in undo_stack[:10]:
                entry_desc = entry.get("description", "?")
                confidence = entry.get("confidence", 0)
                source = entry.get("source", "?")
                lines.append(
                    f"    - {entry_desc} (confidence: {confidence:.1f}, source: {source})"
                )
    else:
        lines.append("  Undo: not available")
    if redo_available:
        desc = data.get("redoDescription")
        depth = data.get("redoDepth")
        line = "  Redo: available"
        if desc:
            line += f" — {desc}"
        if depth is not None:
            line += f" ({depth} steps)"
        lines.append(line)
    else:
        lines.append("  Redo: not available")
    shortcut = data.get("undoShortcut")
    if shortcut:
        lines.append(f"  Undo shortcut: {shortcut}")
    redo_shortcut = data.get("redoShortcut")
    if redo_shortcut:
        lines.append(f"  Redo shortcut: {redo_shortcut}")
    sources = data.get("detectionSources", [])
    if sources:
        lines.append(f"  Detection: {', '.join(sources)}")
    return "\n".join(lines)


def _format_forms_response(data: dict[str, Any] | None) -> str:
    """Format FormsResponse data into LLM-readable text."""
    if not data:
        return "No form data available."

    forms = data.get("forms", [])
    summary = data.get("summary", "")

    if not forms:
        return "No forms detected on the page."

    lines = [f"Forms: {summary}", ""]

    for form in forms:
        form_id = form.get("id", "unknown")
        purpose = form.get("purpose", "")
        is_valid = form.get("isValid", True)
        is_dirty = form.get("isDirty", False)
        submit_btn = form.get("submitButton")

        status_parts = []
        if not is_valid:
            status_parts.append("INVALID")
        if is_dirty:
            status_parts.append("dirty")
        status = f" [{', '.join(status_parts)}]" if status_parts else ""

        header = f"Form: {form_id}"
        if purpose:
            header += f" ({purpose})"
        header += status
        lines.append(header)

        fields = form.get("fields", [])
        for field in fields:
            fid = field.get("id", "?")
            label = field.get("label", fid)
            ftype = field.get("type", "text")
            value = field.get("value", "")
            required = field.get("required", False)
            valid = field.get("valid", True)
            error = field.get("error")
            checked = field.get("checked")
            dirty = field.get("isDirty", False)
            placeholder = field.get("placeholder")

            # Build field line
            parts = [f"  {label} ({ftype})"]

            # Value display
            if checked is not None:
                parts.append(f"checked={checked}")
            elif value:
                display_val = value if len(value) <= 60 else value[:57] + "..."
                parts.append(f'"{display_val}"')
            elif placeholder:
                parts.append(f"[placeholder: {placeholder}]")
            else:
                parts.append("[empty]")

            # Flags
            flags = []
            if required:
                flags.append("required")
            if dirty:
                flags.append("dirty")
            if not valid:
                flags.append("INVALID")
            if flags:
                parts.append(f"[{', '.join(flags)}]")

            # Error message
            if error:
                parts.append(f"ERROR: {error}")

            lines.append(" | ".join(parts))

        if submit_btn:
            lines.append(f"  Submit: {submit_btn}")
        lines.append("")

    return "\n".join(lines)


def _format_fill_form_response(data: dict[str, Any] | None) -> str:
    """Format fill form response into LLM-readable text."""
    if not data:
        return "No fill result data available."

    results = data.get("results", {})
    errors = data.get("errors", {})
    total = len(results)
    succeeded = sum(1 for v in results.values() if v)
    failed = total - succeeded

    lines = [f"Fill result: {succeeded}/{total} fields set successfully"]

    if failed > 0:
        lines.append("")
        lines.append("Failed fields:")
        for field_id, success in results.items():
            if not success:
                error_msg = errors.get(field_id, "unknown error")
                lines.append(f"  {field_id}: {error_msg}")

    validation_errors = data.get("validationErrors", {})
    if validation_errors:
        lines.append("")
        lines.append("Validation errors:")
        for field_id, error in validation_errors.items():
            lines.append(f"  {field_id}: {error}")

    return "\n".join(lines)


def _format_form_diff_response(data: dict[str, Any] | None) -> str:
    """Format form diff response into LLM-readable text."""
    if not data:
        return "No diff data available."

    summary = data.get("summary", "")
    changed = data.get("changed", [])
    added = data.get("added", [])
    removed = data.get("removed", [])

    lines: list[str] = []

    if summary:
        lines.append(summary)
        lines.append("")

    if not changed and not added and not removed:
        lines.append("No form changes detected.")
        return "\n".join(lines)

    if changed:
        lines.append("Changed fields:")
        for field in changed:
            field_id = field.get("id", field.get("fieldId", "?"))
            label = field.get("label", field_id)
            before_val = field.get("before", "")
            after_val = field.get("after", "")
            lines.append(f'  {label}: "{before_val}" -> "{after_val}"')
        lines.append("")

    if added:
        lines.append("Added fields:")
        for field in added:
            if isinstance(field, dict):
                field_id = field.get("id", field.get("fieldId", "?"))
                label = field.get("label", field_id)
                lines.append(f"  {label}")
            else:
                lines.append(f"  {field}")
        lines.append("")

    if removed:
        lines.append("Removed fields:")
        for field in removed:
            if isinstance(field, dict):
                field_id = field.get("id", field.get("fieldId", "?"))
                label = field.get("label", field_id)
                lines.append(f"  {label}")
            else:
                lines.append(f"  {field}")
        lines.append("")

    return "\n".join(lines)


def _format_network_requests_response(data: dict[str, Any] | None) -> str:
    """Format network requests response into LLM-readable text."""
    if not data:
        return "No network request data available."

    requests = data.get("requests", [])
    if not requests:
        return "No network requests recorded."

    in_flight = [r for r in requests if r.get("status") == "in-flight"]
    completed = [r for r in requests if r.get("status") != "in-flight"]

    lines: list[str] = [f"Network Requests: {len(requests)} total"]
    if in_flight:
        lines.append(f"  In-flight: {len(in_flight)}")
    if completed:
        lines.append(f"  Completed: {len(completed)}")
    lines.append("")

    if in_flight:
        lines.append("IN-FLIGHT:")
        for req in in_flight:
            method = req.get("method", "?")
            url = req.get("url", "?")
            lines.append(f"  [IN-FLIGHT] {method} {url}")
            req_id = req.get("requestId") or req.get("id")
            if req_id:
                lines.append(f"    Request ID: {req_id}")
        lines.append("")

    if completed:
        lines.append("COMPLETED:")
        for req in completed:
            status = req.get("status", "?").upper()
            method = req.get("method", "?")
            url = req.get("url", "?")
            status_code = req.get("statusCode") or req.get("status_code")
            duration = req.get("durationMs") or req.get("duration_ms")

            status_parts = [f"[{status}]", method, url]
            if status_code is not None:
                status_parts.append(f"\u2192 {status_code}")
            if duration is not None:
                status_parts.append(f"({duration}ms)")
            lines.append(f"  {' '.join(status_parts)}")

            req_id = req.get("requestId") or req.get("id")
            if req_id:
                lines.append(f"    Request ID: {req_id}")
            error = req.get("error")
            if error:
                lines.append(f"    Error: {error}")

    return "\n".join(lines)


def _format_wait_result(data: dict[str, Any] | None) -> str:
    """Format wait-for-network-request result into LLM-readable text."""
    if not data:
        return "No wait result data available."

    timed_out = data.get("timedOut", data.get("timed_out", False))
    matched = data.get("request") or data.get("matched")

    if timed_out:
        lines = ["Wait timed out - no matching request completed in time."]
        if matched:
            method = matched.get("method", "?")
            url = matched.get("url", "?")
            lines.append(f"  Closest match: {method} {url}")
        return "\n".join(lines)

    if not matched:
        return "Wait completed but no request details available."

    status = matched.get("status", "?").upper()
    method = matched.get("method", "?")
    url = matched.get("url", "?")
    status_code = matched.get("statusCode") or matched.get("status_code")
    duration = matched.get("durationMs") or matched.get("duration_ms")

    parts = [f"[{status}]", method, url]
    if status_code is not None:
        parts.append(f"\u2192 {status_code}")
    if duration is not None:
        parts.append(f"({duration}ms)")

    lines = [f"Matched request: {' '.join(parts)}"]

    req_id = matched.get("requestId") or matched.get("id")
    if req_id:
        lines.append(f"  Request ID: {req_id}")
    error = matched.get("error")
    if error:
        lines.append(f"  Error: {error}")

    return "\n".join(lines)


def format_page_header(page: dict[str, Any]) -> list[str]:
    """Format page/route context as a concise header block."""
    lines: list[str] = []
    title = page.get("title", "")
    url = page.get("url", "")
    if title and url:
        lines.append(f'Page: "{title}" ({url})')
    elif title:
        lines.append(f'Page: "{title}"')
    elif url:
        lines.append(f"Page: {url}")

    route = page.get("route")
    if isinstance(route, dict):
        pattern = route.get("pattern", "")
        params = route.get("params", {})
        if pattern:
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"Route: {pattern} ({param_str})")
            else:
                lines.append(f"Route: {pattern}")
        elif page.get("pathname"):
            lines.append(f"Path: {page['pathname']}")
    elif page.get("pathname"):
        lines.append(f"Path: {page['pathname']}")

    page_context = page.get("pageContext")
    if isinstance(page_context, dict):
        name = page_context.get("name", "")
        section = page_context.get("section", "")
        parts = []
        if name:
            parts.append(f"Name: {name}")
        if section:
            parts.append(f"Section: {section}")
        if parts:
            lines.append(", ".join(parts))

    return lines


def format_modal_header(modal_stack: dict[str, Any]) -> list[str]:
    """Format modal/dialog stack as a concise header block."""
    modals = modal_stack.get("modals", [])
    if not modals:
        return []
    lines: list[str] = []
    count = modal_stack.get("count", len(modals))
    blocking = modal_stack.get("hasBlockingModal", False)
    status = "BLOCKING" if blocking else "non-blocking"
    lines.append(f"Modals: {count} active ({status})")
    top = modal_stack.get("topModal")
    if isinstance(top, dict):
        title = top.get("title", top.get("ariaLabel", ""))
        modal_type = top.get("type", "dialog")
        parts = [f"  Top: [{modal_type}]"]
        if title:
            parts.append(f'"{title}"')
        if top.get("escDismiss"):
            parts.append("(ESC to dismiss)")
        if top.get("primaryAction"):
            parts.append(f'action="{top["primaryAction"]}"')
        lines.append(" ".join(parts))
    return lines


def format_toast_header(toasts: dict[str, Any]) -> list[str]:
    """Format toast/notification snapshot as a concise header block."""
    active = toasts.get("active", [])
    recent = toasts.get("recent", [])
    if not active and not recent:
        return []
    lines: list[str] = []
    if active:
        lines.append(f"Toasts: {len(active)} active")
        for t in active[:5]:  # Show at most 5
            level = t.get("level", "unknown")
            msg = t.get("message", "")
            if len(msg) > 80:
                msg = msg[:77] + "..."
            lines.append(f"  [{level}] {msg}")
    if recent:
        lines.append(f"Recent toasts: {len(recent)} dismissed")
        for t in recent[:3]:  # Show at most 3 recent
            level = t.get("level", "unknown")
            msg = t.get("message", "")
            if len(msg) > 80:
                msg = msg[:77] + "..."
            lines.append(f"  [{level}] {msg} (dismissed)")
    return lines


def format_relationship_header(relationships: dict[str, Any]) -> list[str]:
    """Format element relationships as a concise header block."""
    rels = relationships.get("relationships", [])
    count = relationships.get("count", len(rels))
    if not rels:
        return []
    lines: list[str] = []
    by_origin = relationships.get("byOrigin", {})
    origin_parts = []
    for origin in ("declared", "aria", "html"):
        n = by_origin.get(origin, 0)
        if n:
            origin_parts.append(f"{n} {origin}")
    origin_str = ", ".join(origin_parts) if origin_parts else str(count)
    lines.append(f"Relationships: {count} ({origin_str})")
    for r in rels[:10]:  # Show at most 10
        source = r.get("sourceId", "?")
        target = r.get("targetId", "?")
        rel_type = r.get("type", "?")
        origin = r.get("origin", "")
        origin_tag = f" [{origin}]" if origin else ""
        lines.append(f"  {source} --{rel_type}--> {target}{origin_tag}")
    if count > 10:
        lines.append(f"  +{count - 10} more relationships")
    return lines


def format_drag_drop_header(drag_drop: dict[str, Any]) -> list[str]:
    """Format drag source & drop zone discovery as a concise header block."""
    sources = drag_drop.get("dragSources", [])
    zones = drag_drop.get("dropZones", [])
    counts = drag_drop.get("count", {})
    source_count = counts.get("dragSources", len(sources))
    zone_count = counts.get("dropZones", len(zones))
    if not sources and not zones:
        return []
    lines: list[str] = []
    by_origin = drag_drop.get("byOrigin", {})
    origin_parts = []
    for origin in ("declared", "aria", "dom"):
        n = by_origin.get(origin, 0)
        if n:
            origin_parts.append(f"{n} {origin}")
    origin_str = f" ({', '.join(origin_parts)})" if origin_parts else ""
    lines.append(f"Drag-Drop: {source_count} sources, {zone_count} zones{origin_str}")
    if sources:
        for s in sources[:8]:
            sid = s.get("id", "?")
            label = s.get("label", "")
            data_type = s.get("dataType", "")
            parts = [f"  drag: {sid}"]
            if data_type:
                parts.append(f"[{data_type}]")
            if label and label != sid:
                label_str = label if len(label) <= 40 else label[:37] + "..."
                parts.append(f'"{label_str}"')
            origin = s.get("origin", "")
            if origin:
                parts.append(f"({origin})")
            lines.append(" ".join(parts))
        if source_count > 8:
            lines.append(f"  +{source_count - 8} more drag sources")
    if zones:
        for z in zones[:8]:
            zid = z.get("id", "?")
            label = z.get("label", "")
            effect = z.get("effect", "")
            is_sortable = z.get("isSortable", False)
            contained = z.get("containedDragSources", [])
            parts = [f"  zone: {zid}"]
            if is_sortable:
                parts.append("[sortable]")
            if effect:
                parts.append(f"effect={effect}")
            accepts = z.get("accepts")
            if isinstance(accepts, list) and accepts:
                parts.append(f"accepts=[{', '.join(accepts)}]")
            if contained:
                parts.append(f"items={len(contained)}")
            if label and label != zid:
                label_str = label if len(label) <= 40 else label[:37] + "..."
                parts.append(f'"{label_str}"')
            origin = z.get("origin", "")
            if origin:
                parts.append(f"({origin})")
            lines.append(" ".join(parts))
        if zone_count > 8:
            lines.append(f"  +{zone_count - 8} more drop zones")
    return lines


def format_undo_redo_header(undo_redo: dict[str, Any]) -> list[str]:
    """Format undo/redo awareness as a concise header block."""
    undo_available = undo_redo.get("undoAvailable", False)
    redo_available = undo_redo.get("redoAvailable", False)
    if not undo_available and not redo_available:
        return []
    lines: list[str] = []
    parts: list[str] = []
    if undo_available:
        undo_desc = undo_redo.get("undoDescription")
        depth = undo_redo.get("undoDepth")
        undo_str = "Undo: available"
        if undo_desc:
            undo_str += f" ({undo_desc})"
        if depth is not None:
            undo_str += f" [{depth} steps]"
        parts.append(undo_str)
    if redo_available:
        redo_desc = undo_redo.get("redoDescription")
        depth = undo_redo.get("redoDepth")
        redo_str = "Redo: available"
        if redo_desc:
            redo_str += f" ({redo_desc})"
        if depth is not None:
            redo_str += f" [{depth} steps]"
        parts.append(redo_str)
    lines.append(" | ".join(parts))
    return lines


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
                    "description": "The element's registered ID or agent ref (e.g., '@e1', 'sidebar-nav-item-settings')",
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
                    "description": "The element's registered ID or agent ref (@e1)",
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
                    "description": "The element's registered ID or agent ref (@e1)",
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
                    "description": "The element's registered ID to focus",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID to hover over",
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
                    "description": "The element's registered ID to double-click",
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
                    "description": "The element's registered ID to right-click",
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
                    "description": "The element's registered ID to clear",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID to scroll",
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
                    "description": "The checkbox element's registered ID",
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
                    "description": "The checkbox element's registered ID",
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
                    "description": "The checkbox element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The source element's registered ID to drag",
                },
                "target_element_id": {
                    "type": "string",
                    "description": "The target element's registered ID to drop on",
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
                    "description": "The element's registered ID (element or its parent form)",
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
                    "description": "The element's registered ID (element or its parent form)",
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
        name="ui_clipboard_read",
        description="Read the current system clipboard text content (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ui_clipboard_write",
        description="Write text to the system clipboard (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to write to the clipboard.",
                },
            },
            "required": ["text"],
        },
    ),
    types.Tool(
        name="sdk_clipboard_read",
        description="Read the current system clipboard text content (SDK mode).",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_clipboard_write",
        description="Write text to the system clipboard (SDK mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to write to the clipboard.",
                },
            },
            "required": ["text"],
        },
    ),
    types.Tool(
        name="sdk_forms",
        description="""Get form state from the connected SDK app.

Returns structured data about all forms on the page:
- Form fields with current values, labels, and types
- Validation errors (detected via HTML5 API, ARIA, CSS heuristics)
- Required fields and constraint attributes (pattern, min/max, length)
- Dirty state (whether fields have been modified)
- Submit button identification
- Form purpose inference (Login, Registration, Search, etc.)

Use this to understand form state before filling, to verify field values
after typing, and to detect validation errors after submission.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ui_forms",
        description="""Get form state from the runner's own UI (Control mode).

Same as sdk_forms but for the runner's React frontend.
Returns forms, fields, values, validation errors, and dirty state.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_fill_form",
        description="""Fill multiple form fields atomically in the connected SDK app.

Accepts a map of field IDs to values and sets each field with proper event
dispatching (change, input events). Use sdk_forms first to discover field IDs.

Returns per-field success/failure results with any validation errors.""",
        inputSchema={
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "description": (
                        "Map of element ID to value. Values can be strings, "
                        "booleans (for checkboxes), or string arrays (for multi-select)."
                    ),
                },
                "triggerValidation": {
                    "type": "boolean",
                    "description": (
                        "Whether to trigger validation after filling. Defaults to true."
                    ),
                    "default": True,
                },
                "clearFirst": {
                    "type": "boolean",
                    "description": (
                        "Whether to clear existing values before filling. "
                        "Defaults to true."
                    ),
                    "default": True,
                },
            },
            "required": ["fields"],
        },
    ),
    types.Tool(
        name="ui_fill_form",
        description="""Fill multiple form fields atomically in the runner's own UI (Control mode).

Same as sdk_fill_form but for the runner's React frontend.
Accepts a map of field IDs to values and sets each field with proper event dispatching.
Returns per-field success/failure results.""",
        inputSchema={
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "description": (
                        "Map of element ID to value. Values can be strings, "
                        "booleans (for checkboxes), or string arrays (for multi-select)."
                    ),
                },
                "triggerValidation": {
                    "type": "boolean",
                    "description": (
                        "Whether to trigger validation after filling. Defaults to true."
                    ),
                    "default": True,
                },
                "clearFirst": {
                    "type": "boolean",
                    "description": (
                        "Whether to clear existing values before filling. "
                        "Defaults to true."
                    ),
                    "default": True,
                },
            },
            "required": ["fields"],
        },
    ),
    types.Tool(
        name="sdk_form_snapshot",
        description="""Capture a snapshot of all form state in the connected SDK app.

Returns a FormSnapshot with all forms and their field states (values,
validation, dirty flags). Use before and after an action, then pass both
snapshots to sdk_form_diff to see what changed.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ui_form_snapshot",
        description="""Capture a snapshot of all form state in the runner's own UI (Control mode).

Same as sdk_form_snapshot but for the runner's React frontend.
Returns a FormSnapshot to use with ui_form_diff.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_form_diff",
        description="""Compare two form snapshots to see what changed in the connected SDK app.

Pass the 'before' and 'after' snapshots from sdk_form_snapshot. Returns a
diff showing changed fields with before/after values, added/removed fields,
and a human-readable summary.""",
        inputSchema={
            "type": "object",
            "properties": {
                "before": {
                    "type": "object",
                    "description": "The before snapshot from sdk_form_snapshot.",
                },
                "after": {
                    "type": "object",
                    "description": "The after snapshot from sdk_form_snapshot.",
                },
            },
            "required": ["before", "after"],
        },
    ),
    types.Tool(
        name="ui_form_diff",
        description="""Compare two form snapshots to see what changed in the runner's own UI.

Same as sdk_form_diff but for the runner's React frontend.
Pass the 'before' and 'after' snapshots from ui_form_snapshot.""",
        inputSchema={
            "type": "object",
            "properties": {
                "before": {
                    "type": "object",
                    "description": "The before snapshot from ui_form_snapshot.",
                },
                "after": {
                    "type": "object",
                    "description": "The after snapshot from ui_form_snapshot.",
                },
            },
            "required": ["before", "after"],
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
                    "description": "The element's registered ID or agent ref (e.g., '@e1')",
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
        description="""Click an element in the SDK app by its registered ID.

Use sdk_snapshot or sdk_elements first to find the element_id.
Accepts refs like @e1 from agent_mode snapshots.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element's registered ID or agent ref (@e1)",
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
                    "description": "The element's registered ID or agent ref (@e1)",
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
                    "description": "The element's registered ID to clear",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The checkbox element's registered ID",
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
                    "description": "The checkbox element's registered ID",
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
                    "description": "The checkbox element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The source element's registered ID",
                },
                "target_element_id": {
                    "type": "string",
                    "description": "The target element's registered ID",
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
                    "description": "The element's registered ID",
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
                    "description": "The element's registered ID",
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
        name="sdk_find",
        description="""Find an element by natural language description with spatial and relational context.

More powerful than sdk_ai_search — handles spatial references, container scoping,
ordinals, state filters, and auto-detects active modals.

Examples:
- "close button near Terminal 1 tab"
- "email input in the login form"
- "the third row in the table"
- "disabled save button"
- "save button" (when a modal is open, auto-prefers the modal's save button)

Returns: element ID, confidence, match reasons, and alternatives for disambiguation.""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language element description. "
                        "Supports spatial refs ('near X', 'above Y'), "
                        "containers ('in the form'), ordinals ('third item'), "
                        "and state filters ('disabled button')."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "Optional context hint (e.g., 'in the dialog', 'sidebar')",
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence threshold 0-1 (default: 0.5)",
                },
            },
            "required": ["query"],
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
Supports annotation modes for different visualization styles:
- "interactive" (default): Only interactive elements (buttons, inputs, links)
- "all": All elements including content
- "validation": Color-code by validation state (red=error, green=valid, orange=required)
- "modal": Dim background, highlight only elements inside the topmost modal
- "state": Color-code by element state (blue=focused, gray=disabled, red=error, green=valid)
- "relationships": Draw connector lines between related elements (aria-controls, labels, etc.)
- "accessibility": Accessibility audit overlay (focus order, touch targets, missing labels)
- "boxmodel": DevTools-style box model overlay (margin/border/padding/content boxes)

Includes viewport indicators (scroll arrows, off-screen count, minimap) when viewport data available.

Supports smart cropping:
- "full" (default): Entire screenshot
- "viewport": Crop to visible viewport area
- "modal": Crop to modal bounds (with padding)""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based). Defaults to primary monitor.",
                },
                "mode": {
                    "type": "string",
                    "enum": [
                        "interactive",
                        "all",
                        "validation",
                        "modal",
                        "state",
                        "relationships",
                        "accessibility",
                        "boxmodel",
                    ],
                    "description": "Annotation mode. Default: interactive.",
                },
                "crop": {
                    "type": "string",
                    "enum": ["full", "viewport", "modal"],
                    "description": "Crop mode. Default: full.",
                },
                "scale": {
                    "type": "number",
                    "description": "Scale factor (0.25-2.0). Default: 1.0.",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Output format. Default: png.",
                },
                "quality": {
                    "type": "integer",
                    "description": "JPEG/WebP quality (1-100). Default: 85.",
                },
                "highlight_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only annotate these element IDs/refs. Overrides mode filtering.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_annotated_screenshot",
        description="""Capture a screenshot of the SDK app's monitor with element labels overlaid.

Each visible element gets a numbered overlay (@e1, @e2) matching agent mode refs.
Supports annotation modes for different visualization styles:
- "interactive" (default): Only interactive elements (buttons, inputs, links)
- "all": All elements including content
- "validation": Color-code by validation state (red=error, green=valid, orange=required)
- "modal": Dim background, highlight only elements inside the topmost modal
- "state": Color-code by element state (blue=focused, gray=disabled, red=error, green=valid)
- "relationships": Draw connector lines between related elements (aria-controls, labels, etc.)
- "accessibility": Accessibility audit overlay (focus order, touch targets, missing labels)
- "boxmodel": DevTools-style box model overlay (margin/border/padding/content boxes)

Includes viewport indicators (scroll arrows, off-screen count, minimap) when viewport data available.

Supports smart cropping:
- "full" (default): Entire screenshot
- "viewport": Crop to visible viewport area
- "modal": Crop to modal bounds (with padding)""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based). Defaults to primary monitor.",
                },
                "mode": {
                    "type": "string",
                    "enum": [
                        "interactive",
                        "all",
                        "validation",
                        "modal",
                        "state",
                        "relationships",
                        "accessibility",
                        "boxmodel",
                    ],
                    "description": "Annotation mode. Default: interactive.",
                },
                "crop": {
                    "type": "string",
                    "enum": ["full", "viewport", "modal"],
                    "description": "Crop mode. Default: full.",
                },
                "scale": {
                    "type": "number",
                    "description": "Scale factor (0.25-2.0). Default: 1.0.",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Output format. Default: png.",
                },
                "quality": {
                    "type": "integer",
                    "description": "JPEG/WebP quality (1-100). Default: 85.",
                },
                "highlight_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only annotate these element IDs/refs. Overrides mode filtering.",
                },
            },
            "required": [],
        },
    ),
    # Element Screenshot Tools
    types.Tool(
        name="ui_element_screenshot",
        description="""Capture a cropped screenshot of a specific element in the runner's UI.

Returns an image cropped to the element's bounding rectangle with configurable padding.
Useful for inspecting individual components without full-page noise.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID or @ref to capture.",
                },
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based). Defaults to primary.",
                },
                "padding": {
                    "type": "integer",
                    "description": "Pixels of padding around element. Default: 16.",
                },
                "scale": {
                    "type": "number",
                    "description": "Scale factor. Default: 1.0.",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Output format. Default: png.",
                },
            },
            "required": ["element_id"],
        },
    ),
    types.Tool(
        name="sdk_element_screenshot",
        description="""Capture a cropped screenshot of a specific element in the SDK app.

Returns an image cropped to the element's bounding rectangle with configurable padding.
Useful for inspecting individual components without full-page noise.""",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID or @ref to capture.",
                },
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0-based). Defaults to primary.",
                },
                "padding": {
                    "type": "integer",
                    "description": "Pixels of padding around element. Default: 16.",
                },
                "scale": {
                    "type": "number",
                    "description": "Scale factor. Default: 1.0.",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Output format. Default: png.",
                },
            },
            "required": ["element_id"],
        },
    ),
    # Screenshot Diffing Tools
    types.Tool(
        name="sdk_screenshot_diff",
        description="""Compare the current SDK app screenshot against a saved baseline.

Returns a diff image highlighting changed regions in red, plus change percentage
and pass/fail result. Save a baseline first with sdk_screenshot_baseline_save.""",
        inputSchema={
            "type": "object",
            "properties": {
                "baseline_key": {
                    "type": "string",
                    "description": "Baseline key to compare against (typically a route/page name).",
                },
                "threshold": {
                    "type": "number",
                    "description": "Max allowed change percentage (0.0-1.0). Default: 0.01 (1%).",
                },
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index. Defaults to primary.",
                },
            },
            "required": ["baseline_key"],
        },
    ),
    types.Tool(
        name="sdk_screenshot_baseline_save",
        description="""Save the current SDK app screenshot as a baseline for future comparisons.

Use a descriptive key like the route name (e.g., 'dashboard', 'settings-page').
The baseline persists for the duration of this MCP session.""",
        inputSchema={
            "type": "object",
            "properties": {
                "baseline_key": {
                    "type": "string",
                    "description": "Key to save the baseline under (e.g., route name).",
                },
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index. Defaults to primary.",
                },
            },
            "required": ["baseline_key"],
        },
    ),
    types.Tool(
        name="sdk_screenshot_baseline_list",
        description="""List all saved screenshot baselines.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    # Before/After Comparison Tools
    types.Tool(
        name="sdk_screenshot_before",
        description="""Capture the current SDK app screenshot as the 'before' state.

Call this before performing an action, then use sdk_screenshot_after to
capture the result and get a side-by-side comparison.""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index. Defaults to primary.",
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_screenshot_after",
        description="""Capture the current state and compare with the saved 'before' screenshot.

Returns a side-by-side comparison image showing before and after states,
plus a diff overlay highlighting changes. Must call sdk_screenshot_before first.""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index. Defaults to primary.",
                },
                "layout": {
                    "type": "string",
                    "enum": ["side-by-side", "vertical"],
                    "description": "Comparison layout. Default: side-by-side.",
                },
            },
            "required": [],
        },
    ),
    # Visual Description Tool
    types.Tool(
        name="sdk_visual_description",
        description="""Get a structured text description of the SDK app's visual layout.

Returns a verbal summary: page info, viewport size, layout regions (header/sidebar/main/footer),
element type breakdown, interactive vs content count, modal/toast/error status, scroll position.
Useful as a lightweight alternative to screenshots for understanding page layout.""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index. Defaults to primary.",
                },
            },
            "required": [],
        },
    ),
    # Delta/Incremental Screenshot Tool
    types.Tool(
        name="sdk_screenshot_delta",
        description="""Get an incremental screenshot of the SDK app, sending only changed regions.

On first call, returns the full image. On subsequent calls, returns only the tile
patches that changed since the last call. Dramatically reduces bandwidth for agents
taking many sequential screenshots (e.g., monitoring for visual changes).

Returns: text summary of changes + changed tile data as JSON.""",
        inputSchema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index. Defaults to primary.",
                },
                "reset": {
                    "type": "boolean",
                    "description": "Reset delta state, forcing a full capture. Default: false.",
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
    # =========================================================================
    # Idle Detection Tools
    # =========================================================================
    types.Tool(
        name="get_idle_status",
        description="""Get the current idle status of the connected app.

Returns whether the app is idle (no pending network requests, DOM mutations,
or loading indicators) along with a per-signal breakdown showing each signal's
state and how long it has been stable.

Automatically detects the connection mode: if an SDK app is connected, queries
the SDK app's idle state; otherwise queries the runner's own UI (control mode).

Use this to check if the app has finished updating after an action (e.g.,
clicking a button, navigating, or submitting a form) before taking a snapshot
or making assertions. This is a non-blocking check — use wait_for_idle if you
need to block until the app settles.

Optionally pass a specific signal name to query just that signal.""",
        inputSchema={
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": ["network", "dom", "loading-indicators"],
                    "description": (
                        "If provided, returns status for only this signal. "
                        "Otherwise returns composite status with all signals."
                    ),
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="wait_for_idle",
        description="""Block until the app is idle (all activity signals are stable).

Waits for network requests to complete, DOM mutations to stop, and loading
indicators to disappear. Returns once ALL signals have been simultaneously
idle for min_stable_ms.

Automatically detects the connection mode: if an SDK app is connected, waits
for the SDK app to become idle; otherwise waits for the runner's own UI.

Use this after triggering an action (click, navigation, form submit) to
ensure the app has fully settled before inspecting the UI. This is the
recommended way to handle async UI updates — prefer this over arbitrary
delays.

You can exclude specific signals if they are not relevant (e.g., exclude
'network' if the app uses long-polling or WebSockets that never fully idle).

Returns the final idle status on success, or an error if the timeout is
reached before the app becomes idle.""",
        inputSchema={
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": (
                        "Max time to wait in milliseconds. "
                        "Defaults to 30000 (30 seconds)."
                    ),
                    "default": 30000,
                },
                "min_stable_ms": {
                    "type": "number",
                    "description": (
                        "How long all signals must remain idle before the app is "
                        "considered stable, in milliseconds. Defaults to 500."
                    ),
                    "default": 500,
                },
                "exclude": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["network", "dom", "loading-indicators"],
                    },
                    "description": (
                        "Signal names to ignore when determining idle state. "
                        "Useful when a signal never settles (e.g., long-polling)."
                    ),
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="wait_for_signal",
        description="""Block until a specific idle signal is stable.

Unlike wait_for_idle which waits for ALL signals, this waits for just one:
- 'network': All fetch/XHR requests have completed
- 'dom': No DOM mutations for min_stable_ms
- 'loading-indicators': No visible spinners, progress bars, or skeleton screens

Automatically detects the connection mode: if an SDK app is connected, waits
on the SDK app; otherwise waits on the runner's own UI.

Use this when you only care about a specific type of activity. For example,
wait for 'network' after an API call, or wait for 'dom' after a client-side
state update.

Returns the signal's status on success, or an error if the timeout is reached.""",
        inputSchema={
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": ["network", "dom", "loading-indicators"],
                    "description": "The signal to wait for.",
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Max time to wait in milliseconds. "
                        "Defaults to 30000 (30 seconds)."
                    ),
                    "default": 30000,
                },
                "min_stable_ms": {
                    "type": "number",
                    "description": (
                        "How long the signal must remain idle before considered "
                        "stable, in milliseconds. Defaults to 500."
                    ),
                    "default": 500,
                },
            },
            "required": ["signal"],
        },
    ),
    types.Tool(
        name="wait_for_targets",
        description="""Wait for specific targets (signals or CSS selectors) to become idle.

Accepts an array of targets, where each target is either:
- A signal name string: 'network', 'dom', or 'loading-indicators'
- An indicator object: {"indicator": ".my-spinner"} — waits for the CSS
  selector to become hidden/removed

Automatically detects the connection mode: if an SDK app is connected, waits
on the SDK app; otherwise waits on the runner's own UI.

This is useful when you know exactly what to wait for. For example, wait
for a specific loading spinner to disappear, or wait for both network and
a custom loading indicator simultaneously.

Returns when ALL specified targets are idle, or an error if the timeout is
reached.""",
        inputSchema={
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "string",
                                "enum": ["network", "dom", "loading-indicators"],
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "indicator": {
                                        "type": "string",
                                        "description": (
                                            "CSS selector of a loading indicator "
                                            "to watch (e.g., '.spinner', '#loading')."
                                        ),
                                    },
                                },
                                "required": ["indicator"],
                            },
                        ],
                    },
                    "description": (
                        "Array of targets to wait for. Each target is either a "
                        "signal name string or an object with an 'indicator' CSS selector."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Max time to wait in milliseconds. "
                        "Defaults to 30000 (30 seconds)."
                    ),
                    "default": 30000,
                },
                "min_stable_ms": {
                    "type": "number",
                    "description": (
                        "How long targets must remain idle before considered "
                        "stable, in milliseconds. Defaults to 500."
                    ),
                    "default": 500,
                },
            },
            "required": ["targets"],
        },
    ),
    types.Tool(
        name="diagnose_stuck_screen",
        description="""Diagnose whether the connected app is stuck on a loading screen.

Observes the app for a short window (default 3s) and checks:
- Are loading indicators (spinners, skeletons, progress bars) visible?
- Is the DOM changing (new content being rendered)?
- Are network requests completing or hanging?

Returns a verdict:
- 'stuck': Loading indicators are visible but nothing is changing — the app
  is frozen on a loading screen.
- 'loading': Loading indicators are visible and the DOM is actively changing —
  normal loading in progress.
- 'idle': No loading indicators detected — the app is in a resting state.
- 'unknown': Ambiguous signals.

Includes detailed evidence (what indicators were found, DOM mutation count,
pending network requests) and suggestions for recovery.

Automatically detects the connection mode: if an SDK app is connected,
diagnoses the SDK app; otherwise diagnoses the runner's own UI.

Use this when:
- The app seems to be taking too long to load
- You suspect the app may be stuck after an action
- You want to verify the app has fully loaded before proceeding""",
        inputSchema={
            "type": "object",
            "properties": {
                "observation_window_ms": {
                    "type": "number",
                    "description": (
                        "How long to observe in milliseconds. Longer = more "
                        "confident. Default: 3000 (3 seconds)."
                    ),
                    "default": 3000,
                },
                "dom_mutation_threshold": {
                    "type": "number",
                    "description": (
                        "Minimum DOM mutations to consider the page 'changing'. "
                        "Fewer than this = static. Default: 3."
                    ),
                    "default": 3,
                },
            },
            "required": [],
        },
    ),
    # -------------------------------------------------------------------------
    # Network Request Monitoring
    # -------------------------------------------------------------------------
    types.Tool(
        name="sdk_network_requests",
        description="""List recent network requests from the connected SDK app.

Shows API calls made by the app with their status, response codes, and timing.
Use this to debug API issues, verify that the correct endpoints are being called,
or check for failed requests after an action.

Supports filtering by status (in-flight, completed, failed, cancelled),
HTTP method, URL substring, and failures-only mode.""",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": (
                        "Filter by status: in-flight, completed, failed, cancelled."
                    ),
                },
                "method": {
                    "type": "string",
                    "description": ("Filter by HTTP method (GET, POST, etc.)."),
                },
                "url_pattern": {
                    "type": "string",
                    "description": "Filter by URL substring match.",
                },
                "failures_only": {
                    "type": "boolean",
                    "description": (
                        "Only show failed requests (4xx/5xx/network errors)."
                    ),
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results.",
                    "default": 50,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="ui_network_requests",
        description="""List recent network requests from the runner's own UI (Control mode).

Same as sdk_network_requests but for the runner's React frontend.
Shows API calls with status, response codes, and timing.""",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": (
                        "Filter by status: in-flight, completed, failed, cancelled."
                    ),
                },
                "method": {
                    "type": "string",
                    "description": ("Filter by HTTP method (GET, POST, etc.)."),
                },
                "url_pattern": {
                    "type": "string",
                    "description": "Filter by URL substring match.",
                },
                "failures_only": {
                    "type": "boolean",
                    "description": (
                        "Only show failed requests (4xx/5xx/network errors)."
                    ),
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results.",
                    "default": 50,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_network_requests_in_flight",
        description="""Show currently in-flight network requests in the connected SDK app.

Returns only requests that are currently pending (not yet completed).
Useful for seeing what API calls are still waiting for a response.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ui_network_requests_in_flight",
        description="""Show currently in-flight network requests in the runner's own UI (Control mode).

Same as sdk_network_requests_in_flight but for the runner's React frontend.
Returns only requests that are currently pending.""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="sdk_wait_for_network_request",
        description="""Wait for a network request matching the given criteria to complete in the SDK app.

Useful after clicking a button to wait for the resulting API call to finish.
You can match by URL substring and/or HTTP method. Returns the matched request
details once it completes, or an error if the timeout is reached.""",
        inputSchema={
            "type": "object",
            "properties": {
                "url_pattern": {
                    "type": "string",
                    "description": "URL substring to match.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to match.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds.",
                    "default": 30000,
                },
            },
            "required": [],
        },
    ),
    types.Tool(
        name="ui_wait_for_network_request",
        description="""Wait for a network request matching the given criteria to complete in the runner's own UI.

Same as sdk_wait_for_network_request but for the runner's React frontend (Control mode).
Returns the matched request details once it completes, or an error on timeout.""",
        inputSchema={
            "type": "object",
            "properties": {
                "url_pattern": {
                    "type": "string",
                    "description": "URL substring to match.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to match.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds.",
                    "default": 30000,
                },
            },
            "required": [],
        },
    ),
    # =========================================================================
    # Change Tracking - SDK Mode
    # =========================================================================
    types.Tool(
        name="sdk_save_bookmark",
        description="Save a snapshot bookmark for later diffing. Captures current UI state.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name (unique identifier)",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="sdk_list_bookmarks",
        description="List all saved snapshot bookmarks.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_delete_bookmark",
        description="Delete a saved bookmark.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name to delete",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="sdk_diff_from_bookmark",
        description="Compare current UI state against a saved bookmark. Returns appeared, disappeared, and modified elements.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name to diff against",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="sdk_execute_with_diff",
        description="Execute an element action and capture what changed in the UI. Returns before/after diff with categorization.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID (or @ref) to interact with",
                },
                "action": {
                    "type": "string",
                    "description": "Action: click, type, focus, etc.",
                },
                "value": {
                    "type": "string",
                    "description": "Value for type/setValue actions",
                },
                "settle_timeout": {
                    "type": "integer",
                    "description": "Max ms to wait for UI to settle (default: 3000)",
                },
                "categorize": {
                    "type": "boolean",
                    "description": "Whether to categorize the diff (default: true)",
                },
                "summary_budget": {
                    "type": "integer",
                    "description": "Max chars for budget summary (default: 300)",
                },
            },
            "required": ["element_id", "action"],
        },
    ),
    types.Tool(
        name="sdk_wait_for_change",
        description="Wait for the UI to change matching a predicate. Polls until a matching change is detected or timeout.",
        inputSchema={
            "type": "object",
            "properties": {
                "predicate_type": {
                    "type": "string",
                    "enum": [
                        "anyChange",
                        "elementAppeared",
                        "elementDisappeared",
                        "elementChanged",
                        "countChanged",
                    ],
                    "description": "Type of change to wait for",
                },
                "element_id": {
                    "type": "string",
                    "description": "Element ID for elementAppeared/Disappeared/Changed predicates",
                },
                "property": {
                    "type": "string",
                    "description": "Property to watch for elementChanged predicate",
                },
                "min_count": {
                    "type": "integer",
                    "description": "Minimum element count change for countChanged predicate",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max ms to wait (default: 5000)",
                },
                "poll_interval": {
                    "type": "integer",
                    "description": "Poll interval in ms (default: 200)",
                },
            },
            "required": ["predicate_type"],
        },
    ),
    types.Tool(
        name="sdk_scoped_diff",
        description="Get a diff scoped to a CSS selector region. Optionally diff from a bookmark.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "CSS selector to scope the diff to",
                },
                "from_bookmark": {
                    "type": "string",
                    "description": "Bookmark name to diff against (optional)",
                },
            },
            "required": ["scope"],
        },
    ),
    types.Tool(
        name="sdk_get_bookmark",
        description="Get a specific bookmark's snapshot data by name.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name to retrieve",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="sdk_categorize_last_diff",
        description=(
            "Categorize the last computed diff. Returns category "
            "(no-op, navigation, content-update, data-refresh, error, "
            "modal-dialog, form-validation, state-toggle, loading) and confidence."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_summarize_diff",
        description="Get a budget-aware text summary of UI changes. Summarizes appeared/disappeared/modified elements within a character budget.",
        inputSchema={
            "type": "object",
            "properties": {
                "budget": {
                    "type": "integer",
                    "description": "Max characters for summary (default: 300)",
                },
                "from_bookmark": {
                    "type": "string",
                    "description": "Compare against this bookmark (optional)",
                },
                "include_category": {
                    "type": "boolean",
                    "description": "Include change category header (default: true)",
                },
            },
        },
    ),
    types.Tool(
        name="sdk_structured_changes",
        description="Analyze table and list changes between snapshots. Detects added/removed rows and list items.",
        inputSchema={
            "type": "object",
            "properties": {
                "from_bookmark": {
                    "type": "string",
                    "description": "Compare against this bookmark (optional)",
                },
            },
        },
    ),
    types.Tool(
        name="sdk_change_buffer_enable",
        description="Enable change buffering. All diffs are accumulated until drained.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_change_buffer_disable",
        description="Disable change buffering.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_change_buffer_drain",
        description="Drain the change buffer, returning all accumulated changes since last drain.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_change_buffer_size",
        description="Get the current change buffer size and enabled status.",
        inputSchema={"type": "object", "properties": {}},
    ),
    # =========================================================================
    # Change Tracking - Control Mode (Runner's Own UI)
    # =========================================================================
    types.Tool(
        name="ui_save_bookmark",
        description="Save a snapshot bookmark for later diffing in the runner's own UI (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name (unique identifier)",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="ui_list_bookmarks",
        description="List all saved snapshot bookmarks in the runner's own UI (Control mode).",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_delete_bookmark",
        description="Delete a saved bookmark in the runner's own UI (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name to delete",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="ui_diff_from_bookmark",
        description="Compare current runner UI state against a saved bookmark (Control mode). Returns appeared, disappeared, and modified elements.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name to diff against",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="ui_execute_with_diff",
        description="Execute an element action in the runner's own UI and capture what changed (Control mode). Returns before/after diff with categorization.",
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID (or @ref) to interact with",
                },
                "action": {
                    "type": "string",
                    "description": "Action: click, type, focus, etc.",
                },
                "value": {
                    "type": "string",
                    "description": "Value for type/setValue actions",
                },
                "settle_timeout": {
                    "type": "integer",
                    "description": "Max ms to wait for UI to settle (default: 3000)",
                },
                "categorize": {
                    "type": "boolean",
                    "description": "Whether to categorize the diff (default: true)",
                },
                "summary_budget": {
                    "type": "integer",
                    "description": "Max chars for budget summary (default: 300)",
                },
            },
            "required": ["element_id", "action"],
        },
    ),
    types.Tool(
        name="ui_wait_for_change",
        description="Wait for the runner's own UI to change matching a predicate (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "predicate_type": {
                    "type": "string",
                    "enum": [
                        "anyChange",
                        "elementAppeared",
                        "elementDisappeared",
                        "elementChanged",
                        "countChanged",
                    ],
                    "description": "Type of change to wait for",
                },
                "element_id": {
                    "type": "string",
                    "description": "Element ID for elementAppeared/Disappeared/Changed predicates",
                },
                "property": {
                    "type": "string",
                    "description": "Property to watch for elementChanged predicate",
                },
                "min_count": {
                    "type": "integer",
                    "description": "Minimum element count change for countChanged predicate",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max ms to wait (default: 5000)",
                },
                "poll_interval": {
                    "type": "integer",
                    "description": "Poll interval in ms (default: 200)",
                },
            },
            "required": ["predicate_type"],
        },
    ),
    types.Tool(
        name="ui_scoped_diff",
        description="Get a diff scoped to a CSS selector region in the runner's own UI (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "CSS selector to scope the diff to",
                },
                "from_bookmark": {
                    "type": "string",
                    "description": "Bookmark name to diff against (optional)",
                },
            },
            "required": ["scope"],
        },
    ),
    types.Tool(
        name="ui_get_bookmark",
        description="Get a specific bookmark's snapshot data by name in the runner's own UI (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Bookmark name to retrieve",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="ui_categorize_last_diff",
        description=(
            "Categorize the last computed diff in the runner's own UI (Control mode). "
            "Returns category and confidence."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_summarize_diff",
        description="Get a budget-aware text summary of runner UI changes (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "budget": {
                    "type": "integer",
                    "description": "Max characters for summary (default: 300)",
                },
                "from_bookmark": {
                    "type": "string",
                    "description": "Compare against this bookmark (optional)",
                },
                "include_category": {
                    "type": "boolean",
                    "description": "Include change category header (default: true)",
                },
            },
        },
    ),
    types.Tool(
        name="ui_structured_changes",
        description="Analyze table and list changes in the runner's own UI (Control mode).",
        inputSchema={
            "type": "object",
            "properties": {
                "from_bookmark": {
                    "type": "string",
                    "description": "Compare against this bookmark (optional)",
                },
            },
        },
    ),
    types.Tool(
        name="ui_change_buffer_enable",
        description="Enable change buffering in the runner's own UI (Control mode).",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_change_buffer_disable",
        description="Disable change buffering in the runner's own UI (Control mode).",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_change_buffer_drain",
        description="Drain the change buffer in the runner's own UI (Control mode).",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_change_buffer_size",
        description="Get the change buffer size and enabled status in the runner's own UI (Control mode).",
        inputSchema={"type": "object", "properties": {}},
    ),
    # Undo/Redo awareness
    types.Tool(
        name="ui_undo_state",
        description="Get undo/redo availability and state in the runner's own UI (Control mode). Shows whether undo/redo is available, what it would reverse, stack depth, and detection sources.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_undo",
        description="Execute undo in the runner's own UI (Control mode). Uses the app's undo handler if available, otherwise dispatches Ctrl+Z.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="ui_redo",
        description="Execute redo in the runner's own UI (Control mode). Uses the app's redo handler if available, otherwise dispatches Ctrl+Shift+Z.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_undo_state",
        description="Get undo/redo availability and state from the connected SDK app. Shows whether undo/redo is available, what it would reverse, stack depth, and detection sources.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_undo",
        description="Execute undo in the connected SDK app. Uses the app's undo handler if available, otherwise dispatches Ctrl+Z.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="sdk_redo",
        description="Execute redo in the connected SDK app. Uses the app's redo handler if available, otherwise dispatches Ctrl+Shift+Z.",
        inputSchema={"type": "object", "properties": {}},
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

            # Extract page context if present
            page_header_lines: list[str] = []
            page_data = data.get("page")
            if isinstance(page_data, dict):
                page_header_lines = format_page_header(page_data)

            # Extract modal stack if present
            modal_header_lines: list[str] = []
            modal_data = data.get("modalStack")
            if isinstance(modal_data, dict):
                modal_header_lines = format_modal_header(modal_data)

            # Extract toast snapshot if present
            toast_header_lines: list[str] = []
            toast_data = data.get("toasts")
            if isinstance(toast_data, dict):
                toast_header_lines = format_toast_header(toast_data)

            # Extract relationship context if present
            rel_header_lines: list[str] = []
            rel_data = data.get("relationships")
            if isinstance(rel_data, dict):
                rel_header_lines = format_relationship_header(rel_data)

            # Extract drag-drop context if present
            dnd_header_lines: list[str] = []
            dnd_data = data.get("dragDrop")
            if isinstance(dnd_data, dict):
                dnd_header_lines = format_drag_drop_header(dnd_data)

            # Extract undo/redo context if present
            undo_header_lines: list[str] = []
            undo_data = data.get("undoRedo")
            if isinstance(undo_data, dict):
                undo_header_lines = format_undo_redo_header(undo_data)

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

            # Prepend context headers if available
            header_lines: list[str] = []
            if page_header_lines:
                header_lines.extend(page_header_lines)
            if modal_header_lines:
                header_lines.extend(modal_header_lines)
            if toast_header_lines:
                header_lines.extend(toast_header_lines)
            if rel_header_lines:
                header_lines.extend(rel_header_lines)
            if dnd_header_lines:
                header_lines.extend(dnd_header_lines)
            if undo_header_lines:
                header_lines.extend(undo_header_lines)
            if header_lines:
                lines = header_lines + [""] + lines

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
            msg = f"Clicked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_type":
            element_id = ref_manager.resolve(arguments["element_id"])
            text = arguments["text"]
            response = await ui_client.control_type(element_id, text)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Typed '{text}' into element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_focus":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_focus(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Focused element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_blur":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "blur")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Blurred element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_hover":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_hover(element_id)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Hovered element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_double_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "doubleClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Double-clicked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_right_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "rightClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Right-clicked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_clear":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "clear")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Cleared element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_select":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            params = {"value": value}
            if arguments.get("by_label"):
                params["byLabel"] = True
            response = await ui_client.control_action(element_id, "select", params)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Selected '{value}' in element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

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
            msg = f"Scrolled element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_check":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "check")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Checked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_uncheck":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "uncheck")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Unchecked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_toggle":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "toggle")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Toggled element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_set_value":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            response = await ui_client.control_action(
                element_id, "setValue", {"value": value}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Set value '{value}' on element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

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
            msg = f"Dragged {element_id} to {target_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_submit":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "submit")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Submitted form for element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_reset":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.control_action(element_id, "reset")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Reset form for element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

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

            # Extract page context if present
            sdk_page_header_lines: list[str] = []
            sdk_page_data = data.get("page")
            if isinstance(sdk_page_data, dict):
                sdk_page_header_lines = format_page_header(sdk_page_data)

            # Extract modal stack if present
            sdk_modal_header_lines: list[str] = []
            sdk_modal_data = data.get("modalStack")
            if isinstance(sdk_modal_data, dict):
                sdk_modal_header_lines = format_modal_header(sdk_modal_data)

            # Extract toast snapshot if present
            sdk_toast_header_lines: list[str] = []
            sdk_toast_data = data.get("toasts")
            if isinstance(sdk_toast_data, dict):
                sdk_toast_header_lines = format_toast_header(sdk_toast_data)

            # Extract relationship context if present
            sdk_rel_header_lines: list[str] = []
            sdk_rel_data = data.get("relationships")
            if isinstance(sdk_rel_data, dict):
                sdk_rel_header_lines = format_relationship_header(sdk_rel_data)

            # Extract drag-drop context if present
            sdk_dnd_header_lines: list[str] = []
            sdk_dnd_data = data.get("dragDrop")
            if isinstance(sdk_dnd_data, dict):
                sdk_dnd_header_lines = format_drag_drop_header(sdk_dnd_data)

            # Extract undo/redo context if present
            sdk_undo_header_lines: list[str] = []
            sdk_undo_data = data.get("undoRedo")
            if isinstance(sdk_undo_data, dict):
                sdk_undo_header_lines = format_undo_redo_header(sdk_undo_data)

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

            # Prepend context headers if available
            sdk_header_lines: list[str] = []
            if sdk_page_header_lines:
                sdk_header_lines.extend(sdk_page_header_lines)
            if sdk_modal_header_lines:
                sdk_header_lines.extend(sdk_modal_header_lines)
            if sdk_toast_header_lines:
                sdk_header_lines.extend(sdk_toast_header_lines)
            if sdk_rel_header_lines:
                sdk_header_lines.extend(sdk_rel_header_lines)
            if sdk_dnd_header_lines:
                sdk_header_lines.extend(sdk_dnd_header_lines)
            if sdk_undo_header_lines:
                sdk_header_lines.extend(sdk_undo_header_lines)
            if sdk_header_lines:
                lines = sdk_header_lines + [""] + lines

            if overflow:
                lines.append(f"+{overflow} more elements not shown")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "ui_clipboard_read":
            response = await ui_client.control_clipboard_read()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            text = data.get("text")
            if text is not None:
                return [
                    types.TextContent(type="text", text=f"Clipboard content:\n{text}")
                ]
            else:
                return [
                    types.TextContent(
                        type="text", text="Clipboard is empty (no text content)."
                    )
                ]

        elif name == "ui_clipboard_write":
            text = arguments.get("text", "")
            response = await ui_client.control_clipboard_write(text)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Wrote {len(text)} chars to clipboard."
                )
            ]

        elif name == "sdk_clipboard_read":
            response = await ui_client.sdk_clipboard_read()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            text = data.get("text")
            if text is not None:
                return [
                    types.TextContent(type="text", text=f"Clipboard content:\n{text}")
                ]
            else:
                return [
                    types.TextContent(
                        type="text", text="Clipboard is empty (no text content)."
                    )
                ]

        elif name == "sdk_clipboard_write":
            text = arguments.get("text", "")
            response = await ui_client.sdk_clipboard_write(text)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Wrote {len(text)} chars to clipboard."
                )
            ]

        elif name == "sdk_forms":
            response = await ui_client.sdk_forms()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=_format_forms_response(response.data)
                )
            ]

        elif name == "ui_forms":
            response = await ui_client.control_forms()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=_format_forms_response(response.data)
                )
            ]

        elif name == "sdk_fill_form":
            fields = arguments.get("fields", {})
            trigger_validation = arguments.get("triggerValidation", True)
            clear_first = arguments.get("clearFirst", True)
            response = await ui_client.sdk_fill_form(
                fields=fields,
                trigger_validation=trigger_validation,
                clear_first=clear_first,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=_format_fill_form_response(response.data)
                )
            ]

        elif name == "ui_fill_form":
            fields = arguments.get("fields", {})
            trigger_validation = arguments.get("triggerValidation", True)
            clear_first = arguments.get("clearFirst", True)
            response = await ui_client.control_fill_form(
                fields=fields,
                trigger_validation=trigger_validation,
                clear_first=clear_first,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=_format_fill_form_response(response.data)
                )
            ]

        elif name == "sdk_form_snapshot":
            response = await ui_client.sdk_form_snapshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(response.data, indent=2),
                )
            ]

        elif name == "ui_form_snapshot":
            response = await ui_client.control_form_snapshot()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(response.data, indent=2),
                )
            ]

        elif name == "sdk_form_diff":
            before = arguments.get("before", {})
            after = arguments.get("after", {})
            response = await ui_client.sdk_form_diff(before=before, after=after)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=_format_form_diff_response(response.data)
                )
            ]

        elif name == "ui_form_diff":
            before = arguments.get("before", {})
            after = arguments.get("after", {})
            response = await ui_client.control_form_diff(before=before, after=after)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=_format_form_diff_response(response.data)
                )
            ]

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
            msg = f"Clicked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_type":
            element_id = ref_manager.resolve(arguments["element_id"])
            text = arguments["text"]
            response = await ui_client.sdk_element_action(
                element_id, "type", {"text": text}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Typed '{text}' into element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_clear":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "clear")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Cleared element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_select":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            response = await ui_client.sdk_element_action(
                element_id, "select", {"value": value}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Selected '{value}' in element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_focus":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "focus")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Focused element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_blur":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "blur")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Blurred element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_hover":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "hover")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Hovered element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_double_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "doubleClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Double-clicked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_right_click":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "rightClick")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Right-clicked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

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
            msg = f"Scrolled element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_check":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "check")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Checked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_uncheck":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "uncheck")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Unchecked element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_toggle":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "toggle")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Toggled element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_set_value":
            element_id = ref_manager.resolve(arguments["element_id"])
            value = arguments["value"]
            response = await ui_client.sdk_element_action(
                element_id, "setValue", {"value": value}
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Set value '{value}' on element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_drag":
            element_id = ref_manager.resolve(arguments["element_id"])
            target_id = ref_manager.resolve(arguments["target_element_id"])
            params = {"target": {"elementId": target_id}}
            if "steps" in arguments:
                params["steps"] = arguments["steps"]
            response = await ui_client.sdk_element_action(element_id, "drag", params)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Dragged {element_id} to {target_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_submit":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "submit")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Submitted form for element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_reset":
            element_id = ref_manager.resolve(arguments["element_id"])
            response = await ui_client.sdk_element_action(element_id, "reset")
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Reset form for element: {element_id}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

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

        elif name == "sdk_find":
            query = arguments["query"]
            context = arguments.get("context")
            confidence_threshold = arguments.get("confidence_threshold")
            response = await ui_client.sdk_ai_find(
                query,
                context=context,
                confidence_threshold=confidence_threshold,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            found = data.get("found", False)

            if not found:
                reason = data.get("reason", "No matching element found")
                partial = data.get("partialMatches", [])
                lines = [f"Not found: {reason}"]
                if partial:
                    lines.append("")
                    lines.append("Partial matches:")
                    for p in partial[:3]:
                        pid = p.get("elementId", "?")
                        pconf = p.get("confidence", 0)
                        pdiff = p.get("differentiator", "")
                        lines.append(f"  - {pid} ({pconf:.0%}) {pdiff}")
                return [types.TextContent(type="text", text="\n".join(lines))]

            ambiguous = data.get("ambiguous", False)
            if ambiguous:
                suggestion = data.get("suggestion", "")
                candidates = data.get("candidates", [])
                lines = [f"Ambiguous match for '{query}':", ""]
                for c in candidates:
                    cid = c.get("elementId", "?")
                    cconf = c.get("confidence", 0)
                    cdiff = c.get("differentiator", "")
                    lines.append(f"  - {cid} ({cconf:.0%}) {cdiff}")
                if suggestion:
                    lines.append("")
                    lines.append(suggestion)
                return [types.TextContent(type="text", text="\n".join(lines))]

            # Successful unique match
            element_id = data.get("elementId", "?")
            confidence = data.get("confidence", 0)
            reasons = data.get("matchReasons", [])
            alternatives = data.get("alternatives", [])

            lines = [f"Found: {element_id} ({confidence:.0%})"]
            if reasons:
                lines.append(f"  Matched by: {', '.join(reasons[:5])}")

            # Show element details if available
            element = data.get("element", {})
            el_type = element.get("type", "")
            el_label = element.get("label", "") or element.get("description", "")
            if el_type or el_label:
                lines.append(f"  [{el_type}] {el_label}")

            if alternatives:
                lines.append(f"  +{len(alternatives)} alternative(s)")

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_ai_execute":
            instruction = arguments["instruction"]
            response = await ui_client.sdk_ai_execute(instruction)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            msg = f"Executed: {instruction}"
            msg += format_action_error_info(response.data)
            return [types.TextContent(type="text", text=msg)]

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
            opts = _build_annotation_options(arguments, ref_manager)
            # Get full snapshot for element positions + modal/viewport context
            snap_resp = await ui_client.control_snapshot()
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text", text=f"Error getting snapshot: {snap_resp.error}"
                    )
                ]
            snap_data = snap_resp.data or {}
            snap_elements = snap_data.get("elements", [])

            # Fetch design data for boxmodel mode
            design_data: list[dict[str, Any]] | None = None
            if opts.mode == "boxmodel":
                design_resp = await ui_client.control_design_snapshot()
                if design_resp.success:
                    design_data = (design_resp.data or {}).get("elements", [])

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
            annotated_b64 = annotate_screenshot(
                screenshot_b64,
                snap_elements,
                ss_width,
                ss_height,
                ref_manager,
                options=opts,
                snapshot=snap_data,
                design_data=design_data,
            )
            result_content: list[types.TextContent | types.ImageContent] = [
                types.ImageContent(
                    type="image",
                    data=annotated_b64,
                    mimeType=mime_type_for_format(opts.format),
                )
            ]
            if opts.mode == "accessibility":
                result_content.insert(
                    0,
                    types.TextContent(
                        type="text",
                        text="Accessibility overlay applied. Focus order numbers, "
                        "touch target warnings (orange), and missing label indicators "
                        "(pink) are drawn on the image.",
                    ),
                )
            return result_content

        elif name == "sdk_annotated_screenshot":
            monitor = arguments.get("monitor")
            opts = _build_annotation_options(arguments, ref_manager)
            # Get full snapshot
            snap_resp = await ui_client.sdk_snapshot()
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text", text=f"Error getting snapshot: {snap_resp.error}"
                    )
                ]
            snap_data = snap_resp.data or {}
            snap_elements = snap_data.get("elements", [])

            # Fetch design data for boxmodel mode
            design_data = None
            if opts.mode == "boxmodel":
                design_resp = await ui_client.sdk_design_snapshot()
                if design_resp.success:
                    design_data = (design_resp.data or {}).get("elements", [])

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
            annotated_b64 = annotate_screenshot(
                screenshot_b64,
                snap_elements,
                ss_width,
                ss_height,
                ref_manager,
                options=opts,
                snapshot=snap_data,
                design_data=design_data,
            )
            result_content = [
                types.ImageContent(
                    type="image",
                    data=annotated_b64,
                    mimeType=mime_type_for_format(opts.format),
                )
            ]
            # For accessibility mode, also return text summary of issues
            if opts.mode == "accessibility":
                result_content.insert(
                    0,
                    types.TextContent(
                        type="text",
                        text="Accessibility overlay applied. Focus order numbers, "
                        "touch target warnings (orange), and missing label indicators "
                        "(pink) are drawn on the image.",
                    ),
                )
            return result_content

        # Element-level screenshots
        elif name in ("ui_element_screenshot", "sdk_element_screenshot"):
            element_id = ref_manager.resolve(arguments["element_id"])
            monitor = arguments.get("monitor")
            padding = arguments.get("padding", 16)
            scale = arguments.get("scale", 1.0)
            fmt = arguments.get("format", "png")

            is_ui = name == "ui_element_screenshot"
            snap_resp = (
                await ui_client.control_snapshot()
                if is_ui
                else await ui_client.sdk_snapshot()
            )
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text", text=f"Error getting snapshot: {snap_resp.error}"
                    )
                ]
            snap_elements = (snap_resp.data or {}).get("elements", [])
            target_el = next(
                (e for e in snap_elements if e.get("id") == element_id), None
            )
            if not target_el:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Element '{element_id}' not found in snapshot.",
                    )
                ]

            screenshot_resp = (
                await ui_client.control_annotated_screenshot(monitor=monitor)
                if is_ui
                else await ui_client.sdk_screenshot_raw(monitor=monitor)
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

            cropped = crop_to_element(
                screenshot_b64,
                target_el,
                ss_width,
                ss_height,
                padding=padding,
                scale=scale,
                fmt=fmt,
            )
            if not cropped:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Element '{element_id}' has no bounding rect.",
                    )
                ]
            return [
                types.ImageContent(
                    type="image",
                    data=cropped,
                    mimeType=mime_type_for_format(fmt),
                )
            ]

        # Screenshot baseline management
        elif name == "sdk_screenshot_baseline_save":
            key = arguments["baseline_key"]
            monitor = arguments.get("monitor")
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
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]
            baseline_store.save_baseline(key, base64.b64decode(screenshot_b64))
            return [types.TextContent(type="text", text=f"Baseline saved: '{key}'")]

        elif name == "sdk_screenshot_baseline_list":
            keys = baseline_store.list_baselines()
            if not keys:
                return [types.TextContent(type="text", text="No baselines saved.")]
            lines = [f"Saved baselines ({len(keys)}):"]
            for k in keys:
                lines.append(f"  - {k}")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_screenshot_diff":
            key = arguments["baseline_key"]
            threshold = arguments.get("threshold", 0.01)
            monitor = arguments.get("monitor")

            baseline_bytes = baseline_store.get_baseline(key)
            if baseline_bytes is None:
                return [
                    types.TextContent(
                        type="text",
                        text=f"No baseline found for '{key}'. Save one first with sdk_screenshot_baseline_save.",
                    )
                ]

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
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]

            baseline_b64 = base64.b64encode(baseline_bytes).decode()
            result = diff_screenshots(baseline_b64, screenshot_b64, threshold=threshold)

            status: str | None = "PASSED" if result.passed else "FAILED"
            summary = (
                f"Visual diff: {status}\n"
                f"Change: {result.change_percentage}% "
                f"(threshold: {result.threshold_used * 100:.1f}%)\n"
                f"Changed regions: {len(result.changed_regions)}"
            )

            return [
                types.TextContent(type="text", text=summary),
                types.ImageContent(
                    type="image",
                    data=result.diff_image_b64,
                    mimeType="image/png",
                ),
            ]

        # Before/After comparison
        elif name == "sdk_screenshot_before":
            monitor = arguments.get("monitor")
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
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]
            baseline_store.save_before(base64.b64decode(screenshot_b64))
            return [
                types.TextContent(
                    type="text",
                    text="'Before' screenshot captured. Perform your action, then call sdk_screenshot_after.",
                )
            ]

        elif name == "sdk_screenshot_after":
            monitor = arguments.get("monitor")
            layout = arguments.get("layout", "side-by-side")

            before_bytes = baseline_store.get_before()
            if before_bytes is None:
                return [
                    types.TextContent(
                        type="text",
                        text="No 'before' screenshot saved. Call sdk_screenshot_before first.",
                    )
                ]

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
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]

            before_b64 = base64.b64encode(before_bytes).decode()

            # Create side-by-side comparison
            comparison_b64 = create_before_after(
                before_b64, screenshot_b64, layout=layout
            )

            # Also compute diff
            result = diff_screenshots(before_b64, screenshot_b64)
            summary = (
                f"Before/After comparison ({layout}):\n"
                f"Change: {result.change_percentage}%\n"
                f"Changed regions: {len(result.changed_regions)}"
            )

            baseline_store.clear_before()

            return [
                types.TextContent(type="text", text=summary),
                types.ImageContent(
                    type="image",
                    data=comparison_b64,
                    mimeType="image/png",
                ),
                types.ImageContent(
                    type="image",
                    data=result.diff_image_b64,
                    mimeType="image/png",
                ),
            ]

        # Visual Description (D2)
        elif name == "sdk_visual_description":
            snap_resp = await ui_client.sdk_snapshot()
            if not snap_resp.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error getting snapshot: {snap_resp.error}",
                    )
                ]
            snap_data = snap_resp.data or {}
            snap_elements = snap_data.get("elements", [])
            description = generate_visual_description(snap_elements, snap_data)
            return [types.TextContent(type="text", text=description)]

        # Delta / Incremental Screenshot (D3)
        elif name == "sdk_screenshot_delta":
            monitor = arguments.get("monitor")
            if arguments.get("reset", False):
                delta_encoder.reset()

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
            if not screenshot_b64:
                return [
                    types.TextContent(
                        type="text", text="Error: No screenshot data returned"
                    )
                ]

            delta = delta_encoder.encode_delta(screenshot_b64)

            if delta["is_first"]:
                summary = (
                    f"First capture (full image): {delta['image_width']}x{delta['image_height']}px, "
                    f"{delta['total_tiles']} tiles"
                )
                return [
                    types.TextContent(type="text", text=summary),
                    types.ImageContent(
                        type="image",
                        data=screenshot_b64,
                        mimeType="image/png",
                    ),
                ]
            else:
                ratio_pct = round(delta["change_ratio"] * 100, 1)
                summary = (
                    f"Delta: {delta['changed_count']}/{delta['total_tiles']} tiles changed "
                    f"({ratio_pct}%)"
                )
                if delta["changed_count"] == 0:
                    return [
                        types.TextContent(
                            type="text", text=f"{summary}\nNo visual changes detected."
                        )
                    ]
                else:
                    # Return summary + changed tile details as JSON
                    tile_summary = json.dumps(
                        {
                            "changed_count": delta["changed_count"],
                            "total_tiles": delta["total_tiles"],
                            "change_ratio": delta["change_ratio"],
                            "tiles": [
                                {"x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]}
                                for t in delta["changed_tiles"]
                            ],
                        },
                        indent=2,
                    )
                    return [
                        types.TextContent(
                            type="text", text=f"{summary}\n{tile_summary}"
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
                raw_elems = snap.get("elements", [])
                vp_elements: list[dict[str, object]] = (
                    raw_elems if isinstance(raw_elems, list) else []
                )
                result_lines.append(
                    f"\n  === {label} ({vw}px) — {len(vp_elements)} elements ==="
                )
                for el in vp_elements[:20]:
                    eid = el.get("elementId", "?") if isinstance(el, dict) else "?"
                    rect = el.get("rect", {}) if isinstance(el, dict) else {}
                    w = rect.get("width", "?") if isinstance(rect, dict) else "?"
                    h = rect.get("height", "?") if isinstance(rect, dict) else "?"
                    display = (
                        el.get("styles", {}).get("display", "?")
                        if isinstance(el, dict)
                        else "?"
                    )
                    result_lines.append(f"    {eid}: {w}×{h} display={display}")
                if len(vp_elements) > 20:
                    result_lines.append(f"    ... and {len(vp_elements) - 20} more")
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

        # =====================================================================
        # Idle Detection Tools
        # =====================================================================

        elif name == "get_idle_status":
            signal = arguments.get("signal")
            # Auto-detect mode: use SDK if connected, otherwise control
            sdk_check = await ui_client.sdk_status()
            use_sdk = (
                sdk_check.success
                and sdk_check.data
                and sdk_check.data.get("connected", False)
            )
            if use_sdk:
                response = await ui_client.sdk_idle_status(signal=signal)
            else:
                response = await ui_client.control_idle_status(signal=signal)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            mode_label = "SDK app" if use_sdk else "Runner"
            if signal:
                # Single signal response
                idle = data.get("idle", False)
                stable_ms = data.get("stableForMs", 0)
                lines = [
                    f"Target: {mode_label}",
                    f"Signal: {signal}",
                    f"Idle: {idle}",
                    f"Stable for: {stable_ms}ms",
                ]
                details = data.get("details")
                if details:
                    lines.append(f"Details: {json.dumps(details)}")
            else:
                # Composite response
                idle = data.get("idle", False)
                lines = [f"{mode_label} Idle: {idle}", ""]
                signals = data.get("signals", {})
                if signals:
                    lines.append("Signals:")
                    for sig_name, sig_info in signals.items():
                        if isinstance(sig_info, dict):
                            sig_idle = sig_info.get("idle", False)
                            sig_stable = sig_info.get("stableForMs", 0)
                            status = "idle" if sig_idle else "busy"
                            lines.append(
                                f"  {sig_name}: {status} (stable {sig_stable}ms)"
                            )
                        else:
                            lines.append(f"  {sig_name}: {sig_info}")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "wait_for_idle":
            timeout = arguments.get("timeout", 30000)
            min_stable_ms = arguments.get("min_stable_ms", 500)
            exclude = arguments.get("exclude")
            # Auto-detect mode: use SDK if connected, otherwise control
            sdk_check = await ui_client.sdk_status()
            use_sdk = (
                sdk_check.success
                and sdk_check.data
                and sdk_check.data.get("connected", False)
            )
            if use_sdk:
                response = await ui_client.sdk_wait_for_idle(
                    timeout=timeout,
                    min_stable_ms=min_stable_ms,
                    exclude=exclude,
                )
            else:
                response = await ui_client.control_wait_for_idle(
                    timeout=timeout,
                    min_stable_ms=min_stable_ms,
                    exclude=exclude,
                )
            if not response.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Idle wait failed: {response.error}",
                    )
                ]
            data = response.data or {}
            waited_ms = data.get("waitedMs", 0)
            mode_label = "SDK app" if use_sdk else "App"
            lines = [f"{mode_label} is idle (waited {waited_ms}ms)"]
            signals = data.get("signals", {})
            if signals:
                for sig_name, sig_info in signals.items():
                    if isinstance(sig_info, dict):
                        sig_stable = sig_info.get("stableForMs", 0)
                        lines.append(f"  {sig_name}: stable {sig_stable}ms")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "wait_for_signal":
            signal = arguments["signal"]
            timeout = arguments.get("timeout", 30000)
            min_stable_ms = arguments.get("min_stable_ms", 500)
            # Auto-detect mode: use SDK if connected, otherwise control
            sdk_check = await ui_client.sdk_status()
            use_sdk = (
                sdk_check.success
                and sdk_check.data
                and sdk_check.data.get("connected", False)
            )
            if use_sdk:
                response = await ui_client.sdk_wait_for_signal(
                    signal=signal,
                    timeout=timeout,
                    min_stable_ms=min_stable_ms,
                )
            else:
                response = await ui_client.control_wait_for_signal(
                    signal=signal,
                    timeout=timeout,
                    min_stable_ms=min_stable_ms,
                )
            if not response.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Signal wait failed ({signal}): {response.error}",
                    )
                ]
            data = response.data or {}
            waited_ms = data.get("waitedMs", 0)
            stable_ms = data.get("stableForMs", 0)
            mode_label = " (SDK)" if use_sdk else ""
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Signal '{signal}' is idle{mode_label} "
                        f"(waited {waited_ms}ms, stable {stable_ms}ms)"
                    ),
                )
            ]

        elif name == "wait_for_targets":
            targets = arguments["targets"]
            timeout = arguments.get("timeout", 30000)
            min_stable_ms = arguments.get("min_stable_ms", 500)
            # Auto-detect mode: use SDK if connected, otherwise control
            sdk_check = await ui_client.sdk_status()
            use_sdk = (
                sdk_check.success
                and sdk_check.data
                and sdk_check.data.get("connected", False)
            )
            if use_sdk:
                response = await ui_client.sdk_wait_for_targets(
                    targets=targets,
                    timeout=timeout,
                    min_stable_ms=min_stable_ms,
                )
            else:
                response = await ui_client.control_wait_for_targets(
                    targets=targets,
                    timeout=timeout,
                    min_stable_ms=min_stable_ms,
                )
            if not response.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Targets wait failed: {response.error}",
                    )
                ]
            data = response.data or {}
            waited_ms = data.get("waitedMs", 0)
            target_results = data.get("targets", [])
            mode_label = " (SDK)" if use_sdk else ""
            lines = [f"All targets idle{mode_label} (waited {waited_ms}ms)"]
            if target_results:
                for t in target_results:
                    if isinstance(t, dict):
                        t_name = t.get("target", t.get("name", "?"))
                        t_stable = t.get("stableForMs", 0)
                        lines.append(f"  {t_name}: stable {t_stable}ms")
            return [types.TextContent(type="text", text="\n".join(lines))]

        # =====================================================================
        # Stuck Screen Diagnosis
        # =====================================================================

        elif name == "diagnose_stuck_screen":
            observation_window_ms = arguments.get("observation_window_ms", 3000)
            dom_mutation_threshold = arguments.get("dom_mutation_threshold", 3)
            # Auto-detect mode
            sdk_check = await ui_client.sdk_status()
            use_sdk = (
                sdk_check.success
                and sdk_check.data
                and sdk_check.data.get("connected", False)
            )
            if use_sdk:
                response = await ui_client.sdk_diagnose_stuck(
                    observation_window_ms=observation_window_ms,
                    dom_mutation_threshold=dom_mutation_threshold,
                )
            else:
                response = await ui_client.control_diagnose_stuck(
                    observation_window_ms=observation_window_ms,
                )
            if not response.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Diagnosis failed: {response.error}",
                    )
                ]
            data = response.data or {}
            verdict = data.get("verdict", "unknown")
            confidence = data.get("confidence", 0)
            summary = data.get("summary", "")
            evidence = data.get("evidence", {})
            suggestions = data.get("suggestions", [])
            obs_ms = data.get("observationWindowMs", 0)
            capture_source = data.get("captureSource", "unknown")
            mode_label = "SDK app" if use_sdk else "Runner"

            lines = [
                f"Stuck Screen Diagnosis ({mode_label})",
                f"Verdict: {verdict.upper()} (confidence: {confidence:.0%})",
                "",
                summary,
            ]

            # Evidence section
            lines.append("")
            similarity = evidence.get("screenshotSimilarity")
            if similarity is not None:
                lines.append(
                    f"Screenshot similarity: {similarity:.1%} "
                    f"(changed: {evidence.get('screenshotChanged', False)})"
                )
            lines.append(
                f"UI Bridge responsive: {evidence.get('uiBridgeResponsive', False)}"
            )

            indicators = evidence.get("loadingIndicators", [])
            if indicators:
                lines.append(f"Loading indicators ({len(indicators)}):")
                for ind in indicators[:5]:
                    ind_type = ind.get("type", "unknown")
                    ind_detail = (
                        ind.get("selector")
                        or ind.get("details")
                        or ind.get("element")
                        or ind_type
                    )
                    lines.append(f"  - [{ind_type}] {ind_detail}")
                if len(indicators) > 5:
                    lines.append(f"  ... +{len(indicators) - 5} more")

            lines.append(
                f"Network: {'busy' if evidence.get('networkBusy') else 'idle'} "
                f"({evidence.get('pendingNetworkRequests', 0)} pending)"
            )
            lines.append(f"Capture source: {capture_source} | Observation window: {obs_ms}ms")

            if suggestions:
                lines.append("")
                lines.append("Suggestions:")
                for s in suggestions:
                    lines.append(f"  - {s}")

            diagnosis_result: list[types.TextContent | types.ImageContent] = [
                types.TextContent(type="text", text="\n".join(lines))
            ]

            # Include screenshot as visual evidence
            screenshot_b64 = data.get("screenshot", "")
            if screenshot_b64:
                diagnosis_result.append(
                    types.ImageContent(
                        type="image",
                        data=screenshot_b64,
                        mimeType="image/png",
                    )
                )

            return diagnosis_result

        # =====================================================================
        # Network Request Monitoring Tools
        # =====================================================================

        elif name == "sdk_network_requests":
            status = arguments.get("status")
            method = arguments.get("method")
            url_pattern = arguments.get("url_pattern")
            failures_only = arguments.get("failures_only", False)
            limit = arguments.get("limit", 50)
            response = await ui_client.sdk_network_requests(
                status=status,
                method=method,
                url_pattern=url_pattern,
                failures_only=failures_only,
                limit=limit,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=_format_network_requests_response(response.data),
                )
            ]

        elif name == "ui_network_requests":
            status = arguments.get("status")
            method = arguments.get("method")
            url_pattern = arguments.get("url_pattern")
            failures_only = arguments.get("failures_only", False)
            limit = arguments.get("limit", 50)
            response = await ui_client.control_network_requests(
                status=status,
                method=method,
                url_pattern=url_pattern,
                failures_only=failures_only,
                limit=limit,
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=_format_network_requests_response(response.data),
                )
            ]

        elif name == "sdk_network_requests_in_flight":
            response = await ui_client.sdk_network_requests_in_flight()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=_format_network_requests_response(response.data),
                )
            ]

        elif name == "ui_network_requests_in_flight":
            response = await ui_client.control_network_requests_in_flight()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text",
                    text=_format_network_requests_response(response.data),
                )
            ]

        elif name == "sdk_wait_for_network_request":
            url_pattern = arguments.get("url_pattern")
            method = arguments.get("method")
            timeout = arguments.get("timeout", 30000)
            response = await ui_client.sdk_wait_for_network_request(
                url_pattern=url_pattern,
                method=method,
                timeout=timeout,
            )
            if not response.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Network request wait failed: {response.error}",
                    )
                ]
            return [
                types.TextContent(type="text", text=_format_wait_result(response.data))
            ]

        elif name == "ui_wait_for_network_request":
            url_pattern = arguments.get("url_pattern")
            method = arguments.get("method")
            timeout = arguments.get("timeout", 30000)
            response = await ui_client.control_wait_for_network_request(
                url_pattern=url_pattern,
                method=method,
                timeout=timeout,
            )
            if not response.success:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Network request wait failed: {response.error}",
                    )
                ]
            return [
                types.TextContent(type="text", text=_format_wait_result(response.data))
            ]

        # =============================================================
        # Change Tracking - SDK Mode
        # =============================================================

        elif name == "sdk_save_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.sdk_save_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Bookmark '{bookmark_name}' saved successfully."
                )
            ]

        elif name == "sdk_list_bookmarks":
            response = await ui_client.sdk_list_bookmarks()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            bookmarks: Any = response.data or []
            if not bookmarks:
                return [types.TextContent(type="text", text="No bookmarks saved.")]
            if isinstance(bookmarks, list):
                return [
                    types.TextContent(
                        type="text",
                        text=f"Bookmarks: {', '.join(str(b) for b in bookmarks)}",
                    )
                ]
            return [
                types.TextContent(type="text", text=json.dumps(bookmarks, indent=2))
            ]

        elif name == "sdk_delete_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.sdk_delete_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Bookmark '{bookmark_name}' deleted."
                )
            ]

        elif name == "sdk_diff_from_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.sdk_diff_from_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            changes = data.get("changes", {})
            appeared = changes.get("appeared", [])
            disappeared = changes.get("disappeared", [])
            modified = changes.get("modified", [])
            lines = [f"Diff from bookmark '{bookmark_name}':"]
            lines.append(f"  Appeared: {len(appeared)} elements")
            lines.append(f"  Disappeared: {len(disappeared)} elements")
            lines.append(f"  Modified: {len(modified)} elements")
            if appeared:
                lines.append("  New elements:")
                for el in appeared[:10]:
                    if isinstance(el, dict):
                        lines.append(
                            f"    - {el.get('type', '?')} '{el.get('label', el.get('id', '?'))}'"
                        )
                    else:
                        lines.append(f"    - {el}")
                if len(appeared) > 10:
                    lines.append(f"    ... and {len(appeared) - 10} more")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_execute_with_diff":
            element_id = ref_manager.resolve(arguments["element_id"])
            action_name = arguments["action"]
            value = arguments.get("value")
            settle_timeout = arguments.get("settle_timeout", 3000)
            categorize = arguments.get("categorize", True)
            summary_budget = arguments.get("summary_budget", 300)

            request_body: dict[str, Any] = {
                "elementAction": {
                    "elementId": element_id,
                    "action": action_name,
                },
                "settleTimeout": settle_timeout,
                "categorize": categorize,
                "summaryBudget": summary_budget,
            }
            if value:
                request_body["elementAction"]["value"] = value

            response = await ui_client.sdk_execute_with_diff(request_body)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]

            data = response.data or {}
            lines = [f"Execute with diff: {action_name} on {element_id}"]
            lines.append(f"  Action success: {data.get('actionSuccess', False)}")
            diff = data.get("diff", {})
            changes = diff.get("changes", {})
            lines.append(f"  Appeared: {len(changes.get('appeared', []))}")
            lines.append(f"  Disappeared: {len(changes.get('disappeared', []))}")
            lines.append(f"  Modified: {len(changes.get('modified', []))}")
            if data.get("categorized"):
                cat = data["categorized"]
                lines.append(
                    f"  Category: {cat.get('category')} ({cat.get('confidence', 0) * 100:.0f}%)"
                )
            if data.get("budgetSummary"):
                lines.append(f"  Summary: {data['budgetSummary']}")
            lines.append(f"  Duration: {data.get('durationMs', 0):.0f}ms")
            # Include error info from the underlying action response
            action_result = data.get("actionResult")
            if isinstance(action_result, dict):
                error_info = format_action_error_info(action_result)
                if error_info:
                    lines.append(error_info)
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_wait_for_change":
            predicate: dict[str, Any] = {"type": arguments["predicate_type"]}
            if "element_id" in arguments:
                predicate["elementId"] = ref_manager.resolve(arguments["element_id"])
            if "property" in arguments:
                predicate["property"] = arguments["property"]
            if "min_count" in arguments:
                predicate["minCount"] = arguments["min_count"]
            options: dict[str, Any] = {}
            if "timeout" in arguments:
                options["timeout"] = arguments["timeout"]
            if "poll_interval" in arguments:
                options["pollInterval"] = arguments["poll_interval"]
            response = await ui_client.sdk_wait_for_change(predicate, options or None)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            if data.get("matched"):
                diff = data.get("diff", {})
                changes = diff.get("changes", {})
                return [
                    types.TextContent(
                        type="text",
                        text=f"Change detected! Appeared: {len(changes.get('appeared', []))}, "
                        f"Disappeared: {len(changes.get('disappeared', []))}, "
                        f"Modified: {len(changes.get('modified', []))}",
                    )
                ]
            return [
                types.TextContent(type="text", text="Timed out waiting for change.")
            ]

        elif name == "sdk_scoped_diff":
            scope = arguments["scope"]
            response = await ui_client.sdk_scoped_diff(
                scope, arguments.get("from_bookmark")
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            changes = data.get("changes", {})
            return [
                types.TextContent(
                    type="text",
                    text=f"Scoped diff ('{scope}'): "
                    f"Appeared: {len(changes.get('appeared', []))}, "
                    f"Disappeared: {len(changes.get('disappeared', []))}, "
                    f"Modified: {len(changes.get('modified', []))}",
                )
            ]

        elif name == "sdk_get_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.sdk_get_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            if not data:
                return [
                    types.TextContent(
                        type="text", text=f"Bookmark '{bookmark_name}' not found."
                    )
                ]
            elements = data.get("snapshot", {}).get("elements", [])
            return [
                types.TextContent(
                    type="text",
                    text=f"Bookmark '{bookmark_name}': {len(elements)} elements captured at {data.get('timestamp', '?')}",
                )
            ]

        elif name == "sdk_categorize_last_diff":
            response = await ui_client.sdk_categorize_last_diff()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            if not data:
                return [
                    types.TextContent(
                        type="text", text="No previous diff to categorize."
                    )
                ]
            return [
                types.TextContent(
                    type="text",
                    text=f"Category: {data.get('category', '?')} (confidence: {data.get('confidence', 0) * 100:.0f}%)",
                )
            ]

        elif name == "sdk_summarize_diff":
            budget = arguments.get("budget", 300)
            body: dict[str, Any] = {"budget": budget}
            if "from_bookmark" in arguments:
                body["fromBookmark"] = arguments["from_bookmark"]
            if arguments.get("include_category"):
                body["includeCategory"] = True
            response = await ui_client.sdk_summarize_diff(body)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            summary = (response.data or {}).get("summary", "No changes")
            return [types.TextContent(type="text", text=summary)]

        elif name == "sdk_structured_changes":
            body = {}
            if "from_bookmark" in arguments:
                body["fromBookmark"] = arguments["from_bookmark"]
            response = await ui_client.sdk_structured_changes(body or None)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            lines = ["Structured change analysis:"]
            lines.append(
                f"  Has structured data: {data.get('hasStructuredData', False)}"
            )
            tables = data.get("tableChanges", [])
            lists = data.get("listChanges", [])
            if tables:
                lines.append(f"  Table changes: {len(tables)}")
                for t in tables[:5]:
                    lines.append(
                        f"    - {t.get('tableId', '?')}: "
                        f"+{len(t.get('addedRows', []))} -{len(t.get('removedRows', []))} rows"
                    )
            if lists:
                lines.append(f"  List changes: {len(lists)}")
                for lst in lists[:5]:
                    lines.append(
                        f"    - {lst.get('listId', '?')}: "
                        f"+{len(lst.get('addedItems', []))} -{len(lst.get('removedItems', []))} items"
                    )
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "sdk_change_buffer_enable":
            response = await ui_client.sdk_enable_change_buffer()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Change buffer enabled.")]

        elif name == "sdk_change_buffer_disable":
            response = await ui_client.sdk_disable_change_buffer()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Change buffer disabled.")]

        elif name == "sdk_change_buffer_drain":
            response = await ui_client.sdk_drain_change_buffer()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            changes = data.get("changes", [])
            return [
                types.TextContent(
                    type="text", text=f"Drained {len(changes)} buffered changes."
                )
            ]

        elif name == "sdk_change_buffer_size":
            response = await ui_client.sdk_change_buffer_size()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            return [
                types.TextContent(
                    type="text",
                    text=f"Buffer size: {data.get('size', 0)}, enabled: {data.get('enabled', False)}",
                )
            ]

        # =============================================================
        # Change Tracking - Control Mode (Runner's Own UI)
        # =============================================================

        elif name == "ui_save_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.control_save_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Bookmark '{bookmark_name}' saved successfully."
                )
            ]

        elif name == "ui_list_bookmarks":
            response = await ui_client.control_list_bookmarks()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            bookmarks = response.data or []
            if not bookmarks:
                return [types.TextContent(type="text", text="No bookmarks saved.")]
            if isinstance(bookmarks, list):
                return [
                    types.TextContent(
                        type="text",
                        text=f"Bookmarks: {', '.join(str(b) for b in bookmarks)}",
                    )
                ]
            return [
                types.TextContent(type="text", text=json.dumps(bookmarks, indent=2))
            ]

        elif name == "ui_delete_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.control_delete_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(
                    type="text", text=f"Bookmark '{bookmark_name}' deleted."
                )
            ]

        elif name == "ui_diff_from_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.control_diff_from_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            changes = data.get("changes", {})
            appeared = changes.get("appeared", [])
            disappeared = changes.get("disappeared", [])
            modified = changes.get("modified", [])
            lines = [f"Diff from bookmark '{bookmark_name}':"]
            lines.append(f"  Appeared: {len(appeared)} elements")
            lines.append(f"  Disappeared: {len(disappeared)} elements")
            lines.append(f"  Modified: {len(modified)} elements")
            if appeared:
                lines.append("  New elements:")
                for el in appeared[:10]:
                    if isinstance(el, dict):
                        lines.append(
                            f"    - {el.get('type', '?')} '{el.get('label', el.get('id', '?'))}'"
                        )
                    else:
                        lines.append(f"    - {el}")
                if len(appeared) > 10:
                    lines.append(f"    ... and {len(appeared) - 10} more")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "ui_execute_with_diff":
            element_id = ref_manager.resolve(arguments["element_id"])
            action_name = arguments["action"]
            value = arguments.get("value")
            settle_timeout = arguments.get("settle_timeout", 3000)
            categorize = arguments.get("categorize", True)
            summary_budget = arguments.get("summary_budget", 300)

            request_body = {
                "elementAction": {
                    "elementId": element_id,
                    "action": action_name,
                },
                "settleTimeout": settle_timeout,
                "categorize": categorize,
                "summaryBudget": summary_budget,
            }
            if value:
                request_body["elementAction"]["value"] = value

            response = await ui_client.control_execute_with_diff(request_body)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]

            data = response.data or {}
            lines = [f"Execute with diff: {action_name} on {element_id}"]
            lines.append(f"  Action success: {data.get('actionSuccess', False)}")
            diff = data.get("diff", {})
            changes = diff.get("changes", {})
            lines.append(f"  Appeared: {len(changes.get('appeared', []))}")
            lines.append(f"  Disappeared: {len(changes.get('disappeared', []))}")
            lines.append(f"  Modified: {len(changes.get('modified', []))}")
            if data.get("categorized"):
                cat = data["categorized"]
                lines.append(
                    f"  Category: {cat.get('category')} ({cat.get('confidence', 0) * 100:.0f}%)"
                )
            if data.get("budgetSummary"):
                lines.append(f"  Summary: {data['budgetSummary']}")
            lines.append(f"  Duration: {data.get('durationMs', 0):.0f}ms")
            # Include error info from the underlying action response
            action_result = data.get("actionResult")
            if isinstance(action_result, dict):
                error_info = format_action_error_info(action_result)
                if error_info:
                    lines.append(error_info)
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "ui_wait_for_change":
            predicate = {"type": arguments["predicate_type"]}
            if "element_id" in arguments:
                predicate["elementId"] = ref_manager.resolve(arguments["element_id"])
            if "property" in arguments:
                predicate["property"] = arguments["property"]
            if "min_count" in arguments:
                predicate["minCount"] = arguments["min_count"]
            options = {}
            if "timeout" in arguments:
                options["timeout"] = arguments["timeout"]
            if "poll_interval" in arguments:
                options["pollInterval"] = arguments["poll_interval"]
            response = await ui_client.control_wait_for_change(
                predicate, options or None
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            if data.get("matched"):
                diff = data.get("diff", {})
                changes = diff.get("changes", {})
                return [
                    types.TextContent(
                        type="text",
                        text=f"Change detected! Appeared: {len(changes.get('appeared', []))}, "
                        f"Disappeared: {len(changes.get('disappeared', []))}, "
                        f"Modified: {len(changes.get('modified', []))}",
                    )
                ]
            return [
                types.TextContent(type="text", text="Timed out waiting for change.")
            ]

        elif name == "ui_scoped_diff":
            scope = arguments["scope"]
            response = await ui_client.control_scoped_diff(
                scope, arguments.get("from_bookmark")
            )
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            changes = data.get("changes", {})
            return [
                types.TextContent(
                    type="text",
                    text=f"Scoped diff ('{scope}'): "
                    f"Appeared: {len(changes.get('appeared', []))}, "
                    f"Disappeared: {len(changes.get('disappeared', []))}, "
                    f"Modified: {len(changes.get('modified', []))}",
                )
            ]

        elif name == "ui_get_bookmark":
            bookmark_name = arguments["name"]
            response = await ui_client.control_get_bookmark(bookmark_name)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            if not data:
                return [
                    types.TextContent(
                        type="text", text=f"Bookmark '{bookmark_name}' not found."
                    )
                ]
            elements = data.get("snapshot", {}).get("elements", [])
            return [
                types.TextContent(
                    type="text",
                    text=f"Bookmark '{bookmark_name}': {len(elements)} elements captured at {data.get('timestamp', '?')}",
                )
            ]

        elif name == "ui_categorize_last_diff":
            response = await ui_client.control_categorize_last_diff()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            if not data:
                return [
                    types.TextContent(
                        type="text", text="No previous diff to categorize."
                    )
                ]
            return [
                types.TextContent(
                    type="text",
                    text=f"Category: {data.get('category', '?')} (confidence: {data.get('confidence', 0) * 100:.0f}%)",
                )
            ]

        elif name == "ui_summarize_diff":
            budget = arguments.get("budget", 300)
            body = {"budget": budget}
            if "from_bookmark" in arguments:
                body["fromBookmark"] = arguments["from_bookmark"]
            if arguments.get("include_category"):
                body["includeCategory"] = True
            response = await ui_client.control_summarize_diff(body)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            summary = (response.data or {}).get("summary", "No changes")
            return [types.TextContent(type="text", text=summary)]

        elif name == "ui_structured_changes":
            body = {}
            if "from_bookmark" in arguments:
                body["fromBookmark"] = arguments["from_bookmark"]
            response = await ui_client.control_structured_changes(body or None)
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            lines = ["Structured change analysis:"]
            lines.append(
                f"  Has structured data: {data.get('hasStructuredData', False)}"
            )
            tables = data.get("tableChanges", [])
            lists = data.get("listChanges", [])
            if tables:
                lines.append(f"  Table changes: {len(tables)}")
                for t in tables[:5]:
                    lines.append(
                        f"    - {t.get('tableId', '?')}: "
                        f"+{len(t.get('addedRows', []))} -{len(t.get('removedRows', []))} rows"
                    )
            if lists:
                lines.append(f"  List changes: {len(lists)}")
                for lst in lists[:5]:
                    lines.append(
                        f"    - {lst.get('listId', '?')}: "
                        f"+{len(lst.get('addedItems', []))} -{len(lst.get('removedItems', []))} items"
                    )
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "ui_change_buffer_enable":
            response = await ui_client.control_enable_change_buffer()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Change buffer enabled.")]

        elif name == "ui_change_buffer_disable":
            response = await ui_client.control_disable_change_buffer()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [types.TextContent(type="text", text="Change buffer disabled.")]

        elif name == "ui_change_buffer_drain":
            response = await ui_client.control_drain_change_buffer()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            changes = data.get("changes", [])
            return [
                types.TextContent(
                    type="text", text=f"Drained {len(changes)} buffered changes."
                )
            ]

        elif name == "ui_change_buffer_size":
            response = await ui_client.control_change_buffer_size()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            return [
                types.TextContent(
                    type="text",
                    text=f"Buffer size: {data.get('size', 0)}, enabled: {data.get('enabled', False)}",
                )
            ]

        # Undo/Redo awareness tools
        elif name == "ui_undo_state":
            response = await ui_client.control_undo_state()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=_format_undo_state(response.data))
            ]

        elif name == "ui_undo":
            response = await ui_client.control_undo()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            executed = data.get("executed", False)
            msg = "Undo executed." if executed else "Undo was not available."
            return [types.TextContent(type="text", text=msg)]

        elif name == "ui_redo":
            response = await ui_client.control_redo()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            executed = data.get("executed", False)
            msg = "Redo executed." if executed else "Redo was not available."
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_undo_state":
            response = await ui_client.sdk_undo_state()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            return [
                types.TextContent(type="text", text=_format_undo_state(response.data))
            ]

        elif name == "sdk_undo":
            response = await ui_client.sdk_undo()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            executed = data.get("executed", False)
            msg = "Undo executed." if executed else "Undo was not available."
            return [types.TextContent(type="text", text=msg)]

        elif name == "sdk_redo":
            response = await ui_client.sdk_redo()
            if not response.success:
                return [types.TextContent(type="text", text=f"Error: {response.error}")]
            data = response.data or {}
            executed = data.get("executed", False)
            msg = "Redo executed." if executed else "Redo was not available."
            return [types.TextContent(type="text", text=msg)]

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


def _build_annotation_options(
    arguments: dict[str, Any], rm: RefManager
) -> AnnotationOptions:
    """Build AnnotationOptions from MCP tool arguments."""
    highlight = arguments.get("highlight_elements")
    if highlight:
        highlight = [rm.resolve(eid) for eid in highlight]
    scale = arguments.get("scale", 1.0)
    scale = max(0.25, min(2.0, float(scale)))
    quality = arguments.get("quality", 85)
    quality = max(1, min(100, int(quality)))
    return AnnotationOptions(
        mode=arguments.get("mode", "interactive"),
        highlight_elements=highlight,
        crop=arguments.get("crop", "full"),
        scale=scale,
        format=arguments.get("format", "png"),
        quality=quality,
    )


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
