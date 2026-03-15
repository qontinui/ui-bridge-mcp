"""HTTP client for UI Bridge API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_PORT = 9876
DEFAULT_TIMEOUT = 30.0
ELEMENT_DISCOVERY_TIMEOUT = 60.0


def get_windows_host() -> str:
    """Get the Windows host IP address from WSL.

    In WSL2, the Windows host is accessible via the IP in /etc/resolv.conf.
    Falls back to localhost for native Windows/Mac/Linux.
    """
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except (FileNotFoundError, IndexError):
        pass
    return "localhost"


@dataclass
class UIBridgeResponse:
    """Response from the UI Bridge API."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class UIBridgeClient:
    """HTTP client for the UI Bridge API.

    This client provides access to both:
    - Control mode: Runner's own Tauri webview UI (/ui-bridge/control/*)
    - SDK mode: External SDK-integrated apps via runner proxy (/ui-bridge/sdk/*)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = DEFAULT_RUNNER_PORT,
    ) -> None:
        """Initialize the client.

        Args:
            host: Runner host. Auto-detected from WSL if None.
            port: Runner port. Defaults to 9876.
        """
        self.host = host or os.environ.get("QONTINUI_RUNNER_HOST") or get_windows_host()
        self.port = int(os.environ.get("QONTINUI_RUNNER_PORT", port))
        self.base_url = f"http://{self.host}:{self.port}"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        params: dict[str, str] | None = None,
    ) -> UIBridgeResponse:
        """Make an HTTP request to the UI Bridge API.

        Args:
            method: HTTP method (GET or POST).
            endpoint: API endpoint path.
            json_data: Optional JSON body for POST requests.
            timeout: Request timeout in seconds.
            params: Optional query parameters for GET requests.
        """
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = await client.get(url, params=params, timeout=timeout)
            elif method == "POST":
                response = await client.post(url, json=json_data, timeout=timeout)
            elif method == "DELETE":
                response = await client.delete(url, timeout=timeout)
            else:
                return UIBridgeResponse(
                    success=False, error=f"Unsupported method: {method}"
                )

            response.raise_for_status()
            data = response.json()
            return UIBridgeResponse(
                success=data.get("success", False),
                data=data.get("data"),
                error=data.get("error"),
            )
        except httpx.ConnectError as e:
            return UIBridgeResponse(
                success=False,
                error=f"Cannot connect to runner at {url}. Is qontinui-runner running? Error: {e}",
            )
        except httpx.HTTPStatusError as e:
            return UIBridgeResponse(
                success=False,
                error=f"API error: {e.response.status_code} - {e.response.text}",
            )
        except httpx.TimeoutException:
            return UIBridgeResponse(
                success=False,
                error=f"Request timed out after {timeout}s",
            )
        except Exception as e:
            return UIBridgeResponse(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # Health & Status
    # -------------------------------------------------------------------------

    async def health(self) -> UIBridgeResponse:
        """Check runner health."""
        return await self._request("GET", "/health")

    # -------------------------------------------------------------------------
    # Control Mode - Runner's Own UI (/ui-bridge/control/*)
    # -------------------------------------------------------------------------

    async def control_snapshot(self) -> UIBridgeResponse:
        """Get a full UI snapshot of the runner's webview.

        Returns all registered elements, components, and workflows
        with their current state (visibility, position, text content).
        """
        return await self._request("GET", "/ui-bridge/control/snapshot")

    async def control_clipboard_read(self) -> UIBridgeResponse:
        """Read the current system clipboard content."""
        return await self._request("GET", "/ui-bridge/control/clipboard")

    async def control_clipboard_write(
        self, text: str, html: str | None = None
    ) -> UIBridgeResponse:
        """Write text to the system clipboard."""
        data: dict[str, Any] = {"text": text}
        if html is not None:
            data["html"] = html
        return await self._request("POST", "/ui-bridge/control/clipboard", data)

    async def control_forms(self) -> UIBridgeResponse:
        """Get form state data from the runner's webview.

        Returns all detected forms with field values, validation errors,
        dirty state, required fields, and constraint information.
        """
        return await self._request("GET", "/ui-bridge/control/forms")

    async def control_fill_form(
        self,
        fields: dict[str, Any],
        trigger_validation: bool = True,
        clear_first: bool = True,
    ) -> UIBridgeResponse:
        """Fill multiple form fields atomically in the runner's webview.

        Args:
            fields: Map of element ID to value (string, boolean, or string[]).
            trigger_validation: Whether to trigger validation after filling.
            clear_first: Whether to clear existing values first.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/fill",
            {
                "fields": fields,
                "triggerValidation": trigger_validation,
                "clearFirst": clear_first,
            },
        )

    async def control_form_snapshot(self) -> UIBridgeResponse:
        """Capture a snapshot of all form state in the runner's webview.

        Returns a FormSnapshot with all forms and their field states.
        Use before and after an action to track form changes via control_form_diff.
        """
        return await self._request("POST", "/ui-bridge/control/forms/snapshot")

    async def control_form_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two form snapshots from the runner's webview.

        Args:
            before: The before snapshot from control_form_snapshot.
            after: The after snapshot from control_form_snapshot.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/forms/diff",
            {"before": before, "after": after},
        )

    async def control_discover(
        self, interactive_only: bool = False
    ) -> UIBridgeResponse:
        """Trigger element discovery in the runner's webview.

        Args:
            interactive_only: If True, only return interactive elements.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/discover",
            {"interactive_only": interactive_only},
        )

    async def control_list_elements(self) -> UIBridgeResponse:
        """List all registered UI elements in the runner's webview."""
        return await self._request("GET", "/ui-bridge/control/elements")

    async def control_get_element(self, element_id: str) -> UIBridgeResponse:
        """Get details for a specific element.

        Args:
            element_id: The element's registered ID.

        Returns:
            Element details including bounds, state, actions, etc.
        """
        return await self._request("GET", f"/ui-bridge/control/element/{element_id}")

    async def control_click(self, element_id: str) -> UIBridgeResponse:
        """Click an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "click"},
        )

    async def control_type(self, element_id: str, text: str) -> UIBridgeResponse:
        """Type text into an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
            text: Text to type.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "type", "params": {"text": text}},
        )

    async def control_focus(self, element_id: str) -> UIBridgeResponse:
        """Focus an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "focus"},
        )

    async def control_hover(self, element_id: str) -> UIBridgeResponse:
        """Hover over an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "hover"},
        )

    async def control_action(
        self,
        element_id: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> UIBridgeResponse:
        """Execute any action on an element in the runner's webview.

        Generic method for all UI Bridge actions.

        Args:
            element_id: The element's registered ID.
            action: Action name (click, type, focus, blur, hover, etc.).
            params: Optional action parameters.
        """
        body: dict[str, Any] = {"action": action}
        if params:
            body["params"] = params
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            body,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Undo/Redo Awareness
    # -------------------------------------------------------------------------

    async def control_undo_state(self) -> UIBridgeResponse:
        """Get undo/redo availability and state from the runner's webview.

        Returns whether undo/redo is available, what it would reverse,
        stack depth, and detection sources.
        """
        return await self._request("GET", "/ui-bridge/control/undo-state")

    async def control_undo(self) -> UIBridgeResponse:
        """Execute undo in the runner's webview.

        Uses the developer-declared handler if available, otherwise
        dispatches Ctrl+Z (Cmd+Z on Mac) keyboard event.
        """
        return await self._request("POST", "/ui-bridge/control/undo")

    async def control_redo(self) -> UIBridgeResponse:
        """Execute redo in the runner's webview.

        Uses the developer-declared handler if available, otherwise
        dispatches Ctrl+Shift+Z (Cmd+Shift+Z on Mac) keyboard event.
        """
        return await self._request("POST", "/ui-bridge/control/redo")

    # -------------------------------------------------------------------------
    # SDK Mode - External SDK-Integrated Apps (/ui-bridge/sdk/*)
    # -------------------------------------------------------------------------

    async def sdk_connect(self, url: str) -> UIBridgeResponse:
        """Connect to an SDK-integrated app.

        Args:
            url: The app URL (e.g., 'http://localhost:3001').
        """
        return await self._request("POST", "/ui-bridge/sdk/connect", {"url": url})

    async def sdk_disconnect(self) -> UIBridgeResponse:
        """Disconnect from the SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/disconnect")

    async def sdk_status(self) -> UIBridgeResponse:
        """Check SDK app connection status."""
        return await self._request("GET", "/ui-bridge/sdk/status")

    async def sdk_elements(
        self,
        content_only: bool = False,
        content_types: list[str] | None = None,
    ) -> UIBridgeResponse:
        """List all registered UI elements in the connected SDK app.

        Args:
            content_only: If True, filter to only content (non-interactive) elements.
            content_types: Filter to elements matching these content types
                (e.g., ['heading', 'paragraph', 'badge']).

        Note: These query parameters require SDK handler support.
        The Rust relay currently does not forward query params for GET requests,
        so these parameters will only take effect once the relay is updated.
        """
        params: dict[str, str] | None = None
        if content_only or content_types:
            params = {}
            if content_only:
                params["contentOnly"] = "true"
            if content_types:
                params["contentTypes"] = ",".join(content_types)
        return await self._request("GET", "/ui-bridge/sdk/elements", params=params)

    async def sdk_element(self, element_id: str) -> UIBridgeResponse:
        """Get details for a specific element by its registered ID.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request("GET", f"/ui-bridge/sdk/element/{element_id}")

    async def sdk_element_action(
        self, element_id: str, action: str, params: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Execute an action on an element.

        Args:
            element_id: The element's registered ID.
            action: Action to perform (click, type, focus, hover).
            params: Optional params (e.g., {"text": "hello"} for type).
        """
        body: dict[str, Any] = {"action": action}
        if params:
            body["params"] = params
        return await self._request(
            "POST", f"/ui-bridge/sdk/element/{element_id}/action", body
        )

    async def sdk_snapshot(
        self,
        include_content: bool = True,
    ) -> UIBridgeResponse:
        """Get a complete UI snapshot with all elements and their state.

        Args:
            include_content: Include content (non-interactive) elements in the snapshot.
                Defaults to True. Set to False to only get interactive elements.

        Note: The includeContent query parameter requires SDK handler support.
        The Rust relay currently does not forward query params for GET requests,
        so this parameter will only take effect once the relay is updated or
        the snapshot endpoint is changed to accept POST body params.
        """
        params: dict[str, str] | None = None
        if not include_content:
            params = {"includeContent": "false"}
        return await self._request("GET", "/ui-bridge/sdk/snapshot", params=params)

    async def sdk_clipboard_read(self) -> UIBridgeResponse:
        """Read the current system clipboard content via SDK endpoint."""
        return await self._request("GET", "/ui-bridge/sdk/clipboard")

    async def sdk_clipboard_write(
        self, text: str, html: str | None = None
    ) -> UIBridgeResponse:
        """Write text to the system clipboard via SDK endpoint."""
        data: dict[str, Any] = {"text": text}
        if html is not None:
            data["html"] = html
        return await self._request("POST", "/ui-bridge/sdk/clipboard", data)

    async def sdk_undo_state(self) -> UIBridgeResponse:
        """Get undo/redo availability and state from the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/undo-state")

    async def sdk_undo(self) -> UIBridgeResponse:
        """Execute undo in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/undo")

    async def sdk_redo(self) -> UIBridgeResponse:
        """Execute redo in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/redo")

    async def sdk_forms(self) -> UIBridgeResponse:
        """Get form state data from the connected SDK app.

        Returns all detected forms with field values, validation errors,
        dirty state, required fields, and constraint information.
        """
        return await self._request("GET", "/ui-bridge/sdk/forms")

    async def sdk_fill_form(
        self,
        fields: dict[str, Any],
        trigger_validation: bool = True,
        clear_first: bool = True,
    ) -> UIBridgeResponse:
        """Fill multiple form fields atomically in the connected SDK app.

        Args:
            fields: Map of element ID to value (string, boolean, or string[]).
            trigger_validation: Whether to trigger validation after filling.
            clear_first: Whether to clear existing values first.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/fill",
            {
                "fields": fields,
                "triggerValidation": trigger_validation,
                "clearFirst": clear_first,
            },
        )

    async def sdk_form_snapshot(self) -> UIBridgeResponse:
        """Capture a snapshot of all form state in the connected SDK app.

        Returns a FormSnapshot with all forms and their field states.
        Use before and after an action to track form changes via sdk_form_diff.
        """
        return await self._request("POST", "/ui-bridge/sdk/forms/snapshot")

    async def sdk_form_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two form snapshots from the connected SDK app.

        Args:
            before: The before snapshot from sdk_form_snapshot.
            after: The after snapshot from sdk_form_snapshot.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/forms/diff",
            {"before": before, "after": after},
        )

    async def sdk_discover(
        self,
        interactive_only: bool = False,
        include_content: bool = True,
        content_roles: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Trigger element discovery in the SDK app.

        Args:
            interactive_only: If True, only return interactive elements.
            include_content: Include content (non-interactive) elements in discovery.
                Defaults to True. Ignored if interactive_only is True.
            content_roles: Filter content elements to these roles
                (e.g., ['heading', 'body-text', 'metric']).
                Only applies when content elements are included.
        """
        body: dict[str, Any] = {"interactive_only": interactive_only}
        if not include_content:
            body["includeContent"] = False
        if content_roles:
            body["contentRoles"] = content_roles
        return await self._request(
            "POST",
            "/ui-bridge/sdk/discover",
            body,
        )

    async def sdk_ai_search(
        self,
        text: str,
        content_role: str | None = None,
        content_types: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Search for elements by natural language description.

        Args:
            text: Natural language description of the element.
            content_role: Filter results to elements with this content role
                (e.g., 'heading', 'body-text', 'metric', 'badge').
            content_types: Filter results to elements with these content types
                (e.g., ['heading', 'paragraph', 'metric-value']).
        """
        body: dict[str, Any] = {"text": text}
        if content_role:
            body["contentRole"] = content_role
        if content_types:
            body["contentTypes"] = content_types
        return await self._request("POST", "/ui-bridge/sdk/ai/search", body)

    async def sdk_ai_find(
        self,
        query: str,
        context: str | None = None,
        confidence_threshold: float | None = None,
    ) -> UIBridgeResponse:
        """Find an element by natural language description with spatial/relational context.

        Supports queries like "close button near Terminal 1 tab" or
        "email input in the login form". More capable than sdk_ai_search
        because it handles spatial references, container scoping, ordinals,
        and state filters.

        Args:
            query: Natural language element description.
            context: Optional context hint (e.g., 'in the dialog').
            confidence_threshold: Minimum confidence threshold (0-1).
        """
        body: dict[str, Any] = {"query": query}
        if context:
            body["context"] = {"sectionHint": context}
        if confidence_threshold is not None:
            body["confidenceThreshold"] = confidence_threshold
        return await self._request("POST", "/ui-bridge/sdk/ai/find", body)

    async def sdk_ai_execute(self, instruction: str) -> UIBridgeResponse:
        """Execute an action by natural language instruction.

        Args:
            instruction: Natural language instruction (e.g., 'click the Submit button').
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/execute", {"instruction": instruction}
        )

    async def sdk_ai_assert(
        self, text: str, state: str | None = None
    ) -> UIBridgeResponse:
        """Assert element state using natural language.

        Args:
            text: Element description or text to find.
            state: Expected state (e.g., 'visible', 'hidden', 'enabled').
        """
        body: dict[str, Any] = {"text": text}
        if state:
            body["state"] = state
        return await self._request("POST", "/ui-bridge/sdk/ai/assert", body)

    async def sdk_ai_summary(self) -> UIBridgeResponse:
        """Get an AI-friendly summary of the current page."""
        return await self._request("GET", "/ui-bridge/sdk/ai/summary")

    async def sdk_screenshot(self) -> UIBridgeResponse:
        """Capture a screenshot of the monitor where the SDK app is running."""
        return await self._request("GET", "/ui-bridge/sdk/screenshot")

    async def sdk_components(self) -> UIBridgeResponse:
        """List all registered components in the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/components")

    # -------------------------------------------------------------------------
    # SDK Mode - Cross-App Analysis (/ui-bridge/sdk/ai/analyze/*)
    # -------------------------------------------------------------------------

    async def sdk_ai_analyze_data(self) -> UIBridgeResponse:
        """Extract labeled data values from the connected SDK app's page."""
        return await self._request("GET", "/ui-bridge/sdk/ai/analyze/data")

    async def sdk_ai_analyze_regions(self) -> UIBridgeResponse:
        """Segment the connected SDK app's page into semantic regions."""
        return await self._request("GET", "/ui-bridge/sdk/ai/analyze/regions")

    async def sdk_ai_analyze_structured_data(self) -> UIBridgeResponse:
        """Extract tables and lists from the connected SDK app's page."""
        return await self._request("GET", "/ui-bridge/sdk/ai/analyze/structured-data")

    async def sdk_ai_cross_app_compare(
        self,
        source_snapshot: dict[str, Any],
        target_snapshot: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two semantic snapshots from different apps.

        Args:
            source_snapshot: Semantic snapshot from the source app.
            target_snapshot: Semantic snapshot from the target app.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/ai/analyze/cross-app-compare",
            {
                "sourceSnapshot": source_snapshot,
                "targetSnapshot": target_snapshot,
            },
        )

    async def sdk_ai_snapshot(self) -> UIBridgeResponse:
        """Get a semantic snapshot of the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/ai/snapshot")

    # -------------------------------------------------------------------------
    # SDK Mode - Page Navigation (/ui-bridge/sdk/page/*)
    # -------------------------------------------------------------------------

    async def sdk_page_refresh(self) -> UIBridgeResponse:
        """Refresh the current page in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/page/refresh")

    async def sdk_page_navigate(self, url: str) -> UIBridgeResponse:
        """Navigate the connected SDK app to a URL.

        Args:
            url: The URL to navigate to.
        """
        return await self._request("POST", "/ui-bridge/sdk/page/navigate", {"url": url})

    async def sdk_page_go_back(self) -> UIBridgeResponse:
        """Go back in browser history in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/page/back")

    async def sdk_page_go_forward(self) -> UIBridgeResponse:
        """Go forward in browser history in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/page/forward")

    # -------------------------------------------------------------------------
    # Control Mode - Idle Detection (/ui-bridge/control/idle-*)
    # -------------------------------------------------------------------------

    async def control_idle_status(self, signal: str | None = None) -> UIBridgeResponse:
        """Get current idle status.

        Args:
            signal: Optional specific signal to query (e.g., 'network', 'dom',
                'loading-indicators'). If None, returns composite status with
                all signals.
        """
        if signal:
            return await self._request(
                "GET", f"/ui-bridge/control/idle-status/{signal}"
            )
        return await self._request("GET", "/ui-bridge/control/idle-status")

    async def control_wait_for_idle(
        self,
        timeout: int = 30000,
        min_stable_ms: int = 500,
        exclude: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Block until the app is idle (all signals stable).

        Args:
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long signals must remain idle before considered
                stable. Defaults to 500.
            exclude: Signal names to ignore (e.g., ['network']).
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        if exclude:
            body["exclude"] = exclude
        # HTTP timeout must exceed the wait timeout
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST", "/ui-bridge/control/wait-for-idle", body, timeout=http_timeout
        )

    async def control_diagnose_stuck(
        self,
        observation_window_ms: int = 3000,
    ) -> UIBridgeResponse:
        """Diagnose whether the runner is stuck on a loading screen.

        Uses native screenshot capture — works even if React hasn't mounted.

        Args:
            observation_window_ms: How long to observe in ms. Defaults to 3000.
        """
        body: dict[str, Any] = {
            "observationWindowMs": observation_window_ms,
        }
        # HTTP timeout must exceed the observation window + screenshot capture + DOM probes
        http_timeout = (observation_window_ms / 1000) + 15.0
        return await self._request(
            "POST",
            "/ui-bridge/control/diagnose-stuck",
            body,
            timeout=http_timeout,
        )

    async def control_wait_for_signal(
        self,
        signal: str,
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Block until a specific signal is idle.

        Args:
            signal: Signal name ('network', 'dom', or 'loading-indicators').
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long the signal must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            f"/ui-bridge/control/wait-for-idle/{signal}",
            body,
            timeout=http_timeout,
        )

    async def control_wait_for_targets(
        self,
        targets: list[str | dict[str, str]],
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Wait for specific targets to become idle.

        Args:
            targets: Array of signal names (strings) or indicator objects
                (e.g., {"indicator": ".loading-spinner"}).
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long targets must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "targets": targets,
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            "/ui-bridge/control/wait-for-targets",
            body,
            timeout=http_timeout,
        )

    # -------------------------------------------------------------------------
    # SDK Mode - Idle Detection (/ui-bridge/sdk/idle-*)
    # -------------------------------------------------------------------------

    async def sdk_idle_status(self, signal: str | None = None) -> UIBridgeResponse:
        """Get current idle status from connected SDK app.

        Args:
            signal: Optional specific signal to query (e.g., 'network', 'dom',
                'loading-indicators'). If None, returns composite status with
                all signals.
        """
        if signal:
            return await self._request("GET", f"/ui-bridge/sdk/idle-status/{signal}")
        return await self._request("GET", "/ui-bridge/sdk/idle-status")

    async def sdk_wait_for_idle(
        self,
        timeout: int = 30000,
        min_stable_ms: int = 500,
        exclude: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Block until the SDK app is idle (all signals stable).

        Args:
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long signals must remain idle before considered
                stable. Defaults to 500.
            exclude: Signal names to ignore (e.g., ['network']).
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        if exclude:
            body["exclude"] = exclude
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST", "/ui-bridge/sdk/wait-for-idle", body, timeout=http_timeout
        )

    async def sdk_diagnose_stuck(
        self,
        observation_window_ms: int = 3000,
        dom_mutation_threshold: int = 3,
    ) -> UIBridgeResponse:
        """Diagnose whether the connected SDK app is stuck on a loading screen.

        Args:
            observation_window_ms: How long to observe in ms. Defaults to 3000.
            dom_mutation_threshold: Fewer DOM mutations than this = not changing.
                Defaults to 3.
        """
        body: dict[str, Any] = {
            "observationWindowMs": observation_window_ms,
            "domMutationThreshold": dom_mutation_threshold,
        }
        http_timeout = (observation_window_ms / 1000) + 15.0
        return await self._request(
            "POST",
            "/ui-bridge/sdk/diagnose-stuck",
            body,
            timeout=http_timeout,
        )

    async def sdk_wait_for_signal(
        self,
        signal: str,
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Block until a specific signal is idle in the SDK app.

        Args:
            signal: Signal name ('network', 'dom', or 'loading-indicators').
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long the signal must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            f"/ui-bridge/sdk/wait-for-idle/{signal}",
            body,
            timeout=http_timeout,
        )

    async def sdk_wait_for_targets(
        self,
        targets: list[str | dict[str, str]],
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Wait for specific targets to become idle in the SDK app.

        Args:
            targets: Array of signal names (strings) or indicator objects
                (e.g., {"indicator": ".loading-spinner"}).
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long targets must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "targets": targets,
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            "/ui-bridge/sdk/wait-for-targets",
            body,
            timeout=http_timeout,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Network Request Monitoring (/ui-bridge/control/network-*)
    # -------------------------------------------------------------------------

    async def control_network_requests(
        self,
        status: str | None = None,
        method: str | None = None,
        url_pattern: str | None = None,
        failures_only: bool = False,
        limit: int = 50,
    ) -> UIBridgeResponse:
        """List recent network requests from the runner's webview.

        Args:
            status: Filter by status: in-flight, completed, failed, cancelled.
            method: Filter by HTTP method (GET, POST, etc.).
            url_pattern: Filter by URL substring match.
            failures_only: Only show failed requests (4xx/5xx/network errors).
            limit: Max number of results. Defaults to 50.
        """
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if method:
            params["method"] = method
        if url_pattern:
            params["url_pattern"] = url_pattern
        if failures_only:
            params["failures_only"] = "true"
        if limit != 50:
            params["limit"] = str(limit)
        return await self._request(
            "GET",
            "/ui-bridge/control/network-requests",
            params=params or None,
        )

    async def control_network_requests_in_flight(self) -> UIBridgeResponse:
        """Show currently in-flight network requests from the runner's webview."""
        return await self._request(
            "GET", "/ui-bridge/control/network-requests/in-flight"
        )

    async def control_wait_for_network_request(
        self,
        url_pattern: str | None = None,
        method: str | None = None,
        timeout: int = 30000,
    ) -> UIBridgeResponse:
        """Wait for a network request matching the given criteria to complete.

        Args:
            url_pattern: URL substring to match.
            method: HTTP method to match.
            timeout: Timeout in milliseconds. Defaults to 30000.
        """
        body: dict[str, Any] = {}
        if url_pattern:
            body["url_pattern"] = url_pattern
        if method:
            body["method"] = method
        if timeout != 30000:
            body["timeout"] = timeout
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            "/ui-bridge/control/network-requests/wait",
            body or None,
            timeout=http_timeout,
        )

    # -------------------------------------------------------------------------
    # SDK Mode - Network Request Monitoring (/ui-bridge/sdk/network-*)
    # -------------------------------------------------------------------------

    async def sdk_network_requests(
        self,
        status: str | None = None,
        method: str | None = None,
        url_pattern: str | None = None,
        failures_only: bool = False,
        limit: int = 50,
    ) -> UIBridgeResponse:
        """List recent network requests from the connected SDK app.

        Args:
            status: Filter by status: in-flight, completed, failed, cancelled.
            method: Filter by HTTP method (GET, POST, etc.).
            url_pattern: Filter by URL substring match.
            failures_only: Only show failed requests (4xx/5xx/network errors).
            limit: Max number of results. Defaults to 50.
        """
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if method:
            params["method"] = method
        if url_pattern:
            params["url_pattern"] = url_pattern
        if failures_only:
            params["failures_only"] = "true"
        if limit != 50:
            params["limit"] = str(limit)
        return await self._request(
            "GET",
            "/ui-bridge/sdk/network-requests",
            params=params or None,
        )

    async def sdk_network_requests_in_flight(self) -> UIBridgeResponse:
        """Show currently in-flight network requests from the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/network-requests/in-flight")

    async def sdk_wait_for_network_request(
        self,
        url_pattern: str | None = None,
        method: str | None = None,
        timeout: int = 30000,
    ) -> UIBridgeResponse:
        """Wait for a network request matching the given criteria to complete.

        Args:
            url_pattern: URL substring to match.
            method: HTTP method to match.
            timeout: Timeout in milliseconds. Defaults to 30000.
        """
        body: dict[str, Any] = {}
        if url_pattern:
            body["url_pattern"] = url_pattern
        if method:
            body["method"] = method
        if timeout != 30000:
            body["timeout"] = timeout
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            "/ui-bridge/sdk/network-requests/wait",
            body or None,
            timeout=http_timeout,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Design Review (/ui-bridge/control/design/*)
    # -------------------------------------------------------------------------

    async def control_design_snapshot(
        self,
        element_ids: list[str] | None = None,
        include_pseudo_elements: bool = False,
    ) -> UIBridgeResponse:
        """Get design data for all or filtered elements (control/runner path).

        Args:
            element_ids: Optional list of element IDs to include.
            include_pseudo_elements: Whether to include ::before/::after styles.
        """
        body: dict[str, Any] = {}
        if element_ids:
            body["elementIds"] = element_ids
        if include_pseudo_elements:
            body["includePseudoElements"] = True
        return await self._request(
            "POST", "/ui-bridge/control/design/snapshot", body or None
        )

    # -------------------------------------------------------------------------
    # SDK Mode - Design Review (/ui-bridge/sdk/design/*)
    # -------------------------------------------------------------------------

    async def sdk_design_element_styles(self, element_id: str) -> UIBridgeResponse:
        """Get extended computed styles for a specific element.

        Args:
            element_id: The element ID to inspect.
        """
        return await self._request(
            "GET", f"/ui-bridge/sdk/design/element/{element_id}/styles"
        )

    async def sdk_design_state_styles(
        self,
        element_id: str,
        states: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Get styles across interaction states (hover, focus, active, disabled).

        Args:
            element_id: The element ID to inspect.
            states: Interaction states to capture. Defaults to all.
        """
        body: dict[str, Any] = {}
        if states:
            body["states"] = states
        return await self._request(
            "POST",
            f"/ui-bridge/sdk/design/element/{element_id}/state-styles",
            body or None,
        )

    async def sdk_design_snapshot(
        self,
        element_ids: list[str] | None = None,
        include_pseudo_elements: bool = False,
    ) -> UIBridgeResponse:
        """Get design data for all or filtered elements.

        Args:
            element_ids: Optional list of element IDs to include.
            include_pseudo_elements: Whether to include ::before/::after styles.
        """
        body: dict[str, Any] = {}
        if element_ids:
            body["elementIds"] = element_ids
        if include_pseudo_elements:
            body["includePseudoElements"] = True
        return await self._request(
            "POST", "/ui-bridge/sdk/design/snapshot", body or None
        )

    async def sdk_design_responsive(
        self,
        viewports: dict[str, int] | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Capture design snapshots at multiple viewport widths.

        Args:
            viewports: Map of label to width in px. Defaults to standard breakpoints.
            element_ids: Optional list of element IDs to include.
        """
        body: dict[str, Any] = {}
        if viewports:
            body["viewports"] = viewports
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request("POST", "/ui-bridge/sdk/design/responsive", body)

    async def sdk_design_audit(
        self,
        guide: dict[str, Any] | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Run a style audit against a loaded or provided style guide.

        Args:
            guide: Inline style guide config. Uses loaded guide if not provided.
            element_ids: Optional list of element IDs to audit.
        """
        body: dict[str, Any] = {}
        if guide:
            body["guide"] = guide
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request("POST", "/ui-bridge/sdk/design/audit", body or None)

    async def sdk_design_load_guide(self, guide: dict[str, Any]) -> UIBridgeResponse:
        """Load a style guide for subsequent audits.

        Args:
            guide: The style guide configuration (StyleGuideConfig).
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/design/style-guide/load", {"guide": guide}
        )

    async def sdk_design_get_guide(self) -> UIBridgeResponse:
        """Get the currently loaded style guide."""
        return await self._request("GET", "/ui-bridge/sdk/design/style-guide")

    async def sdk_design_clear_guide(self) -> UIBridgeResponse:
        """Clear the currently loaded style guide."""
        return await self._request("DELETE", "/ui-bridge/sdk/design/style-guide")

    # -------------------------------------------------------------------------
    # SDK Mode - Quality Evaluation
    # -------------------------------------------------------------------------

    async def sdk_design_evaluate(
        self,
        context: str | None = None,
        custom_context: dict[str, Any] | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Run holistic UI quality evaluation.

        Args:
            context: Built-in context name (general, minimal, data-dense, mobile, accessibility).
            custom_context: Custom context object with metric weights/thresholds.
            element_ids: Optional list of element IDs to evaluate.
        """
        body: dict[str, Any] = {}
        if context:
            body["context"] = context
        if custom_context:
            body["customContext"] = custom_context
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request(
            "POST", "/ui-bridge/sdk/design/evaluate", body or None
        )

    async def sdk_design_evaluate_contexts(self) -> UIBridgeResponse:
        """Get available quality evaluation contexts."""
        return await self._request("GET", "/ui-bridge/sdk/design/evaluate/contexts")

    async def sdk_design_save_baseline(
        self,
        label: str | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Save current element state as a baseline for diff comparison.

        Args:
            label: Optional label for the baseline.
            element_ids: Optional list of element IDs to include.
        """
        body: dict[str, Any] = {}
        if label:
            body["label"] = label
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request(
            "POST", "/ui-bridge/sdk/design/evaluate/baseline", body or None
        )

    async def sdk_design_diff_baseline(
        self,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Diff current elements against saved baseline.

        Args:
            element_ids: Optional list of element IDs to diff.
        """
        body: dict[str, Any] = {}
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request(
            "POST", "/ui-bridge/sdk/design/evaluate/diff", body or None
        )

    # -------------------------------------------------------------------------
    # Control Mode - Change Tracking (/ui-bridge/control/ai/*)
    # -------------------------------------------------------------------------

    async def control_save_bookmark(self, name: str) -> UIBridgeResponse:
        """Save a snapshot bookmark for later diffing (control mode).

        Args:
            name: Unique bookmark name.
        """
        return await self._request(
            "POST", "/ui-bridge/control/ai/bookmarks", {"name": name}
        )

    async def control_list_bookmarks(self) -> UIBridgeResponse:
        """List all saved snapshot bookmarks (control mode)."""
        return await self._request("GET", "/ui-bridge/control/ai/bookmarks")

    async def control_get_bookmark(self, name: str) -> UIBridgeResponse:
        """Get a specific bookmark by name (control mode).

        Args:
            name: Bookmark name.
        """
        return await self._request("GET", f"/ui-bridge/control/ai/bookmark/{name}")

    async def control_delete_bookmark(self, name: str) -> UIBridgeResponse:
        """Delete a saved bookmark (control mode).

        Args:
            name: Bookmark name to delete.
        """
        return await self._request("DELETE", f"/ui-bridge/control/ai/bookmark/{name}")

    async def control_diff_from_bookmark(self, name: str) -> UIBridgeResponse:
        """Diff current UI state against a saved bookmark (control mode).

        Args:
            name: Bookmark name to diff against.
        """
        return await self._request("GET", f"/ui-bridge/control/ai/bookmark/{name}/diff")

    async def control_execute_with_diff(
        self, request: dict[str, Any]
    ) -> UIBridgeResponse:
        """Execute an element action and capture what changed (control mode).

        Args:
            request: Request body with elementAction, settleTimeout, etc.
        """
        return await self._request(
            "POST", "/ui-bridge/control/ai/execute-with-diff", request
        )

    async def control_wait_for_change(
        self, predicate: dict[str, Any], options: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Wait for a change matching a predicate (control mode).

        Args:
            predicate: Change predicate (e.g. {"type": "anyChange"}).
            options: Optional wait options (timeout, pollInterval, etc.).
        """
        body: dict[str, Any] = {"predicate": predicate}
        if options:
            body["options"] = options
        return await self._request(
            "POST", "/ui-bridge/control/ai/wait-for-change", body
        )

    async def control_categorize_last_diff(self) -> UIBridgeResponse:
        """Categorize the last computed diff (control mode)."""
        return await self._request("GET", "/ui-bridge/control/ai/categorize-last-diff")

    async def control_scoped_diff(
        self,
        scope: str,
        from_bookmark: str | None = None,
    ) -> UIBridgeResponse:
        """Get a scoped diff (control mode).

        Args:
            scope: Scope for the diff.
            from_bookmark: Optional bookmark name to diff against.
        """
        body: dict[str, Any] = {"scope": scope}
        if from_bookmark:
            body["fromBookmark"] = from_bookmark
        return await self._request("POST", "/ui-bridge/control/ai/scoped-diff", body)

    async def control_summarize_diff(self, body: dict[str, Any]) -> UIBridgeResponse:
        """Get a budget-aware summary of UI changes (control mode).

        Args:
            body: Request body with budget, fromBookmark, includeCategory.
        """
        return await self._request("POST", "/ui-bridge/control/ai/summarize-diff", body)

    async def control_structured_changes(
        self, body: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Analyze table and list changes (control mode).

        Args:
            body: Optional request body with fromBookmark.
        """
        return await self._request(
            "POST", "/ui-bridge/control/ai/structured-changes", body
        )

    async def control_enable_change_buffer(self) -> UIBridgeResponse:
        """Enable change buffering (control mode)."""
        return await self._request("POST", "/ui-bridge/control/ai/change-buffer/enable")

    async def control_disable_change_buffer(self) -> UIBridgeResponse:
        """Disable change buffering (control mode)."""
        return await self._request(
            "POST", "/ui-bridge/control/ai/change-buffer/disable"
        )

    async def control_drain_change_buffer(self) -> UIBridgeResponse:
        """Drain the change buffer (control mode)."""
        return await self._request("POST", "/ui-bridge/control/ai/change-buffer/drain")

    async def control_change_buffer_size(self) -> UIBridgeResponse:
        """Get change buffer size and enabled status (control mode)."""
        return await self._request("GET", "/ui-bridge/control/ai/change-buffer/size")

    # -------------------------------------------------------------------------
    # SDK Mode - Change Tracking (/ui-bridge/sdk/ai/*)
    # -------------------------------------------------------------------------

    async def sdk_save_bookmark(self, name: str) -> UIBridgeResponse:
        """Save a snapshot bookmark for later diffing (SDK mode).

        Args:
            name: Unique bookmark name.
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/bookmarks", {"name": name}
        )

    async def sdk_list_bookmarks(self) -> UIBridgeResponse:
        """List all saved snapshot bookmarks (SDK mode)."""
        return await self._request("GET", "/ui-bridge/sdk/ai/bookmarks")

    async def sdk_get_bookmark(self, name: str) -> UIBridgeResponse:
        """Get a specific bookmark by name (SDK mode).

        Args:
            name: Bookmark name.
        """
        return await self._request("GET", f"/ui-bridge/sdk/ai/bookmark/{name}")

    async def sdk_delete_bookmark(self, name: str) -> UIBridgeResponse:
        """Delete a saved bookmark (SDK mode).

        Args:
            name: Bookmark name to delete.
        """
        return await self._request("DELETE", f"/ui-bridge/sdk/ai/bookmark/{name}")

    async def sdk_diff_from_bookmark(self, name: str) -> UIBridgeResponse:
        """Diff current UI state against a saved bookmark (SDK mode).

        Args:
            name: Bookmark name to diff against.
        """
        return await self._request("GET", f"/ui-bridge/sdk/ai/bookmark/{name}/diff")

    async def sdk_execute_with_diff(self, request: dict[str, Any]) -> UIBridgeResponse:
        """Execute an element action and capture what changed (SDK mode).

        Args:
            request: Request body with elementAction, settleTimeout, etc.
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/execute-with-diff", request
        )

    async def sdk_wait_for_change(
        self, predicate: dict[str, Any], options: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Wait for a change matching a predicate (SDK mode).

        Args:
            predicate: Change predicate (e.g. {"type": "anyChange"}).
            options: Optional wait options (timeout, pollInterval, etc.).
        """
        body: dict[str, Any] = {"predicate": predicate}
        if options:
            body["options"] = options
        return await self._request("POST", "/ui-bridge/sdk/ai/wait-for-change", body)

    async def sdk_categorize_last_diff(self) -> UIBridgeResponse:
        """Categorize the last computed diff (SDK mode)."""
        return await self._request("GET", "/ui-bridge/sdk/ai/categorize-last-diff")

    async def sdk_scoped_diff(
        self,
        scope: str,
        from_bookmark: str | None = None,
    ) -> UIBridgeResponse:
        """Get a scoped diff (SDK mode).

        Args:
            scope: Scope for the diff.
            from_bookmark: Optional bookmark name to diff against.
        """
        body: dict[str, Any] = {"scope": scope}
        if from_bookmark:
            body["fromBookmark"] = from_bookmark
        return await self._request("POST", "/ui-bridge/sdk/ai/scoped-diff", body)

    async def sdk_summarize_diff(self, body: dict[str, Any]) -> UIBridgeResponse:
        """Get a budget-aware summary of UI changes (SDK mode).

        Args:
            body: Request body with budget, fromBookmark, includeCategory.
        """
        return await self._request("POST", "/ui-bridge/sdk/ai/summarize-diff", body)

    async def sdk_structured_changes(
        self, body: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Analyze table and list changes (SDK mode).

        Args:
            body: Optional request body with fromBookmark.
        """
        return await self._request("POST", "/ui-bridge/sdk/ai/structured-changes", body)

    async def sdk_enable_change_buffer(self) -> UIBridgeResponse:
        """Enable change buffering (SDK mode)."""
        return await self._request("POST", "/ui-bridge/sdk/ai/change-buffer/enable")

    async def sdk_disable_change_buffer(self) -> UIBridgeResponse:
        """Disable change buffering (SDK mode)."""
        return await self._request("POST", "/ui-bridge/sdk/ai/change-buffer/disable")

    async def sdk_drain_change_buffer(self) -> UIBridgeResponse:
        """Drain the change buffer (SDK mode)."""
        return await self._request("POST", "/ui-bridge/sdk/ai/change-buffer/drain")

    async def sdk_change_buffer_size(self) -> UIBridgeResponse:
        """Get change buffer size and enabled status (SDK mode)."""
        return await self._request("GET", "/ui-bridge/sdk/ai/change-buffer/size")

    # -------------------------------------------------------------------------
    # Agent Mode - Annotated Screenshots
    # -------------------------------------------------------------------------

    async def control_annotated_screenshot(
        self, monitor: int | None = None, runner: bool = False
    ) -> UIBridgeResponse:
        """Get a screenshot of the runner's monitor for annotation.

        Returns screenshot base64, width, and height in one response.

        Args:
            monitor: Monitor index (0-based). Defaults to primary.
            runner: If True, capture the runner's Tauri window instead of monitor.
        """
        params: dict[str, str] | None = None
        if monitor is not None:
            params = params or {}
            params["monitor"] = str(monitor)
        if runner:
            params = params or {}
            params["runner"] = "true"
        return await self._get("/ui-bridge/control/annotated-screenshot", params=params)

    async def control_query_selector(
        self,
        selector: str,
        action: str | None = None,
        index: int | None = None,
    ) -> UIBridgeResponse:
        """Query DOM elements by CSS selector.

        Args:
            selector: CSS selector to query.
            action: Optional action to perform on matched element(s).
            index: Index of matched element to target (0-based).
        """
        body: dict[str, Any] = {"selector": selector}
        if action is not None:
            body["action"] = action
        if index is not None:
            body["index"] = index
        return await self._request(
            "POST", "/ui-bridge/sdk/control/query-selector", body
        )

    async def control_page_evaluate(self, expression: str) -> UIBridgeResponse:
        """Evaluate a JavaScript expression in the webview.

        The expression is validated by a regex on the frontend side.
        Only safe property access expressions are allowed.

        Args:
            expression: JavaScript expression to evaluate.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/control/page-evaluate",
            {"expression": expression},
        )

    async def sdk_screenshot_raw(self, monitor: int | None = None) -> UIBridgeResponse:
        """Get raw screenshot data from the SDK app's monitor.

        Returns screenshot base64, width, and height for annotation.

        Args:
            monitor: Monitor index (0-based). Defaults to primary.
        """
        params: dict[str, str] | None = None
        if monitor is not None:
            params = {"monitor": str(monitor)}
        return await self._request("GET", "/ui-bridge/sdk/screenshot", params=params)

    # -------------------------------------------------------------------------
    # SDK Mode - Media Discovery & Analysis (/ui-bridge/sdk/ai/media/*)
    # -------------------------------------------------------------------------

    async def sdk_media_find(
        self,
        media_type: str | None = None,
        broken_only: bool = False,
        missing_alt_only: bool = False,
        src_pattern: str | None = None,
        oversize_threshold: float | None = None,
    ) -> UIBridgeResponse:
        """Find media elements with optional filters.

        Args:
            media_type: Filter by type ('image', 'video', 'svg', 'canvas', 'picture').
            broken_only: Only return media that failed to load.
            missing_alt_only: Only return images missing alt text.
            src_pattern: Regex pattern to match against source URL.
            oversize_threshold: Filter images where natural/rendered ratio exceeds this.
        """
        body: dict[str, Any] = {"mediaOnly": True}
        if media_type:
            body["mediaType"] = media_type
        if broken_only:
            body["brokenOnly"] = True
        if missing_alt_only:
            body["missingAltOnly"] = True
        if src_pattern:
            body["srcPattern"] = src_pattern
        if oversize_threshold is not None:
            body["oversizeThreshold"] = oversize_threshold
        return await self._request("POST", "/ui-bridge/sdk/ai/media/find", body)

    async def sdk_media_audit_accessibility(self) -> UIBridgeResponse:
        """Run an accessibility audit on media elements.

        Returns images missing alt text, images with generic alt text,
        and decorative images that should have empty alt.
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/media/audit/accessibility"
        )

    async def sdk_media_audit_performance(self) -> UIBridgeResponse:
        """Run a performance audit on media elements.

        Returns oversized images, images with large transfer sizes,
        and below-fold images not using lazy loading.
        """
        return await self._request("POST", "/ui-bridge/sdk/ai/media/audit/performance")

    async def sdk_media_snapshot(
        self,
        element_id: str,
        max_size: int | None = None,
    ) -> UIBridgeResponse:
        """Capture a visual snapshot of a media element as base64 PNG.

        Args:
            element_id: The media element ID to capture.
            max_size: Maximum dimension in pixels (default: 512).
        """
        body: dict[str, Any] = {"elementId": element_id}
        if max_size is not None:
            body["maxSize"] = max_size
        return await self._request("POST", "/ui-bridge/sdk/ai/media/snapshot", body)

    async def sdk_media_compare(
        self,
        snapshot_a: dict[str, Any],
        snapshot_b: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two media snapshots pixel-by-pixel.

        Args:
            snapshot_a: First snapshot from sdk_media_snapshot.
            snapshot_b: Second snapshot from sdk_media_snapshot.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/ai/media/compare",
            {"snapshotA": snapshot_a, "snapshotB": snapshot_b},
        )

    async def sdk_media_analyze(
        self,
        element_id: str,
        max_size: int | None = None,
    ) -> UIBridgeResponse:
        """Capture a media element's visual content + metadata for AI analysis.

        Returns base64 PNG image data and structured context (alt text, src,
        parent context, sibling labels, dimensions) formatted for direct use
        with Claude's vision API.

        Args:
            element_id: The media element ID to analyze.
            max_size: Maximum image dimension in pixels (default: 512).
        """
        body: dict[str, Any] = {"elementId": element_id}
        if max_size is not None:
            body["maxSize"] = max_size
        return await self._request("POST", "/ui-bridge/sdk/ai/media/analyze", body)

    async def sdk_media_analyze_batch(
        self,
        element_ids: list[str],
        max_size: int | None = None,
    ) -> UIBridgeResponse:
        """Capture multiple media elements for comparison.

        Args:
            element_ids: List of media element IDs to analyze.
            max_size: Maximum image dimension per element (default: 512).
        """
        body: dict[str, Any] = {"elementIds": element_ids}
        if max_size is not None:
            body["maxSize"] = max_size
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/media/analyze/batch", body
        )

    async def sdk_media_analyze_page(
        self,
        max_elements: int | None = None,
        max_size: int | None = None,
        include_context: bool = True,
    ) -> UIBridgeResponse:
        """Capture ALL visible media on the page for AI analysis.

        Returns all visible media with context, enabling full-page visual
        audits like "are all images appropriate and loading correctly?"

        Args:
            max_elements: Maximum number of elements to capture (default: 20).
            max_size: Maximum image dimension per element (default: 512).
            include_context: Include parent/sibling context (default: True).
        """
        body: dict[str, Any] = {}
        if max_elements is not None:
            body["maxElements"] = max_elements
        if max_size is not None:
            body["maxSize"] = max_size
        if not include_context:
            body["includeContext"] = False
        return await self._request("POST", "/ui-bridge/sdk/ai/media/analyze/page", body)

    async def _get(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> UIBridgeResponse:
        """Make a GET request (convenience wrapper)."""
        return await self._request("GET", endpoint, params=params, timeout=timeout)
