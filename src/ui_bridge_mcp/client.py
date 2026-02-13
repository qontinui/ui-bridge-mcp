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
            element_id: The element's data-ui-id.

        Returns:
            Element details including bounds, state, actions, etc.
        """
        return await self._request("GET", f"/ui-bridge/control/element/{element_id}")

    async def control_click(self, element_id: str) -> UIBridgeResponse:
        """Click an element in the runner's webview.

        Args:
            element_id: The element's data-ui-id.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "click"},
        )

    async def control_type(self, element_id: str, text: str) -> UIBridgeResponse:
        """Type text into an element in the runner's webview.

        Args:
            element_id: The element's data-ui-id.
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
            element_id: The element's data-ui-id.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "focus"},
        )

    async def control_hover(self, element_id: str) -> UIBridgeResponse:
        """Hover over an element in the runner's webview.

        Args:
            element_id: The element's data-ui-id.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "hover"},
        )

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
        """Get details for a specific element by its data-ui-id.

        Args:
            element_id: The element's data-ui-id.
        """
        return await self._request("GET", f"/ui-bridge/sdk/element/{element_id}")

    async def sdk_element_action(
        self, element_id: str, action: str, params: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Execute an action on an element.

        Args:
            element_id: The element's data-ui-id.
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
