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
    - External mode: External browser tabs via Chrome extension (/extension/*)
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
    ) -> UIBridgeResponse:
        """Make an HTTP request to the UI Bridge API."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = await client.get(url, timeout=timeout)
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
    # External Mode - Browser Tabs via Chrome Extension (/extension/*)
    # -------------------------------------------------------------------------

    async def extension_status(self) -> UIBridgeResponse:
        """Check Chrome extension connection status."""
        return await self._request("GET", "/extension/status")

    async def extension_list_tabs(self) -> UIBridgeResponse:
        """List available browser tabs from the Chrome extension."""
        return await self._request(
            "POST",
            "/extension/command",
            {"action": "listTabs"},
        )

    async def extension_select_tab(self, tab_id: int) -> UIBridgeResponse:
        """Select a browser tab for subsequent operations.

        Args:
            tab_id: The tab ID from extension_list_tabs().
        """
        return await self._request(
            "POST",
            "/extension/command",
            {"action": "selectTab", "params": {"tabId": tab_id}},
        )

    async def extension_get_active_tab(self) -> UIBridgeResponse:
        """Get the currently active browser tab."""
        return await self._request(
            "POST",
            "/extension/command",
            {"action": "getActiveTab"},
        )

    async def extension_get_elements(self, timeout_secs: int = 30) -> UIBridgeResponse:
        """Get all elements from the selected browser tab.

        Args:
            timeout_secs: Timeout for element discovery.

        Returns:
            List of elements with their properties, actions, and bounds.
        """
        return await self._request(
            "POST",
            "/extension/command",
            {"action": "getElements", "timeout_secs": timeout_secs},
            timeout=float(timeout_secs) + 10.0,
        )

    async def extension_click(
        self, selector: str, timeout_secs: int = 10
    ) -> UIBridgeResponse:
        """Click an element in the browser tab.

        Args:
            selector: CSS selector for the element.
            timeout_secs: Timeout for finding the element.
        """
        return await self._request(
            "POST",
            "/extension/command",
            {
                "action": "click",
                "params": {"selector": selector},
                "timeout_secs": timeout_secs,
            },
        )

    async def extension_type(
        self, selector: str, text: str, timeout_secs: int = 10
    ) -> UIBridgeResponse:
        """Type text into an element in the browser tab.

        Args:
            selector: CSS selector for the element.
            text: Text to type.
            timeout_secs: Timeout for finding the element.
        """
        return await self._request(
            "POST",
            "/extension/command",
            {
                "action": "type",
                "params": {"selector": selector, "text": text},
                "timeout_secs": timeout_secs,
            },
        )

    async def extension_screenshot(self) -> UIBridgeResponse:
        """Take a screenshot of the current browser tab.

        Returns:
            Base64-encoded PNG image.
        """
        return await self._request(
            "POST",
            "/extension/command",
            {"action": "screenshot"},
        )
