# UI Bridge MCP

MCP (Model Context Protocol) server for UI Bridge - enables AI to inspect and interact with UI elements.

## Overview

This MCP server provides tools for:

- **Inspecting UI elements** - Get element positions, bounds, visibility, and state
- **Interacting with elements** - Click, type, focus, hover
- **Two modes of operation**:
  - **Control mode**: Interact with the qontinui-runner's own Tauri webview
  - **SDK mode**: Interact with external apps via the UI Bridge SDK

## Installation

```bash
# Using pip
pip install ui-bridge-mcp

# Using poetry
poetry add ui-bridge-mcp
```

## Prerequisites

The MCP server requires the **qontinui-runner** to be running on port 9876.

For external app access, the target application must have the UI Bridge SDK integrated.

## Configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ui-bridge": {
      "command": "ui-bridge-mcp"
    }
  }
}
```

### Claude Code

Add to your MCP settings:

```json
{
  "mcpServers": {
    "ui-bridge": {
      "command": "ui-bridge-mcp"
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QONTINUI_RUNNER_HOST` | `localhost` (or WSL host) | Runner hostname |
| `QONTINUI_RUNNER_PORT` | `9876` | Runner port |

## Available Tools

### Health Check

| Tool | Description |
|------|-------------|
| `ui_health` | Check if qontinui-runner is accessible |

### Control Mode (Runner's Own UI)

| Tool | Description |
|------|-------------|
| `ui_snapshot` | Get complete UI state with all elements |
| `ui_discover` | Force element discovery/registration |
| `ui_get_element` | Get detailed info for a specific element |
| `ui_click` | Click an element by ID |
| `ui_type` | Type text into an input element |
| `ui_focus` | Focus an element |

### SDK Mode (External Apps)

| Tool | Description |
|------|-------------|
| `sdk_connect` | Connect to an SDK-integrated app by URL |
| `sdk_disconnect` | Disconnect from the current app |
| `sdk_status` | Check SDK connection status |
| `sdk_elements` | Get all elements from the connected app |
| `sdk_snapshot` | Get a full UI snapshot from the connected app |
| `sdk_click` | Click an element by ID |
| `sdk_type` | Type into an element by ID |

> **Note:** The MCP server also includes legacy `extension_*` tools for browser tab access via a Chrome extension. These are deprecated and will be removed in a future release. Use the SDK tools instead.

## Usage Examples

### Inspect Runner UI

```
AI: Let me check what's on the runner's Settings page.

1. First, get a snapshot of the UI:
   ui_snapshot

2. Click the Settings button:
   ui_click element_id="sidebar-nav-item-settings"

3. Get another snapshot to see the new page:
   ui_snapshot
```

### Inspect an SDK-Integrated App

```
AI: Let me check the login form on localhost:3000.

1. Connect to the app:
   sdk_connect url="http://localhost:3000"

2. Get all elements:
   sdk_elements

3. Type into the email field:
   sdk_type element_id="email-input" text="test@example.com"

4. Click the submit button:
   sdk_click element_id="login-button"
```

## Element IDs

Elements in the runner's UI have `data-ui-id` attributes that follow patterns like:

- `sidebar-nav-item-{name}` - Sidebar navigation items
- `button-{action}` - Action buttons
- `input-{field}` - Input fields
- `dialog-{name}` - Dialog components

Use `ui_snapshot` to discover all available element IDs.

## Architecture

```
Claude/AI
    |  MCP Protocol
ui-bridge-mcp (this server)
    |  HTTP
qontinui-runner (port 9876)
    |-- /ui-bridge/control/* --> Runner's Tauri webview
    |-- /ui-bridge/sdk/*     --> SDK-integrated apps (direct HTTP)
```

## Development

```bash
# Install dependencies
poetry install

# Run linting
poetry run ruff check .
poetry run mypy src

# Format code
poetry run black src
poetry run isort src
```

## License

MIT
