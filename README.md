# UI Bridge MCP

MCP (Model Context Protocol) server for UI Bridge - enables AI to inspect and interact with UI elements.

## Overview

This MCP server provides tools for:

- **Inspecting UI elements** - Get element positions, bounds, visibility, and state
- **Interacting with elements** - Click, type, focus, hover
- **Two modes of operation**:
  - **Control mode**: Interact with the qontinui-runner's own Tauri webview
  - **External mode**: Interact with external browser tabs via Chrome extension

## Installation

```bash
# Using pip
pip install ui-bridge-mcp

# Using poetry
poetry add ui-bridge-mcp
```

## Prerequisites

The MCP server requires the **qontinui-runner** to be running on port 9876.

For external browser tab access, you also need the Chrome extension connected.

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

### External Mode (Browser Tabs)

| Tool | Description |
|------|-------------|
| `extension_status` | Check Chrome extension connection |
| `extension_list_tabs` | List available browser tabs |
| `extension_select_tab` | Select a tab for subsequent operations |
| `extension_get_elements` | Get all elements from selected tab |
| `extension_click` | Click element by CSS selector |
| `extension_type` | Type into element by CSS selector |
| `extension_screenshot` | Capture screenshot of current tab |

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

### Inspect External Website

```
AI: Let me check the login form on localhost:3001.

1. Check extension is connected:
   extension_status

2. List browser tabs:
   extension_list_tabs

3. Select the tab with localhost:3001:
   extension_select_tab tab_id=123456

4. Get page elements:
   extension_get_elements

5. Type into the email field:
   extension_type selector="input[name='email']" text="test@example.com"
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
    ↓ MCP Protocol
ui-bridge-mcp (this server)
    ↓ HTTP
qontinui-runner (port 9876)
    ├── /ui-bridge/control/* → Runner's Tauri webview
    └── /extension/* → Chrome extension → External browser tabs
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
