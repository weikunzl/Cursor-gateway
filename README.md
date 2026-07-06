<div align="center">

# Cursor Gateway

**OpenAI and Anthropic compatible proxy for the Cursor API**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

Use Cursor models from Claude Code, OpenCode, Codex, Cline, Roo Code, and other OpenAI or Anthropic compatible clients.

[Features](#features) • [Quick Start](#quick-start) • [Configuration](#configuration) • [Docker](#docker)

</div>

---

## Features

| Feature | Description |
|---------|-------------|
| OpenAI-compatible API | `/v1/chat/completions` with streaming |
| Anthropic-compatible API | `/v1/messages` with streaming |
| Auto credential detection | Reads Cursor `state.vscdb` on macOS/Linux/Windows |
| Model aliases | Maps Claude Code model IDs to Cursor models |
| Tool calling | Supports function calling and inline tool extraction |
| Thinking support | Handles Cursor thinking blocks for compatible clients |
| VPN/Proxy support | HTTP/SOCKS5 proxy for restricted networks |

## Quick Start

### Prerequisites

- Python 3.10+
- [Cursor](https://cursor.com/) installed and logged in

### Installation

```bash
git clone https://github.com/weikunzl/Cursor-gateway.git
cd cursor-gateway

python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# Edit .env and set PROXY_API_KEY

./scripts/cursor-gateway.sh start
```

The server starts on `http://localhost:8001` by default.

### Connect a Client

Use your gateway URL and `PROXY_API_KEY` as the API key:

```bash
# OpenAI style
Authorization: Bearer <PROXY_API_KEY>

# Anthropic style
x-api-key: <PROXY_API_KEY>
```

Example endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`

## Configuration

Copy `.env.example` to `.env` and configure:

```env
PROXY_API_KEY="my-super-secret-password-123"
```

### Credentials

By default the gateway reads Cursor credentials from the local database:

| Platform | Default path |
|----------|--------------|
| macOS | `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb` |
| Linux | `~/.config/Cursor/User/globalStorage/state.vscdb` |
| Windows | `%APPDATA%/Cursor/User/globalStorage/state.vscdb` |

Override with:

```env
CURSOR_DB_FILE="/path/to/state.vscdb"
```

Or provide a token directly:

```env
CURSOR_ACCESS_TOKEN="your_jwt_token"
CURSOR_MACHINE_ID="your_machine_id"
```

### Optional Settings

```env
CURSOR_SERVER_HOST="0.0.0.0"
CURSOR_SERVER_PORT=8001
CURSOR_API_HOST="https://api2.cursor.sh"
VPN_PROXY_URL="http://127.0.0.1:7890"
LOG_LEVEL=INFO
```

## Docker

```bash
cp .env.example .env
./scripts/cursor-gateway.sh docker up -d --build
./scripts/cursor-gateway.sh docker logs -f
```

Docker mode mounts your host Cursor database into the container. On macOS, use OrbStack and ensure `DOCKER_CONTEXT=orbstack`.

## Development

```bash
pytest -v
pytest tests/unit/test_cursor_model_resolver.py -v
```

See `AGENTS.md` for contributor guidance.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
