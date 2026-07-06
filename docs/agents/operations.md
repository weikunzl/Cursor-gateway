# Agent Reference: Operations and Configuration

## Running the Server

```bash
python main.py
python main.py --port 8001
./scripts/cursor-gateway.sh start
./scripts/cursor-gateway.sh docker up -d --build
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Core Environment Variables

```bash
PROXY_API_KEY="my-super-secret-password-123"
CURSOR_DB_FILE="/path/to/state.vscdb"
CURSOR_ACCESS_TOKEN="your_jwt_token"
CURSOR_MACHINE_ID="your_machine_id"
CURSOR_SERVER_HOST="0.0.0.0"
CURSOR_SERVER_PORT=8001
CURSOR_API_HOST="https://api2.cursor.sh"
VPN_PROXY_URL="http://127.0.0.1:7890"
LOG_LEVEL=INFO
```

## API Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`

## API Authentication

OpenAI style: `Authorization: Bearer {PROXY_API_KEY}`

Anthropic style: `x-api-key: {PROXY_API_KEY}`

## Docker

```bash
cp .env.example .env
docker compose up -d --build
```

Docker mounts the host Cursor database read-only and exposes port `8001` by default.
