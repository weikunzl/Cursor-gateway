# Agent Reference: Operations and Configuration

This file expands the command, configuration, Docker, and API notes summarized in `AGENTS.md`.

## Running the Server

```bash
python main.py
python main.py --port 9000
python main.py --host 127.0.0.1 --port 9000
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Dependencies

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Main runtime and test dependencies include:

- fastapi.
- uvicorn.
- httpx.
- loguru.
- requests.
- python-dotenv.
- tiktoken.
- pytest.
- pytest-asyncio.
- hypothesis.

## Core Environment Variables

Configuration is usually loaded from `.env`. See `.env.example` for the complete template.

```bash
PROXY_API_KEY="my-super-secret-password-123"

KIRO_CREDS_FILE="~/.aws/sso/cache/kiro-auth-token.json"
REFRESH_TOKEN="your_refresh_token"
KIRO_CLI_DB_FILE="~/.local/share/kiro-cli/data.sqlite3"

PROFILE_ARN="arn:aws:codewhisperer:us-east-1:..."
KIRO_REGION="us-east-1"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"
VPN_PROXY_URL="http://127.0.0.1:7890"
DEBUG_MODE="off"
```

## Configuration Priority

1. CLI arguments.
2. Environment variables.
3. Built-in defaults.

Example: `python main.py --port 9000` overrides `SERVER_PORT`.

## Docker

Build and run:

```bash
docker build -t kiro-gateway .
docker run -d \
  -p 8000:8000 \
  -e PROXY_API_KEY="your-secret-key" \
  -e REFRESH_TOKEN="your-refresh-token" \
  --name kiro-gateway \
  kiro-gateway
```

Docker Compose:

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
docker-compose up -d --build
docker-compose --env-file .env.production up -d
```

Credential mounts:

```bash
docker run -d \
  -p 8000:8000 \
  -v ~/.aws/sso/cache:/home/kiro/.aws/sso/cache:ro \
  -e KIRO_CREDS_FILE=/home/kiro/.aws/sso/cache/kiro-auth-token.json \
  -e PROXY_API_KEY="your-secret-key" \
  --name kiro-gateway \
  kiro-gateway

docker run -d \
  -p 8000:8000 \
  -v ~/.local/share/kiro-cli:/home/kiro/.local/share/kiro-cli:ro \
  -e KIRO_CLI_DB_FILE=/home/kiro/.local/share/kiro-cli/data.sqlite3 \
  -e PROXY_API_KEY="your-secret-key" \
  --name kiro-gateway \
  kiro-gateway
```

Docker expectations:

- non-root `kiro` user.
- `/health` health check.
- volume mounts for credentials and debug logs.
- restart support.
- all authentication methods supported.

## CI/CD

The Docker workflow is in `.github/workflows/docker.yml` when present.

Expected pipeline behavior:

- run tests before Docker build.
- test Docker image health checks.
- publish to GitHub Container Registry on main.
- generate coverage reports where configured.

## API Endpoints

OpenAI-compatible:

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Anthropic-compatible:

- `POST /v1/messages`

## API Authentication

OpenAI style:

```bash
Authorization: Bearer {PROXY_API_KEY}
```

Anthropic style:

```bash
x-api-key: {PROXY_API_KEY}
```

## Debug Logging

`DEBUG_MODE` supports:

- `off`: disabled, recommended for normal production.
- `errors`: save failed request logs, recommended for troubleshooting.
- `all`: save every request, useful for local development only.

Debug logs are saved under `debug_logs/`.

## Proxy Support

Use `VPN_PROXY_URL` for restricted networks or corporate proxies:

```bash
VPN_PROXY_URL="http://127.0.0.1:7890"
VPN_PROXY_URL="socks5://127.0.0.1:1080"
VPN_PROXY_URL="http://user:pass@proxy:8080"
```
