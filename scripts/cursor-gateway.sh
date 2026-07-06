#!/usr/bin/env bash
# Cursor Gateway launcher — supports native Python and OrbStack Docker modes.
#
# Native (default, reads Cursor DB directly on macOS):
#   ./scripts/cursor-gateway.sh native start
#   ./scripts/cursor-gateway.sh native stop
#   ./scripts/cursor-gateway.sh native status
#
# Docker / OrbStack (mounts host Cursor DB into Linux container):
#   ./scripts/cursor-gateway.sh docker up -d --build
#   ./scripts/cursor-gateway.sh docker down
#   ./scripts/cursor-gateway.sh docker logs -f
#
# Shorthand:
#   ./scripts/cursor-gateway.sh start          # native
#   ./scripts/cursor-gateway.sh start --docker # orbstack

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
PID_FILE="${ROOT_DIR}/.cursor-gateway.native.pid"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
MAIN="${ROOT_DIR}/main.py"
DEFAULT_PORT="${CURSOR_SERVER_PORT:-8001}"

usage() {
  cat <<'EOF'
Cursor Gateway — two run modes

  native   Run directly with local .venv (recommended on macOS)
  docker   Run in OrbStack/Docker container

Commands:
  ./scripts/cursor-gateway.sh native start [--port PORT]
  ./scripts/cursor-gateway.sh native stop
  ./scripts/cursor-gateway.sh native status
  ./scripts/cursor-gateway.sh native restart [--port PORT]

  ./scripts/cursor-gateway.sh docker up -d [--build]
  ./scripts/cursor-gateway.sh docker down
  ./scripts/cursor-gateway.sh docker logs [-f]
  ./scripts/cursor-gateway.sh docker ps
  ./scripts/cursor-gateway.sh docker <any docker compose args>

Shorthand:
  ./scripts/cursor-gateway.sh start [--docker] [--port PORT]
  ./scripts/cursor-gateway.sh stop [--docker]
  ./scripts/cursor-gateway.sh status

Notes:
  - Native and Docker both default to port 8001; run only one at a time.
  - Docker mode uses OrbStack context and clears DOCKER_HOST overrides.
EOF
}

require_env_file() {
  if [[ ! -f "${ROOT_DIR}/.env" ]]; then
    echo "Warning: ${ROOT_DIR}/.env not found. Copy from .env.example" >&2
  fi
}

require_venv() {
  if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Error: virtual environment not found at ${ROOT_DIR}/.venv" >&2
    echo "Create it with:" >&2
    echo "  python3.10 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
}

port_in_use() {
  local port="$1"
  lsof -iTCP:"${port}" -sTCP:LISTEN -n -P >/dev/null 2>&1
}

listener_summary() {
  local port="$1"
  lsof -iTCP:"${port}" -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR==2 {print $1, $2}'
}

setup_docker_env() {
  unset DOCKER_HOST
  export DOCKER_CONTEXT="${DOCKER_CONTEXT:-orbstack}"
}

ensure_docker() {
  setup_docker_env
  if ! docker info >/dev/null 2>&1; then
    echo "Error: Cannot connect to Docker (OrbStack)." >&2
    echo "  1. Start OrbStack" >&2
    echo "  2. docker context use orbstack" >&2
    echo "  3. unset DOCKER_HOST  # if it points to Docker Desktop" >&2
    exit 1
  fi
}

docker_compose() {
  ensure_docker
  require_env_file
  cd "${ROOT_DIR}"
  docker compose -f "${COMPOSE_FILE}" "$@"
}

native_start() {
  local port="${DEFAULT_PORT}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port|-p)
        port="$2"
        shift 2
        ;;
      *)
        echo "Unknown option for native start: $1" >&2
        exit 1
        ;;
    esac
  done

  require_env_file
  require_venv

  if port_in_use "${port}"; then
    echo "Error: port ${port} is already in use: $(listener_summary "${port}")" >&2
    echo "Stop the other service first, or choose another port with --port." >&2
    exit 1
  fi

  cd "${ROOT_DIR}"
  nohup "${VENV_PYTHON}" "${MAIN}" --port "${port}" > "${ROOT_DIR}/cursor-gateway.native.log" 2>&1 &
  echo $! > "${PID_FILE}"

  sleep 1
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Error: native gateway failed to start. See cursor-gateway.native.log" >&2
    exit 1
  fi

  echo "Native Cursor Gateway started on http://localhost:${port} (PID $(cat "${PID_FILE}"))"
  echo "Logs: ${ROOT_DIR}/cursor-gateway.native.log"
}

native_stop() {
  local stopped=0

  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      stopped=1
      echo "Stopped native Cursor Gateway (PID ${pid})"
    fi
    rm -f "${PID_FILE}"
  fi

  if port_in_use "${DEFAULT_PORT}"; then
    local summary
    summary="$(listener_summary "${DEFAULT_PORT}")"
    if [[ "${summary}" == Python* ]] || [[ "${summary}" == *python* ]]; then
      local pid
      pid="$(echo "${summary}" | awk '{print $2}')"
      kill "${pid}" 2>/dev/null || true
      stopped=1
      echo "Stopped process on port ${DEFAULT_PORT} (PID ${pid})"
    fi
  fi

  if [[ "${stopped}" -eq 0 ]]; then
    echo "Native Cursor Gateway is not running."
  fi
}

native_status() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Native: running (PID $(cat "${PID_FILE}"), port ${DEFAULT_PORT})"
    return 0
  fi

  if port_in_use "${DEFAULT_PORT}"; then
    echo "Native: port ${DEFAULT_PORT} in use by $(listener_summary "${DEFAULT_PORT}")"
    return 0
  fi

  echo "Native: stopped"
}

docker_status() {
  ensure_docker
  if docker ps --format '{{.Names}}' | grep -qx 'cursor-gateway'; then
    docker ps --filter name=cursor-gateway --format 'Docker: running ({{.Status}}, ports {{.Ports}})'
  else
    echo "Docker: stopped"
  fi
}

cmd_start() {
  local mode="native"
  local args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --docker)
        mode="docker"
        shift
        ;;
      --port|-p)
        args+=("$1" "$2")
        shift 2
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${mode}" == "docker" ]]; then
    docker_compose up -d "${args[@]}"
  else
    native_start "${args[@]}"
  fi
}

cmd_stop() {
  if [[ "${1:-}" == "--docker" ]]; then
    docker_compose down
  else
    native_stop
  fi
}

cmd_status() {
  native_status
  docker_status || true
}

main() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 0
  fi

  case "$1" in
    native)
      shift
      case "${1:-}" in
        start) shift; native_start "$@" ;;
        stop) native_stop ;;
        restart) shift; native_stop; native_start "$@" ;;
        status) native_status ;;
        *) usage; exit 1 ;;
      esac
      ;;
    docker)
      shift
      docker_compose "$@"
      ;;
    start) shift; cmd_start "$@" ;;
    stop) shift; cmd_stop "$@" ;;
    status) cmd_status ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown command: $1" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
