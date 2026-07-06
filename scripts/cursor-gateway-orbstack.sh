#!/usr/bin/env bash
# Backward-compatible wrapper for Docker/OrbStack mode.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cursor-gateway.sh" docker "$@"
