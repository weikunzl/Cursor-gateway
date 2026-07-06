"""
Cursor Gateway - OpenAI/Anthropic-compatible interface for Cursor API.

Application entry point. Creates FastAPI app and connects routes.

Usage:
    python main.py
    python main.py --port 8001
    python main.py --host 127.0.0.1 --port 8001
"""

import argparse
import logging
import sys
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from cursor.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    CURSOR_ACCESS_TOKEN,
    CURSOR_DB_PATH,
    CURSOR_API_HOST,
    CURSOR_MODELS_RPC,
    PROXY_API_KEY,
    LOG_LEVEL,
    SERVER_HOST,
    SERVER_PORT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    STREAMING_READ_TIMEOUT,
    MODEL_ALIASES,
    HIDDEN_FROM_LIST,
    FALLBACK_MODELS,
    VPN_PROXY_URL,
)
from cursor.auth import CursorAuthManager
from cursor.cache import ModelInfoCache
from cursor.model_resolver import ModelResolver
from cursor.routes_openai import router as openai_router
from cursor.routes_anthropic import router as anthropic_router
from cursor.exceptions import validation_exception_handler


# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


class InterceptHandler(logging.Handler):
    """Intercepts logs from standard logging and redirects to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type is not None and exc_type.__name__ in (
                "CancelledError", "KeyboardInterrupt",
            ):
                return

        msg = record.getMessage()
        if any(exc in msg for exc in ("CancelledError", "KeyboardInterrupt")):
            return

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging_intercept():
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


setup_logging_intercept()


# --- VPN/Proxy Configuration ---
if VPN_PROXY_URL:
    proxy_url = VPN_PROXY_URL if "://" in VPN_PROXY_URL else f"http://{VPN_PROXY_URL}"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    no_proxy = os.environ.get("NO_PROXY", "")
    local = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = f"{no_proxy},{local}" if no_proxy else local
    logger.info(f"Proxy configured: {proxy_url}")


# --- Configuration Validation ---
def validate_configuration() -> None:
    """Validates that required configuration is present."""
    has_token = bool(CURSOR_ACCESS_TOKEN)
    has_db = bool(CURSOR_DB_PATH)

    if has_db:
        from pathlib import Path
        if not Path(CURSOR_DB_PATH).expanduser().exists():
            has_db = False
            logger.warning(f"Cursor database not found: {CURSOR_DB_PATH}")

    if not has_token and not has_db:
        logger.error("")
        logger.error("=" * 60)
        logger.error("  CONFIGURATION ERROR")
        logger.error("=" * 60)
        logger.error("  No Cursor credentials configured!")
        logger.error("")
        logger.error("  Options:")
        logger.error("    1. Log in to Cursor IDE (auto-detected)")
        logger.error("    2. Set CURSOR_ACCESS_TOKEN in .env.cursor")
        logger.error("    3. Set CURSOR_DB_FILE to your state.vscdb path")
        logger.error("")
        logger.error("  See .env.cursor.example for details.")
        logger.error("=" * 60)
        logger.error("")
        sys.exit(1)


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cursor Gateway...")

    # Create shared HTTP/2 client
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(
        connect=30.0,
        read=STREAMING_READ_TIMEOUT,
        write=30.0,
        pool=30.0,
    )
    app.state.http_client = httpx.AsyncClient(
        http2=True,
        limits=limits,
        timeout=timeout,
        follow_redirects=True,
    )
    logger.info("Shared HTTP/2 client created")

    # Create AuthManager
    app.state.auth_manager = CursorAuthManager(
        db_path=CURSOR_DB_PATH if CURSOR_DB_PATH else None,
    )

    # Create model cache
    app.state.model_cache = ModelInfoCache()

    # Try to fetch models from Cursor API
    logger.info("Loading models from Cursor API...")
    try:
        from cursor.utils import get_cursor_headers
        from cursor.protobuf import wrap_connect_envelope

        headers = get_cursor_headers(app.state.auth_manager)
        url = f"{CURSOR_API_HOST}{CURSOR_MODELS_RPC}"

        # AvailableModels is a simple RPC - send empty request
        empty_request = wrap_connect_envelope(b"")

        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            response = await client.post(url, content=empty_request, headers=headers)

            if response.status_code == 200:
                # Try to parse the response - it may be protobuf or JSON
                data = response.content
                # For now, use fallback models since parsing the protobuf
                # model list response requires more reverse engineering
                logger.info("Connected to Cursor API successfully")
                await app.state.model_cache.update(FALLBACK_MODELS)
                logger.info(f"Using {len(FALLBACK_MODELS)} configured models")
            else:
                raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not fetch models from Cursor API: {e}")
        logger.info("Using fallback model list")
        await app.state.model_cache.update(FALLBACK_MODELS)

    # Create model resolver
    app.state.model_resolver = ModelResolver(
        cache=app.state.model_cache,
        aliases=MODEL_ALIASES,
        hidden_from_list=HIDDEN_FROM_LIST,
    )
    logger.info(f"Model resolver initialized with {app.state.model_cache.size} models")

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        await app.state.http_client.aclose()
    except Exception as e:
        logger.warning(f"Error closing HTTP client: {e}")


# --- FastAPI Application ---
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(openai_router)
app.include_router(anthropic_router)


# --- Uvicorn log config ---
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "default": {"class": "main.InterceptHandler"},
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_TITLE} - {APP_DESCRIPTION}")
    parser.add_argument("-H", "--host", type=str, default=None, help=f"Server host (default: {DEFAULT_SERVER_HOST})")
    parser.add_argument("-p", "--port", type=int, default=None, help=f"Server port (default: {DEFAULT_SERVER_PORT})")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {APP_VERSION}")
    return parser.parse_args()


def print_startup_banner(host: str, port: int) -> None:
    GREEN = "\033[92m"
    WHITE = "\033[97m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    display_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"

    print()
    print(f"  {WHITE}{BOLD}🖱️  {APP_TITLE} v{APP_VERSION}{RESET}")
    print()
    print(f"  {WHITE}Server running at:{RESET}")
    print(f"  {GREEN}{BOLD}➜  {url}{RESET}")
    print()
    print(f"  {DIM}API Docs:      {url}/docs{RESET}")
    print(f"  {DIM}Health Check:  {url}/health{RESET}")
    print()
    print(f"  {DIM}{'─' * 48}{RESET}")
    print(f"  {WHITE}OpenAI API:     {url}/v1/chat/completions{RESET}")
    print(f"  {WHITE}Anthropic API:  {url}/v1/messages{RESET}")
    print(f"  {DIM}{'─' * 48}{RESET}")
    print()


if __name__ == "__main__":
    import uvicorn

    validate_configuration()

    args = parse_cli_args()

    final_host = args.host if args.host else SERVER_HOST
    final_port = args.port if args.port else SERVER_PORT

    print_startup_banner(final_host, final_port)

    logger.info(f"Starting Uvicorn on {final_host}:{final_port}...")

    uvicorn.run(
        "main:app",
        host=final_host,
        port=final_port,
        log_config=UVICORN_LOG_CONFIG,
    )
