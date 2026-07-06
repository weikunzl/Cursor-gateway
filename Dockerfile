# Cursor Gateway - Docker Image (OrbStack / Docker compatible)

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CURSOR_SERVER_HOST=0.0.0.0 \
    CURSOR_SERVER_PORT=8001

RUN groupadd -r cursor && useradd -r -g cursor cursor

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=cursor:cursor cursor/ cursor/
COPY --chown=cursor:cursor main.py .

RUN mkdir -p debug_logs && chown -R cursor:cursor debug_logs

USER cursor

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=5)"

CMD ["python", "main.py"]
