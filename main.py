"""AI Avatar Video Generator - FastAPI application entry point.

Wires together the REST and WebSocket routers, configures CORS for the
React frontend, and manages the Redis connection lifecycle.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_task_manager
from api.routes import tasks, ws
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upload size limit (bytes) matching the largest validator constraint
# (video appearance asset <= 200MB). Enforced by the ASGI server / proxy.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Allowed frontend origins for local development.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup: ensure output directories exist and warm the Redis connection.
    settings.ensure_directories()
    try:
        tm = get_task_manager()
        # Touch the Redis client so connection issues surface early.
        tm._redis.ping()
        logger.info("Connected to Redis at %s", settings.redis_url)
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("Redis not reachable at startup: %s", e)

    yield

    # Shutdown: close Redis connection pool if present.
    try:
        get_task_manager()._redis.close()
    except Exception:  # pragma: no cover
        pass


app = FastAPI(
    title="AI Avatar Video Generator",
    version="1.0.0",
    description="本地零費用 AI 虛擬人物影片生成系統 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)
app.include_router(ws.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    """Lightweight liveness probe."""
    return {"status": "ok"}
