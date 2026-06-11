"""
main.py — Application entry point
Data Analyst Copilot

Usage:
    python main.py                          # development
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2   # production
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# orchestrator.py, validation.py, and all other modules inside the
# analyst_copilot/ inner package use flat imports:
#   from interfaces import ...
#   from validation import ...
# They resolve only when the inner package directory is on sys.path.
# conftest.py does this for tests; this block does it for the server.
_inner_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyst_copilot")
if _inner_pkg not in sys.path:
    sys.path.insert(0, _inner_pkg)

# ── Load .env before any project import that reads env vars ──────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

# ── Application ───────────────────────────────────────────────────────────────
from app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    # reload is only enabled in development.
    # Set APP_ENV=development (or UVICORN_RELOAD=true) to activate.
    dev_mode = os.getenv("APP_ENV", "production").lower() == "development"
    hot_reload = dev_mode or os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=hot_reload,
        log_level="debug" if dev_mode else "info",
    )
