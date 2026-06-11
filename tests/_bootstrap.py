"""Test bootstrap for running individual test files as scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INNER_PACKAGE = str(ROOT / "analyst_copilot")

if INNER_PACKAGE not in sys.path:
    sys.path.insert(0, INNER_PACKAGE)

os.environ.setdefault("GEMINI_API_KEY", "gemini-test-placeholder")
os.environ.setdefault("DATABASE_URL", "sqlite:///data/test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
