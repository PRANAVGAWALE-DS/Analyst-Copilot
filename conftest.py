"""
conftest.py  (project root)
Adds the inner analyst_copilot/ package to sys.path so that tests can
import modules with their flat names:

    from interfaces import ...
    from validation import ...
    from orchestrator import ...

Place this file at:
    C:\\Users\\victas\\analyst_copilot\\conftest.py   ← ROOT level, not inside tests/

pytest discovers this before any test collection begins, so the path
injection is in place before the first import statement in any test file.
"""

import os
import sys

# Insert the inner package directory so flat imports resolve correctly.
# Before: analyst_copilot/ (root) is on sys.path → `import interfaces` fails
# After:  analyst_copilot/analyst_copilot/ is on sys.path → resolves correctly
_inner_pkg = os.path.join(os.path.dirname(__file__), "analyst_copilot")
if _inner_pkg not in sys.path:
    sys.path.insert(0, _inner_pkg)

# Stub required env vars so modules that read them at import time don't crash.
os.environ.setdefault("GEMINI_API_KEY", "gemini-test-placeholder")
os.environ.setdefault("DATABASE_URL", "sqlite:///data/test.db")
# P3-11 FIX: was "redis://localhost:6379/0".
# build_session_store() checks `if not url` before attempting any connection.
# A non-empty REDIS_URL caused it to always try RedisSessionStore, call
# ping() with a 2-second asyncio.wait_for timeout, and only then fall back
# to InMemorySessionStore — adding a 2s dead wait to every test run on a
# machine without Redis.  An empty string takes the immediate in-memory
# path with zero network overhead.
# Tests that specifically need a live Redis connection set REDIS_URL
# themselves; the CI redis service sets it via the job-level env block.
os.environ.setdefault("REDIS_URL", "")
