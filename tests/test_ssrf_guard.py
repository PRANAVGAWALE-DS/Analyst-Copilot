"""
test_ssrf_guard.py — SEC-3 verification
Confirms ALLOWED_DB_HOSTS fails closed in production.

sys.path requirements:
  "."                — allows `from app import IngestRequest`
  "analyst_copilot"  — allows flat imports inside orchestrator.py:
                        `from interfaces import ...`
                        `from validation import ...`

Run from the project root:
    python tests/test_ssrf_guard.py
"""

import os
import sys

# Both paths required: project root for package imports, inner dir for flat imports
for p in (".", "analyst_copilot"):
    if p not in sys.path:
        sys.path.insert(0, p)

# Set env BEFORE any analyst_copilot module is imported — validators read env at
# class-definition time (Pydantic v2 field validators run on first model import).
os.environ.pop("ALLOWED_DB_HOSTS", None)
os.environ["APP_ENV"] = "production"

from pydantic import ValidationError  # noqa: E402

from app import IngestRequest  # noqa: E402

print("=== SEC-3: ALLOWED_DB_HOSTS fail-closed ===")

# ── Test 1: Production + no ALLOWED_DB_HOSTS → network URL rejected ──────────
try:
    IngestRequest(
        schema_id="test",
        database_url="postgresql://user:pass@internal-db.corp:5432/prod",
        overwrite=False,
    )
    print("  FAIL: accepted network URL with no ALLOWED_DB_HOSTS in production!")
except (ValidationError, ValueError):
    print("  PASS: network URL rejected in production")

# ── Test 2: SQLite local path → always accepted (no hostname to check) ────────
try:
    IngestRequest(
        schema_id="test",
        database_url="sqlite:///./data/test.db",
        overwrite=False,
    )
    print("  PASS: SQLite local path accepted")
except (ValidationError, ValueError) as e:
    print(f"  FAIL: SQLite incorrectly rejected — {e}")

# ── Test 3: ALLOWED_DB_HOSTS set → approved host accepted ────────────────────
os.environ["ALLOWED_DB_HOSTS"] = "approved-db.internal"
os.environ["APP_ENV"] = "production"

for mod in list(sys.modules.keys()):
    if "analyst_copilot" in mod:
        del sys.modules[mod]

from app import IngestRequest as IR2  # noqa: E402

try:
    IR2(
        schema_id="test",
        database_url="postgresql://user:pass@approved-db.internal:5432/prod",
        overwrite=False,
    )
    print("  PASS: whitelisted host accepted when ALLOWED_DB_HOSTS is set")
except (ValidationError, ValueError) as e:
    print(f"  FAIL: whitelisted host incorrectly rejected — {e}")

# ── Test 4: Non-whitelisted host → rejected ───────────────────────────────────
try:
    IR2(
        schema_id="test",
        database_url="postgresql://user:pass@rogue-server.evil:5432/prod",
        overwrite=False,
    )
    print("  FAIL: non-whitelisted host accepted!")
except (ValidationError, ValueError):
    print("  PASS: non-whitelisted host rejected")

# ── Test 5: Development mode → any network URL accepted ───────────────────────
os.environ.pop("ALLOWED_DB_HOSTS", None)
os.environ["APP_ENV"] = "development"

for mod in list(sys.modules.keys()):
    if "analyst_copilot" in mod:
        del sys.modules[mod]

from app import IngestRequest as IR3  # noqa: E402

try:
    IR3(
        schema_id="test",
        database_url="postgresql://localhost:5432/dev_db",
        overwrite=False,
    )
    print("  PASS: dev mode accepts network URL")
except (ValidationError, ValueError) as e:
    print(f"  FAIL: dev mode incorrectly rejected — {e}")

# Cleanup
os.environ.pop("ALLOWED_DB_HOSTS", None)
os.environ["APP_ENV"] = "development"
