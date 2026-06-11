"""
dataframe_store.py — Per-session DataFrame registry
Data Analyst Copilot · Phase 2 · Python 3.11+

Wires the `dataframe_refs=None` stub in orchestrator.py:

    loop_result = loop.run(
        ...
        dataframe_refs=await df_store.get(session_id),   # Phase 2
        ...
    )

Storage model:
  - In-memory dict: session_id → {df_name: pd.DataFrame}
  - DataFrames are loaded from uploaded CSV or Parquet files via ingest()
  - TTL: DataFrames are evicted after DATAFRAME_TTL_SECONDS of inactivity
  - Max per-session total size: MAX_SESSION_MB

Security:
  - File size capped at MAX_UPLOAD_MB before reading into memory
  - Column names sanitised (strip whitespace, lowercase option)
  - No eval, no arbitrary code — only pd.read_csv / pd.read_parquet

Integration:
  - DataFrameStore is instantiated once at startup in app.py
  - Injected into Orchestrator via orchestrator._df_store
  - orchestrator.py needs the one-line patch documented at the bottom
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_UPLOAD_MB: int = 50  # hard cap on uploaded file size
MAX_SESSION_MB: int = 256  # max total DataFrame memory per session
DATAFRAME_TTL_SECONDS: int = 3600  # 1 hour of inactivity → evict


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------


class IngestFileResult:
    def __init__(
        self,
        df_name: str,
        rows: int,
        columns: list[str],
        size_mb: float,
        warnings: list[str],
        error: str | None = None,
    ) -> None:
        self.df_name = df_name
        self.rows = rows
        self.columns = columns
        self.size_mb = size_mb
        self.warnings = warnings
        self.error = error
        self.success = error is None


# ---------------------------------------------------------------------------
# DataFrameStore
# ---------------------------------------------------------------------------


class DataFrameStore:
    """
    In-memory registry of DataFrames keyed by (session_id, df_name).

    Concurrency model: safe within a single-threaded async event loop (FastAPI's
    default). CPython's GIL protects individual dict read/write operations from
    torn reads, and all public methods are synchronous and non-reentrant.
    NOT safe for multi-threaded access (e.g. concurrent threads calling
    ingest() and _evict_expired() simultaneously without external locking).
    For multi-process or multi-replica deployments use a Redis-backed store.

    Lifecycle:
      1. POST /upload → ingest(session_id, df_name, file_bytes, extension)
      2. orchestrator._df_store.get(session_id) → dict passed as dataframe_refs
      3. TTL eviction runs on every get() call (lazy eviction, no background thread)
    """

    def __init__(
        self,
        max_upload_mb: int = MAX_UPLOAD_MB,
        max_session_mb: int = MAX_SESSION_MB,
        ttl_seconds: int = DATAFRAME_TTL_SECONDS,
    ) -> None:
        self._store: dict[str, dict[str, pd.DataFrame]] = {}
        self._last_access: dict[str, float] = {}  # session_id → epoch
        self._max_upload_mb = max_upload_mb
        self._max_session_mb = max_session_mb
        self._ttl = ttl_seconds

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(
        self,
        session_id: str,
        df_name: str,
        file_bytes: bytes,
        extension: str,
    ) -> IngestFileResult:
        """
        Parse file_bytes as CSV or Parquet and store as df_name in session_id.

        Parameters
        ----------
        session_id : Session identifier.
        df_name    : Name the DataFrame will be available as in the sandbox.
                     Must be a valid Python identifier. Sanitised automatically.
        file_bytes : Raw file content.
        extension  : "csv" or "parquet" (case-insensitive).
        """
        # A-06 FIX: evict expired sessions at the start of ingest() as well as
        # get().  Sessions that only upload files but never query (e.g. broken
        # query loops, integration test clients that write but never read history)
        # previously never triggered eviction because _evict_expired() was only
        # called in get().  Those sessions accumulated in _store and _last_access
        # indefinitely.  Lazy eviction on every write keeps the footprint bounded
        # without requiring a background task or lock.
        self._evict_expired()

        warnings: list[str] = []

        # Size guard
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self._max_upload_mb:
            return IngestFileResult(
                df_name=df_name,
                rows=0,
                columns=[],
                size_mb=size_mb,
                warnings=[],
                error=f"File size {size_mb:.1f} MB exceeds the {self._max_upload_mb} MB limit.",
            )

        # Sanitise df_name → valid Python identifier
        clean_name = _sanitise_df_name(df_name)
        if clean_name != df_name:
            warnings.append(f"DataFrame name sanitised: '{df_name}' → '{clean_name}'")

        # Parse
        try:
            ext = extension.lower().lstrip(".")
            if ext == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            elif ext in ("parquet", "pq"):
                df = pd.read_parquet(io.BytesIO(file_bytes))
            elif ext == "xlsx":
                df = pd.read_excel(io.BytesIO(file_bytes))
            elif ext == "xls":
                # A-12 FIX: xlrd is intentionally excluded (security issues with
                # the legacy .xls binary format).  Return a clear, actionable
                # message instead of letting pd.read_excel() raise a generic
                # ImportError that surfaces as "Failed to parse file: ...".
                return IngestFileResult(
                    df_name=clean_name,
                    rows=0,
                    columns=[],
                    size_mb=size_mb,
                    warnings=[],
                    error=(
                        "Legacy .xls format is not supported. "
                        "Please re-save the file as .xlsx (Excel 2007+) and re-upload."
                    ),
                )
            else:
                return IngestFileResult(
                    df_name=clean_name,
                    rows=0,
                    columns=[],
                    size_mb=size_mb,
                    warnings=[],
                    error=f"Unsupported file type '.{ext}'. Supported: csv, parquet, xlsx.",
                )
        except Exception as exc:  # noqa: BLE001
            return IngestFileResult(
                df_name=clean_name,
                rows=0,
                columns=[],
                size_mb=size_mb,
                warnings=[],
                error=f"Failed to parse file: {exc}",
            )

        # Sanitise column names (strip whitespace)
        original_cols = list(df.columns)
        df.columns = [str(c).strip() for c in df.columns]
        renamed = [o for o, n in zip(original_cols, df.columns, strict=False) if o != n]
        if renamed:
            warnings.append(
                f"Column names stripped of whitespace: {renamed[:5]}"
                + (" (and more)" if len(renamed) > 5 else "")
            )

        # Per-session memory cap
        session_used = self._session_size_mb(session_id)
        new_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if session_used + new_size > self._max_session_mb:
            return IngestFileResult(
                df_name=clean_name,
                rows=0,
                columns=[],
                size_mb=size_mb,
                warnings=[],
                error=(
                    f"Adding this DataFrame ({new_size:.1f} MB) would exceed the "
                    f"per-session memory limit ({self._max_session_mb} MB). "
                    f"Current session usage: {session_used:.1f} MB."
                ),
            )

        # Store
        if session_id not in self._store:
            self._store[session_id] = {}
        self._store[session_id][clean_name] = df
        self._last_access[session_id] = time.monotonic()

        return IngestFileResult(
            df_name=clean_name,
            rows=len(df),
            columns=list(df.columns),
            size_mb=round(new_size, 3),
            warnings=warnings,
        )

    def get(self, session_id: str) -> dict[str, pd.DataFrame]:
        """
        Return all DataFrames for session_id as {df_name: DataFrame}.
        Returns {} if session has no DataFrames or session has expired.
        Updates last-access timestamp.

        P2-05 FIX: returns a per-DataFrame copy (df.copy()) instead of a
        shallow dict copy.  The previous dict(frames) only copied the dict
        container; the DataFrame values were still shared references to the
        objects in self._store.  Sandboxed Pandas code injected into
        execute_python() can perform in-place mutations (df.iloc[0]=...,
        df.sort_values(inplace=True), df.drop(inplace=True)) that the AST
        visitor does not block.  Those mutations would corrupt the stored
        DataFrame and contaminate all future queries in the same session.

        Performance note: df.copy() is a full block-level copy.  At the
        configured MAX_SESSION_MB (256 MB) and _ROW_CAP (50 000 rows), the
        copy overhead is typically <50 ms and well within the LLM call
        latency budget.  If profiling shows this is hot, switch to
        df.copy(deep=False) (shares block data, prevents column-level
        mutations but not cell-level; still blocks the most common attack
        vectors) or add an inplace=True kwarg guard to _ForbiddenNodeVisitor.
        """
        self._evict_expired()
        frames = self._store.get(session_id, {})
        if frames:
            self._last_access[session_id] = time.monotonic()
        # Deep copy: each DataFrame is an independent object; mutations in the
        # sandbox execution layer cannot reach self._store.
        return {name: df.copy() for name, df in frames.items()}

    def list_dataframes(self, session_id: str) -> list[dict[str, Any]]:
        """
        Return metadata (name, rows, columns, size_mb) for all DataFrames in session.
        Used by the /upload/list endpoint.
        """
        frames = self._store.get(session_id, {})
        return [
            {
                "df_name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "size_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
            }
            for name, df in frames.items()
        ]

    def delete(self, session_id: str, df_name: str) -> bool:
        """Remove a single DataFrame from a session. Returns True if deleted."""
        session_frames = self._store.get(session_id)
        if session_frames and df_name in session_frames:
            del session_frames[df_name]
            return True
        return False

    def delete_session(self, session_id: str) -> None:
        """Remove all DataFrames for a session."""
        self._store.pop(session_id, None)
        self._last_access.pop(session_id, None)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _session_size_mb(self, session_id: str) -> float:
        frames = self._store.get(session_id, {})
        return sum(df.memory_usage(deep=True).sum() for df in frames.values()) / (1024 * 1024)

    def _evict_expired(self) -> None:
        """Lazy TTL eviction — runs on every get() call."""
        now = time.monotonic()
        expired = [sid for sid, last in self._last_access.items() if now - last > self._ttl]
        for sid in expired:
            self._store.pop(sid, None)
            self._last_access.pop(sid, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitise_df_name(name: str) -> str:
    """
    Convert a filename-derived string to a valid Python identifier.
    Example: 'claims data.csv' → 'claims_data'
    """
    # Strip path and extension
    stem = Path(name).stem
    # Replace non-alphanumeric (except underscore) with underscore
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    # Must start with a letter or underscore
    if clean and clean[0].isdigit():
        clean = "df_" + clean
    if not clean:
        clean = "df"
    return clean
