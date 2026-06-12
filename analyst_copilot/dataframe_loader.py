"""
dataframe_loader.py — DataFrame Loader for Pandas Executor
Data Analyst Copilot · Phase 2 · Python 3.11+

Problem: orchestrator.py passes dataframe_refs=None to ExecutionLoop.run(),
so the Pandas executor always runs in an empty namespace. This module fills
that gap by loading actual DataFrames from either:
  A. The analytical DB (SELECT * FROM {table} LIMIT {cap}) — default path.
  B. Files on disk (CSV / Parquet) — fallback for file-based schemas.

Design decision: load on demand, cache in-process per (schema_id, table).
Trade-off: first Pandas query for a table pays the load cost; subsequent
queries in the same process hit the cache. Cache is invalidated on /ingest.

Interface consumed by orchestrator.py:
    loader = DataFrameLoader(engine)
    df_refs: dict[str, pd.DataFrame] = await loader.load(
        tables=["claims", "policies"],
        schema_id="ins_prod_v3",
        row_limit=50_000,
    )
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from observability import ObservabilityStack

# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class _CacheEntry:
    def __init__(self, df: pd.DataFrame, loaded_at: float) -> None:
        self.df = df
        self.loaded_at = loaded_at


# ---------------------------------------------------------------------------
# DataFrameLoader
# ---------------------------------------------------------------------------

_DEFAULT_ROW_LIMIT = 50_000
_DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes — stale data is acceptable
_MAX_MEMORY_MB = 512  # total in-process DataFrame cache ceiling


class DataFrameLoader:
    """
    Loads DataFrames from a SQLAlchemy engine (or file paths) and caches them
    in-process for the lifetime of the server.

    Thread-safety: all cache operations use asyncio.Lock — safe for FastAPI's
    single-event-loop model. Not safe for multi-process deployments; use a
    Redis-backed cache in that case (Phase 3 extension).

    Parameters
    ----------
    engine          : Read-only SQLAlchemy Engine (same one used by execute_sql).
    file_root       : Optional directory to search for CSV/Parquet files when
                      the table is not found in the DB. Filename must be
                      {table_name}.csv or {table_name}.parquet.
    cache_ttl       : Seconds before a cached DataFrame is considered stale.
    row_limit       : Maximum rows loaded per table. Enforced via LIMIT clause.
    obs             : Optional ObservabilityStack for trace logging.
    """

    def __init__(
        self,
        engine: Any,
        *,
        file_root: str | Path | None = None,
        cache_ttl: int = _DEFAULT_CACHE_TTL_SECONDS,
        row_limit: int = _DEFAULT_ROW_LIMIT,
        obs: ObservabilityStack | None = None,
    ) -> None:
        self._engine = engine
        self._file_root = Path(file_root) if file_root else None
        self._cache_ttl = cache_ttl
        self._row_limit = row_limit
        self._obs = obs
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        # Tracks in-flight loads keyed by cache_key so concurrent callers
        # for the same table await a single load instead of each issuing one.
        # A-10 FIX: run_in_executor() returns asyncio.Future, not asyncio.Task.
        # ensure_future() on a Future is a no-op (returns the same object).
        # Corrected type annotation to Future[pd.DataFrame].
        self._inflight: dict[str, asyncio.Future[pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(
        self,
        tables: list[str],
        schema_id: str,
        row_limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load and return a dict of {table_name: DataFrame} for the given tables.

        Tables already in the cache (and not stale) are returned immediately.
        Missing or stale tables are loaded from DB or file, then cached.

        Empty tables return an empty DataFrame with correct column names.
        Tables that fail to load are omitted from the result with a warning logged.
        """
        limit = row_limit or self._row_limit
        result: dict[str, pd.DataFrame] = {}

        for table in tables:
            cache_key = self._cache_key(schema_id, table)
            df = await self._get_or_load(cache_key, table, limit)
            if df is not None:
                result[table] = df

        return result

    async def invalidate(self, schema_id: str) -> None:
        """
        Invalidate all cached DataFrames for a schema.
        Called by the ingestion pipeline after a schema is re-ingested.
        Acquires the cache lock to avoid racing with concurrent loads.
        """
        prefix = f"{schema_id}::"
        async with self._lock:
            keys_to_drop = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_drop:
                del self._cache[k]

    def cache_stats(self) -> dict[str, Any]:
        """Return in-process cache statistics for the /health endpoint."""
        total_mb = sum(
            entry.df.memory_usage(deep=True).sum() / 1_048_576
            for entry in self._cache.values()
        )
        return {
            "cached_tables": len(self._cache),
            "total_memory_mb": round(total_mb, 2),
            "max_memory_mb": _MAX_MEMORY_MB,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_or_load(
        self,
        cache_key: str,
        table: str,
        limit: int,
    ) -> pd.DataFrame | None:
        # Fast path: cache hit (under lock to avoid torn reads)
        async with self._lock:
            entry = self._cache.get(cache_key)
            if entry:
                if (time.monotonic() - entry.loaded_at) < self._cache_ttl:
                    return entry.df
                # A-05 FIX: evict the stale entry immediately.  Previously the
                # stale entry stayed in self._cache until overwritten, meaning
                # its memory was counted in the ceiling check even though it was
                # about to be replaced.  A session with 400 MB of expired
                # DataFrames would incorrectly block a 150 MB fresh load.
                del self._cache[cache_key]

            # If a load for this key is already in flight, attach to it
            # instead of launching another — prevents thundering herd.
            if cache_key in self._inflight:
                task = self._inflight[cache_key]
            else:
                # F05: use get_running_loop() — get_event_loop() is deprecated.
                # A-10 FIX: run_in_executor returns an asyncio.Future directly;
                # wrapping it in ensure_future() is a documented no-op for
                # Futures (returns the same object).  Removed the redundant call.
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(None, self._load_table, table, limit)
                self._inflight[cache_key] = task

        # Await outside the lock so other coroutines remain unblocked
        try:
            df = await task
        except Exception as exc:  # noqa: BLE001
            self._warn(f"Failed to load DataFrame for table '{table}': {exc}")
            async with self._lock:
                self._inflight.pop(cache_key, None)
            return None

        async with self._lock:
            self._inflight.pop(cache_key, None)
            # Check memory ceiling before caching
            current_mb = sum(
                e.df.memory_usage(deep=True).sum() / 1_048_576
                for e in self._cache.values()
            )
            new_mb = df.memory_usage(deep=True).sum() / 1_048_576
            if current_mb + new_mb <= _MAX_MEMORY_MB:
                self._cache[cache_key] = _CacheEntry(df=df, loaded_at=time.monotonic())

        return df

    def _load_table(self, table: str, limit: int) -> pd.DataFrame:
        """
        Synchronous table load — runs in executor to avoid blocking event loop.

        Priority:
          1. DB query: SELECT * FROM {table} LIMIT {limit}
          2. File fallback: {file_root}/{table}.parquet or {table}.csv
        """
        # Attempt DB load first
        try:
            return self._load_from_db(table, limit)
        except Exception as db_exc:  # noqa: BLE001
            self._warn(
                f"DB load failed for '{table}' ({db_exc}), trying file fallback."
            )

        # File fallback
        if self._file_root:
            for ext in (".parquet", ".csv"):
                candidate = self._file_root / f"{table}{ext}"
                if candidate.exists():
                    df: pd.DataFrame = (
                        pd.read_parquet(str(candidate))
                        if ext == ".parquet"
                        else pd.read_csv(str(candidate))
                    )
                    return df.head(limit)

        raise FileNotFoundError(
            f"Table '{table}' not found in DB or file root "
            f"({self._file_root or 'no file root configured'})."
        )

    def _load_from_db(self, table: str, limit: int) -> pd.DataFrame:
        """Load a table from the DB engine using pandas.read_sql.

        BUG-2 FIX: identifier quoting is now dialect-aware.
        PostgreSQL / SQLite / DuckDB use double-quotes: "table"
        MySQL / MariaDB use backticks: `table`
        The previous hardcoded double-quote caused MySQL to interpret the
        identifier as a string alias, silently returning wrong results.
        """
        dialect = getattr(self._engine, "dialect", None)
        dialect_name = getattr(dialect, "name", "").lower()
        # PostgreSQL, SQLite, DuckDB, MSSQL (and unknown engines)
        # all accept ANSI double-quote identifier quoting.
        quoted = f"`{table}`" if dialect_name in ("mysql", "mariadb") else f'"{table}"'
        sql = f"SELECT * FROM {quoted} LIMIT {limit}"
        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn)

    @staticmethod
    def _cache_key(schema_id: str, table: str) -> str:
        return f"{schema_id}::{table}"

    def _warn(self, message: str) -> None:
        # A-04 FIX: was file=sys.stderr; changed to sys.stdout.
        # DATAFRAME_LOADER_WARNING events (DB load failure, file fallback
        # activation) are operationally significant — a Pandas query returning
        # wrong results because it silently fell back to a stale CSV is hard to
        # diagnose without this signal.  All other structured logs in the
        # application write to stdout; routing here to stderr made these events
        # invisible in most production log aggregators (CloudWatch, Loki, Datadog).
        print(
            json.dumps(
                {
                    "event": "DATAFRAME_LOADER_WARNING",
                    "message": message,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
            ),
            file=sys.stdout,
            flush=True,
        )
