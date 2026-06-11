"""
tests/test_dataframe_loader.py
Phase 2 — DataFrameLoader unit tests.

Uses a file-backed SQLite engine (not :memory:) because DataFrameLoader
runs _load_table in a ThreadPoolExecutor — each thread gets its own
connection, and SQLite in-memory DBs are per-connection, so the table
created in the fixture connection is invisible in the worker thread.
File-backed SQLite shares state across all connections to the same path.
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import pytest
from dataframe_loader import DataFrameLoader
from sqlalchemy import create_engine, text

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sqlite_engine(tmp_path):
    """File-backed SQLite engine with a small claims table."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE claims ("
                "  claim_id INTEGER PRIMARY KEY,"
                "  policy_type TEXT,"
                "  claim_amount REAL,"
                "  customer_id INTEGER"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO claims VALUES "
                "(1,'auto',1200.0,101),"
                "(2,'home',3500.0,102),"
                "(3,'life',800.0,103),"
                "(4,'auto',2100.0,101),"
                "(5,'home',4200.0,104)"
            )
        )
        conn.commit()
    return engine


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    """Write a small policies CSV in tmp_path for file-fallback tests."""
    p = tmp_path / "policies.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy_id", "policy_type", "premium"])
        writer.writeheader()
        writer.writerows(
            [
                {"policy_id": 1, "policy_type": "auto", "premium": 500.0},
                {"policy_id": 2, "policy_type": "home", "premium": 800.0},
                {"policy_id": 3, "policy_type": "life", "premium": 1200.0},
            ]
        )
    return tmp_path


# ── DB load path ──────────────────────────────────────────────────────────────


class TestDBLoad:
    @pytest.mark.asyncio
    async def test_loads_table_from_db(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine)
        result = await loader.load(["claims"], schema_id="test")
        assert "claims" in result
        assert isinstance(result["claims"], pd.DataFrame)
        assert len(result["claims"]) == 5

    @pytest.mark.asyncio
    async def test_correct_columns_returned(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine)
        result = await loader.load(["claims"], schema_id="test")
        df = result["claims"]
        assert "policy_type" in df.columns
        assert "claim_amount" in df.columns

    @pytest.mark.asyncio
    async def test_row_limit_enforced(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, row_limit=3)
        result = await loader.load(["claims"], schema_id="test")
        assert len(result["claims"]) <= 3

    @pytest.mark.asyncio
    async def test_missing_table_omitted(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine)
        result = await loader.load(["nonexistent_table"], schema_id="test")
        assert "nonexistent_table" not in result

    @pytest.mark.asyncio
    async def test_mixed_tables_partial_result(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine)
        result = await loader.load(["claims", "nonexistent_table"], schema_id="test")
        assert "claims" in result
        assert "nonexistent_table" not in result


# ── Cache behaviour ───────────────────────────────────────────────────────────


class TestCache:
    @pytest.mark.asyncio
    async def test_second_load_hits_cache(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, cache_ttl=60)
        r1 = await loader.load(["claims"], schema_id="test")
        r2 = await loader.load(["claims"], schema_id="test")
        # Same DataFrame object returned from cache
        assert r1["claims"] is r2["claims"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_clears_entry(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, cache_ttl=60)
        await loader.load(["claims"], schema_id="test")
        assert len(loader._cache) == 1
        await loader.invalidate("test")
        assert len(loader._cache) == 0

    @pytest.mark.asyncio
    async def test_invalidate_only_affects_target_schema(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, cache_ttl=60)
        await loader.load(["claims"], schema_id="schema_a")
        await loader.load(["claims"], schema_id="schema_b")
        assert len(loader._cache) == 2
        await loader.invalidate("schema_a")
        assert len(loader._cache) == 1
        assert any("schema_b" in k for k in loader._cache)

    @pytest.mark.asyncio
    async def test_stale_cache_reloads(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, cache_ttl=0)
        r1 = await loader.load(["claims"], schema_id="test")
        await asyncio.sleep(0.01)
        r2 = await loader.load(["claims"], schema_id="test")
        # TTL=0 means every load is stale — fresh object each time
        assert r1["claims"] is not r2["claims"]


# ── File fallback ─────────────────────────────────────────────────────────────


class TestFileFallback:
    @pytest.mark.asyncio
    async def test_csv_file_loaded_when_db_fails(self, sqlite_engine, csv_dir: Path) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, file_root=csv_dir)
        # 'policies' does not exist in SQLite → falls back to policies.csv
        result = await loader.load(["policies"], schema_id="test")
        assert "policies" in result
        assert len(result["policies"]) == 3
        assert "premium" in result["policies"].columns

    @pytest.mark.asyncio
    async def test_no_file_root_returns_empty(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, file_root=None)
        result = await loader.load(["policies"], schema_id="test")
        assert "policies" not in result


# ── Cache stats ───────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_cache_stats_reflect_loaded_tables(self, sqlite_engine) -> None:
        loader = DataFrameLoader(engine=sqlite_engine, cache_ttl=60)
        await loader.load(["claims"], schema_id="test")
        stats = loader.cache_stats()
        assert stats["cached_tables"] == 1
        assert stats["total_memory_mb"] >= 0  # tiny test DF may report 0 bytes
        assert "max_memory_mb" in stats


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
