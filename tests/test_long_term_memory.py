"""
tests/test_long_term_memory.py
Phase 3 — LongTermMemory unit tests.

Updated to match long_term_memory.py (synchronous FAISS/JSON implementation):

  API changes from previous test version:
  - store()    : (session_id, schema_id, nl_query, sql, insight) → bool
                  No was_successful / code_type parameters.
  - search()   : replaces retrieve(); returns list[MemorySearchResult]
  - Results    : result.record.{nl_query,sql,schema_id,insight}
                  result.similarity, result.is_exact_hit
  - stats()    : {total_records, active_records, index_size, dirty}
                  No per-schema breakdown.
  - Stale purge: rebuild_if_stale() → bool  (replaces async prune())
                  Backdating done via JSON metadata file, not SQLite.
  - No async   : all methods are synchronous; pytest-asyncio not required.
  - Skip conds : PII in nl_query, duplicate nl_query+schema_id, MAX_RECORDS.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC

import _bootstrap  # noqa: F401  — adjusts sys.path for src layout
import numpy as np
import pytest
from long_term_memory import LongTermMemory

# ── Stub embedder ─────────────────────────────────────────────────────────────


class _StubEmbedder:
    """
    Deterministic unit-vector embedder seeded by the query string.

    Identical queries  → identical vectors  → inner-product ≈ 1.0  (exact hit)
    Different queries  → near-orthogonal    → inner-product ≈ 0.0  (no hit)

    DIM is intentionally small (64) for test speed. Pass dimension=DIM to
    LongTermMemory so the FAISS index is built at the correct dimensionality.
    """

    DIM = 64

    def embed_query(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def embedder() -> _StubEmbedder:
    return _StubEmbedder()


@pytest.fixture()
def memory(tmp_path, embedder: _StubEmbedder) -> LongTermMemory:
    """Fresh in-memory LongTermMemory instance backed by a tmp_path directory."""
    return LongTermMemory(
        embedder=embedder,
        index_dir=tmp_path / "lt_memory",
        dimension=_StubEmbedder.DIM,  # must match embedder output
        k_retrieve=3,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _store(
    memory: LongTermMemory,
    nl: str,
    sql: str = "SELECT 1",
    schema_id: str = "ins_v1",
    session_id: str = "sess1",
    insight: str = "",
) -> bool:
    return memory.store(session_id, schema_id, nl, sql, insight)


# ── Store + search round-trip ─────────────────────────────────────────────────


class TestStoreSearch:
    def test_stored_entry_is_retrieved(self, memory: LongTermMemory) -> None:
        nl = "What was the average claim amount by policy type?"
        sql = "SELECT policy_type, AVG(claim_amount) FROM claims GROUP BY policy_type"
        _store(memory, nl, sql)
        results = memory.search(nl, "ins_v1")
        assert len(results) >= 1
        assert results[0].record.nl_query == nl
        assert results[0].record.sql == sql

    def test_schema_id_preserved_on_record(self, memory: LongTermMemory) -> None:
        _store(memory, "count claims", "SELECT COUNT(*) FROM claims")
        results = memory.search("count claims", "ins_v1")
        assert results[0].record.schema_id == "ins_v1"

    def test_insight_preserved_on_record(self, memory: LongTermMemory) -> None:
        _store(memory, "count claims", insight="42 total claims")
        results = memory.search("count claims", "ins_v1")
        assert results[0].record.insight == "42 total claims"

    def test_similarity_score_is_positive(self, memory: LongTermMemory) -> None:
        nl = "total claims by region"
        _store(memory, nl, "SELECT region, COUNT(*) FROM claims GROUP BY region")
        results = memory.search(nl, "ins_v1")
        assert results[0].similarity > 0.0

    def test_identical_query_is_exact_hit(self, memory: LongTermMemory) -> None:
        """Identical nl_query → inner-product ≈ 1.0 → is_exact_hit=True."""
        nl = "total claims by region"
        _store(memory, nl)
        results = memory.search(nl, "ins_v1")
        assert results[0].is_exact_hit is True

    def test_k_caps_result_count(self, memory: LongTermMemory) -> None:
        for i in range(10):
            _store(memory, f"query variant {i}", f"SELECT {i}")
        results = memory.search("query variant 0", "ins_v1", k=2)
        assert len(results) <= 2

    def test_multiple_similar_queries_returned(self, memory: LongTermMemory) -> None:
        queries = [
            "average premium by type",
            "average premium by region",
            "average premium overall",
        ]
        for q in queries:
            _store(memory, q, f"SELECT AVG(premium) -- {q}")
        results = memory.search("average premium by type", "ins_v1", k=3)
        # At minimum the exact-match entry must be returned
        assert len(results) >= 1

    def test_store_returns_true_on_success(self, memory: LongTermMemory) -> None:
        stored = _store(memory, "some query")
        assert stored is True


# ── Deduplication ─────────────────────────────────────────────────────────────


class TestDedup:
    def test_duplicate_nl_query_same_schema_not_stored(self, memory: LongTermMemory) -> None:
        nl = "count all claims"
        _store(memory, nl, "SELECT COUNT(*) FROM claims")
        second = _store(memory, nl, "SELECT COUNT(*) FROM claims", session_id="sess2")
        assert second is False
        assert memory.stats()["total_records"] == 1

    def test_same_nl_query_different_schema_stored(self, memory: LongTermMemory) -> None:
        """Same nl_query on a different schema_id is a distinct record."""
        nl = "count records"
        a = _store(memory, nl, "SELECT COUNT(*) FROM claims", schema_id="schema_a")
        b = _store(memory, nl, "SELECT COUNT(*) FROM orders", schema_id="schema_b")
        assert a is True
        assert b is True
        assert memory.stats()["total_records"] == 2


# ── PII guard ─────────────────────────────────────────────────────────────────


class TestPiiGuard:
    @pytest.mark.parametrize(
        "pii_query",
        [
            "claims for user@example.com last month",  # email
            "policy for SSN 123-45-6789",  # SSN
            "customer 555-867-5309 claim history",  # phone
            "card 4111 1111 1111 1111 transactions",  # credit card
        ],
    )
    def test_pii_query_not_stored(self, memory: LongTermMemory, pii_query: str) -> None:
        stored = _store(memory, pii_query)
        assert stored is False
        assert memory.stats()["total_records"] == 0


# ── Schema isolation ──────────────────────────────────────────────────────────


class TestSchemaIsolation:
    def test_schema_a_result_not_returned_for_schema_b(self, memory: LongTermMemory) -> None:
        nl = "total revenue last quarter"
        _store(memory, nl, "SELECT SUM(revenue) FROM sales", schema_id="schema_a")
        results = memory.search(nl, "schema_b")
        assert results == []

    def test_each_schema_returns_own_sql(self, memory: LongTermMemory) -> None:
        nl = "count records"
        _store(memory, nl, "SELECT COUNT(*) FROM claims", schema_id="schema_a")
        _store(memory, nl, "SELECT COUNT(*) FROM orders", schema_id="schema_b")
        ra = memory.search(nl, "schema_a")
        rb = memory.search(nl, "schema_b")
        assert any("claims" in r.record.sql for r in ra)
        assert any("orders" in r.record.sql for r in rb)


# ── Unknown schema ────────────────────────────────────────────────────────────


class TestUnknownSchema:
    def test_returns_empty_list_for_unknown_schema(self, memory: LongTermMemory) -> None:
        results = memory.search("any query", "nonexistent_schema")
        assert results == []

    def test_returns_empty_list_when_index_is_empty(self, memory: LongTermMemory) -> None:
        results = memory.search("any query", "ins_v1")
        assert results == []


# ── Stats ─────────────────────────────────────────────────────────────────────


class TestStats:
    def test_total_records_reflects_stored_count(self, memory: LongTermMemory) -> None:
        for i in range(3):
            _store(memory, f"query {i}", f"SELECT {i}")
        stats = memory.stats()
        assert stats["total_records"] == 3

    def test_active_records_equals_total_when_nothing_stale(self, memory: LongTermMemory) -> None:
        for i in range(3):
            _store(memory, f"query {i}", f"SELECT {i}")
        stats = memory.stats()
        assert stats["active_records"] == stats["total_records"]

    def test_index_size_matches_total_records(self, memory: LongTermMemory) -> None:
        for i in range(3):
            _store(memory, f"query {i}", f"SELECT {i}")
        stats = memory.stats()
        assert stats["index_size"] == stats["total_records"]

    def test_stats_empty_when_nothing_stored(self, memory: LongTermMemory) -> None:
        stats = memory.stats()
        assert stats["total_records"] == 0
        assert stats["active_records"] == 0
        assert stats["index_size"] == 0

    def test_dirty_flag_set_after_store(self, memory: LongTermMemory) -> None:
        # store() uses a save-on-write policy: it calls save() immediately
        # after appending each record, which resets _dirty=False before
        # returning. The test therefore verifies the save-on-write contract:
        # after a successful store(), the data is persisted (dirty=False means
        # save() ran) and the record count is correct.
        #
        # To observe _dirty=True transiently, we must bypass save(). We use
        # unittest.mock to patch save() out for this one call, then check that
        # _dirty was set to True inside the lock before save() was called.
        import unittest.mock as mock

        dirty_during_store: list[bool] = []

        def _capture_dirty_then_save(self_=memory):
            # Called instead of save() — capture _dirty state at that moment
            dirty_during_store.append(memory._dirty)
            # Do NOT call the real save() — we're observing the transient state

        with mock.patch.object(memory, "save", side_effect=_capture_dirty_then_save):
            stored = _store(memory, "any query about total claims")

        assert stored is True, "store() returned False — check guard conditions"
        # _dirty was True when save() was invoked (before save() reset it)
        assert len(dirty_during_store) == 1, "save() was not called exactly once"
        assert dirty_during_store[0] is True, "_dirty was not True when save() ran"

    def test_dirty_flag_cleared_after_save(self, memory: LongTermMemory) -> None:
        _store(memory, "any query")
        memory.save()
        assert memory.stats()["dirty"] is False


# ── rebuild_if_stale ──────────────────────────────────────────────────────────


class TestRebuildIfStale:
    """
    Backdating is done by writing directly to the JSON metadata file
    (lt_memory.meta.json), then calling load() to reload the mutated state.
    This mirrors the implementation's storage layer (FAISS + JSON, not SQLite).
    """

    def _backdate_all(self, mem: LongTermMemory, days_ago: int = 91) -> None:
        """Overwrite all created_at timestamps in the JSON metadata file."""
        from datetime import datetime, timedelta

        old_ts = (datetime.now(tz=UTC) - timedelta(days=days_ago)).isoformat()
        mem.save()
        with open(mem._meta_path) as f:
            records = json.load(f)
        for r in records:
            r["created_at"] = old_ts
        with open(mem._meta_path, "w") as f:
            json.dump(records, f)

    def test_rebuild_returns_true_when_all_records_stale(
        self, tmp_path, embedder: _StubEmbedder
    ) -> None:
        mem = LongTermMemory(
            embedder=embedder,
            index_dir=tmp_path / "lt_rebuild_true",
            dimension=_StubEmbedder.DIM,
        )
        for i in range(5):
            _store(mem, f"query {i}", schema_id="s1")
        self._backdate_all(mem)

        mem.load()  # reload to pick up backdated timestamps
        rebuilt = mem.rebuild_if_stale()
        assert rebuilt is True

    def test_stale_records_removed_after_rebuild(self, tmp_path, embedder: _StubEmbedder) -> None:
        mem = LongTermMemory(
            embedder=embedder,
            index_dir=tmp_path / "lt_rebuild_clean",
            dimension=_StubEmbedder.DIM,
        )
        for i in range(5):
            _store(mem, f"query {i}", schema_id="s1")
        self._backdate_all(mem)
        mem.load()
        mem.rebuild_if_stale()

        assert mem.stats()["total_records"] == 0
        assert mem.search("query 0", "s1") == []

    def test_rebuild_returns_false_when_stale_fraction_below_threshold(
        self, tmp_path, embedder: _StubEmbedder
    ) -> None:
        """1 stale out of 10 = 10% < 20% threshold → no rebuild."""
        from datetime import datetime, timedelta

        mem = LongTermMemory(
            embedder=embedder,
            index_dir=tmp_path / "lt_no_rebuild",
            dimension=_StubEmbedder.DIM,
        )
        for i in range(10):
            _store(mem, f"query {i}", schema_id="s1")
        mem.save()

        # Backdate only the first record (10% stale)
        old_ts = (datetime.now(tz=UTC) - timedelta(days=91)).isoformat()
        with open(mem._meta_path) as f:
            records = json.load(f)
        records[0]["created_at"] = old_ts
        with open(mem._meta_path, "w") as f:
            json.dump(records, f)

        mem.load()
        rebuilt = mem.rebuild_if_stale()
        assert rebuilt is False

    def test_rebuild_returns_false_when_no_records(self, memory: LongTermMemory) -> None:
        """Empty index must not raise; returns False."""
        assert memory.rebuild_if_stale() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
