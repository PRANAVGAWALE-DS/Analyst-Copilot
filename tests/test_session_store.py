"""
tests/test_session_store.py
Tests for session_store.py.

Uses InMemorySessionStore throughout (no Redis required).
Tests enforce the exact interface contract that orchestrator.py depends on.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import pytest
from interfaces import ErrorDetail, TurnRecord
from session_store import InMemorySessionStore, build_session_store


@pytest.fixture()
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


def _make_turn(nl_query: str = "test query", retry: int = 0) -> TurnRecord:
    return TurnRecord(
        nl_query=nl_query,
        generated_code="SELECT 1",
        code_type="sql",
        row_count=1,
        insight="Looks good.",
        retry_count=retry,
        error=None,
    )


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_returns_string_id(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    @pytest.mark.asyncio
    async def test_each_call_returns_unique_id(self, store: InMemorySessionStore) -> None:
        ids = {await store.create_session() for _ in range(10)}
        assert len(ids) == 10


class TestGetOrCreate:
    @pytest.mark.asyncio
    async def test_returns_existing_session(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        returned = await store.get_or_create(sid)
        assert returned == sid

    @pytest.mark.asyncio
    async def test_creates_new_when_none(self, store: InMemorySessionStore) -> None:
        sid = await store.get_or_create(None)
        assert isinstance(sid, str)

    @pytest.mark.asyncio
    async def test_creates_new_when_unknown_id(self, store: InMemorySessionStore) -> None:
        sid = await store.get_or_create("nonexistent-id-xyz")
        # Should return a new session, not the given unknown id
        assert isinstance(sid, str)


class TestAppendAndGetHistory:
    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_list(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        history = await store.get_history(sid, n=10)
        assert history == []

    @pytest.mark.asyncio
    async def test_history_contains_appended_turn(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        turn = _make_turn("What is the total revenue?")
        await store.append_turn(sid, turn)
        history = await store.get_history(sid, n=10)
        assert len(history) == 1
        assert history[0]["nl_query"] == "What is the total revenue?"

    @pytest.mark.asyncio
    async def test_history_oldest_first(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        for i in range(5):
            await store.append_turn(sid, _make_turn(f"query {i}"))
        history = await store.get_history(sid, n=5)
        queries = [h["nl_query"] for h in history]
        assert queries == [f"query {i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_history_respects_n_limit(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        for i in range(15):
            await store.append_turn(sid, _make_turn(f"query {i}"))
        history = await store.get_history(sid, n=5)
        assert len(history) == 5
        # Should be the 5 most recent
        assert history[-1]["nl_query"] == "query 14"

    @pytest.mark.asyncio
    async def test_history_keys_match_orchestrator_expectation(
        self, store: InMemorySessionStore
    ) -> None:
        """orchestrator.py reads: nl_query, generated_code, code_type, insight, retry_count."""
        sid = await store.create_session()
        await store.append_turn(sid, _make_turn())
        history = await store.get_history(sid, n=1)
        required_keys = {
            "nl_query",
            "generated_code",
            "code_type",
            "insight",
            "retry_count",
        }
        assert required_keys.issubset(set(history[0].keys()))


class TestGetFullHistory:
    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_session(self, store: InMemorySessionStore) -> None:
        with pytest.raises(KeyError):
            await store.get_full_history("unknown-session")

    @pytest.mark.asyncio
    async def test_returns_turn_records(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        turn = _make_turn("revenue by quarter")
        await store.append_turn(sid, turn)
        records = await store.get_full_history(sid)
        assert len(records) == 1
        assert isinstance(records[0], TurnRecord)
        assert records[0].nl_query == "revenue by quarter"

    @pytest.mark.asyncio
    async def test_turn_with_error_preserved(self, store: InMemorySessionStore) -> None:
        sid = await store.create_session()
        turn = TurnRecord(
            nl_query="bad query",
            generated_code="SELECT nonexistent_col FROM t",
            code_type="sql",
            row_count=None,
            insight="I wasn't able to answer that.",
            retry_count=3,
            error=ErrorDetail(
                error_code="UNRESOLVED_COLUMN",
                message="Column 'nonexistent_col' not found.",
            ),
        )
        await store.append_turn(sid, turn)
        records = await store.get_full_history(sid)
        assert records[0].error is not None
        assert records[0].error.error_code == "UNRESOLVED_COLUMN"


class TestBuildSessionStore:
    @pytest.mark.asyncio
    async def test_blank_redis_url_uses_in_memory_store(self) -> None:
        store = await build_session_store(redis_url="")
        assert isinstance(store, InMemorySessionStore)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
