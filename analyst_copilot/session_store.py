"""
session_store.py — Redis-backed SessionStore
Data Analyst Copilot · Python 3.11+ · Section 2G

Implements the SessionStore interface declared in orchestrator.py:

    class SessionStore:
        async def get_history(session_id, n=10) -> list[dict]
        async def append_turn(session_id, turn: TurnRecord) -> None
        async def create_session() -> str
        async def get_or_create(session_id | None) -> str

Storage layout in Redis:
  session:{session_id}:meta  — JSON hash of session metadata (schema_id, created_at)
  session:{session_id}:turns — Redis LIST of JSON-serialised TurnRecord objects
                               RPUSH on write → LRANGE 0 -1 returns oldest-first
                               (M-20 FIX: docstring previously said LPUSH — the code
                               uses RPUSH.  RPUSH appends to the tail; LRANGE 0 -1
                               returns elements in insertion order = oldest-first.)

TTL: 7 days per session key (refreshed on each append_turn call).

Fallback: InMemorySessionStore is provided for testing and dev environments
where Redis is not available (REDIS_URL not set or connection fails).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from interfaces import TurnRecord

_SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days


# ---------------------------------------------------------------------------
# Redis-backed implementation
# ---------------------------------------------------------------------------


class RedisSessionStore:
    """
    Production session store backed by Redis.

    Parameters
    ----------
    redis_url : Redis connection URL, e.g. "redis://localhost:6379/0".
                Reads from REDIS_URL env var if not supplied.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        import os

        import redis.asyncio as aioredis

        url: str = (
            redis_url or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"
        )
        # socket_connect_timeout + socket_timeout cap the TCP handshake and
        # per-command wait. Without these, an unreachable Redis host (e.g. when
        # running `docker run` standalone without --network) causes the startup
        # ping to block for the OS default (~30 s) before failing over to the
        # in-memory store.
        self._redis: aioredis.Redis[str] = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    async def ping(self) -> bool:
        """Health check — returns True if Redis is reachable within 2 s."""
        import asyncio

        try:
            return await asyncio.wait_for(self._redis.ping(), timeout=2.0)
        except Exception:  # noqa: BLE001
            return False

    def _meta_key(self, session_id: str) -> str:
        return f"session:{session_id}:meta"

    def _turns_key(self, session_id: str) -> str:
        return f"session:{session_id}:turns"

    async def create_session(self, schema_id: str = "") -> str:
        session_id = str(uuid.uuid4())
        meta = {
            "session_id": session_id,
            "schema_id": schema_id,
            "created_at": _now(),
        }
        await self._redis.set(
            self._meta_key(session_id),
            json.dumps(meta),
            ex=_SESSION_TTL_SECONDS,
        )
        return session_id

    async def get_or_create(
        self,
        session_id: str | None,
        schema_id: str = "",
    ) -> str:
        if session_id and await self._redis.exists(self._meta_key(session_id)):
            return session_id
        # Create new (or recreate if expired)
        return await self.create_session(schema_id=schema_id)

    async def append_turn(self, session_id: str, turn: TurnRecord) -> None:
        """
        Append a TurnRecord to the session's turn list.
        Uses RPUSH so that LRANGE 0 -1 returns turns in oldest-first order.
        Refreshes the session TTL.
        """
        serialised = turn.model_dump_json()
        pipe = self._redis.pipeline()
        pipe.rpush(self._turns_key(session_id), serialised)
        pipe.expire(self._turns_key(session_id), _SESSION_TTL_SECONDS)
        pipe.expire(self._meta_key(session_id), _SESSION_TTL_SECONDS)
        await pipe.execute()

    async def get_history(
        self,
        session_id: str,
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return the last n turns as plain dicts (for prompt injection).
        Returns [] if session does not exist or has no turns.
        Oldest-first (LRANGE from end).
        """
        turns_key = self._turns_key(session_id)
        total = await self._redis.llen(turns_key)
        if total == 0:
            return []

        # LRANGE indices: last n elements (oldest-first within that slice)
        start = max(0, total - n)
        raw_turns = await self._redis.lrange(turns_key, start, -1)

        history: list[dict[str, Any]] = []
        for raw in raw_turns:
            try:
                turn_data = json.loads(raw)
                # Return a lightweight dict for prompt use — omit error details
                history.append(
                    {
                        "nl_query": turn_data.get("nl_query", ""),
                        "generated_code": turn_data.get("generated_code", ""),
                        "code_type": turn_data.get("code_type", "sql"),
                        "insight": turn_data.get("insight", ""),
                        "retry_count": turn_data.get("retry_count", 0),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue  # skip corrupted entries

        return history

    async def get_full_history(self, session_id: str) -> list[TurnRecord]:
        """
        Return all TurnRecord objects for a session (for /history endpoint).
        Raises KeyError if session not found.
        """
        if not await self._redis.exists(self._meta_key(session_id)):
            raise KeyError(f"Session '{session_id}' not found.")

        turns_key = self._turns_key(session_id)
        raw_turns = await self._redis.lrange(turns_key, 0, -1)

        records: list[TurnRecord] = []
        for raw in raw_turns:
            try:
                records.append(TurnRecord.model_validate_json(raw))
            except Exception:  # noqa: BLE001
                continue

        return records


# ---------------------------------------------------------------------------
# In-memory fallback (dev / test)
# ---------------------------------------------------------------------------


class InMemorySessionStore:
    """
    In-memory SessionStore for local development and testing.
    Not suitable for multi-process or multi-replica deployments.

    M4 FIX: TTL eviction added (lazy, runs on every get_history call).
    The RedisSessionStore has a 7-day TTL enforced at Redis level; without
    eviction the in-memory store grows without bound across long dev sessions.
    """

    # Match the Redis-backed store's TTL so behaviour is consistent across
    # backends. Reduce for memory-constrained dev environments via
    # constructor parameter.
    _DEFAULT_TTL: int = _SESSION_TTL_SECONDS

    def __init__(self, ttl_seconds: int = _SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._turns: dict[str, list[TurnRecord]] = {}
        self._last_access: dict[str, float] = {}
        self._ttl = ttl_seconds

    def _evict_expired(self) -> None:
        """Lazy TTL eviction — runs on every get_history() call."""
        now = time.monotonic()
        expired = [
            sid for sid, last in self._last_access.items() if now - last > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._turns.pop(sid, None)
            self._last_access.pop(sid, None)

    async def create_session(self, schema_id: str = "") -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "session_id": session_id,
            "schema_id": schema_id,
            "created_at": _now(),
        }
        self._turns[session_id] = []
        self._last_access[session_id] = time.monotonic()
        return session_id

    async def get_or_create(
        self,
        session_id: str | None,
        schema_id: str = "",
    ) -> str:
        if session_id and session_id in self._sessions:
            return session_id
        return await self.create_session(schema_id=schema_id)

    async def append_turn(self, session_id: str, turn: TurnRecord) -> None:
        # M-21 FIX: call _evict_expired() here as well as in get_history().
        # Previously, sessions that only received append_turn() calls — batch
        # pipelines, fire-and-forget writers — never triggered eviction because
        # _evict_expired() was gated behind get_history().  Those sessions
        # accumulated in _sessions, _turns, and _last_access indefinitely.
        # Lazy eviction on every write keeps the footprint bounded without
        # adding a background task or lock.
        self._evict_expired()
        if session_id not in self._turns:
            self._turns[session_id] = []
        self._turns[session_id].append(turn)
        self._last_access[session_id] = time.monotonic()

    async def get_history(
        self,
        session_id: str,
        n: int = 10,
    ) -> list[dict[str, Any]]:
        self._evict_expired()
        turns = self._turns.get(session_id, [])
        if turns:
            self._last_access[session_id] = time.monotonic()
        recent = turns[-n:]
        return [
            {
                "nl_query": t.nl_query,
                "generated_code": t.generated_code,
                "code_type": t.code_type,
                "insight": t.insight,
                "retry_count": t.retry_count,
            }
            for t in recent
        ]

    async def get_full_history(self, session_id: str) -> list[TurnRecord]:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        return list(self._turns.get(session_id, []))


# ---------------------------------------------------------------------------
# Factory — picks Redis or in-memory based on env / availability
# ---------------------------------------------------------------------------


async def build_session_store(
    redis_url: str | None = None,
) -> RedisSessionStore | InMemorySessionStore:
    """
    Returns a RedisSessionStore if Redis is reachable, else InMemorySessionStore.
    Logs the choice to stdout so the startup log makes it visible.
    """
    import os
    import sys

    url = (
        redis_url if redis_url is not None else os.environ.get("REDIS_URL", "")
    ).strip()
    if not url:
        fallback = InMemorySessionStore()
        print(
            json.dumps(
                {
                    "event": "SESSION_STORE_INIT",
                    "backend": "in_memory",
                    "reason": "REDIS_URL is not set — using in-memory store (dev/test only).",
                }
            ),
            file=sys.stdout,
        )
        return fallback

    try:
        store = RedisSessionStore(redis_url=url)
    except Exception as exc:  # noqa: BLE001
        # P3-05 FIX: was `except ValueError`.
        # RedisSessionStore.__init__ executes `import redis.asyncio as aioredis`
        # at call time.  If the redis package is not installed but REDIS_URL is
        # configured, this raises ImportError — which fell outside the old
        # ValueError catch and crashed _startup() with a confusing traceback.
        # Broadening to Exception ensures any constructor failure (ImportError,
        # TypeError on bad URL format, etc.) falls back gracefully to
        # InMemorySessionStore with a log event, instead of killing the server.
        fallback = InMemorySessionStore()
        print(
            json.dumps(
                {
                    "event": "SESSION_STORE_INIT",
                    "backend": "in_memory",
                    "reason": f"RedisSessionStore init failed ({type(exc).__name__}: {exc}) — using in-memory store (dev/test only).",
                }
            ),
            file=sys.stdout,
        )
        return fallback

    if await store.ping():
        print(
            json.dumps(
                {
                    "event": "SESSION_STORE_INIT",
                    "backend": "redis",
                    "url": url.split("@")[-1],  # strip credentials if any
                }
            ),
            file=sys.stdout,
        )
        return store

    fallback = InMemorySessionStore()
    print(
        json.dumps(
            {
                "event": "SESSION_STORE_INIT",
                "backend": "in_memory",
                "reason": "Redis not reachable — using in-memory store (dev/test only).",
            }
        ),
        file=sys.stdout,
    )
    return fallback


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()
