"""
long_term_memory.py — Cross-session vector memory
Data Analyst Copilot · Phase 3 · Python 3.11+

Stores successful (nl_query, sql, insight) triples from past sessions and
retrieves them by semantic similarity at RETRIEVAL state before generation.

Why this improves generation quality:
  - The LLM sees an analogous past query+SQL pair as a few-shot example.
  - Avoids re-generating the same SQL for recurring queries.
  - Reduces token cost: cached SQL is injected directly, skipping generation.

Storage layout:
  data/lt_memory/
    lt_memory.faiss      — FAISS HNSW index (one global index, not per-schema)
    lt_memory.meta.json  — list of MemoryRecord dicts (parallel to FAISS rows)

Retrieval trigger (orchestrator integration):
  Called at the start of _retrieval_state() in orchestrator.py.
  If a memory hit (similarity > SIMILARITY_THRESHOLD) is found:
    - The retrieved SQL is injected into the generation prompt as a few-shot example.
    - If similarity > EXACT_HIT_THRESHOLD the SQL is used directly (skip LLM call).

Staleness policy:
  - Records older than RECORD_TTL_DAYS are excluded from search results.
  - Full index rebuild is triggered when >20% of records are stale.

Privacy:
  - NL queries containing detected PII (emails, phone numbers, etc.) are NOT stored.
  - PII detection is heuristic-only (regex). Not a complete PII filter.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SIMILARITY_THRESHOLD: float = 0.85  # minimum cosine similarity for a useful hit
EXACT_HIT_THRESHOLD: float = 0.97  # similarity above this → skip LLM, use cached SQL
RECORD_TTL_DAYS: int = 30
MAX_RECORDS: int = 10_000  # cap to bound memory footprint
HNSW_M: int = 32
HNSW_EF_CONSTRUCTION: int = 200
HNSW_EF_SEARCH: int = 40


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    record_id: str
    session_id: str
    schema_id: str
    nl_query: str
    sql: str
    insight: str
    created_at: str  # ISO-8601 UTC
    score: float = 0.0  # populated at retrieval time


@dataclass
class MemorySearchResult:
    record: MemoryRecord
    similarity: float
    is_exact_hit: bool


# ---------------------------------------------------------------------------
# PII guard
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),  # email
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),  # credit card
]


def _contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """
    FAISS HNSW-backed vector memory for cross-session query history.

    Parameters
    ----------
    embedder    : SchemaEmbedder instance (reuses the same model as retrieval).
    index_dir   : Directory to persist the index and metadata.
    dimension   : Embedding dimension (must match embedder.dimension).
    """

    def __init__(
        self,
        embedder: Any,
        index_dir: str | Path = "data/lt_memory",
        # FIX: default corrected from 768 → 1024.
        # 768 is the output dimension of bge-BASE-en-v1.5.
        # bge-LARGE-en-v1.5 (the configured model) outputs 1024 dimensions.
        # This default must match EMBEDDING_DIM in .env; app.py now passes
        # it explicitly so a model change only requires updating EMBEDDING_DIM.
        dimension: int = 1024,
        k_retrieve: int = 3,
    ) -> None:
        self._embedder = embedder
        self._dir = Path(index_dir)
        self._dim = dimension
        self._k_retrieve = k_retrieve
        self._index: Any = None  # faiss.IndexHNSWFlat
        self._records: list[MemoryRecord] = []
        self._dirty = False
        # threading.Lock (not asyncio.Lock) because store/search run inside
        # executor threads (asyncio.to_thread), not directly on the event loop.
        self._lock = threading.Lock()
        # M-10 FIX: maintain dedup keys as a persistent set rather than
        # recomputing SHA-256 for every existing record on every store() call.
        # At MAX_RECORDS=10_000 the old approach ran 10k hash ops + held the
        # lock for the entire duration on every successful query turn.
        # The set is populated in load() and appended in store().
        self._dedup_keys: set[str] = set()

    def _embed_cpu(self, text: str) -> np.ndarray:
        """
        Embed a single query string, forcing the sentence-transformers model
        onto CPU before calling encode(), then restoring the original device.

        Why this is necessary
        ---------------------
        store() and search() run inside asyncio.to_thread() worker threads.
        On a 4 GB GPU the main retrieval path already holds ~1.34 GB of VRAM
        (bge-large-en-v1.5). A concurrent encode() call from a worker thread
        either pushes VRAM over the limit or hits a CUDA context issue on the
        non-main thread — either way it crashes silently inside the caller's
        contextlib.suppress(Exception), so nothing is stored and no error
        is ever visible.

        This method serialises GPU access: store()/search() each hold
        self._lock, so only one CPU←→GPU movement can happen at a time.
        The overhead (~100-200 ms on CPU vs ~30 ms on GPU) is invisible to
        the user because store() is called after the response is delivered.

        Gemini / no local model path
        ----------------------------
        If the embedder has no _model attribute (Gemini API path), encode()
        runs over HTTP and has no GPU contention — fall through to the normal
        embed_query call.
        """
        _model = getattr(self._embedder, "_model", None)
        if _model is None:
            # Gemini or other HTTP-based embedder — no GPU contention
            return self._embedder.embed_query(text).reshape(1, -1).astype(np.float32)

        try:
            original_device = next(_model.parameters()).device
            if original_device.type == "cpu":
                # Already on CPU — no movement needed
                return self._embedder.embed_query(text).reshape(1, -1).astype(np.float32)
            _model.to("cpu")
            try:
                return self._embedder.embed_query(text).reshape(1, -1).astype(np.float32)
            finally:
                _model.to(original_device)
        except Exception:  # noqa: BLE001
            # Device movement failed — try direct call as last resort
            return self._embedder.embed_query(text).reshape(1, -1).astype(np.float32)

    # ── Index paths ───────────────────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self._dir / "lt_memory.faiss"

    @property
    def _meta_path(self) -> Path:
        return self._dir / "lt_memory.meta.json"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load index and metadata from disk. Creates empty index if not found."""
        import faiss

        self._dir.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path) as f:
                raw = json.load(f)
            # M-12 FIX: guard against corrupt / schema-mismatched records.
            # The original list comprehension raised TypeError on any record
            # with a missing or renamed field and propagated as an unhandled
            # exception through asyncio.to_thread into the orchestrator's
            # silent catch-all — producing INTERNAL_ERROR on every query
            # for the lifetime of the process with no logged cause.
            records: list[MemoryRecord] = []
            for r in raw:
                try:
                    records.append(MemoryRecord(**r))
                except (TypeError, KeyError):
                    # Skip malformed records silently — the rest of the index
                    # remains usable. A full rebuild (rebuild_if_stale) will
                    # drop them permanently on the next maintenance cycle.
                    continue
            self._records = records
            # M-10: rebuild dedup key set from the loaded records.
            self._dedup_keys = {
                hashlib.sha256(f"{r.schema_id}::{r.nl_query}".encode()).hexdigest()
                for r in self._records
            }
        else:
            self._index = self._new_index()
            self._records = []
            self._dedup_keys = set()

    def save(self) -> None:
        """
        Persist index and metadata to disk.

        BUG-06 FIX: disk I/O is now performed OUTSIDE self._lock.  Previously
        save() was called from inside `with self._lock:` blocks in store() and
        rebuild_if_stale(), blocking concurrent search() calls for the full
        duration of faiss.write_index() + JSON write (20–80 ms on SSDs).

        Strategy:
          1. Acquire the lock briefly to serialise the FAISS index to bytes in
             memory (faiss.serialize_index() — no I/O, sub-millisecond) and
             snapshot the records list.  Lock released immediately after.
          2. Write bytes and JSON to temp files outside the lock (search()
             proceeds concurrently during this window).
          3. Atomic replace: temp → final path so readers never see a partial file
             (Path.replace() uses os.replace() which is atomic and overwrites the
             target even on Windows, unlike Path.rename() which raises FileExistsError
             when the target already exists on Windows — WIN-01 FIX).
          4. Re-acquire the lock briefly to clear self._dirty.

        Callers (store, rebuild_if_stale) must call save() OUTSIDE their own
        `with self._lock:` blocks, since threading.Lock is not reentrant.
        """
        import faiss

        # ── Phase 1: snapshot under lock (no I/O) ────────────────────────────
        with self._lock:
            if self._index is None:
                return
            # faiss.serialize_index() produces the same binary representation
            # as faiss.write_index() but writes to a numpy uint8 array,
            # letting us safely release the lock before touching the filesystem.
            index_bytes: bytes = faiss.serialize_index(self._index).tobytes()
            records_snapshot: list[dict] = [asdict(r) for r in self._records]

        # ── Phase 2: write to disk outside lock (search() is unblocked) ──────
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_index = self._index_path.with_suffix(".tmp")
        tmp_meta = self._meta_path.with_suffix(".tmp")
        try:
            with open(tmp_index, "wb") as fh:
                fh.write(index_bytes)
            tmp_index.replace(self._index_path)  # WIN-01 FIX: replace() overwrites on Windows
            with open(tmp_meta, "w") as fh:
                json.dump(records_snapshot, fh)
            tmp_meta.replace(self._meta_path)  # WIN-01 FIX: replace() overwrites on Windows
        finally:
            # Clean up temp files if rename failed (e.g. disk-full crash)
            for _p in (tmp_index, tmp_meta):
                with contextlib.suppress(OSError):
                    _p.unlink(missing_ok=True)

        # ── Phase 3: clear dirty flag under lock ─────────────────────────────
        with self._lock:
            self._dirty = False

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        session_id: str,
        schema_id: str,
        nl_query: str,
        sql: str,
        insight: str,
    ) -> bool:
        """
        Embed nl_query and add to the index.

        Returns False and skips if:
          - nl_query contains detected PII
          - An identical nl_query+schema_id is already stored (dedup)
          - MAX_RECORDS has been reached
        """
        if _contains_pii(nl_query):
            return False

        with self._lock:
            if self._index is None:
                self.load()

            if len(self._records) >= MAX_RECORDS:
                return False

            # Dedup: skip if identical nl_query+schema_id already stored.
            # M-10 FIX: O(1) set lookup — self._dedup_keys is maintained
            # incrementally in load() and store() rather than recomputed
            # from scratch (O(N) SHA-256 ops) on every store() call.
            dedup_key = hashlib.sha256(f"{schema_id}::{nl_query}".encode()).hexdigest()
            if dedup_key in self._dedup_keys:
                return False

            vec = self._embed_cpu(nl_query)
            self._index.add(vec)  # type: ignore[arg-type]

            record = MemoryRecord(
                record_id=dedup_key[:16],
                session_id=session_id,
                schema_id=schema_id,
                nl_query=nl_query,
                sql=sql,
                insight=insight,
                created_at=datetime.now(tz=UTC).isoformat(),
            )
            self._records.append(record)
            self._dedup_keys.add(dedup_key)  # M-10: keep set in sync
            self._dirty = True

        # BUG-06 FIX: save() is now called OUTSIDE the lock.  save() acquires
        # its own brief internal lock only for the in-memory serialisation
        # phase; the disk I/O runs without holding any lock.
        # threading.Lock is not reentrant: calling save() (which now acquires
        # self._lock) from inside a `with self._lock:` block would deadlock.
        #
        # Durability rationale (unchanged): save on every successful store so
        # fire-and-forget callers (_persist_turn) never lose data if the
        # process exits between stores.
        self.save()

        return True

    def invalidate(self, schema_id: str, nl_query: str) -> bool:
        """
        Remove a single stored record by nl_query + schema_id key.

        Called when an LTM exact-hit result fails downstream validation
        (e.g. METRIC_OUT_OF_RANGE), indicating the cached SQL is stale or
        was stored before the current prompt fixes landed.  Removing the
        entry lets the next store() call for the same query replace it
        with freshly-generated correct SQL.

        HNSW has no in-place deletion API, so the FAISS index is rebuilt
        from the remaining live records — same pattern as rebuild_if_stale().
        O(N × embedding_cost), fast in practice because MAX_RECORDS is small.

        Returns True if the record was found and removed, False otherwise.
        Safe to call when the record does not exist.
        """
        dedup_key = hashlib.sha256(f"{schema_id}::{nl_query}".encode()).hexdigest()

        with self._lock:
            if self._index is None:
                self.load()

            if dedup_key not in self._dedup_keys:
                return False

            # Remove the matching record — match on content, not truncated id.
            self._records = [
                r
                for r in self._records
                if not (r.nl_query == nl_query and r.schema_id == schema_id)
            ]
            self._dedup_keys.discard(dedup_key)

            # Rebuild FAISS index from remaining records.
            live_texts = [r.nl_query for r in self._records]
            new_index = self._new_index()
            if live_texts:
                _model = getattr(self._embedder, "_model", None)
                _orig_device = None
                try:
                    if _model is not None:
                        _orig_device = next(_model.parameters()).device
                        if _orig_device.type != "cpu":
                            _model.to("cpu")
                    # P3-07 FIX: was self._embedder.embed(live_texts).
                    # store() encodes via _embed_cpu() → embed_query() (single string).
                    # embed() is the batch variant; if the two methods ever diverge
                    # in normalisation or pooling, stored vectors (embed_query) and
                    # rebuilt vectors (embed) would be in different sub-spaces, causing
                    # silent HNSW inner-product score degradation after every invalidate.
                    # Using embed_query() per record eliminates the divergence.
                    # Device is already on CPU from the block above; no extra round-trip.
                    vecs = np.vstack(
                        [
                            self._embedder.embed_query(t).reshape(1, -1).astype(np.float32)
                            for t in live_texts
                        ]
                    )
                finally:
                    if (
                        _model is not None
                        and _orig_device is not None
                        and _orig_device.type != "cpu"
                    ):
                        _model.to(_orig_device)
                new_index.add(vecs)  # type: ignore[arg-type]
            self._index = new_index
            self._dirty = True

        # save() outside the lock — consistent with store() and rebuild_if_stale().
        self.save()
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        nl_query: str,
        schema_id: str,
        k: int = 3,
    ) -> list[MemorySearchResult]:
        """
        Search for the top-k most similar past queries for the given schema.

        Returns [] if no records exist or no result exceeds SIMILARITY_THRESHOLD.
        Results are filtered to the same schema_id and non-stale records.
        """
        with self._lock:
            if self._index is None:
                self.load()

            if not self._records:
                return []

            # Filter active (non-stale) records for this schema
            active_indices = self._active_indices(schema_id)
            if not active_indices:
                return []

            vec = self._embed_cpu(nl_query)
            self._index.hnsw.efSearch = HNSW_EF_SEARCH

            # Search with larger k and post-filter to schema_id + staleness
            search_k = min(k * 5, len(self._records))
            distances, indices = self._index.search(vec, search_k)  # type: ignore[arg-type]

            active_set = set(active_indices)
            results: list[MemorySearchResult] = []

            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx == -1 or idx not in active_set:
                    continue
                similarity = float(dist)  # HNSW with IP metric: dist = inner product
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                record = self._records[idx]
                results.append(
                    MemorySearchResult(
                        record=record,
                        similarity=similarity,
                        is_exact_hit=similarity >= EXACT_HIT_THRESHOLD,
                    )
                )
                if len(results) == k:
                    break

            return sorted(results, key=lambda r: r.similarity, reverse=True)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def rebuild_if_stale(self) -> bool:
        """
        Rebuild the index from scratch if >20% of records are stale.
        Returns True if a rebuild was performed.

        BUG-4 FIX: the entire body is now executed under self._lock.
        Previously, self._index and self._records were mutated without holding
        the lock, racing with concurrent store() / search() calls (both of
        which do acquire the lock). The misleading comment "rebuild_if_stale()
        holds self._lock" in the previous version was factually wrong — no lock
        was acquired. Acquiring it here makes the claim true.
        """
        with self._lock:
            if not self._records:
                return False

            stale_count = len(self._records) - len(self._active_indices(schema_id=None))
            stale_fraction = stale_count / len(self._records)

            if stale_fraction < 0.20:
                return False

            # Remove stale records — P3-F FIX: compare datetime objects, not
            # ISO-8601 strings.  String comparison works for +00:00 vs +00:00
            # but breaks silently if any record uses the 'Z' suffix variant.
            cutoff_dt = _cutoff_timestamp()
            live_records = [
                r for r in self._records if datetime.fromisoformat(r.created_at) >= cutoff_dt
            ]
            live_texts = [r.nl_query for r in live_records]

            new_index = self._new_index()
            if live_texts:
                # Move model to CPU for batch re-embedding — same GPU contention
                # fix as _embed_cpu. The lock is held here so the device movement
                # is safe from concurrent store()/search() access.
                _model = getattr(self._embedder, "_model", None)
                _orig_device = None
                try:
                    if _model is not None:
                        _orig_device = next(_model.parameters()).device
                        if _orig_device.type != "cpu":
                            _model.to("cpu")
                    # P3-07 FIX: was self._embedder.embed(live_texts).
                    # See invalidate() for the full rationale.  Both rebuild
                    # paths must use embed_query() to stay consistent with
                    # the store() path which calls _embed_cpu() → embed_query().
                    vecs = np.vstack(
                        [
                            self._embedder.embed_query(t).reshape(1, -1).astype(np.float32)
                            for t in live_texts
                        ]
                    )
                finally:
                    if (
                        _model is not None
                        and _orig_device is not None
                        and _orig_device.type != "cpu"
                    ):
                        _model.to(_orig_device)
                new_index.add(vecs)  # type: ignore[arg-type]

            self._index = new_index
            self._records = live_records
            # P2-06 FIX: rebuild _dedup_keys from the surviving records.
            # Previously _dedup_keys was never updated here, so the SHA-256
            # hashes of evicted records remained in the set.  Any NL query
            # that matched a stale record was permanently blocked from being
            # re-stored — store() found its hash in _dedup_keys and returned
            # False even though the corresponding record no longer existed in
            # self._records or the FAISS index.
            # This mirrors the rebuild performed in load() (line 229) and
            # invalidate() (line 391) so all three mutation paths are consistent.
            self._dedup_keys = {
                hashlib.sha256(f"{r.schema_id}::{r.nl_query}".encode()).hexdigest()
                for r in live_records
            }
            self._dirty = True
            # BUG-06 FIX: save() now owns its own brief lock acquisition;
            # calling it inside this with-block would deadlock (non-reentrant
            # threading.Lock).  Save after releasing the lock.

        # save() outside the lock — consistent with store() fix above.
        self.save()
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "active_records": len(self._active_indices(schema_id=None)),
            "index_size": self._index.ntotal if self._index else 0,
            "dirty": self._dirty,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _new_index(self) -> Any:
        import faiss

        index = faiss.IndexHNSWFlat(self._dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        return index

    def _active_indices(self, schema_id: str | None) -> list[int]:
        """
        Return indices of records that are non-stale and match schema_id.
        If schema_id is None, return all non-stale indices.

        P3-F FIX: datetime comparison via _cutoff_timestamp() (which now
        returns a datetime object) replaces fragile ISO-8601 string comparison.
        """
        cutoff_dt = _cutoff_timestamp()
        return [
            i
            for i, r in enumerate(self._records)
            if datetime.fromisoformat(r.created_at) >= cutoff_dt
            and (schema_id is None or r.schema_id == schema_id)
        ]


def _cutoff_timestamp() -> datetime:
    """
    Return the earliest datetime a record must have to be considered active.

    P3-F FIX: returns a datetime object, not an ISO-8601 string.
    String comparison of ISO-8601 datetimes breaks silently if one side uses
    the 'Z' suffix and the other uses '+00:00'.  All callers now do:
        datetime.fromisoformat(r.created_at) >= _cutoff_timestamp()
    which is unambiguous regardless of suffix format.
    """
    from datetime import timedelta

    return datetime.now(tz=UTC) - timedelta(days=RECORD_TTL_DAYS)
