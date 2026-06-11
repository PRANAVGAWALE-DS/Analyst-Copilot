"""
retrieval.py — RetrievalLayer, SchemaEmbedder, FAISSIndexer, IngestionPipeline
Data Analyst Copilot · Python 3.11+ · Sections 2A-2D

Implements the RetrievalLayer interface from orchestrator.py:

    class RetrievalLayer:
        async def retrieve(nl_query, schema_id, k=5) -> list[SchemaChunk]
        async def get_schema_columns(schema_id) -> set[str]
        async def get_table_policies(schema_id) -> dict[str, TablePolicy]

Also exposes:
  SchemaRegistry      — in-memory registry of ingested SchemaProfile objects
  IngestionPipeline   — extract → profile → chunk → embed → index
  SchemaEmbedder      — sentence-transformers local embedder with cache
  FAISSIndexer        — HNSW index wrapper

Design choice: HNSW over Flat/IVF (Section 2C).
  Reason: at scale target (500 schemas × ≤200 tables = ≤100,000 chunks),
  HNSW provides O(log N) query time with high recall@5, vs Flat's O(N).
  IVF requires a pre-training pass and degrades on small corpora.
  Trade-off: higher memory (~1.1 × dimension × 4 bytes × N).
  At 1024-dim (BGE-large-en-v1.5), 100k chunks = ~400MB — acceptable on 16GB RAM.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken as _tiktoken
from interfaces import (
    FKRelationship,
    SchemaChunk,
    SchemaColumn,
)
from validation import TablePolicy

# Must run before any lazy import of faiss / sentence_transformers.
# Both are imported lazily inside methods (SchemaEmbedder.embed /
# FAISSIndexer.save / FAISSIndexer.load), so setting this at module
# load time is sufficient to suppress GPU initialisation on CPU-only
# deployments.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Gap-9: token budget for individual schema chunks.
# Hard cap: 400 tokens per chunk. Truncation priority (design §2B):
#   1. Strip column_description annotations first (advisory only).
#   2. Strip sample_values next (nice-to-have context).
#   3. Column names and data types are NEVER truncated.
_CHUNK_TOKEN_CAP = 400
# Encoder is initialised lazily on first call to _count_chunk_tokens().
# This prevents a startup crash when the tiktoken cache file hasn't been
# downloaded yet (e.g. first container boot without the Dockerfile pre-warm).
# The Dockerfile pre-warms the cache at build time so this path is only hit
# in dev environments without a pre-built image.
_chunk_encoder: _tiktoken.Encoding | None = None


def _count_chunk_tokens(text: str) -> int:
    global _chunk_encoder
    if _chunk_encoder is None:
        _chunk_encoder = _tiktoken.get_encoding("cl100k_base")
    return len(_chunk_encoder.encode(text))


# ---------------------------------------------------------------------------
# SchemaProfile (internal — not in interfaces.py)
# ---------------------------------------------------------------------------


@dataclass
class ColumnMeta:
    name: str
    data_type: str
    nullable: bool
    null_rate: float | None = None
    cardinality_estimate: int | None = None
    sample_values: list[Any] = field(default_factory=list)
    is_pii: bool = False
    column_description: str | None = None


@dataclass
class TableMeta:
    table_name: str
    columns: list[ColumnMeta]
    foreign_keys: list[dict[str, str]] = field(default_factory=list)
    row_count_estimate: int | None = None
    is_pii_flagged: bool = False
    business_description: str | None = None


@dataclass
class SchemaProfile:
    schema_id: str
    dialect: str
    tables: list[TableMeta]
    ingested_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


# ---------------------------------------------------------------------------
# SchemaRegistry
# ---------------------------------------------------------------------------


class SchemaRegistry:
    """
    In-memory registry of ingested SchemaProfile objects.
    Populated by IngestionPipeline.run().
    """

    def __init__(self) -> None:
        self._profiles: dict[str, SchemaProfile] = {}

    def put(self, schema_id: str, profile: SchemaProfile) -> None:
        self._profiles[schema_id] = profile

    def get(self, schema_id: str) -> SchemaProfile | None:
        return self._profiles.get(schema_id)

    def list_ids(self) -> list[str]:
        return list(self._profiles.keys())

    def all_columns(self, schema_id: str) -> set[str]:
        profile = self._profiles.get(schema_id)
        if profile is None:
            return set()
        return {col.name.lower() for tbl in profile.tables for col in tbl.columns}

    def table_policies(self, schema_id: str) -> dict[str, TablePolicy]:
        profile = self._profiles.get(schema_id)
        if profile is None:
            return {}
        return {
            tbl.table_name: TablePolicy(
                table_name=tbl.table_name,
                pii_flagged=tbl.is_pii_flagged,
            )
            for tbl in profile.tables
        }


# ---------------------------------------------------------------------------
# SchemaEmbedder
# ---------------------------------------------------------------------------

_EMBED_CACHE_FILENAME = "embed_cache.json"


class SchemaEmbedder:
    """
    Embeds text chunks using a sentence-transformers model (local inference).

    Model: BAAI/bge-large-en-v1.5 (default) — 1024 dimensions, strong
    retrieval recall on domain-specific text.
    Alternative: "gemini:text-embedding-004" — swap prefix to use
    Google Gemini's embedding API instead (set EMBEDDING_MODEL env var).

    Embedding cache: SHA-256(text) → vector, persisted as JSON alongside
    the FAISS index at data/faiss_index/embed_cache.json by default.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path = "data/faiss_index",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self._cache_path = Path(cache_dir) / _EMBED_CACHE_FILENAME
        self._legacy_cache_path = Path("data") / _EMBED_CACHE_FILENAME
        self._batch_size = batch_size
        self._model: Any = None
        self._cache: dict[str, list[float]] = {}
        self._is_gemini = self._model_name.startswith("gemini:")

    def _load_model(self) -> None:
        if self._model is not None or self._is_gemini:
            return

        os.environ.setdefault("SAFETENSORS_FAST_GPU", "0")
        # P3-C FIX: setdefault — respect any GPU env var the caller set
        # explicitly before startup.  The module-level setdefault already
        # disables GPU for fresh processes; this is only a safety-net guard.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        self._model = self._model.to("cpu")

    def _load_cache(self) -> None:
        self._migrate_legacy_cache()
        if self._cache_path.exists():
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    # M2 FIX: JSON instead of pickle.
                    # pickle.load() executes arbitrary Python during
                    # deserialization — a malicious embed_cache file achieves
                    # RCE at startup.  The cache stores {sha256_hex: list[float]},
                    # which is natively JSON-serialisable with no loss of fidelity.
                    self._cache = json.load(f)
            except Exception:  # noqa: BLE001
                self._cache = {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    def _migrate_legacy_cache(self) -> None:
        """
        Preserve embeddings created before the cache lived beside FAISS files.
        Copy only when the consolidated cache is absent.
        """
        if self._cache_path.exists() or not self._legacy_cache_path.exists():
            return
        if self._legacy_cache_path.resolve() == self._cache_path.resolve():
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._legacy_cache_path, self._cache_path)

    def warmup(self) -> None:
        """
        Pre-load the embedding model and cache into memory.
        Call once at application startup so the first request doesn't pay
        the cold-start penalty (BAAI/bge-large-en-v1.5 loads in ~35–40s).
        """
        self._load_model()
        self._load_cache()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns shape (N, D) float32 ndarray."""
        self._load_model()
        self._load_cache()

        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        miss_indices = [i for i, k in enumerate(keys) if k not in self._cache]

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            vectors = self._embed_batch(miss_texts)
            for idx, vec in zip(miss_indices, vectors, strict=False):
                self._cache[keys[idx]] = vec.tolist()
            self._save_cache()

        return np.array([self._cache[k] for k in keys], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        if self._is_gemini:
            return self._normalize_l2(self._embed_gemini(texts))
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _embed_gemini(self, texts: list[str]) -> np.ndarray:
        import os

        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        model = self._model_name.replace("gemini:", "")
        if model == "text-embedding-004":
            model = "gemini-embedding-001"

        # FIX: default corrected from 768 → 1024 to match bge-large-en-v1.5.
        # For Gemini models, output_dimensionality truncates via Matryoshka;
        # 1024 is valid for text-embedding-004 and gemini-embedding-001.
        output_dimensionality = int(os.environ.get("EMBEDDING_DIM", "1024"))
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=output_dimensionality,
        )
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = client.models.embed_content(
                model=model,
                contents=batch,
                config=config,
            )
            embeddings = resp.embeddings or []
            vecs.extend([embedding.values or [] for embedding in embeddings])
        return np.array(vecs, dtype=np.float32)

    def _normalize_l2(self, vecs: np.ndarray) -> np.ndarray:
        """
        H5 FIX: L2-normalise embedding vectors in-place.
        FAISS HNSW with METRIC_INNER_PRODUCT equals cosine similarity ONLY for
        unit-norm vectors.  sentence-transformers enforces this via
        normalize_embeddings=True; the Gemini API returns raw un-normalised
        vectors, making similarity scores and thresholds meaningless without
        this step.  A zero-vector (degenerate edge-case) is left unchanged to
        avoid division-by-zero.
        """
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vecs / norms

    @property
    def dimension(self) -> int:
        self._load_model()
        if self._is_gemini:
            # ML-6 FIX: return the actual output dimension as controlled by
            # EMBEDDING_DIM env var, not the hardcoded 768 default.
            # Gemini models support Matryoshka truncation — the output dimension
            # is whatever output_dimensionality was passed in _embed_gemini()
            # (defaults to EMBEDDING_DIM, not the 768 bare-model default).
            # Returning 768 while the index was built at 1024 would cause any
            # caller using .dimension to set a FAISS index size to build a
            # mis-sized index.
            return int(os.environ.get("EMBEDDING_DIM", "1024"))
        return self._model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# FAISSIndexer — HNSW index per schema_id
# ---------------------------------------------------------------------------


class FAISSIndexer:
    """
    Maintains one FAISS HNSW index per schema_id, stored under index_dir.

    Index files:
      {index_dir}/{schema_id}.faiss   — the FAISS binary index
      {index_dir}/{schema_id}.meta    — JSON list of chunk metadata dicts

    HNSW parameters:
      M=32 (neighbours per node) — higher M increases recall at cost of memory.
      ef_construction=200        — higher = better recall at build time.
      ef_search=50               — set at query time for recall/speed tradeoff.
    """

    _HNSW_M = 32
    _HNSW_EF_CONSTRUCTION = 200
    _HNSW_EF_SEARCH = 50

    def __init__(
        self,
        index_dir: str | Path = "data/faiss_index",
        # FIX: default corrected from 768 → 1024 (bge-large-en-v1.5 output dim).
        dimension: int = 1024,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._dimension = dimension
        self._indices: dict[str, Any] = {}  # schema_id → faiss.Index
        self._chunk_meta: dict[str, list[dict]] = {}  # schema_id → list of chunk dicts

    def _index_path(self, schema_id: str) -> Path:
        return self._index_dir / f"{schema_id}.faiss"

    def _meta_path(self, schema_id: str) -> Path:
        return self._index_dir / f"{schema_id}.meta"

    def _load(self, schema_id: str) -> bool:
        """Load an existing index from disk. Returns True if loaded."""
        import faiss

        idx_p = self._index_path(schema_id)
        meta_p = self._meta_path(schema_id)
        if not idx_p.exists() or not meta_p.exists():
            return False
        self._indices[schema_id] = faiss.read_index(str(idx_p))
        with open(meta_p) as f:
            self._chunk_meta[schema_id] = json.load(f)
        return True

    def add(
        self,
        schema_id: str,
        embeddings: np.ndarray,
        chunks: list[dict[str, Any]],
        force: bool = False,
    ) -> None:
        """
        Build or replace the HNSW index for schema_id.

        Parameters
        ----------
        embeddings : shape (N, D) float32
        chunks     : list of dicts with at minimum {"chunk_id", "table_name", "text"}
        force      : If True, rebuild even if the index already exists on disk.
        """
        import faiss

        if not force and self._load(schema_id):
            return

        self._index_dir.mkdir(parents=True, exist_ok=True)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be a 2-D array, got shape {embeddings.shape}.")
        # C3-adjacent FIX: correct self._dimension BEFORE passing it to
        # IndexHNSWFlat.  The old order set self._dimension after constructing
        # the index, meaning wrong: EMBEDDING_DIM=768 while model outputs 1024
        # was used to build the index — FAISS then raised an assertion error when
        # embeddings.shape[1] didn't match.
        if embeddings.shape[1] != self._dimension:
            self._dimension = int(embeddings.shape[1])

        # Build HNSW index — use METRIC_INNER_PRODUCT so ranking is equivalent
        # to cosine similarity for normalized vectors (sentence-transformers
        # normalizes by default). Consistent with LongTermMemory._new_index().
        index = faiss.IndexHNSWFlat(self._dimension, self._HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._HNSW_EF_CONSTRUCTION
        index.add(embeddings.astype(np.float32))  # type: ignore[arg-type]

        # Persist
        faiss.write_index(index, str(self._index_path(schema_id)))
        with open(self._meta_path(schema_id), "w") as f:
            json.dump(chunks, f)

        self._indices[schema_id] = index
        self._chunk_meta[schema_id] = chunks

    def list_indexed_schemas(self) -> list[str]:
        """
        Return schema_ids for which both .faiss and .meta files exist on disk.
        Used by RetrievalLayer._bootstrap_registry() to reconstruct
        SchemaProfile entries after a process restart without a full re-ingest.
        """
        if not self._index_dir.exists():
            return []
        return [
            p.stem
            for p in self._index_dir.glob("*.faiss")
            if (self._index_dir / f"{p.stem}.meta").exists()
        ]

    def search(
        self,
        schema_id: str,
        query_vec: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for top-k most similar chunks to query_vec.

        Returns list of chunk dicts sorted by similarity (most similar first).
        Returns [] if schema_id has no index.
        """

        if schema_id not in self._indices and not self._load(schema_id):
            return []

        index = self._indices[schema_id]
        index.hnsw.efSearch = self._HNSW_EF_SEARCH
        q = query_vec.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(q, k)  # type: ignore[arg-type]

        meta = self._chunk_meta.get(schema_id, [])
        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx == -1:
                continue
            if idx < len(meta):
                results.append({**meta[idx], "_score": float(dist)})
        return results


# ---------------------------------------------------------------------------
# Chunk builder — Section 2B: table-level hybrid chunking
# ---------------------------------------------------------------------------


def _hash_sample(val: Any) -> str:
    return hashlib.sha256(str(val).encode()).hexdigest()[:16]


def build_chunks(profile: SchemaProfile) -> list[dict[str, Any]]:
    """
    One chunk per table. Text representation optimised for embedding.

    Gap-9 FIX: enforces a hard 400-token cap per chunk.
    Truncation priority (design §2B — column names/types are NEVER truncated):
      Pass 1 — full chunk (names + types + nullability + cardinality + samples + descriptions + FKs)
      Pass 2 — strip column_description annotations
      Pass 3 — strip sample_values
      Pass 4 — if still over (pathological: 200+ column tables), log a warning
                and emit the chunk anyway — names + types are always kept.
    """
    chunks: list[dict[str, Any]] = []

    for tbl in profile.tables:

        def _build_text(
            _tbl: TableMeta,
            include_descriptions: bool,
            include_samples: bool,
        ) -> tuple[str, list[dict[str, str | None]], list[str]]:
            """Build chunk text + metadata with optional description/sample stripping."""
            col_lines = []
            columns_meta: list[dict[str, str | None]] = []
            for col in _tbl.columns:
                null_s = f", null_rate={col.null_rate:.1%}" if col.null_rate is not None else ""
                card_s = (
                    f", cardinality≈{col.cardinality_estimate}" if col.cardinality_estimate else ""
                )
                if include_samples:
                    samples = col.sample_values
                    if col.is_pii:
                        samples = [_hash_sample(v) for v in samples]
                    samp_s = f", samples={samples[:3]}" if samples else ""
                else:
                    samp_s = ""

                desc_s = (
                    f" NOTE: {col.column_description}"
                    if include_descriptions and col.column_description
                    else ""
                )
                col_lines.append(
                    f"  - {col.name}: {col.data_type}"
                    f"{'  NOT NULL' if not col.nullable else ''}"
                    f"{null_s}{card_s}{samp_s}{desc_s}"
                )
                columns_meta.append(
                    {
                        "name": col.name,
                        "description": (col.column_description if include_descriptions else None),
                    }
                )

            fk_lines = []
            for fk in _tbl.foreign_keys:
                fk_lines.append(
                    f"  - {_tbl.table_name}.{fk['from_column']} → "
                    f"{fk['to_table']}.{fk['to_column']}"
                )

            parts = [
                f"Table: {_tbl.table_name}",
                f"Schema: {profile.schema_id}",
                f"Row count (estimate): {_tbl.row_count_estimate or 'unknown'}",
            ]
            if _tbl.business_description:
                parts.append(f"Description: {_tbl.business_description}")
            parts.append("Columns:")
            parts.extend(col_lines)
            if fk_lines:
                parts.append("Foreign keys:")
                parts.extend(fk_lines)

            return "\n".join(parts), columns_meta, fk_lines

        # --- Token-capped build: three passes ---
        # Pass 1: full fidelity
        text, columns_meta, _ = _build_text(tbl, include_descriptions=True, include_samples=True)
        if _count_chunk_tokens(text) > _CHUNK_TOKEN_CAP:
            # Pass 2: strip column descriptions
            text, columns_meta, _ = _build_text(
                tbl, include_descriptions=False, include_samples=True
            )
        if _count_chunk_tokens(text) > _CHUNK_TOKEN_CAP:
            # Pass 3: strip samples too
            text, columns_meta, _ = _build_text(
                tbl, include_descriptions=False, include_samples=False
            )
        if _count_chunk_tokens(text) > _CHUNK_TOKEN_CAP:
            # Pass 4: pathological case (200+ column table). Emit as-is with a
            # warning — column names and types are never truncated per design §2B.
            import sys

            # M-11 FIX: was file=sys.stderr; changed to sys.stdout.
            # Every other structured log in the application writes to stdout.
            # Docker and most log aggregators (CloudWatch, Loki, Datadog) capture
            # stdout by default; stderr is a separate stream that is often not
            # collected, making this overflow event invisible in production.
            print(
                json.dumps(
                    {
                        "event": "CHUNK_TOKEN_OVERFLOW",
                        "table": tbl.table_name,
                        "schema_id": profile.schema_id,
                        "tokens": _count_chunk_tokens(text),
                        "cap": _CHUNK_TOKEN_CAP,
                        "message": (
                            "Chunk exceeds 400-token cap after all truncation passes. "
                            "Column names and types are preserved. Consider splitting "
                            "this table's schema into logical sub-groups."
                        ),
                    }
                ),
                file=sys.stdout,
                flush=True,
            )

        chunks.append(
            {
                "chunk_id": f"{profile.schema_id}::{tbl.table_name}",
                "schema_id": profile.schema_id,
                "table_name": tbl.table_name,
                "text": text,
                "column_names": [c.name for c in tbl.columns],
                "columns_meta": columns_meta,
                "fk_targets": [fk["to_table"] for fk in tbl.foreign_keys],
                "fk_relationships": [
                    {
                        "from_column": fk["from_column"],
                        "to_table": fk["to_table"],
                        "to_column": fk["to_column"],
                    }
                    for fk in tbl.foreign_keys
                ],
                "is_pii_flagged": tbl.is_pii_flagged,
                "business_description": tbl.business_description,
                "row_count_estimate": tbl.row_count_estimate,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# RetrievalLayer — implements orchestrator.py interface
# ---------------------------------------------------------------------------


class RetrievalLayer:
    """
    Production RetrievalLayer. Implements the interface from orchestrator.py.

    Parameters
    ----------
    embedder  : SchemaEmbedder instance.
    indexer   : FAISSIndexer instance.
    registry  : SchemaRegistry instance.
    """

    def __init__(
        self,
        embedder: SchemaEmbedder,
        indexer: FAISSIndexer,
        registry: SchemaRegistry,
    ) -> None:
        self._embedder = embedder
        self._indexer = indexer
        self._registry = registry
        self._bootstrapped = False

    def warmup(self) -> None:
        """
        Pre-load the embedding model and bootstrap the registry from any
        persisted .meta files.  Call once at application startup — moves
        cold-start latency (~35–40s for bge-large) out of the first request.
        """
        self._embedder.warmup()
        self._bootstrap_registry()

    def _bootstrap_registry(self) -> None:
        """
        Reconstruct SchemaProfile entries from persisted .meta files for any
        schema_id not already in the registry.  This restores column metadata
        after a process restart without requiring a full re-ingest against
        the live database.

        Each .meta file is a JSON list of chunk dicts produced by build_chunks().
        Every chunk carries 'column_names' and 'fk_targets' — sufficient to
        populate SchemaRegistry.all_columns() which is what validate_sql uses
        for grounding checks.
        """
        if self._bootstrapped:
            return
        self._bootstrapped = True

        for schema_id in self._indexer.list_indexed_schemas():
            if self._registry.get(schema_id) is not None:
                continue  # already populated (e.g. ingest ran in this process)

            meta_path = self._indexer._meta_path(schema_id)
            try:
                with open(meta_path) as f:
                    chunks: list[dict] = json.load(f)
            except Exception:  # noqa: BLE001
                continue

            tables: list[TableMeta] = []
            for chunk in chunks:
                tbl_name = chunk.get("table_name", "")
                col_names: list[str] = chunk.get("column_names", [])
                columns = [
                    ColumnMeta(name=c, data_type="unknown", nullable=True) for c in col_names
                ]
                fks = [
                    {"from_column": "", "to_table": to_tbl, "to_column": ""}
                    for to_tbl in chunk.get("fk_targets", [])
                ]
                tables.append(
                    TableMeta(
                        table_name=tbl_name,
                        columns=columns,
                        foreign_keys=fks,
                        is_pii_flagged=chunk.get("is_pii_flagged", False),
                    )
                )

            # dialect is not stored in .meta; "unknown" is safe — it's only
            # used by _quote() in IngestionPipeline, not in retrieval paths.
            profile = SchemaProfile(
                schema_id=schema_id,
                dialect="unknown",
                tables=tables,
            )
            self._registry.put(schema_id, profile)

    async def retrieve(
        self,
        nl_query: str,
        schema_id: str,
        k: int = 5,
    ) -> list[SchemaChunk]:
        """
        Returns top-K SchemaChunk objects for the given NL query.
        Returns [] if schema_id is not indexed (orchestrator asks clarification).

        This is an async method matching the orchestrator interface, but
        embedding and FAISS search are CPU-bound — asyncio.get_running_loop()
        .run_in_executor() is used so the event loop is not blocked.

        _bootstrap_registry() is called here as a lazy fallback in case
        warmup() was not called at startup (dev/test paths).
        """
        self._bootstrap_registry()
        import asyncio

        loop = asyncio.get_running_loop()
        query_vec = await loop.run_in_executor(None, self._embedder.embed_query, nl_query)
        raw_chunks = await loop.run_in_executor(None, self._indexer.search, schema_id, query_vec, k)

        if not raw_chunks:
            return []

        return [_chunk_dict_to_schema_chunk(c) for c in raw_chunks]

    async def get_schema_columns(self, schema_id: str) -> set[str]:
        # Bootstrap guard: /execute calls this directly without going through
        # retrieve(), so lazy bootstrap must fire here too.
        self._bootstrap_registry()
        return self._registry.all_columns(schema_id)

    async def get_table_policies(self, schema_id: str) -> dict[str, TablePolicy]:
        self._bootstrap_registry()
        return self._registry.table_policies(schema_id)


def _chunk_dict_to_schema_chunk(chunk: dict[str, Any]) -> SchemaChunk:
    """Convert a raw chunk dict from FAISSIndexer.search() to a SchemaChunk."""
    # Prefer columns_meta (richer); fall back to column_names for old .meta files
    if "columns_meta" in chunk:
        columns = [
            SchemaColumn(
                name=cm["name"],
                type="unknown",
                nullable=True,
                description=cm.get("description"),
            )
            for cm in chunk["columns_meta"]
        ]
    else:
        columns = [
            SchemaColumn(name=name, type="unknown", nullable=True)
            for name in chunk.get("column_names", [])
        ]
    if "fk_relationships" in chunk:
        # New format: full FK metadata with correct column names
        fks = [
            FKRelationship(
                column=fk["from_column"],
                references=f"{fk['to_table']}.{fk['to_column']}",
            )
            for fk in chunk["fk_relationships"]
        ]
    else:
        # Legacy .meta files only stored the target table; from_column is
        # unknown and ".id" is assumed. Re-ingest to get accurate FK data.
        fks = [
            FKRelationship(column="", references=f"{target}.id")
            for target in chunk.get("fk_targets", [])
        ]
    return SchemaChunk(
        table=chunk["table_name"],
        schema_id=chunk["schema_id"],
        columns=columns,
        fk_relationships=fks,
        pii_flagged=chunk.get("is_pii_flagged", False),
        business_description=chunk.get("business_description"),
        row_count_estimate=chunk.get("row_count_estimate"),
    )


# ---------------------------------------------------------------------------
# IngestionPipeline — Section 2A
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """
    Orchestrates: connect → extract → profile → chunk → embed → index → register.
    """

    _SAMPLE_LIMIT = 5
    _CARDINALITY_LIMIT = 50_000

    def __init__(
        self,
        registry: SchemaRegistry,
        embedder: SchemaEmbedder,
        indexer: FAISSIndexer,
    ) -> None:
        self._registry = registry
        self._embedder = embedder
        self._indexer = indexer

    def ingest(
        self,
        schema_id: str,
        database_url: str,
        dialect: str = "postgres",
        table_allowlist: list[str] | None = None,
        pii_tables: list[str] | None = None,
        force_reingest: bool = False,
        table_description_overrides: dict[str, str] | None = None,
        column_description_overrides: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronous ingest entry point (runs DB queries, CPU-bound embedding).
        Wrap in run_in_executor() if calling from an async context.

        Returns a summary dict: {tables_ingested, chunks_indexed, ingestion_time_ms, warnings}.
        """
        from sqlalchemy import create_engine, inspect

        t0 = time.perf_counter()
        warnings: list[str] = []
        pii_set = {t.lower() for t in (pii_tables or [])}

        try:
            engine = create_engine(database_url, pool_pre_ping=True)
            inspector = inspect(engine)
        except Exception as exc:  # noqa: BLE001
            return {
                "tables_ingested": 0,
                "chunks_indexed": 0,
                "ingestion_time_ms": 0,
                "warnings": [f"Connection failed: {exc}"],
            }

        all_tables = inspector.get_table_names()
        tables = [t for t in all_tables if t in table_allowlist] if table_allowlist else all_tables

        table_metas: list[TableMeta] = []
        with engine.connect() as conn:
            for tbl_name in tables:
                try:
                    meta = self._profile_table(conn, inspector, tbl_name, dialect, pii_set)
                    if table_description_overrides:
                        meta.business_description = table_description_overrides.get(
                            tbl_name, meta.business_description
                        )
                    if column_description_overrides:
                        col_overrides = column_description_overrides.get(tbl_name, {})
                        if col_overrides:
                            for col in meta.columns:
                                if col.name in col_overrides:
                                    col.column_description = col_overrides[col.name]
                    table_metas.append(meta)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Failed to profile '{tbl_name}': {exc}")

        profile = SchemaProfile(schema_id=schema_id, dialect=dialect, tables=table_metas)
        self._registry.put(schema_id, profile)

        chunks = build_chunks(profile)

        # Guard: if no tables were profiled (empty DB or all tables failed),
        # skip embedding and indexing rather than passing a shape-(0,) array
        # to FAISSIndexer.add() which raises ValueError.
        if not chunks:
            return {
                "tables_ingested": len(table_metas),
                "chunks_indexed": 0,
                "ingestion_time_ms": int((time.perf_counter() - t0) * 1000),
                "warnings": warnings
                + (
                    [
                        "No tables found — schema index is empty. "
                        "Run scripts/generate_insurance_data.py to populate the database."
                    ]
                    if not table_metas
                    else []
                ),
            }

        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed(texts)
        self._indexer.add(schema_id, embeddings, chunks, force=force_reingest)

        return {
            "tables_ingested": len(table_metas),
            "chunks_indexed": len(chunks),
            "ingestion_time_ms": int((time.perf_counter() - t0) * 1000),
            "warnings": warnings,
        }

    def _profile_table(
        self,
        conn: Any,
        inspector: Any,
        table_name: str,
        dialect: str,
        pii_set: set[str],
    ) -> TableMeta:
        from sqlalchemy import text

        is_pii = table_name.lower() in pii_set
        raw_cols = inspector.get_columns(table_name)
        raw_fks = inspector.get_foreign_keys(table_name)
        q = _quote(table_name, dialect)

        try:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {q}")).scalar()
        except Exception:  # noqa: BLE001
            row_count = None

        columns: list[ColumnMeta] = []
        for col in raw_cols:
            col_name = col["name"]
            null_rate = self._null_rate(conn, table_name, col_name, row_count, dialect)
            samples = self._sample_values(conn, table_name, col_name, dialect, is_pii)
            columns.append(
                ColumnMeta(
                    name=col_name,
                    data_type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    null_rate=null_rate,
                    sample_values=samples,
                    is_pii=is_pii,
                )
            )

        fks = [
            {
                "from_column": (fk["constrained_columns"][0] if fk["constrained_columns"] else ""),
                "to_table": fk["referred_table"],
                "to_column": (fk["referred_columns"][0] if fk["referred_columns"] else ""),
            }
            for fk in raw_fks
            if fk.get("constrained_columns") and fk.get("referred_columns")
        ]

        return TableMeta(
            table_name=table_name,
            columns=columns,
            foreign_keys=fks,
            row_count_estimate=row_count,
            is_pii_flagged=is_pii,
        )

    def _null_rate(
        self, conn: Any, tbl: str, col: str, row_count: int | None, dialect: str
    ) -> float | None:
        from sqlalchemy import text

        if not row_count:
            return None
        try:
            q = f"SELECT COUNT(*) FROM {_quote(tbl, dialect)} WHERE {_quote(col, dialect)} IS NULL"
            null_count = conn.execute(text(q)).scalar() or 0
            return round(null_count / row_count, 4)
        except Exception:  # noqa: BLE001
            return None

    def _sample_values(
        self, conn: Any, tbl: str, col: str, dialect: str, is_pii: bool
    ) -> list[Any]:
        from sqlalchemy import text

        try:
            q = (
                f"SELECT DISTINCT {_quote(col, dialect)} FROM {_quote(tbl, dialect)} "
                f"WHERE {_quote(col, dialect)} IS NOT NULL LIMIT {self._SAMPLE_LIMIT}"
            )
            rows = conn.execute(text(q)).fetchall()
            samples = [r[0] for r in rows]
            if is_pii:
                samples = [_hash_sample(v) for v in samples]
            return samples
        except Exception:  # noqa: BLE001
            return []


def _quote(name: str, dialect: str) -> str:
    """Return a properly quoted identifier for the given SQL dialect.

    Supports postgres, sqlite, bigquery, snowflake (double-quote) and
    mysql / mariadb (backtick).  Falls back to ANSI double-quote for any
    unrecognised dialect so callers always receive a non-None string.
    """
    if dialect in ("postgres", "sqlite", "bigquery", "snowflake"):
        return f'"{name}"'
    if dialect == "mysql":
        return f"`{name}`"
    # ANSI double-quote fallback — safe for any SQL-92-compliant engine.
    return f'"{name}"'
