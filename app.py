"""
app.py — FastAPI application factory
Data Analyst Copilot · Python 3.11+

FIX (v2): Lifespan wiring changed from app.router.lifespan_context override
(unreliable on a pre-constructed app) to FastAPI.router.on_startup /
on_shutdown hooks, which are guaranteed to fire in FastAPI 0.110.x regardless
of how the base app was constructed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import sys
import time
from datetime import UTC
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from analyst_copilot.dataframe_loader import DataFrameLoader
    from analyst_copilot.dataframe_store import DataFrameStore
    from analyst_copilot.long_term_memory import LongTermMemory

import pandas as pd
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Path,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from analyst_copilot.interfaces import (
    ErrorDetail,
    ExecuteRequest,
    HistoryResponse,
    QueryRequest,
    QueryResponse,
    TurnRecord,
)
from analyst_copilot.observability import ObservabilityStack
from analyst_copilot.orchestrator import Orchestrator
from analyst_copilot.prompts import LLMClient, build_llm_client
from analyst_copilot.retrieval import (
    FAISSIndexer,
    IngestionPipeline,
    RetrievalLayer,
    SchemaEmbedder,
    SchemaRegistry,
)
from analyst_copilot.session_store import (
    InMemorySessionStore,
    RedisSessionStore,
    build_session_store,
)

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _require_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            f"Add it to your .env file and restart the server."
        )
    return val


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _slowapi_config_filename() -> str:
    """Use an ASCII-only config file so SlowAPI does not reread .env."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".slowapi.env")
    if not os.path.exists(path):
        with open(path, "w", encoding="ascii") as f:
            f.write("# SlowAPI config placeholder.\n")
    return path


# ---------------------------------------------------------------------------
# App state container
# ---------------------------------------------------------------------------


class AppState:
    obs: ObservabilityStack
    sessions: RedisSessionStore | InMemorySessionStore
    registry: SchemaRegistry
    embedder: SchemaEmbedder
    indexer: FAISSIndexer
    retrieval: RetrievalLayer
    ingestion: IngestionPipeline
    llm: LLMClient
    orchestrator: Orchestrator
    db_engine: Any
    # Phase 2 / Phase 3 extensions — annotated here so type checkers and
    # IDE completion can resolve them (they are set during _startup).
    df_loader: DataFrameLoader
    # Gap-4: DataFrameStore for user-uploaded CSV/Parquet files.
    # Wired into Orchestrator so uploaded DataFrames reach execute_python().
    df_store: DataFrameStore
    long_term_memory: LongTermMemory


# Module-level state holder — set during startup, read by route handlers.
# Using a module-level variable (instead of app.state) avoids the lifespan
# context wiring issue that caused the 503 on the pre-constructed app object.
_app_state: AppState | None = None

# Guard against double-invocation: uvicorn --reload shares the _interfaces_app
# singleton across reloader + worker processes, causing add_event_handler to
# accumulate duplicate registrations each time create_app() is called.
_startup_called: bool = False


# ---------------------------------------------------------------------------
# Startup / shutdown (on_event pattern — reliable on pre-constructed apps)
# ---------------------------------------------------------------------------


async def _startup() -> None:
    global _app_state, _startup_called
    if _startup_called or _app_state is not None:
        _log({"event": "STARTUP_SKIPPED", "reason": "already_initialised"})
        return

    # _startup_called is set to True only after _app_state is successfully
    # assigned (bottom of try block). If anything raises before that point,
    # the except resets it to False so uvicorn --reload can retry startup
    # instead of silently skipping it and serving 503 forever.
    try:
        state = AppState()

        # SEC-5 FIX: warn loudly when API_KEY is not set in production.
        # An empty API_KEY causes the auth middleware to skip all authentication
        # (the guard is `if required_key:`, so "" bypasses it entirely).
        # This is intentional in development but must not happen silently in prod.
        _api_key = os.environ.get("API_KEY", "").strip()
        _app_env = os.environ.get("APP_ENV", "production").lower()
        if not _api_key and _app_env != "development":
            _log(
                {
                    "event": "STARTUP_WARNING",
                    "message": (
                        "API_KEY is not set. ALL requests will be accepted without "
                        "authentication. Set API_KEY in .env before exposing this "
                        "service to untrusted networks."
                    ),
                }
            )

        # 1. Observability
        state.obs = ObservabilityStack()
        _log({"event": "STARTUP", "step": "observability_ok"})

        # 2. Session store
        state.sessions = await build_session_store(
            redis_url=_env("REDIS_URL", "redis://localhost:6379/0")
        )
        _log({"event": "STARTUP", "step": "session_store_ok"})

        # 3. Schema registry + retrieval
        state.registry = SchemaRegistry()
        state.embedder = SchemaEmbedder(
            model_name=_env("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            cache_dir=_env("FAISS_INDEX_DIR", "data/faiss_index"),
            batch_size=int(_env("EMBEDDING_BATCH_SIZE", "64")),
        )
        state.indexer = FAISSIndexer(
            index_dir=_env("FAISS_INDEX_DIR", "data/faiss_index"),
            # FIX: default corrected from 768 → 1024 (bge-large outputs 1024 dims).
            dimension=int(_env("EMBEDDING_DIM", "1024")),
        )
        state.retrieval = RetrievalLayer(
            embedder=state.embedder,
            indexer=state.indexer,
            registry=state.registry,
        )
        state.ingestion = IngestionPipeline(
            registry=state.registry,
            embedder=state.embedder,
            indexer=state.indexer,
        )
        # warmup() is CPU-bound (loads BAAI/bge-large-en-v1.5 ~35-40s and
        # bootstraps the SchemaRegistry from persisted .meta files).
        # Running in executor keeps the event loop responsive during startup.
        _warmup_loop = asyncio.get_running_loop()
        await _warmup_loop.run_in_executor(None, state.retrieval.warmup)
        _log({"event": "STARTUP", "step": "retrieval_ok"})

        # 4. DB engine (read-only)
        from sqlalchemy import text as _sa_text

        from analyst_copilot.validation import _make_readonly_engine

        db_url = _env("DATABASE_URL", "sqlite:///data/dev.db")
        state.db_engine = _make_readonly_engine(db_url)
        # Eagerly verify the DB connection at startup so DNS / credential
        # failures surface here with a clear log entry rather than on the
        # first user request as an opaque DB_UNAVAILABLE error.
        with state.db_engine.connect() as _probe_conn:
            _probe_conn.execute(_sa_text("SELECT 1"))
        _log(
            {
                "event": "STARTUP",
                "step": "db_engine_ok",
                "dialect": db_url.split(":")[0],
            }
        )

        # 5. LLM + orchestrator
        provider = _env("LLM_PROVIDER", "groq")
        # H-11 FIX: validate provider against the supported set before
        # attempting key lookup.  Any value outside this set previously fell
        # through to the Gemini branch silently, producing a confusing
        # "GEMINI_API_KEY not set" error even when the operator intended groq.
        _SUPPORTED_PROVIDERS = {"groq", "gemini"}
        if provider not in _SUPPORTED_PROVIDERS:
            raise RuntimeError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                f"Supported values: {', '.join(sorted(_SUPPORTED_PROVIDERS))}."
            )
        if provider == "groq":
            llm_api_key = _require_env("GROQ_API_KEY")
        else:
            llm_api_key = _require_env("GEMINI_API_KEY")
        llm_model = _env("LLM_MODEL") or None  # None → factory picks provider default
        state.llm = build_llm_client(
            provider=provider,
            api_key=llm_api_key,
            default_model=llm_model,
        )
        # Phase 2: DataFrame loader for Pandas executor (DB-backed tables)
        from analyst_copilot.dataframe_loader import DataFrameLoader

        state.df_loader = DataFrameLoader(
            engine=state.db_engine,
            file_root=_env("FILE_DATA_ROOT", "data/files") or None,
            cache_ttl=int(_env("DF_CACHE_TTL_SECONDS", "300")),
            row_limit=int(_env("DF_ROW_LIMIT", "50000")),
        )
        _log({"event": "STARTUP", "step": "dataframe_loader_ok"})

        # Gap-4 FIX: DataFrameStore for user-uploaded CSV/Parquet files.
        # Previously instantiated but never wired into Orchestrator, so
        # uploaded DataFrames could not reach execute_python() — causing
        # NameError for every upload-based Pandas query.
        from analyst_copilot.dataframe_store import DataFrameStore

        state.df_store = DataFrameStore(
            max_upload_mb=int(_env("MAX_UPLOAD_MB", "50")),
            max_session_mb=int(_env("MAX_SESSION_MB", "256")),
            ttl_seconds=int(_env("DATAFRAME_TTL_SECONDS", "3600")),
        )
        _log({"event": "STARTUP", "step": "dataframe_store_ok"})

        # Phase 3: Long-term memory
        from analyst_copilot.long_term_memory import LongTermMemory

        state.long_term_memory = LongTermMemory(
            embedder=state.embedder,
            index_dir=_env("LT_MEMORY_DIR", "data/lt_memory"),
            # FIX: pass EMBEDDING_DIM explicitly.  Without this, LongTermMemory
            # used its hardcoded default (768) while the embedder produced 1024-dim
            # vectors.  _new_index() built IndexHNSWFlat(768); the first store()
            # call passed a 1024-dim vector → FAISS assertion error.
            # Now a single EMBEDDING_DIM change in .env propagates everywhere.
            dimension=int(_env("EMBEDDING_DIM", "1024")),
            k_retrieve=int(_env("LT_MEMORY_K", "3")),
        )

        # M5 FIX: load persisted index from disk and prune stale records.
        # rebuild_if_stale() was defined but never called from any lifecycle
        # hook, so stale records accumulated indefinitely.  Run in executor
        # to keep the event loop responsive during startup.
        def _init_lt_memory() -> None:
            state.long_term_memory.load()
            state.long_term_memory.rebuild_if_stale()

        await _warmup_loop.run_in_executor(None, _init_lt_memory)
        _log({"event": "STARTUP", "step": "long_term_memory_ok"})

        state.orchestrator = Orchestrator(
            llm=state.llm,
            retrieval=state.retrieval,
            session_store=state.sessions,
            engine=state.db_engine,
            obs=state.obs,
            model=_env("LLM_MODEL", "llama-3.3-70b-versatile"),
            df_loader=state.df_loader,
            df_store=state.df_store,  # Gap-4 FIX: wire uploaded DataFrame store
            long_term_memory=state.long_term_memory,
        )
        _log({"event": "STARTUP", "step": "orchestrator_ok"})

        _app_state = state
        _startup_called = True  # set only after _app_state is fully assigned
        _log(
            {
                "event": "STARTUP_COMPLETE",
                "model": _env("LLM_MODEL", "llama-3.3-70b-versatile"),
            }
        )

    except Exception:
        _startup_called = False  # allow retry on next uvicorn --reload
        _app_state = None
        raise


async def _shutdown() -> None:
    global _app_state, _startup_called
    _log({"event": "SHUTDOWN"})
    if _app_state is not None:
        with contextlib.suppress(Exception):
            _app_state.db_engine.dispose()
    _app_state = None
    _startup_called = False


def _log(record: dict) -> None:
    from datetime import datetime

    record["timestamp"] = datetime.now(tz=UTC).isoformat()
    print(json.dumps(record, default=str), file=sys.stdout, flush=True)


# ---------------------------------------------------------------------------
# State accessor — fails fast with a clear 503 if startup did not complete
# ---------------------------------------------------------------------------


def _state() -> AppState:
    if _app_state is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server initialisation did not complete. "
                "Check the uvicorn terminal for startup errors."
            ),
        )
    return _app_state


# ---------------------------------------------------------------------------
# Error code → HTTP status mapping
# ---------------------------------------------------------------------------
# The API contract (interfaces.py) was originally specified as "always 200
# for caught errors". This is revised here: the structured `error` field is
# still always populated in the response body, but the HTTP status code now
# correctly reflects the error category so that standard HTTP clients,
# middleware, and monitoring tools can react without body inspection.
#
# Mapping rationale:
#   422 — client sent a query that references data that doesn't exist in the
#          schema (UNRESOLVED_REFERENCE / UNRESOLVED_COLUMN). The request is
#          well-formed JSON but the semantic content is invalid.
#   403 — client asked for a mutation or a blocked/PII table. Forbidden.
#   502 — LLM returned TERMINAL_ERROR or UNRECOVERABLE after retries. The
#          upstream (LLM) failed, not the client.
#   503 — the LLM API is unreachable or the executor is initialising.

_ERROR_HTTP_STATUS: dict[str, int] = {
    "UNRESOLVED_REFERENCE": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "UNRESOLVED_COLUMN": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "POLICY_VIOLATION": status.HTTP_403_FORBIDDEN,
    "MUTATION_STATEMENT": status.HTTP_403_FORBIDDEN,
    "TERMINAL_ERROR": status.HTTP_502_BAD_GATEWAY,
    "EXECUTION_TIMEOUT": status.HTTP_503_SERVICE_UNAVAILABLE,
    "DB_UNAVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
    "LLM_UNAVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
    "LLM_PARSE_ERROR": status.HTTP_502_BAD_GATEWAY,
    "LLM_EMPTY_RESPONSE": status.HTTP_502_BAD_GATEWAY,
    "INTERNAL_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
}


def _http_status_for(error_code: str | None) -> int:
    if error_code is None:
        return status.HTTP_200_OK
    return _ERROR_HTTP_STATUS.get(error_code, status.HTTP_502_BAD_GATEWAY)


# ---------------------------------------------------------------------------
# Ingest request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    schema_id: str
    # FIX: database_url is now optional.
    # When omitted the server falls back to its own DATABASE_URL env var —
    # the same connection that was already validated at startup.  This removes
    # the need for callers to repeat the connection string on every /ingest
    # call and eliminates the 422 error that occurred when the field was left
    # out of the request body.
    # The SSRF validator still runs on the resolved URL (see model_validator
    # below) so the security properties are unchanged.
    database_url: str | None = None
    dialect: str = "postgres"
    table_allowlist: list[str] | None = None
    pii_tables: list[str] = []
    force_reingest: bool = False
    # Semantic descriptions injected into FAISS chunk text so the LLM
    # understands column intent (e.g. payments.paid_amount vs claims.claim_amount).
    # table_descriptions: {table_name: "business description"}
    # column_descriptions: {table_name: {col_name: "column description"}}
    table_descriptions: dict[str, str] | None = None
    column_descriptions: dict[str, dict[str, str]] | None = None

    @model_validator(mode="after")
    def _resolve_and_validate_db_url(self) -> IngestRequest:
        """
        1. If database_url was omitted, resolve it from DATABASE_URL env var.
        2. Run SSRF validation on the resolved URL regardless of source.

        Using model_validator (fires after all field assignments) instead of
        field_validator so the resolution step runs first and the SSRF check
        always sees a non-None string.
        """
        import pathlib

        # Step 1 — resolve
        if not self.database_url:
            url = os.environ.get("DATABASE_URL", "").strip()
            if not url:
                raise ValueError(
                    "database_url was not provided and DATABASE_URL env var is "
                    "not set. Pass database_url explicitly or set DATABASE_URL "
                    "in .env."
                )
            self.database_url = url

        # Step 2 — SSRF guard (same logic as the previous field_validator)
        v = self.database_url
        _ALLOWED_SCHEMES = {"postgresql", "postgres", "sqlite", "mysql", "mssql"}
        parsed = urlparse(v)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"Unsupported DB scheme '{parsed.scheme}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_SCHEMES))}."
            )

        if parsed.scheme.lower() == "sqlite":
            data_root = pathlib.Path(os.environ.get("SQLITE_DATA_ROOT", "data")).resolve()
            raw_path = parsed.path.lstrip("/") or "."
            try:
                resolved = (data_root / raw_path).resolve()
                resolved.relative_to(data_root)
            except ValueError as err:
                raise ValueError(
                    f"SQLite path '{parsed.path}' escapes the allowed data "
                    f"root '{data_root}'. Set SQLITE_DATA_ROOT to override."
                ) from err
            return self

        # Network databases: hostname allowlist.
        # SEC-3 FIX: fail-CLOSED when ALLOWED_DB_HOSTS is unset.
        allowed_hosts_raw = os.environ.get("ALLOWED_DB_HOSTS", "").strip()
        if not allowed_hosts_raw:
            app_env = os.environ.get("APP_ENV", "production").lower()
            if app_env != "development":
                raise ValueError(
                    "ALLOWED_DB_HOSTS is not set. Network database connections are "
                    "rejected in production to prevent SSRF. "
                    "Set ALLOWED_DB_HOSTS=<your-db-host> in .env, or set "
                    "APP_ENV=development to allow all hosts in local dev mode."
                )
            return self

        allowed_hosts = {h.strip().lower() for h in allowed_hosts_raw.split(",") if h.strip()}
        _LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}
        host = (parsed.hostname or "").lower()
        if host in _LOOPBACK:
            host = "localhost"
        if host not in allowed_hosts:
            raise ValueError(f"DB host '{parsed.hostname}' is not on the ALLOWED_DB_HOSTS list.")
        return self


class IngestResponse(BaseModel):
    schema_id: str
    tables_ingested: int
    chunks_indexed: int
    ingestion_time_ms: int
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Route implementations
# ---------------------------------------------------------------------------


async def _query(body: QueryRequest, request: Request) -> QueryResponse:
    state = _state()
    result = await state.orchestrator.run(body)
    http_code = _http_status_for(result.error.error_code if result.error else None)
    if http_code != status.HTTP_200_OK:
        return JSONResponse(
            content=json.loads(result.model_dump_json()),
            status_code=http_code,
        )
    return result


async def _execute(body: ExecuteRequest, request: Request) -> QueryResponse:
    state = _state()
    session_id = body.session_id or await state.sessions.get_or_create(
        None, schema_id=body.schema_id
    )
    schema_columns = await state.retrieval.get_schema_columns(body.schema_id)
    table_policies = await state.retrieval.get_table_policies(body.schema_id)

    from analyst_copilot.validation import (
        PreExecutionPolicy,
        execute_python,
        execute_sql,
        validate_python,
        validate_sql,
    )

    if body.code_type == "sql":
        val = validate_sql(body.code, schema_columns)
        if not val.valid:
            return _error_response(
                session_id,
                body.code,
                "sql",
                val.error_type or "VALIDATION_ERROR",
                val.error_message or "Validation failed.",
            )
        policy = PreExecutionPolicy(table_policies)
        pol = policy.check(body.code)
        if not pol.valid:
            return _error_response(
                session_id,
                body.code,
                "sql",
                "POLICY_VIOLATION",
                pol.error_message or "Policy check failed.",
            )
        if body.dry_run:
            return _dry_run_response(session_id, body.code, "sql")
        exec_r = execute_sql(body.code, state.db_engine)
    else:
        uploaded_refs: dict = {}
        if hasattr(state, "df_store"):
            uploaded_refs = state.df_store.get(session_id)
        uploaded_columns = {
            str(col).lower() for df in uploaded_refs.values() for col in getattr(df, "columns", [])
        }

        val = validate_python(
            body.code,
            schema_columns | uploaded_columns,
            dataframe_refs=set(uploaded_refs),
        )
        if not val.valid:
            return _error_response(
                session_id,
                body.code,
                "pandas",
                val.error_type or "VALIDATION_ERROR",
                val.error_message or "Validation failed.",
            )
        if body.dry_run:
            return _dry_run_response(session_id, body.code, "pandas")
        # Load DataFrames so Pandas code runs in a populated namespace.
        # Falls back to {} if df_loader is unavailable or the schema has no tables.
        df_refs: dict = {}
        if hasattr(state, "df_loader"):
            try:
                schema_profile = state.registry.get(body.schema_id)
                if schema_profile:
                    tables = [t.table_name for t in schema_profile.tables]
                    df_refs = await state.df_loader.load(tables=tables, schema_id=body.schema_id)
            except Exception:  # noqa: BLE001
                df_refs = {}
        df_refs = {**df_refs, **uploaded_refs}
        exec_r = execute_python(body.code, dataframe_refs=df_refs)

    if not exec_r.success:
        return _error_response(
            session_id,
            body.code,
            body.code_type,
            exec_r.error_type or "EXECUTION_ERROR",
            exec_r.error_message or "Execution failed.",
        )

    if body.code_type == "sql":
        result_rows = exec_r.result or []
        total_rows = len(result_rows)
        result_rows = result_rows[:10_000]
    else:
        res = exec_r.dataframe
        if isinstance(res, pd.DataFrame):
            total_rows = len(res)
            result_rows = res.head(100).to_dict("records")
        elif isinstance(res, pd.Series):
            total_rows = len(res)
            result_rows = res.reset_index().head(100).to_dict("records")
        else:
            total_rows = 1
            result_rows = [{"result": res}]

    insight = await state.llm.generate_insight(
        nl_query=body.code,
        result_preview=result_rows[:5],
        row_count=total_rows,
        result_warnings=[],
        error=None,
        model=_env("LLM_MODEL", "llama-3.3-70b-versatile"),
    )

    return QueryResponse(
        session_id=session_id,
        generated_code=body.code,
        code_type=body.code_type,
        result_preview=result_rows[:100],
        row_count=total_rows,
        insight=insight,
        execution_time_ms=exec_r.execution_time_ms,
        retry_count=0,
        warnings=[],
        error=None,
    )


async def _history(
    session_id: Annotated[str, Path(min_length=1)],
    request: Request,
) -> HistoryResponse:
    state = _state()
    try:
        turns = await state.sessions.get_full_history(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        ) from exc
    turn_records = (
        turns if turns and isinstance(turns[0], TurnRecord) else [TurnRecord(**t) for t in turns]
    )
    return HistoryResponse(session_id=session_id, turns=turn_records)


async def _ingest(body: IngestRequest, request: Request) -> IngestResponse:
    state = _state()
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    # body.database_url is guaranteed non-None here: the model_validator
    # resolved it from DATABASE_URL env var if the caller omitted it, and
    # raised ValueError (→ 422) if neither source had a value.
    assert body.database_url, "database_url must be resolved before reaching _ingest"
    result = await loop.run_in_executor(
        None,
        lambda: state.ingestion.ingest(
            schema_id=body.schema_id,
            database_url=body.database_url,  # type: ignore[arg-type]
            dialect=body.dialect,
            table_allowlist=body.table_allowlist,
            pii_tables=body.pii_tables,
            force_reingest=body.force_reingest,
            table_description_overrides=body.table_descriptions,
            column_description_overrides=body.column_descriptions,
        ),
    )
    # IngestionPipeline.ingest() does not return ingestion_time_ms — inject it.
    # setdefault preserves the value if the pipeline ever starts returning it.
    result.setdefault("ingestion_time_ms", int((time.monotonic() - t0) * 1000))
    return IngestResponse(schema_id=body.schema_id, **result)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


# ── Upload handlers (Gap-4) ───────────────────────────────────────────────────


async def _upload_file(
    request: Request,
    session_id: str,
    file: Annotated[UploadFile, File(...)],
) -> dict:
    """
    POST /upload?session_id=<id>
    Accepts a CSV, Parquet, or XLSX file and stores it in the DataFrameStore
    for the given session. The DataFrame is then available by filename stem
    in the Pandas execution namespace for that session.
    """
    state = _state()

    file_bytes = await file.read()
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "csv"

    result = state.df_store.ingest(
        session_id=session_id,
        df_name=filename,
        file_bytes=file_bytes,
        extension=ext,
    )
    if result.success:
        return {
            "success": True,
            "df_name": result.df_name,
            "rows": result.rows,
            "columns": result.columns,
            "size_mb": result.size_mb,
            "warnings": result.warnings,
        }
    return {"success": False, "error": result.error, "warnings": result.warnings}


async def _upload_list(request: Request, session_id: str) -> dict:
    """GET /upload/list?session_id=<id> — list all DataFrames for a session."""
    state = _state()
    return {"dataframes": state.df_store.list_dataframes(session_id)}


async def _upload_delete(request: Request, df_name: str, session_id: str) -> dict:
    """DELETE /upload/{df_name}?session_id=<id> — remove a DataFrame."""
    state = _state()
    deleted = state.df_store.delete(session_id, df_name)
    return {"deleted": deleted, "df_name": df_name}


def create_app() -> FastAPI:
    # ARCH-4 FIX: use the lifespan asynccontextmanager pattern instead of
    # add_event_handler("startup"/"shutdown"). The on_event approach relies on
    # the module-level _startup_called bool which is not reset between
    # thread-based uvicorn reloads on Windows — causing the second startup
    # invocation to return early without re-initialising _app_state and
    # silently serving 503 forever. The lifespan context manager is scoped
    # to the app instance, not the module, so it is always called correctly.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # noqa: ARG001
        await _startup()
        try:
            yield
        finally:
            await _shutdown()

    # H-07 FIX: disable OpenAPI schema and interactive docs in production.
    # /openapi.json exposes the full API surface (endpoint paths, every
    # request/response schema including IngestRequest.database_url and
    # pii_tables) to any unauthenticated caller that can reach the service.
    # In development (APP_ENV=development) docs remain enabled for convenience.
    _is_production = os.environ.get("APP_ENV", "production").lower() == "production"
    _docs_url: str | None = None if _is_production else "/docs"
    _redoc_url: str | None = None if _is_production else "/redoc"
    _openapi_url: str | None = None if _is_production else "/openapi.json"

    app = FastAPI(
        title="Data Analyst Copilot",
        version="1.0.0",
        description=(
            "LLM-powered data analyst copilot — converts natural language "
            "analytical questions into SQL or Pandas workflows against "
            "registered schemas."
        ),
        lifespan=_lifespan,
        docs_url=_docs_url,
        redoc_url=_redoc_url,
        openapi_url=_openapi_url,
    )

    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        _log(
            {
                "event": "UNHANDLED_EXCEPTION",
                "path": str(request.url.path),
                "error": str(exc),
                "type": type(exc).__name__,
            }
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred."},
        )

    # ── API Key authentication ────────────────────────────────────────────────
    # In production the docs/redoc/openapi routes do not exist (disabled above),
    # so they never reach this middleware.  In development they are exempt so
    # the browser can reach them without an API key.
    _AUTH_EXEMPT: frozenset[str] = frozenset({"/", "/health"}) | (
        frozenset({"/docs", "/redoc", "/openapi.json"}) if not _is_production else frozenset()
    )

    @app.middleware("http")
    async def _api_key_middleware(request: Request, call_next):
        if request.url.path not in _AUTH_EXEMPT:
            required_key = os.environ.get("API_KEY", "").strip()
            if required_key:
                provided = request.headers.get("X-API-Key", "")
                # H3 FIX: use hmac.compare_digest for constant-time comparison.
                # Plain string != short-circuits on the first differing byte,
                # leaking key length/prefix via response-time differences.
                if not hmac.compare_digest(provided.encode("utf-8"), required_key.encode("utf-8")):
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={
                            "detail": "Invalid or missing API key. "
                            "Provide it via the X-API-Key header."
                        },
                    )
        return await call_next(request)

    # No stub-removal dance required (there are no pre-registered stubs).
    app.add_api_route(
        "/query",
        _query,
        methods=["POST"],
        response_model=QueryResponse,
        responses={
            # All error responses carry the same QueryResponse envelope so
            # clients always parse the same shape. The `error` field is
            # populated; `result_preview` and `row_count` are null.
            422: {
                "model": QueryResponse,
                "description": "Unresolvable column or reference in query.",
            },
            403: {
                "model": QueryResponse,
                "description": "Mutation or policy-blocked table requested.",
            },
            502: {
                "model": QueryResponse,
                "description": "LLM or execution failure after retries.",
            },
            503: {
                "model": QueryResponse,
                "description": "LLM API unreachable or server initialising.",
            },
        },
    )
    app.add_api_route("/execute", _execute, methods=["POST"], response_model=QueryResponse)
    app.add_api_route(
        "/history/{session_id}",
        _history,
        methods=["GET"],
        response_model=HistoryResponse,
    )
    app.add_api_route("/ingest", _ingest, methods=["POST"], response_model=IngestResponse)

    # ── Upload endpoints (Gap-4) ──────────────────────────────────────────────
    app.add_api_route("/upload", _upload_file, methods=["POST"])
    app.add_api_route("/upload/list", _upload_list, methods=["GET"])
    app.add_api_route("/upload/{df_name}", _upload_delete, methods=["DELETE"])

    @app.get("/")
    async def root():
        return {"status": "ok", "docs": "/docs"}

    @app.get("/health", include_in_schema=False)
    async def _health() -> JSONResponse:
        initialized = _app_state is not None
        # B-05 FIX: return HTTP 503 when the server has not finished
        # initialising so load-balancers and Docker HEALTHCHECK correctly
        # mark the container as unhealthy rather than routing traffic to a
        # half-started instance.  HTTP 200 is reserved for "ready to serve".
        return JSONResponse(
            status_code=200 if initialized else 503,
            content={
                "status": "ok" if initialized else "starting",
                "initialized": initialized,
                "model": _env("LLM_MODEL", "llama-3.3-70b-versatile"),
            },
        )

    # ALLOWED_ORIGINS must be explicitly set in production.
    # When unset, CORS is disabled (no cross-origin requests are allowed).
    # Dev: set ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
    raw_origins = _env("ALLOWED_ORIGINS", "")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    # P3-01 FIX: guard against ALLOWED_ORIGINS=* with allow_credentials=True.
    # Starlette's CORSMiddleware raises ValueError at middleware-init time when
    # allow_credentials=True is combined with allow_origins=["*"], producing a
    # confusing traceback with no indication of the root cause.  We surface a
    # clear RuntimeError at startup instead so the operator knows exactly what
    # to change.  Using credentials with a wildcard origin is also a security
    # anti-pattern (it would let any website make credentialed cross-origin
    # requests on behalf of authenticated users).
    if "*" in allowed_origins:
        raise RuntimeError(
            "ALLOWED_ORIGINS contains '*' (wildcard), but allow_credentials=True "
            "is required by this API.  Starlette prohibits this combination — it "
            "would allow any website to make credentialed cross-origin requests.  "
            "Replace '*' with explicit origins, e.g.:\n"
            "  ALLOWED_ORIGINS=https://your-app.example.com,https://localhost:3000"
        )

    if not allowed_origins:
        _log(
            {
                "event": "STARTUP_WARNING",
                "message": (
                    "ALLOWED_ORIGINS is not set — CORS is disabled. "
                    "Set ALLOWED_ORIGINS in .env to enable browser clients."
                ),
            }
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        # SEC-6 FIX: restrict to the actual methods and headers this API uses.
        # allow_methods=["*"] + allow_credentials=True means any origin in
        # allow_origins can make credentialed requests with arbitrary methods
        # and headers (including auth tokens). Explicit allowlists prevent
        # accidental exposure if allow_origins is ever widened.
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # DEP-2 FIX: per-key rate limiting to protect the Groq/Gemini TPM budget.
    # Groq free tier: ~6,000 TPM for llama-3.3-70b — a single concurrent
    # burst exhausts this and returns 429 for all other users.
    # Default: 10 requests/minute per API key (or per IP when no key is set).
    # Override with RATE_LIMIT_PER_MINUTE env var.
    # Requires: pip install slowapi (added to requirements.txt)
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address

        def _rate_limit_key(request: Request) -> str:
            """Key by API key when present, fall back to IP address."""
            return request.headers.get("X-API-Key", "") or get_remote_address(request)

        _rpm = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10"))
        limiter = Limiter(
            key_func=_rate_limit_key,
            default_limits=[f"{_rpm}/minute"],
            config_filename=_slowapi_config_filename(),
            storage_uri="memory://",
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        # SlowAPIMiddleware enforces default_limits on ALL routes automatically.
        # Without it, default_limits only apply to routes decorated with
        # @limiter.limit() — leaving most routes unprotected.
        app.add_middleware(SlowAPIMiddleware)
        _log({"event": "RATE_LIMITER_INIT", "limit": f"{_rpm}/minute"})
    except ImportError:
        _log(
            {
                "event": "RATE_LIMITER_SKIP",
                "message": (
                    "slowapi not installed — rate limiting disabled. " "Run: pip install slowapi"
                ),
            }
        )

    return app


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_response(session_id, code, code_type, error_code, message) -> QueryResponse:
    return QueryResponse(
        session_id=session_id,
        generated_code=code,
        code_type=code_type,
        result_preview=None,
        row_count=None,
        insight=f"I wasn't able to complete that: {message}",
        execution_time_ms=0,
        retry_count=0,
        warnings=[],
        error=ErrorDetail(error_code=error_code, message=message, attempted_code=code),
    )


def _dry_run_response(session_id: str, code: str, code_type: str) -> QueryResponse:
    return QueryResponse(
        session_id=session_id,
        generated_code=code,
        code_type=code_type,
        result_preview=None,
        row_count=None,
        insight="Dry run complete. Code passed all validation checks.",
        execution_time_ms=0,
        retry_count=0,
        warnings=[],
        error=None,
    )
