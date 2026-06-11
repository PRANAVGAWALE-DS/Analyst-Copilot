"""
orchestrator.py — Agent State Machine
Data Analyst Copilot · Python 3.11+ · Pydantic v2

Implements the full state machine from Section 2H:
  INTAKE → RETRIEVAL → GENERATION → VALIDATION → EXECUTION
  → RESULT_CHECK → INSIGHT → TERMINAL

  Error paths:
  VALIDATION / EXECUTION → ERROR_CORRECT → GENERATION (loop, max 3 attempts)
  RETRIEVAL (0 chunks)   → INTAKE (clarification)
  RESULT_CHECK (empty)   → INTAKE (clarification)
  attempt ≥ max          → TERMINAL_ERROR

Wires together:
  - prompts.py       (LLMClient, PromptRenderer, all prompt templates)
  - validation.py    (ExecutionLoop, validate_sql, validate_python)
  - interfaces.py    (QueryRequest, QueryResponse, ErrorDetail, TurnRecord)
  - observability.py (ObservabilityStack, TraceLogger)

No pseudocode. All imports included.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import difflib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dataframe_loader import DataFrameLoader
    from dataframe_store import DataFrameStore
    from long_term_memory import LongTermMemory

from interfaces import (
    ErrorDetail,
    GeneratePandasOutput,
    GenerateSQLOutput,
    QueryRequest,
    QueryResponse,
    SchemaChunk,
    TurnRecord,
)
from observability import AgentState, ObservabilityStack, TraceLogger
from prompts import (
    ERROR_CORRECT_SYSTEM_PROMPT,
    NL_TO_PANDAS_SYSTEM_PROMPT,
    NL_TO_SQL_SYSTEM_PROMPT,
    GenerationRequest,
    LLMClient,
    PromptRenderer,
    count_tokens,
    enforce_token_budget,
)
from sql_postprocessor import postprocess_sql
from sqlalchemy.engine import Engine
from validation import (
    AttemptRecord,
    ExecutionLoop,
    ExecutionResult,
    LoopResult,
    TablePolicy,
    validate_group_by_cardinality,
    validate_metric_ranges,
    validate_result,
)

# ---------------------------------------------------------------------------
# Input sanitiser (INTAKE state)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = re.compile(
    # SEC-1 FIX: removed \bdisregard\b, \boverride\b, \bact\s+as\b.
    # These are common business English words that appear in legitimate queries:
    #   "What is the premium override for home policies?"
    #   "Which agents act as brokers?"
    #   "Claims where we disregard the deductible."
    # SQL injection is already handled downstream by sqlglot parse + mutation
    # guard — keeping broad vocabulary patterns here only adds false positives.
    # Retained patterns target structural prompt-injection signals only:
    #   -- / /* */       SQL comment injection
    #   <tag>            HTML/XML tag injection
    #   ignore previous  Classic system-prompt override phrase
    #   you are now      Role-switch injection
    #   new instructions Direct instruction-override phrase
    #   forget your      System prompt erasure
    #   system prompt    Direct system-prompt reference
    r"(--|/\*|\*/|/\*!|<[^>]+>|"
    r"\bignore\s+previous\b|\byou\s+are\s+now\b|\bnew\s+instructions\b|"
    r"\bforget\s+your\b|\bsystem\s+prompt\b)",
    re.IGNORECASE,
)
_SEMICOLON_OUTSIDE_QUOTES = re.compile(r";(?=(?:[^'\"]*['\"][^'\"]*['\"])*[^'\"]*$)")


def sanitise_input(text: str) -> tuple[str, bool]:
    """
    Strip known prompt injection and SQL injection patterns.

    Returns (sanitised_text, injection_detected).
    injection_detected=True signals the orchestrator to return a user-facing
    warning instead of proceeding with generation.
    """
    detected = bool(_INJECTION_PATTERNS.search(text))
    cleaned = _INJECTION_PATTERNS.sub("", text)
    cleaned = _SEMICOLON_OUTSIDE_QUOTES.sub("", cleaned).strip()
    return cleaned, detected


# ---------------------------------------------------------------------------
# Retrieval layer stub — replace with real FAISS + metadata filter in production
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalLayer(Protocol):
    """
    Structural interface for the retrieval layer.

    M-07 FIX: was a plain stub class with `raise NotImplementedError` — not
    a proper ABC or Protocol.  Duck-typing violations (missing a method on a
    concrete implementation) were only caught at call-time, not at
    construction.  Promoted to @runtime_checkable Protocol so that:
      1. mypy enforces structural compatibility at the call site.
      2. isinstance(obj, RetrievalLayer) works at runtime for injection checks.
      3. No inheritance is required — existing retrieval.py SchemaEmbedder
         satisfies the protocol purely by having the correct method signatures.

    In production: wraps a FAISS HNSW index pre-filtered by schema_id,
    returning SchemaChunk objects for the top-K hits.
    Failure mode: returns [] if the schema_id is not found.
    The orchestrator transitions to INTAKE (clarification) on empty results.
    """

    async def retrieve(
        self,
        nl_query: str,
        schema_id: str,
        k: int = 5,
    ) -> list[SchemaChunk]:
        """Returns top-K SchemaChunk objects for the given query and schema."""
        ...

    async def get_schema_columns(self, schema_id: str) -> set[str]:
        """
        Returns the full set of column names for all tables in the schema.
        Used by validate_sql() and validate_python() for column existence checks.
        """
        ...

    async def get_table_policies(self, schema_id: str) -> dict[str, TablePolicy]:
        """Returns PII and access-level flags per table, keyed by table name."""
        ...


# ---------------------------------------------------------------------------
# Session store stub
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStore(Protocol):
    """
    Structural interface for session history storage (Redis in production).

    M-07 FIX: was a plain stub with concrete method bodies for
    create_session() and get_or_create() — meaning a mock that forgot to
    implement get_history() would silently inherit the stub's
    `raise NotImplementedError` and only fail at call time.  Promoted to
    @runtime_checkable Protocol:
      1. mypy catches missing methods on concrete implementations at type-
         check time, not at runtime.
      2. isinstance(obj, SessionStore) works for injection guards.
      3. RedisSessionStore and InMemorySessionStore in session_store.py
         already satisfy this protocol structurally — no changes to those
         files required.

    ARCH-2 FIX retained: create_session() and get_or_create() include
    schema_id: str = "" to match both concrete implementations.
    """

    async def get_history(self, session_id: str, n: int = 10) -> list[dict[str, Any]]:
        """Returns last n turns as dicts (for injection into prompt context)."""
        ...

    async def append_turn(self, session_id: str, turn: TurnRecord) -> None: ...

    async def create_session(self, schema_id: str = "") -> str:
        """Create a new session and return its ID."""
        ...

    async def get_or_create(self, session_id: str | None, schema_id: str = "") -> str: ...


# ---------------------------------------------------------------------------
# Orchestrator state
# ---------------------------------------------------------------------------


@dataclass
class TurnState:
    """
    Mutable state for one query turn. Passed between state handlers.
    All fields start with None and are populated as states are traversed.
    """

    session_id: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nl_query: str = ""
    nl_query_clean: str = ""  # after sanitisation
    schema_id: str = ""
    execution_mode: Literal["sql", "pandas", "auto"] = "auto"
    dry_run: bool = False
    dialect: str = "postgres"  # propagated from QueryRequest.dialect

    # Retrieval
    chunks: list[SchemaChunk] = field(default_factory=list)
    schema_columns: set[str] = field(default_factory=set)
    table_policies: dict[str, TablePolicy] = field(default_factory=dict)
    session_history: list[dict] = field(default_factory=list)
    # Gap-7: last-3 execution results from this session (for multi-turn
    # "use the previous result" queries). Populated from Orchestrator._result_cache
    # in _retrieval_state before prompt rendering.
    prior_results: list[dict] = field(default_factory=list)

    # Generation
    generated_sql: str | None = None
    generated_code: str | None = None
    code_type: Literal["sql", "pandas"] = "sql"
    generation_confidence: float = 1.0
    generation_assumptions: list[str] = field(default_factory=list)

    # Validation + Execution
    execution_result: ExecutionResult | None = None
    attempt_records: list[AttemptRecord] = field(default_factory=list)
    attempt_count: int = 0

    # Result check
    result_rows: list[dict[str, Any]] = field(default_factory=list)
    result_warnings: list[str] = field(default_factory=list)
    # Pre-computed analytical metrics injected into the INSIGHT prompt.
    # Populated by _result_check after row validation; consumed by _insight_state.
    # Empty dict when the result shape is not amenable to metric computation
    # (e.g. scalar single-row results, non-numeric columns only).
    result_metrics: dict[str, Any] = field(default_factory=dict)

    # Insight
    insight: str = ""

    # Phase 3: long-term memory examples injected into generation prompt
    lt_examples: list = field(default_factory=list)
    # Set True when _retrieval_state bypassed generation via an LTM exact hit.
    # _result_check uses this to invalidate the cached entry when downstream
    # validation fails — prevents the stale SQL from being served indefinitely.
    lt_exact_hit_used: bool = False

    # Terminal
    error: ErrorDetail | None = None
    terminal_state: AgentState = "INTAKE"

    # Semantic aggregation guard: set by _generation when the LLM omitted a
    # required aggregate function.  Stored separately from state.error so the
    # for-loop can route this to _error_correct_agg instead of hard-breaking.
    sem_agg_missing: tuple[str, str] | None = None  # (fn_name, trigger_word)

    # Postprocessor W1 fan-out hint: set by _generation when postprocess_sql
    # detects a JOIN fan-out that cannot be auto-fixed by AST rewrite.
    # Mirrors sem_agg_missing: routes to _error_correct_postprocessor before
    # ExecutionLoop runs so the LLM can regenerate with an explicit CTE hint.
    # Cleared by run() after the correction pass (or on final attempt).
    postprocessor_hint: str | None = None

    @property
    def active_code(self) -> str:
        """The code string that was most recently generated."""
        return self.generated_code or self.generated_sql or ""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _format_lt_examples(lt_examples: list) -> str:
    """
    Format LTM MemorySearchResult list as a few-shot block for injection into
    NL_TO_SQL_SYSTEM_PROMPT via the {lt_examples} placeholder.

    Each entry is a MemorySearchResult with:
      .similarity   float   cosine similarity score
      .record       MemoryRecord with .nl_query and .sql

    Returns a safe placeholder string when no examples are available so the
    prompt section always renders cleanly without an empty {lt_examples} slot.
    """
    if not lt_examples:
        return "No prior examples available."
    lines: list[str] = []
    for i, hit in enumerate(lt_examples, 1):
        lines.append(
            f"Example {i} (similarity={hit.similarity:.2f}):\n"
            f"  Question: {hit.record.nl_query}\n"
            f"  SQL: {hit.record.sql}"
        )
    return "\n\n".join(lines)


def _extract_hard_constraints(chunks: list[SchemaChunk]) -> str:
    """
    Build a numbered HARD_CONSTRAINTS string from retrieved schema chunks.

    Sources (in order per chunk):
      - SchemaChunk.business_description  (table-level semantic note)
      - SchemaColumn.description          (column-level data quality note)

    Only non-None, non-empty descriptions are included. The result is
    injected into NL_TO_SQL_SYSTEM_PROMPT via the {hard_constraints}
    placeholder so constraints surface with the same structural prominence
    as the numbered RULES, rather than being buried inside the schema_context
    JSON blob where the LLM may treat them as advisory metadata.

    Returns a safe placeholder string when no chunks carry any descriptions,
    so the prompt section always renders cleanly.
    """
    lines: list[str] = []
    n = 1
    for chunk in chunks:
        if chunk.business_description and chunk.business_description.strip():
            lines.append(
                f"CONSTRAINT {n} — {chunk.table} (table): " f"{chunk.business_description.strip()}"
            )
            n += 1
        for col in chunk.columns:
            if col.description and col.description.strip():
                lines.append(
                    f"CONSTRAINT {n} — {chunk.table}.{col.name}: " f"{col.description.strip()}"
                )
                n += 1
    return "\n".join(lines) if lines else "No schema-specific constraints for this query."


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3

# P3-B FIX: module-level constant — was rebuilt with re.compile() on every
# call to _generation().  At 50 req/s that is ~500 unnecessary regex compiles
# per second.  Patterns are immutable; compile once at import time.
_AGG_INTENT_MAP: list[tuple[re.Pattern[str], re.Pattern[str], str]] = [
    (
        re.compile(r"\b(average|avg|mean)\b", re.I),
        re.compile(r"\bAVG\s*\(", re.I),
        "AVG",
    ),
    (
        re.compile(r"\b(total|sum)\b", re.I),
        re.compile(r"\bSUM\s*\(", re.I),
        "SUM",
    ),
    (
        re.compile(r"\b(count|how many|number of)\b", re.I),
        re.compile(r"\bCOUNT\s*\(", re.I),
        "COUNT",
    ),
    (
        re.compile(r"\b(maximum|max)\b", re.I),
        re.compile(r"\bMAX\s*\(", re.I),
        "MAX",
    ),
    (
        re.compile(r"\b(minimum|min)\b", re.I),
        re.compile(r"\bMIN\s*\(", re.I),
        "MIN",
    ),
]


class Orchestrator:
    """
    Implements the full agent state machine from Section 2H.

    Each state is a private async method that accepts and mutates a TurnState.
    State transitions are explicit — there are no implicit falls through.

    Parameters
    ----------
    llm             : LLMClient instance (Gemini wrapper).
    retrieval       : RetrievalLayer implementation.
    session_store   : SessionStore implementation.
    engine          : Read-only SQLAlchemy engine for the SQL executor.
    obs             : ObservabilityStack (trace store + alert evaluator).
    model           : LLM model string passed to all GenerationRequest calls.
    """

    def __init__(
        self,
        llm: LLMClient,
        retrieval: RetrievalLayer,
        session_store: SessionStore,
        engine: Engine,
        obs: ObservabilityStack,
        model: str = "llama-3.3-70b-versatile",
        df_loader: DataFrameLoader | None = None,
        df_store: DataFrameStore | None = None,
        long_term_memory: LongTermMemory | None = None,
    ) -> None:
        self._llm = llm
        self._retrieval = retrieval
        self._sessions = session_store
        self._engine = engine
        self._obs = obs
        self._model = model
        self._df_loader = df_loader  # DB-backed DataFrame injection
        # Gap-4 FIX: wire DataFrameStore so user-uploaded CSV/Parquet files
        # are injected into the Pandas execution namespace. Without this,
        # uploaded files are stored but never reach execute_python(), causing
        # NameError for every upload-based Pandas query.
        self._df_store = df_store
        self._long_term_memory = long_term_memory
        # Gap-7: per-session last-3 execution result cache.
        # {session_id: deque([{"code": str, "result_preview": list, "code_type": str}, ...])}
        # Stored in-process only — not persisted to Redis (results can be large).
        # TTL is managed implicitly: entries are per-session and evicted when
        # the session is closed. Max 3 entries per session (deque maxlen).
        #
        # H-10 FIX: the outer dict was unbounded — every unique session_id
        # accumulated indefinitely for the process lifetime.  Under adversarial
        # conditions (clients generating random session IDs) this is a trivial
        # heap exhaustion attack.  Use a size-capped OrderedDict with LRU
        # eviction: once _RESULT_CACHE_MAX_SESSIONS is reached, the oldest
        # session is dropped before inserting the new one.
        _max = int(os.environ.get("RESULT_CACHE_MAX_SESSIONS", "1000"))
        self._result_cache: collections.OrderedDict[str, collections.deque[dict[str, Any]]] = (
            collections.OrderedDict()
        )
        self._result_cache_max: int = _max
        # Gap-10: feature flag for LTM exact-hit bypass (skips full LLM generation).
        self._use_lt_exact_hit = os.environ.get("USE_LT_EXACT_HIT", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, request: QueryRequest) -> QueryResponse:
        """
        Full turn: INTAKE → … → TERMINAL or TERMINAL_ERROR.
        Always returns a QueryResponse — never raises.
        """
        session_id = await self._sessions.get_or_create(request.session_id)
        state = TurnState(
            session_id=session_id,
            nl_query=request.nl_query,
            schema_id=request.schema_id,
            execution_mode=request.execution_mode,
            dry_run=request.dry_run,
            dialect=request.dialect,
        )
        t0 = time.monotonic()
        self._obs.trace_store.start_trace(
            session_id, state.turn_id, nl_query_length=len(request.nl_query)
        )

        try:
            await self._intake(state)
            if state.error:
                return self._build_response(state, int((time.monotonic() - t0) * 1000))

            await self._retrieval_state(state)
            if state.error or state.terminal_state == "INTAKE":
                return self._build_response(state, int((time.monotonic() - t0) * 1000))

            # Generation → Validation → Execution retry loop
            loop = ExecutionLoop(
                max_attempts=MAX_ATTEMPTS,
                engine=self._engine,
                table_policies=state.table_policies,
            )

            # When error_correct successfully produces corrected code it stores
            # it in state.generated_sql / state.generated_code. Without this
            # flag, the next for-iteration calls _generation() again which
            # unconditionally overwrites that corrected code with a fresh LLM
            # call — making the entire error_correct round-trip a no-op.
            # If _retrieval_state populated state.generated_sql from LTM
            # (lt_exact_hit=True), terminal_state is already "VALIDATION" —
            # skip the first _generation() call so the cached SQL is not
            # overwritten by a fresh LLM round-trip.
            _skip_generation: bool = state.terminal_state == "VALIDATION"

            for attempt in range(MAX_ATTEMPTS):
                state.attempt_count = attempt
                # Reset per-attempt; populated by UNRESOLVED_COLUMN fast-path
                # when a fuzzy match exists; passed to _error_correct so the
                # correction prompt includes explicit substitution hints.
                _column_hints: dict[str, str | None] = {}

                if _skip_generation:
                    # Use the corrected code produced by _error_correct on the
                    # previous iteration. Reset the flag so we regenerate on
                    # the next attempt if this corrected code also fails.
                    _skip_generation = False
                else:
                    await self._generation(state, attempt)
                    if state.error:
                        break

                    # Semantic aggregation guard fired: the LLM generated SQL
                    # without the required aggregate function.  This is a
                    # recoverable model error — route to _error_correct_agg
                    # with a targeted constraint-injection prompt, then retry
                    # generation with the corrected SQL.  Only do this when
                    # retries remain; on the final attempt escalate to terminal.
                    if state.sem_agg_missing is not None:
                        if attempt < MAX_ATTEMPTS - 1:
                            await self._error_correct_agg(state, attempt)
                            state.sem_agg_missing = None
                            if state.error:
                                break
                            _skip_generation = bool(state.active_code)
                            continue
                        else:
                            _fn_name, _matched_word = state.sem_agg_missing
                            state.error = ErrorDetail(
                                error_code="SEMANTIC_AGG_MISSING",
                                message=(
                                    f"The question asks for '{_matched_word}' but the "
                                    f"generated SQL contains no {_fn_name}() function. "
                                    f"Please try rephrasing your question."
                                ),
                                attempted_code=state.active_code,
                            )
                            break

                    # LOW_CONFIDENCE sets terminal_state=INTAKE and returns from
                    # _generation without setting state.error, so the check above
                    # would not fire and ExecutionLoop would run with code="".
                    # Also guard the edge-case where active_code is empty for any
                    # other reason (belt-and-suspenders alongside Bug 1 fix).
                    if state.terminal_state == "INTAKE" or not state.active_code:
                        break

                    # W1 postprocessor fan-out: needs_retry is True when the SQL
                    # joins fan-out tables without pre-aggregation CTEs and the
                    # fix requires LLM regeneration (not just AST mutation).
                    # Route to _error_correct_postprocessor before ExecutionLoop,
                    # identical to the sem_agg_missing pattern above.
                    if state.postprocessor_hint is not None:
                        if attempt < MAX_ATTEMPTS - 1:
                            await self._error_correct_postprocessor(state, attempt)
                            state.postprocessor_hint = None
                            if state.error:
                                break
                            _skip_generation = bool(state.active_code)
                            continue
                        else:
                            # Final attempt: hint is stale, clear it and proceed
                            # to execution — a warning is already in result_warnings
                            # from the postprocessor.
                            state.postprocessor_hint = None

                # Gap-4 FIX: merge two DataFrame sources for the Pandas path.
                # Source 1: DB-loaded tables via DataFrameLoader (schema tables).
                # Source 2: User-uploaded files via DataFrameStore (session uploads).
                # Uploaded files take precedence — {**db_refs, **uploaded_refs}
                # allows a user to shadow a DB table with a local CSV override.
                df_refs: dict | None = None
                if state.code_type == "pandas":
                    db_refs: dict = {}
                    if self._df_loader is not None:
                        tables = [c.table for c in state.chunks]
                        try:
                            db_refs = await self._df_loader.load(
                                tables=tables,
                                schema_id=state.schema_id,
                            )
                        except Exception:  # noqa: BLE001
                            db_refs = {}  # fail open — continue with uploaded only

                    uploaded_refs: dict = {}
                    if self._df_store is not None:
                        uploaded_refs = self._df_store.get(state.session_id)

                    # Merge: uploaded takes precedence over DB-loaded
                    df_refs = {**db_refs, **uploaded_refs} if (db_refs or uploaded_refs) else None

                # P2-07 FIX: if Pandas mode was selected (by keyword or
                # explicit execution_mode) but neither source has any
                # DataFrames, abort immediately with a clear INTAKE message.
                # Without this guard the execution loop would receive an empty
                # namespace, generate a NameError on the first variable
                # reference, enter the error-correction retry cycle, exhaust
                # MAX_ATTEMPTS, and finally surface as TERMINAL_ERROR — giving
                # the user no hint that they need to upload a file first.
                if state.code_type == "pandas" and not df_refs:
                    state.terminal_state = "INTAKE"
                    state.insight = (
                        "To run a Pandas analysis I need at least one DataFrame "
                        "to work with. Please upload a CSV, Parquet, or Excel file "
                        "using the /upload endpoint and then retry your question."
                    )
                    return state

                loop_result = loop.run(
                    code=state.active_code,
                    code_type=state.code_type,
                    schema_columns=state.schema_columns,
                    dataframe_refs=df_refs,
                    dialect=_dialect_for_request(request),
                    dry_run=request.dry_run,
                )

                if loop_result.success:
                    state.execution_result = loop_result.execution_result
                    state.attempt_records = loop_result.attempt_history
                    # RESULT_CAPPED: check whether this is a group-by cardinality
                    # mismatch (retryable) before treating it as a plain warning.
                    # A capped result on a "for each X / compare / breakdown" query
                    # means the SQL grouped at the wrong level — per-row instead of
                    # per-category.  Route to ERROR_CORRECT the same way the
                    # postprocessor fan-out and sem_agg_missing paths do.
                    if (
                        loop_result.runtime_validation
                        and loop_result.runtime_validation.issue == "RESULT_CAPPED"
                    ):
                        _exec_rows = (
                            loop_result.execution_result.result or []
                            if loop_result.execution_result
                            else []
                        )
                        _cardinality_rt = validate_group_by_cardinality(
                            _exec_rows, state.nl_query_clean
                        )
                        if _cardinality_rt is not None and attempt < MAX_ATTEMPTS - 1:
                            # Discard the stale capped result — we are retrying.
                            state.execution_result = None
                            _correction_loop_result = LoopResult(
                                success=False,
                                attempt_history=loop_result.attempt_history,
                                final_error_type="GROUPBY_CARDINALITY_MISMATCH",
                                final_error_message=_cardinality_rt.message,
                                retry_count=loop_result.retry_count,
                            )
                            _code_before_correction = state.active_code
                            await self._error_correct(state, _correction_loop_result)
                            if state.error:
                                break
                            _skip_generation = bool(
                                state.active_code and state.active_code != _code_before_correction
                            )
                            continue
                        else:
                            # Not a group-by intent, or no retries left — surface warning
                            state.result_warnings.append(
                                loop_result.runtime_validation.message or ""
                            )
                    # Surface non-blocking advisory (e.g. NON_SARGABLE_FILTER)
                    # in QueryResponse.warnings so the caller is informed
                    # without blocking execution or triggering error_correct.
                    if loop_result.validation_warning:
                        state.result_warnings.append(loop_result.validation_warning)

                    # Gap-7: store successful execution result in the per-session
                    # last-3 cache. This enables multi-turn "use the previous
                    # result" queries where the model needs access to prior data.
                    if state.session_id not in self._result_cache:
                        # H-10 FIX: evict the oldest session entry when the cap
                        # is reached before inserting a new one.
                        if len(self._result_cache) >= self._result_cache_max:
                            self._result_cache.popitem(last=False)
                        self._result_cache[state.session_id] = collections.deque(maxlen=3)
                    else:
                        # Touch: move to end so it is the most-recently-used entry.
                        self._result_cache.move_to_end(state.session_id)
                    self._result_cache[state.session_id].append(
                        {
                            "code": state.active_code,
                            "code_type": state.code_type,
                            "result_preview": (
                                state.execution_result.result[:10]
                                if state.execution_result and state.execution_result.result
                                else []
                            ),
                        }
                    )
                    break

                # Determine if retryable
                state.attempt_records = loop_result.attempt_history
                if (
                    loop_result.final_error_type
                    in (
                        "POLICY_VIOLATION",
                        "MUTATION_STATEMENT",
                        "DB_UNAVAILABLE",  # DB connection failure — not fixable by LLM
                        "EXECUTION_TIMEOUT",  # infrastructure/runtime timeout; do not ask LLM to rewrite blindly
                    )
                ):
                    # Not retryable — go straight to terminal error
                    state.error = ErrorDetail(
                        error_code=loop_result.final_error_type or "UNKNOWN",
                        message=loop_result.final_error_message or "",
                        attempted_code=state.active_code,
                    )
                    break

                # Fast-path: if validation caught UNRESOLVED_COLUMN and none of
                # the offending names have any fuzzy match in the registered
                # schema columns, error_correct cannot help (the LLM would
                # return UNRECOVERABLE anyway). Skip the extra ~35s LLM call
                # and go directly to TERMINAL_ERROR.
                #
                # cutoff=0.85 here (vs 0.6 used for user-facing "Did you mean?"
                # suggestions). The fast-path must only let through near-exact
                # typos (e.g. "claim_amnt" → "claim_amount", ratio ≈ 0.94).
                # A lower threshold causes false-positive matches on prefix
                # coincidences: difflib("policy_type","policy_id") ≈ 0.70,
                # which incorrectly allows error_correct to run even though it
                # cannot resolve a structurally absent column.
                if loop_result.final_error_type == "UNRESOLVED_COLUMN":
                    unresolved = loop_result.unresolved_columns
                    if unresolved:
                        close = _fuzzy_match_columns(unresolved, state.schema_columns, cutoff=0.85)
                        if not any(close.get(u) for u in unresolved):
                            state.error = ErrorDetail(
                                error_code="TERMINAL_ERROR",
                                message=(
                                    "I wasn't able to generate a working query for "
                                    "this question. The requested field(s) "
                                    f"({', '.join(repr(u) for u in unresolved)}) "
                                    "are not available in this dataset."
                                ),
                                attempted_code=state.active_code,
                                attempt_history=[r.code for r in state.attempt_records],
                            )
                            break
                        # Close match(es) exist — capture for _error_correct
                        # so the correction prompt includes explicit column
                        # substitution hints (e.g. premium_amt→premium_amount).
                        # Take single best match per column (matches[0]).
                        # _fuzzy_match_columns returns dict[str, list[str]];
                        # _error_correct expects dict[str, str | None].
                        _column_hints = {col: m[0] if m else None for col, m in close.items()}

                if loop_result.final_error_type == "EMPTY_RESULT":
                    # Escalate to INTAKE for clarification — not an ERROR_CORRECT case
                    state.terminal_state = "INTAKE"
                    state.insight = await self._llm.generate_clarification(
                        state.nl_query_clean,
                        trigger="EMPTY_RESULT",
                        context=loop_result.final_error_message or "",
                    )
                    break

                if attempt < MAX_ATTEMPTS - 1:
                    # Inject error context for ERROR_CORRECT on next iteration
                    _code_before = state.active_code
                    await self._error_correct(
                        state, loop_result, column_hints=_column_hints or None
                    )
                    # _error_correct sets state.error on UNRECOVERABLE (or on
                    # an internal LLM failure). Either way, do not advance to
                    # the next generation attempt — the error is terminal.
                    if state.error:
                        break
                    # H2 FIX: only skip generation when error_correct actually
                    # produced different code.  If _error_correct returned a
                    # parsed response with an empty sql/code field (silent no-op),
                    # active_code is unchanged.  Setting _skip_generation=True in
                    # that case would re-submit the identical broken code on the
                    # next attempt, wasting a retry slot without making progress.
                    # Falling through (skip_generation=False) causes _generation
                    # to run on the next iteration, giving the LLM a fresh attempt.
                    if state.active_code and state.active_code != _code_before:
                        _skip_generation = True
                    elif state.active_code:
                        # error_correct produced non-empty code identical to its
                        # input — the model converged without improvement.  A
                        # fresh _generation() with the same prompt reproduces the
                        # same broken SQL (confirmed empirically), burning a token
                        # slot with no chance of recovery.  Terminate instead.
                        state.error = ErrorDetail(
                            error_code="TERMINAL_ERROR",
                            message=(
                                f"I wasn't able to generate a working query. "
                                f"Error correction converged after {attempt + 1} attempt(s)."
                            ),
                            attempted_code=state.active_code,
                            attempt_history=[r.code for r in state.attempt_records],
                        )
                        break
                    # else: state.active_code is empty (silent no-op) —
                    # original H2 intent: fall through to _generation().
                else:
                    # Exhausted retries
                    state.error = ErrorDetail(
                        error_code="TERMINAL_ERROR",
                        message=(
                            f"I wasn't able to generate a working query after "
                            f"{MAX_ATTEMPTS} attempts."
                        ),
                        attempted_code=state.active_code,
                        attempt_history=[r.code for r in state.attempt_records],
                    )

            # RESULT_CHECK + INSIGHT (if execution succeeded)
            if state.error is None and state.execution_result and state.execution_result.success:
                await self._result_check(state)
                # _result_check sets state.insight directly when IMPLAUSIBLE_VALUE
                # is detected (null-dominant metric columns). A non-empty string
                # here means the data quality guard already wrote the user-facing
                # message — do not overwrite it with an LLM narration of bad data.
                if not state.insight:
                    await self._insight_state(state)

            elif state.error is None and state.terminal_state != "INTAKE":
                # Defensive dead branch: state.error is None but execution_result
                # is also None (or failed), and no clarification redirect was
                # triggered. Not reachable in normal flow — all code-generating
                # paths either set state.error, populate execution_result, or
                # set terminal_state="INTAKE". Exists as a silent safety net
                # so the RESULT_CHECK / INSIGHT block is never entered on a
                # partially-constructed state.
                pass

        except Exception as exc:
            # H-09 FIX: capture and emit the full exception context before
            # setting the INTERNAL_ERROR response. The original bare `except`
            # discarded the exception object entirely — no type, no message,
            # no stack information was logged, making production incidents
            # effectively undiagnosable from logs alone.
            print(
                json.dumps(
                    {
                        "event": "ORCHESTRATOR_UNHANDLED_EXCEPTION",
                        "session_id": state.session_id,
                        "turn_id": state.turn_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    default=str,
                ),
                file=sys.stdout,
                flush=True,
            )
            # Catch-all: orchestrator never lets an exception reach the API layer
            state.error = ErrorDetail(
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred. Please try again.",
            )

        # Generate a user-facing insight for error states using the same
        # INSIGHT_SYSTEM_PROMPT (Rule 4 handles the error case). This keeps
        # `insight` as a human-readable explanation rather than a raw error
        # string, and separates the error detail (in `error.message`) from
        # the primary-persona summary (in `insight`).
        # Model-failure errors get a fixed, accurate insight rather than an
        # LLM-generated one that could mislead the user into thinking their
        # question was wrong. SEMANTIC_AGG_MISSING is a model failure: the
        # user phrased the question correctly; the model omitted the aggregate
        # function. An LLM insight on this error says 'rephrase to include AVG'
        # which incorrectly blames the user.
        _STATIC_ERROR_INSIGHTS: dict[str, str] = {
            "SEMANTIC_AGG_MISSING": (
                "The query engine generated SQL without the required aggregate "
                "function and could not correct it after retrying. This is a "
                "model limitation \u2014 your question was correctly phrased. "
                "Switch to a more capable model (e.g. llama-3.3-70b-versatile) "
                "or try rephrasing slightly."
            ),
        }
        if state.error is not None and not state.insight:
            static_insight = _STATIC_ERROR_INSIGHTS.get(state.error.error_code)
            if static_insight:
                state.insight = static_insight
            else:
                with contextlib.suppress(Exception):  # _build_response fallback covers this
                    state.insight = await self._llm.generate_insight(
                        nl_query=state.nl_query_clean,
                        result_preview=[],
                        row_count=0,
                        result_warnings=[],
                        error=state.error.message,
                        model=self._model,
                    )

        latency_ms = int((time.monotonic() - t0) * 1000)
        terminal = "TERMINAL_ERROR" if state.error else state.terminal_state or "TERMINAL"

        self._obs.record_turn_outcome(
            turn_id=state.turn_id,
            terminal_state=terminal,
            latency_ms=latency_ms,
            executable=state.execution_result is not None and state.execution_result.success,
            hit_max_retries=state.attempt_count >= MAX_ATTEMPTS - 1,
            schema_id=state.schema_id or "",
            code_type=state.code_type or "",
        )

        response = self._build_response(state, latency_ms)
        await self._persist_turn(state, response)
        return response

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _intake(self, state: TurnState) -> None:
        """
        INTAKE state: sanitise input, detect injection.
        Transition: always → RETRIEVAL (or error on injection).
        """
        with TraceLogger(
            self._obs.trace_store, state.session_id, state.turn_id, state="INTAKE"
        ) as tl:
            tl.set_input({"nl_query_length": len(state.nl_query)})
            clean, injection_detected = sanitise_input(state.nl_query)
            state.nl_query_clean = clean

            if injection_detected:
                state.error = ErrorDetail(
                    error_code="POLICY_VIOLATION",
                    message=("Your query contains characters I can't process. " "Please rephrase."),
                )
                tl.set_output({"injection_detected": True})
            else:
                tl.set_output({"injection_detected": False, "cleaned_length": len(clean)})

    async def _retrieval_state(self, state: TurnState) -> None:
        """
        RETRIEVAL state: fetch top-K schema chunks + session history concurrently.
        Transition: chunks ≥ 1 → GENERATION; chunks = 0 → INTAKE (clarification).
        """
        with TraceLogger(
            self._obs.trace_store, state.session_id, state.turn_id, state="RETRIEVAL"
        ) as tl:
            tl.set_input({"schema_id": state.schema_id, "k": 5})

            # Concurrent retrieval + session load (Section 9.3 async pattern)
            try:
                chunks, history, schema_columns, table_policies = await asyncio.gather(
                    self._retrieval.retrieve(state.nl_query_clean, state.schema_id, k=5),
                    self._sessions.get_history(state.session_id, n=10),
                    self._retrieval.get_schema_columns(state.schema_id),
                    self._retrieval.get_table_policies(state.schema_id),
                )
            except Exception as exc:
                import traceback as _tb

                state.error = ErrorDetail(
                    error_code="RETRIEVAL_ERROR",
                    message=(
                        f"Could not load schema '{state.schema_id}'. "
                        "Check that the schema has been ingested via /ingest."
                    ),
                )
                # FIX: was str(exc) — AssertionError and similar exceptions
                # have no message, so str(exc) == "" making the log useless.
                # Now includes the exception type and a condensed traceback so
                # retrieval_exception in STATE_TRANSITION output is actionable.
                tl.set_output(
                    {
                        "retrieval_exception": (f"{type(exc).__name__}: {exc!r}"),
                        "retrieval_traceback": _tb.format_exc().splitlines()[-5:],
                    }
                )
                return

            state.chunks = chunks
            state.session_history = history
            state.schema_columns = schema_columns
            state.table_policies = table_policies

            # Gap-7: inject last-3 prior execution results for multi-turn context.
            # These are available to the prompt renderer so the LLM can reference
            # prior result shapes for "now filter that" / "use the previous result"
            # type queries without needing to re-execute.
            cached = self._result_cache.get(state.session_id)
            state.prior_results = list(cached) if cached else []

            # Apply token budget enforcement
            trimmed_chunks, trimmed_history, total_tokens = enforce_token_budget(chunks, history)
            state.chunks = trimmed_chunks
            state.session_history = trimmed_history

            if not state.chunks:
                state.terminal_state = "INTAKE"
                state.insight = await self._llm.generate_clarification(
                    state.nl_query_clean,
                    trigger="NO_SCHEMA_MATCH",
                    context=f"schema_id={state.schema_id}",
                )
                tl.set_output({"chunks_returned": 0, "transition": "INTAKE"})
            else:
                # Advance the state machine — without this the gate at run()
                # (`if state.terminal_state == "INTAKE"`) fires every time and
                # GENERATION is never entered (terminal_state defaults to "INTAKE").
                state.terminal_state = "GENERATION"

                # Phase 3: retrieve long-term memory examples as few-shot context
                lt_examples: list = []
                if self._long_term_memory is not None:
                    lt_raw = await asyncio.to_thread(
                        self._long_term_memory.search,
                        state.nl_query_clean,
                        state.schema_id,
                        3,
                    )
                    lt_examples = lt_raw

                    # Gap-10 FIX: USE_LT_EXACT_HIT fast path.
                    # When enabled and the top LTM result is an exact hit
                    # (similarity >= 0.97), inject the cached SQL directly into
                    # state and skip the full LLM generation call.
                    # This eliminates an ~2-4s LLM round-trip for repeat queries
                    # against stable schemas.
                    # Risk: near-duplicate queries with different intent (e.g.
                    # "claims this quarter" vs "claims last quarter") may score
                    # >= 0.97 on poor embeddings. Keep disabled by default;
                    # enable only after validating on your query distribution.
                    #
                    # H-08 + H-12 FIX (two coupled bugs):
                    #   H-12: getattr(lt_examples[0], "generated_code", None)
                    #         always returned None — MemorySearchResult has no
                    #         generated_code attribute; the code lives at
                    #         lt_examples[0].record.sql (MemoryRecord.sql).
                    #         This made the entire if-block permanently unreachable.
                    #   H-08: state.active_code is a read-only @property with no
                    #         setter; assigning to it raises AttributeError at
                    #         runtime.  Write the backing fields directly instead.
                    if (
                        self._use_lt_exact_hit
                        and lt_examples
                        and lt_examples[0].is_exact_hit  # direct attr
                        and lt_examples[0].record.sql  # correct path
                    ):
                        cached_code: str = lt_examples[0].record.sql
                        # MemoryRecord has no code_type field — infer from content.
                        _inferred_type: Literal["sql", "pandas"] = (
                            "pandas"
                            if cached_code.lstrip().startswith(("df", "import", "result ="))
                            and "SELECT" not in cached_code.upper()
                            else "sql"
                        )
                        if _inferred_type == "pandas":
                            state.generated_code = cached_code
                            state.generated_sql = None
                        else:
                            state.generated_sql = cached_code
                            state.generated_code = None
                        state.code_type = _inferred_type
                        state.terminal_state = "VALIDATION"  # skip to VALIDATION
                        state.lt_exact_hit_used = True  # mark for _result_check invalidation
                        tl.set_output(
                            {
                                "chunks_returned": len(state.chunks),
                                "estimated_tokens": total_tokens,
                                "lt_examples": len(lt_examples),
                                "lt_exact_hit": True,
                                "transition": "VALIDATION",
                            }
                        )
                        return  # bypass GENERATION

                state.lt_examples = lt_examples  # consumed by _generation prompt renderer

                tl.set_output(
                    {
                        "chunks_returned": len(state.chunks),
                        "estimated_tokens": total_tokens,
                        "lt_examples": len(lt_examples),
                        "lt_exact_hit": False,
                        "transition": "GENERATION",
                    }
                )

    async def _generation(self, state: TurnState, attempt: int) -> None:
        """
        GENERATION state: call LLM with NL_TO_SQL or NL_TO_PANDAS prompt.
        Sets state.generated_sql / state.generated_code and state.code_type.

        Tool selection policy (Section 2H):
          SQL is default. Pandas only when execution_mode='pandas' or
          the query contains multi-step transformation markers.
        """
        code_type = _select_code_type(state)
        state.code_type = code_type

        if code_type == "sql":
            prompt = PromptRenderer.render(
                NL_TO_SQL_SYSTEM_PROMPT,
                hard_constraints=_extract_hard_constraints(state.chunks),
                schema_context=[c.model_dump() for c in state.chunks],
                dialect=state.dialect,
                session_history=state.session_history,
                nl_query=state.nl_query_clean,
                lt_examples=_format_lt_examples(state.lt_examples),
            )
            response_model = GenerateSQLOutput
        else:
            df_refs = [c.table for c in state.chunks]
            prompt = PromptRenderer.render(
                NL_TO_PANDAS_SYSTEM_PROMPT,
                dataframe_refs=df_refs,
                schema_context=[c.model_dump() for c in state.chunks],
                session_history=state.session_history,
                nl_query=state.nl_query_clean,
            )
            response_model = GeneratePandasOutput

        with TraceLogger(
            self._obs.trace_store,
            state.session_id,
            state.turn_id,
            state="GENERATION",
            attempt=attempt,
            model=self._model,
        ) as tl:
            tl.set_input(
                {
                    "code_type": code_type,
                    "schema_chunks": len(state.chunks),
                    "prompt_tokens": count_tokens(prompt),
                }
            )

            req = GenerationRequest(
                prompt_type="nl_to_sql" if code_type == "sql" else "nl_to_pandas",
                system_prompt=prompt,
                response_model=response_model,
                model=self._model,
            )

            try:
                gen_resp = await self._llm.generate(req)
            except Exception as exc:
                state.error = ErrorDetail(
                    error_code="LLM_UNAVAILABLE",
                    message="The query service is temporarily unavailable. Please try again.",
                )
                tl.set_output({"error": str(exc)})
                return

            tl.set_tokens(gen_resp.prompt_tokens, gen_resp.completion_tokens)

            if gen_resp.parse_error:
                state.error = ErrorDetail(
                    error_code="LLM_PARSE_ERROR",
                    message=("I received an unexpected response format. Retrying automatically."),
                )
                tl.set_output({"parse_error": gen_resp.parse_error})
                return

            parsed = gen_resp.parsed

            # UNRESOLVED_REFERENCE — LLM could not ground the query
            if isinstance(parsed, GenerateSQLOutput | GeneratePandasOutput):
                if parsed.error_code == "UNRESOLVED_REFERENCE":
                    unresolved = parsed.unresolved
                    close_matches = _fuzzy_match_columns(unresolved, state.schema_columns)
                    state.error = ErrorDetail(
                        error_code="UNRESOLVED_REFERENCE",
                        message=_unresolved_message(unresolved, close_matches),
                    )
                    tl.set_output({"unresolved": unresolved})
                    return

                # MUTATION_REQUESTED — LLM's Rule 4 blocked a write operation.
                # Without this check the code falls through to the sql/code
                # assignment block, finds parsed.sql is empty, and silently
                # returns — triggering Bug 1 before the fix above.
                if parsed.error_code in ("MUTATION_REQUESTED", "MUTATION_STATEMENT"):
                    state.error = ErrorDetail(
                        error_code="MUTATION_STATEMENT",
                        message=(
                            "I can only run read-only queries. "
                            "This question implies modifying data, which isn't permitted."
                        ),
                    )
                    tl.set_output({"mutation_blocked": True})
                    return

                # LOW CONFIDENCE — surface both interpretations
                if parsed.confidence < 0.7:
                    state.generation_assumptions = parsed.assumptions
                    state.generation_confidence = parsed.confidence
                    state.insight = await self._llm.generate_clarification(
                        state.nl_query_clean,
                        trigger="LOW_CONFIDENCE",
                        context=json.dumps(parsed.assumptions),
                    )
                    state.terminal_state = "INTAKE"
                    tl.set_output({"low_confidence": True, "confidence": parsed.confidence})
                    return

            if isinstance(parsed, GenerateSQLOutput) and parsed.sql:
                state.generated_sql = parsed.sql

                # Post-processor: F1 integer-division cast, F2 NULLIF guard,
                # W1 fan-out warning. Wrapped in try/except so a sqlglot
                # failure never blocks execution (belt-and-suspenders).
                try:
                    _pp = postprocess_sql(state.generated_sql)
                    if _pp.was_modified:
                        state.generated_sql = _pp.sql
                    if _pp.warnings:
                        state.result_warnings.extend(_pp.warnings)
                    # W1 fan-out requires LLM regeneration (AST rewrite cannot
                    # restructure CTEs). Store the hint so run() can route to
                    # _error_correct_postprocessor before ExecutionLoop runs.
                    state.postprocessor_hint = _pp.retry_hint if _pp.needs_retry else None
                except Exception:  # noqa: BLE001
                    pass  # post-processor failure must never block execution

                state.generation_confidence = parsed.confidence
                state.generation_assumptions = parsed.assumptions

                # Proactive grounding check: the LLM sometimes generates SQL
                # with hallucinated column names while also self-reporting
                # all_columns_verified=False in grounding_check. Catching this
                # here avoids a round-trip through ExecutionLoop and surfaces a
                # clearer error message with fuzzy-match suggestions.
                gc = parsed.grounding_check
                if gc and not gc.all_columns_verified and gc.unresolved_references:
                    close_matches = _fuzzy_match_columns(
                        gc.unresolved_references, state.schema_columns
                    )
                    state.error = ErrorDetail(
                        error_code="UNRESOLVED_REFERENCE",
                        message=_unresolved_message(gc.unresolved_references, close_matches),
                    )
                    tl.set_output(
                        {
                            "grounding_check_failed": True,
                            "unresolved": gc.unresolved_references,
                        }
                    )
                    return

                # Semantic aggregation guard: detect questions that ask for
                # an aggregate (average, sum, count, etc.) but received SQL
                # with no corresponding aggregate function. Syntactically
                # valid SQL that answers the wrong question passes all
                # structural validators and executes silently, returning
                # misleading results with a fabricated insight. Catch it
                # here before ExecutionLoop runs.
                #
                # NOTE: we do NOT set state.error here.  Setting state.error
                # causes the for-loop to break immediately (before reaching
                # the _error_correct branch), making this a hard terminal with
                # no retry.  Instead we set state.sem_agg_missing so the
                # for-loop can call _error_correct_agg on the next iteration.
                # Strip sort-direction phrases before intent matching to avoid
                # false positives: "ordered from highest to lowest" should not
                # trigger MAX or MIN — those are sort keywords, not aggregates.
                _nl_for_agg = re.sub(
                    r"\b(highest\s+to\s+lowest|lowest\s+to\s+highest"
                    r"|high\s+to\s+low|low\s+to\s+high)\b",
                    "",
                    state.nl_query_clean,
                    flags=re.I,
                )
                for _intent_re, _sql_re, _fn_name in _AGG_INTENT_MAP:
                    if _intent_re.search(_nl_for_agg) and not _sql_re.search(parsed.sql):
                        # BUG-04 FIX: _intent_re matched _nl_for_agg (the
                        # sort-phrase-stripped version) but may not match
                        # nl_query_clean if the intent keyword was adjacent to
                        # a stripped phrase at a word boundary.  Use the first
                        # non-None match to avoid AttributeError on .group(0).
                        _raw_match = _intent_re.search(state.nl_query_clean) or _intent_re.search(
                            _nl_for_agg
                        )
                        if _raw_match is None:
                            continue
                        _matched_word = _raw_match.group(0)
                        state.sem_agg_missing = (_fn_name, _matched_word)
                        tl.set_output(
                            {
                                "semantic_agg_check_failed": True,
                                "missing_agg": _fn_name,
                                "trigger_word": _matched_word,
                            }
                        )
                        return

                tl.set_output(
                    {
                        "sql_length": len(parsed.sql),
                        "confidence": parsed.confidence,
                        "sql": parsed.sql,
                    }
                )

            elif isinstance(parsed, GeneratePandasOutput) and parsed.code:
                state.generated_code = parsed.code
                state.generation_confidence = parsed.confidence
                state.generation_assumptions = parsed.assumptions
                tl.set_output({"code_length": len(parsed.code), "confidence": parsed.confidence})

            else:
                # LLM returned valid JSON that parsed without error but contained no
                # executable code and no recognised error_code.  Surface this
                # explicitly rather than silently returning with active_code == ""
                # (which would feed an empty string to ExecutionLoop and produce a
                # misleading "The query completed successfully." response).
                if state.error is None and state.terminal_state != "INTAKE":
                    state.error = ErrorDetail(
                        error_code="LLM_EMPTY_RESPONSE",
                        message=(
                            "The query engine returned an empty response. "
                            "Please try rephrasing your question."
                        ),
                        attempted_code="",
                    )
                    tl.set_output({"empty_response": True})

    async def _error_correct_postprocessor(self, state: TurnState, attempt: int) -> None:
        """
        Targeted correction pass for W1 postprocessor fan-out warnings.

        Fires before ExecutionLoop when postprocess_sql detected that the LLM
        joined fan-out tables (payments → claims → policies) without the required
        pre-aggregation CTEs.  The postprocessor cannot fix this by AST rewrite
        alone — it requires schema-aware CTE restructuring by the LLM.

        Injects the postprocessor's retry_hint (which describes exactly which
        join pair triggered the warning and the required CTE fix) into the
        ERROR_CORRECT prompt so the LLM rewrites the FROM/JOIN block.

        On success: state.generated_sql is overwritten with the corrected SQL.
        On failure: returns without setting state.error so fresh generation
                    runs on the next attempt.
        """
        if state.postprocessor_hint is None:
            raise RuntimeError(
                "_error_correct_postprocessor called with postprocessor_hint=None; "
                "only call when the W1 hint is set."
            )

        attempt_history_str = (
            f"Attempt {attempt}:\n{state.active_code}\n"
            f"Validation error: W1_FANOUT — JOIN fan-out inflates SUM aggregates\n"
            f"Execution error: none (query would execute but return wrong totals)"
        )

        prompt = PromptRenderer.render(
            ERROR_CORRECT_SYSTEM_PROMPT,
            hard_constraints=_extract_hard_constraints(state.chunks),
            nl_query=state.nl_query_clean,
            code_type=state.code_type,
            schema_context=[c.model_dump() for c in state.chunks],
            attempt_history=attempt_history_str,
            error_type="W1_FANOUT",
            error_message=state.postprocessor_hint,
            error_line="",
        )

        with TraceLogger(
            self._obs.trace_store,
            state.session_id,
            state.turn_id,
            state="ERROR_CORRECT",
            attempt=attempt,
            model=self._model,
        ) as tl:
            tl.set_input({"error_type": "W1_FANOUT", "attempt": attempt})

            req = GenerationRequest(
                prompt_type="error_correct",
                system_prompt=prompt,
                response_model=GenerateSQLOutput,
                model=self._model,
                max_tokens=1_500,
            )

            try:
                gen_resp = await self._llm.generate(req)
            except Exception as exc:  # noqa: BLE001
                tl.set_output({"error": str(exc)})
                return  # non-fatal: fresh generation on next attempt

            tl.set_tokens(gen_resp.prompt_tokens, gen_resp.completion_tokens)

            if gen_resp.parse_error or not gen_resp.parsed:
                tl.set_output(
                    {
                        "parse_error": gen_resp.parse_error,
                        "raw_content": gen_resp.content[:500],
                    }
                )
                return  # non-fatal: fresh generation on next attempt

            parsed = gen_resp.parsed

            if hasattr(parsed, "error_code") and parsed.error_code == "UNRECOVERABLE":
                # W1 fan-out is always recoverable given a correct CTE pattern;
                # UNRECOVERABLE here means the LLM cannot resolve schema constraints.
                # Log it but do not terminate — fall through to fresh generation.
                tl.set_output({"unrecoverable": True})
                return

            if isinstance(parsed, GenerateSQLOutput) and parsed.sql:
                state.generated_sql = parsed.sql
                tl.set_output({"corrected_sql_length": len(parsed.sql)})

    async def _error_correct_agg(self, state: TurnState, attempt: int) -> None:
        """
        Targeted correction pass for SEMANTIC_AGG_MISSING.

        Unlike the general _error_correct (which requires a loop_result from
        ExecutionLoop), this method fires before ExecutionLoop runs — the LLM
        produced syntactically valid SQL that passed structural validation but
        omitted the required aggregate function.  We inject a single, precise
        constraint and ask the model to rewrite the SELECT clause.

        On success: state.generated_sql is overwritten with the corrected SQL.
        On failure: state.error is set (terminal).
        """
        # M-15 pattern: assert is compiled out under python -O; use explicit guard.
        if state.sem_agg_missing is None:
            raise RuntimeError(
                "_error_correct_agg called with sem_agg_missing=None; "
                "this is a caller bug — only call when the guard is set."
            )
        fn_name, trigger_word = state.sem_agg_missing

        constraint_msg = (
            f"Your previous SQL was missing the required aggregate function.\n"
            f"The question contains '{trigger_word}' — the SELECT clause MUST "
            f"use {fn_name}(...) as specified in Rule 14.\n"
            f"Rewrite the query so the SELECT clause contains {fn_name}().\n"
            f"Do not change any other part of the query.\n\n"
            f"Rejected SQL:\n{state.active_code}"
        )

        # Reuse ERROR_CORRECT_SYSTEM_PROMPT — it already handles the rewrite
        # contract and UNRECOVERABLE sentinel.  We synthesise a minimal
        # attempt_history entry from the rejected SQL so Rule 1 is satisfied.
        attempt_history_str = (
            f"Attempt {attempt}:\n{state.active_code}\n"
            f"Validation error: SEMANTIC_AGG_MISSING — {fn_name}() absent\n"
            f"Execution error: none"
        )

        prompt = PromptRenderer.render(
            ERROR_CORRECT_SYSTEM_PROMPT,
            hard_constraints=_extract_hard_constraints(state.chunks),
            nl_query=state.nl_query_clean,
            code_type=state.code_type,
            schema_context=[c.model_dump() for c in state.chunks],
            attempt_history=attempt_history_str,
            error_type="SEMANTIC_AGG_MISSING",
            error_message=constraint_msg,
            error_line="",
        )

        response_model = GenerateSQLOutput if state.code_type == "sql" else GeneratePandasOutput

        with TraceLogger(
            self._obs.trace_store,
            state.session_id,
            state.turn_id,
            state="ERROR_CORRECT",
            attempt=attempt,
            model=self._model,
        ) as tl:
            tl.set_input(
                {
                    "error_type": "SEMANTIC_AGG_MISSING",
                    "missing_agg": fn_name,
                    "attempt": attempt,
                }
            )

            req = GenerationRequest(
                prompt_type="error_correct",
                system_prompt=prompt,
                response_model=response_model,
                model=self._model,
                max_tokens=1_500,
            )

            try:
                gen_resp = await self._llm.generate(req)
            except Exception as exc:
                tl.set_output({"error": str(exc)})
                # Non-fatal: let the next generation attempt run fresh.
                return

            tl.set_tokens(gen_resp.prompt_tokens, gen_resp.completion_tokens)

            if gen_resp.parse_error or not gen_resp.parsed:
                tl.set_output(
                    {
                        "parse_error": gen_resp.parse_error,
                        "raw_content": gen_resp.content[:500],
                    }
                )
                # Non-fatal: fall through to fresh generation on next attempt.
                return

            parsed = gen_resp.parsed

            if hasattr(parsed, "error_code") and parsed.error_code == "UNRECOVERABLE":
                state.error = ErrorDetail(
                    error_code="TERMINAL_ERROR",
                    message=(
                        "I wasn't able to generate a working query for this question. "
                        f"Reason: {getattr(parsed, 'reason', 'unknown')}."
                    ),
                    attempted_code=state.active_code or None,
                    attempt_history=[r.code for r in state.attempt_records],
                )
                tl.set_output({"unrecoverable": True})
                return

            if isinstance(parsed, GenerateSQLOutput) and parsed.sql:
                state.generated_sql = parsed.sql
                tl.set_output({"corrected_sql_length": len(parsed.sql)})
            elif isinstance(parsed, GeneratePandasOutput) and parsed.code:
                state.generated_code = parsed.code
                tl.set_output({"corrected_code_length": len(parsed.code)})

    async def _error_correct(
        self,
        state: TurnState,
        loop_result: Any,
        column_hints: dict[str, str | None] | None = None,
    ) -> None:
        """
        ERROR_CORRECT state: inject error context and call the error correction prompt.
        The corrected code overwrites state.generated_sql / state.generated_code
        so that the next iteration of the generation → validation → execution loop
        uses it automatically.

        column_hints: fuzzy-matched {bad_col: good_col} pairs from the
        UNRESOLVED_COLUMN fast-path (cutoff=0.85).  Injected into the error
        message as explicit substitution instructions so the LLM corrects the
        right column name rather than re-hallucinating a different wrong variant.
        Handles name-typo errors (e.g. premium_amt → premium_amount).

        A separate table-aware listing is always injected for UNRESOLVED_COLUMN
        regardless of column_hints, covering the 'wrong table' class of error
        (column exists in schema but not on the aliased table) which fuzzy-match
        alone cannot detect.
        """
        attempt_history_str = "\n---\n".join(
            f"Attempt {r.attempt}:\n{r.code}\n"
            f"Validation error: {r.validation_error or 'none'}\n"
            f"Execution error: {r.execution_error or 'none'}"
            for r in state.attempt_records
        )

        error_message = loop_result.final_error_message or ""

        # Inject name-substitution hints for UNRESOLVED_COLUMN
        if column_hints:
            suggestions = [
                f"  '{bad}' \u2192 use '{good}' instead"
                for bad, good in column_hints.items()
                if good and bad != good
            ]
            if suggestions:
                error_message += "\nColumn name corrections (use these exact names):\n" + "\n".join(
                    suggestions
                )

        # Inject per-table column listing for UNRESOLVED_COLUMN.
        # This handles the 'wrong table' class: the column name is valid in
        # the schema but does not exist on the specific aliased table the LLM
        # referenced.  Fuzzy-match hints return the same column name (ratio=1.0)
        # for these, providing no useful signal.  The explicit listing lets the
        # LLM see which columns actually belong to each table and self-correct.
        if loop_result.final_error_type == "UNRESOLVED_COLUMN":
            table_cols: dict[str, list[str]] = {}
            for chunk in state.chunks:
                if chunk.table and chunk.columns:
                    table_cols.setdefault(chunk.table, []).extend(col.name for col in chunk.columns)
            if table_cols:
                error_message += (
                    "\nExact columns available per table"
                    " (use ONLY column names from this list):\n"
                    + "\n".join(
                        f"  {tbl}: {', '.join(cols)}" for tbl, cols in sorted(table_cols.items())
                    )
                )

        prompt = PromptRenderer.render(
            ERROR_CORRECT_SYSTEM_PROMPT,
            hard_constraints=_extract_hard_constraints(state.chunks),
            nl_query=state.nl_query_clean,
            code_type=state.code_type,
            schema_context=[c.model_dump() for c in state.chunks],
            attempt_history=attempt_history_str,
            error_type=loop_result.final_error_type or "",
            error_message=error_message,
            error_line="",
        )

        response_model = GenerateSQLOutput if state.code_type == "sql" else GeneratePandasOutput

        with TraceLogger(
            self._obs.trace_store,
            state.session_id,
            state.turn_id,
            state="ERROR_CORRECT",
            attempt=state.attempt_count,
            model=self._model,
        ) as tl:
            tl.set_input(
                {
                    "error_type": loop_result.final_error_type,
                    "attempt": state.attempt_count,
                }
            )

            req = GenerationRequest(
                prompt_type="error_correct",
                system_prompt=prompt,
                response_model=response_model,
                model=self._model,
                # max_tokens tuned for Groq rate limits:
                # - 1_000 (default) truncates mid-JSON for complex corrections
                # - 2_500 reserves too many output tokens against Groq's
                #   6000 TPM budget for llama-3.1-8b-instant, causing
                #   23s+ queuing even when only 66 tokens are generated
                # - 1_500 fits the longest corrected SQL + JSON wrapper
                #   while keeping rate-limit pressure ~40% lower than 2500
                max_tokens=1_500,
            )

            try:
                gen_resp = await self._llm.generate(req)
            except Exception as exc:
                tl.set_output({"error": str(exc)})
                return

            tl.set_tokens(gen_resp.prompt_tokens, gen_resp.completion_tokens)

            if gen_resp.parse_error or not gen_resp.parsed:
                # ERROR_CORRECT returned malformed JSON even after the LLM's
                # own backoff retries. Do not silently advance to the next
                # generation attempt — that would retry without error context,
                # making the correction loop invisible in the trace. Treat
                # this as terminal so the caller gets a clear error signal.
                tl.set_output(
                    {
                        "parse_error": gen_resp.parse_error,
                        # First 500 chars captured for alias-table diagnosis.
                        # If this is a new llama quirk, add it to _ERROR_CODE_ALIASES
                        # in prompts.py once the pattern is confirmed in the trace.
                        "raw_content": gen_resp.content[:500],
                        "completion_tokens": gen_resp.completion_tokens,
                    }
                )
                state.error = ErrorDetail(
                    error_code="TERMINAL_ERROR",
                    message=(
                        "I wasn't able to correct the query — "
                        "the correction step returned an unexpected response."
                    ),
                    attempted_code=state.active_code or None,
                    attempt_history=[r.code for r in state.attempt_records],
                )
                return

            parsed = gen_resp.parsed

            # UNRECOVERABLE from ERROR_CORRECT prompt
            if hasattr(parsed, "error_code") and parsed.error_code == "UNRECOVERABLE":
                state.error = ErrorDetail(
                    error_code="TERMINAL_ERROR",
                    message=(
                        "I wasn't able to generate a working query for this question. "
                        f"Reason: {getattr(parsed, 'reason', 'unknown')}."
                    ),
                    attempted_code=state.active_code or None,
                    attempt_history=[r.code for r in state.attempt_records],
                )
                tl.set_output({"unrecoverable": True})
                return

            if isinstance(parsed, GenerateSQLOutput) and parsed.sql:
                state.generated_sql = parsed.sql
                tl.set_output({"corrected_sql_length": len(parsed.sql)})
            elif isinstance(parsed, GeneratePandasOutput) and parsed.code:
                state.generated_code = parsed.code
                tl.set_output({"corrected_code_length": len(parsed.code)})

    async def _result_check(self, state: TurnState) -> None:
        """
        RESULT_CHECK state: validate result shape and plausibility.
        Populates state.result_rows and appends to state.result_warnings.
        Transition: valid → INSIGHT; EMPTY_RESULT → INTAKE.
        """
        with TraceLogger(
            self._obs.trace_store, state.session_id, state.turn_id, state="RESULT_CHECK"
        ) as tl:
            exec_result = state.execution_result
            if exec_result is None or not exec_result.success:
                tl.set_output({"skipped": True})
                return

            rows: list[dict[str, Any]] = exec_result.result or []
            rt = validate_result(rows)

            if not rt.valid and rt.issue == "EMPTY_RESULT":
                state.terminal_state = "INTAKE"
                state.insight = await self._llm.generate_clarification(
                    state.nl_query_clean,
                    trigger="EMPTY_RESULT",
                    context=rt.message or "",
                )
                tl.set_output({"issue": "EMPTY_RESULT"})
                return

            state.result_rows = rows

            if rt.issue == "RESULT_CAPPED" and rt.message:
                state.result_warnings.append(rt.message)

            if rt.issue == "IMPLAUSIBLE_VALUE" and rt.message:
                # Null-dominant result: the query succeeded structurally but all
                # primary metric columns are NULL (e.g. claim_amount on pending
                # claims). Add the quality warning to result_warnings so the
                # caller sees it in QueryResponse.warnings, then short-circuit
                # insight generation by setting state.insight directly.
                #
                # run() checks `if not state.insight` before calling _insight_state,
                # so a non-empty string here suppresses the LLM call entirely.
                # This prevents the model from narrating incidental columns
                # (e.g. total_premium) as if they were the requested metrics.
                state.result_warnings.append(rt.message)
                state.insight = rt.message
                state.terminal_state = "TERMINAL"
                tl.set_output({"issue": "IMPLAUSIBLE_VALUE", "insight_suppressed": True})
                return

            # Compute analytical metrics before INSIGHT so the LLM receives
            # pre-calculated ratios and variation classification rather than
            # inferring them from the 5-row result_preview.
            # P1-02 FIX: wrapped in try/except — _compute_result_metrics contains
            # numeric conversions that can raise ValueError on heterogeneous column
            # types (e.g. a COALESCE column that is int in row 0 but 'N/A' in row 5).
            # Degrade gracefully to an empty dict; the LLM falls back to describing
            # result_preview directly, which is an acceptable degradation.
            try:
                state.result_metrics = _compute_result_metrics(rows, nl_query=state.nl_query_clean)
            except Exception:  # noqa: BLE001
                state.result_metrics = {}

            # Domain metric range check: catch rate/ratio columns with values
            # outside business-plausible bounds (e.g. loss_ratio > 2.0).
            # This fires AFTER _compute_result_metrics so result_metrics is
            # populated even when we suppress insight — useful for debugging.
            # Like IMPLAUSIBLE_VALUE: valid=True, non-blocking, suppresses LLM
            # insight to prevent confident narration of wrong numbers.
            _rt_domain = validate_metric_ranges(rows, sql=state.active_code)
            if _rt_domain is not None and _rt_domain.issue == "METRIC_OUT_OF_RANGE":
                state.result_warnings.append(_rt_domain.message or "")
                state.insight = _rt_domain.message or ""
                state.terminal_state = "TERMINAL"
                # If this run used an LTM exact hit, the stale cached SQL is
                # what triggered the validation failure.  Invalidate the entry
                # so the next store() call for the same query overwrites it
                # with the freshly-generated correct SQL.
                if state.lt_exact_hit_used and self._long_term_memory is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(
                            self._long_term_memory.invalidate,
                            state.schema_id,
                            state.nl_query_clean,
                        )
                tl.set_output({"issue": "METRIC_OUT_OF_RANGE", "insight_suppressed": True})
                return

            tl.set_output(
                {
                    "row_count": len(rows),
                    "issue": rt.issue,
                }
            )

    async def _insight_state(self, state: TurnState) -> None:
        """
        INSIGHT state: generate a plain-English summary for the primary persona.
        """
        with TraceLogger(
            self._obs.trace_store,
            state.session_id,
            state.turn_id,
            state="INSIGHT",
            model=self._model,
        ) as tl:
            tl.set_input({"row_count": len(state.result_rows)})

            # Extract business descriptions from the retrieved schema chunks.
            # These give the LLM domain vocabulary to explain WHY patterns
            # exist rather than only reporting WHAT the numbers are.
            # model_dump() is the established pattern in this file — safe to
            # use without importing SchemaChunk attribute names directly.
            schema_descriptions: dict[str, str] = {}
            for chunk in state.chunks:
                # SchemaChunk Pydantic field is "table" — model_dump() produces
                # the key "table", not "table_name". Using get("table_name") here
                # always returned "" (silent miss), so schema_descriptions was
                # always empty and generate_insight() never received domain context.
                chunk_dict = chunk.model_dump()
                table = chunk_dict.get("table", "") or getattr(chunk, "table", "")
                desc = chunk_dict.get("business_description", "") or ""
                if table and desc:
                    schema_descriptions[table] = desc

            state.insight = await self._llm.generate_insight(
                nl_query=state.nl_query_clean,
                result_preview=state.result_rows[:5],
                row_count=len(state.result_rows),
                result_warnings=state.result_warnings,
                error=None,
                model=self._model,
                result_metrics=state.result_metrics,
                schema_descriptions=schema_descriptions or None,
            )
            state.terminal_state = "TERMINAL"
            tl.set_output({"insight_length": len(state.insight)})

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    def _build_response(self, state: TurnState, latency_ms: int) -> QueryResponse:
        result_preview = state.result_rows[:100] if state.result_rows else None
        row_count = len(state.result_rows) if state.result_rows else None

        # When the system returns a clarification question, use it as the insight
        insight = state.insight
        if not insight and state.error:
            insight = state.error.message
        elif not insight:
            insight = "The query completed successfully."

        return QueryResponse(
            session_id=state.session_id,
            generated_code=state.active_code,
            code_type=state.code_type,
            result_preview=result_preview,
            row_count=row_count,
            insight=insight,
            execution_time_ms=latency_ms,
            retry_count=state.attempt_count,
            warnings=state.result_warnings,
            error=state.error,
        )

    async def _persist_turn(self, state: TurnState, response: QueryResponse) -> None:
        """Appends TurnRecord to session history and long-term memory. Fire-and-forget."""
        turn = TurnRecord(
            turn_id=state.turn_id,
            nl_query=state.nl_query,
            generated_code=state.active_code,
            code_type=state.code_type,
            row_count=response.row_count,
            insight=response.insight,
            retry_count=state.attempt_count,
            error=response.error,
        )
        with contextlib.suppress(
            Exception
        ):  # persistence failure must never affect the user response
            await self._sessions.append_turn(state.session_id, turn)

        # Phase 3: store successful turns in long-term memory for few-shot retrieval
        if self._long_term_memory is not None and response.error is None:
            with contextlib.suppress(
                Exception
            ):  # memory store failure must never affect the user response
                await asyncio.to_thread(
                    self._long_term_memory.store,
                    state.session_id,
                    state.schema_id,
                    state.nl_query_clean,
                    state.active_code,
                    state.insight,
                )


# ---------------------------------------------------------------------------
# Tool selection policy
# ---------------------------------------------------------------------------

# P3-02 FIX: removed "merge" and "concat" from _PANDAS_KEYWORDS.
#
# These are common English words that appear in legitimate business questions
# that SQL handles natively:
#   "merge the duplicate policy records"  → SQL: GROUP BY / DISTINCT
#   "concat the agent names into one"     → SQL: STRING_AGG / LISTAGG
#   "how do we merge our two books"       → SQL: UNION ALL
#
# Both words triggered Pandas mode via simple whitespace-split token matching,
# causing the LLM to generate DataFrame code when a SQL query was more
# appropriate, then failing at execution because the relevant tables were not
# available as DataFrames in the sandbox namespace.
#
# The remaining keywords are unambiguous Pandas/NumPy operation names that
# have no SQL equivalent and are extremely unlikely in plain English sentences:
#   melt, reshape, rolling, cumsum, pct_change, shift, resample, pivot_table
#
# "pivot" is also removed — it is a real SQL concept (PIVOT ... FOR ... IN)
# and appears in "pivot the report" meaning "transpose the view", which is
# equally expressible in SQL.  Users who want Pandas can set
# execution_mode='pandas' explicitly.
_PANDAS_KEYWORDS = frozenset(
    {
        "melt",
        "reshape",
        "rolling",
        "cumsum",
        "pct_change",
        "shift",
        "resample",
        "pivot_table",  # more specific than "pivot" — unambiguously Pandas API
    }
)


def _select_code_type(state: TurnState) -> Literal["sql", "pandas"]:
    """
    Section 2H tool selection policy:
      SQL default. Pandas only when:
        (a) execution_mode='pandas' (explicit override)
        (b) multi-step transformation keywords present in the query
    """
    if state.execution_mode == "pandas":
        return "pandas"
    if state.execution_mode == "sql":
        return "sql"
    # auto mode: check for transformation keywords
    query_tokens = set(state.nl_query_clean.lower().split())
    if query_tokens & _PANDAS_KEYWORDS:
        return "pandas"
    return "sql"


def _dialect_for_request(request: QueryRequest) -> str:
    return request.dialect


# ---------------------------------------------------------------------------
# Column fuzzy matching for UNRESOLVED_REFERENCE user messages
# ---------------------------------------------------------------------------


def _fuzzy_match_columns(
    unresolved: list[str],
    known_columns: set[str],
    n: int = 3,
    cutoff: float = 0.6,
) -> dict[str, list[str]]:
    """
    For each unresolved term, return the top-n closest column names by
    difflib sequence matching ratio. Used to populate the user-facing
    "Did you mean: …?" suggestion list.
    """
    result: dict[str, list[str]] = {}
    cols_lower = {c.lower(): c for c in known_columns}
    for term in unresolved:
        matches = difflib.get_close_matches(term.lower(), cols_lower.keys(), n=n, cutoff=cutoff)
        result[term] = [cols_lower[m] for m in matches]
    return result


def _unresolved_message(
    unresolved: list[str],
    close_matches: dict[str, list[str]],
) -> str:
    parts = []
    for term in unresolved:
        suggestions = close_matches.get(term, [])
        if suggestions:
            parts.append(
                f"I couldn't find a column matching '{term}'. "
                f"Did you mean one of: {', '.join(suggestions)}?"
            )
        else:
            parts.append(f"I couldn't find a column or table matching '{term}' in the schema.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Post-execution metrics engine
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    """
    P1-02 FIX: type-safe float coercion for _compute_result_metrics.

    _compute_result_metrics inspects rows[0] to classify column types, but
    actual row values can differ (e.g. a column is int in row 0, but 'N/A' or
    None in later rows).  A bare float() on a non-numeric string raises
    ValueError, which propagated through _result_check (no try/except at the
    call site) all the way to orchestrator.run()'s outer except-clause —
    surfacing as INTERNAL_ERROR for any query whose result contained mixed
    column types (common with COALESCE / CASE WHEN expressions).

    Returns None on any conversion failure.  Callers filter None values out
    of their list comprehensions so metrics degrade gracefully rather than
    crashing the whole metrics pass.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compute_result_metrics(
    rows: list[dict[str, Any]],
    nl_query: str = "",
) -> dict[str, Any]:
    """
    Compute analytical metrics from an execution result before LLM narration.

    Operates on results with this shape:
      - 2+ rows  (ratio / spread not meaningful for single rows)
      - At least one numeric column  (the aggregate value to analyse)
      - Optionally one or more string columns  (the grouping dimension)

    Returns an empty dict when the shape doesn't match — the LLM then
    falls back to describing result_preview directly.  This is intentional:
    it is better to return no metrics than to compute misleading ones on
    multi-dimensional or non-aggregated result sets.

    Numeric columns that appear to be row-count columns (name matches
    ``count_*`` / ``*_count`` / ``n_*``) are de-prioritised.

    When nl_query is provided, the primary value column is selected by
    scoring each non-count numeric column against the tokenised NL query.
    A column whose name tokens overlap the most with the query is preferred.
    This prevents the metrics engine from narrating an incidental column
    (e.g. total_premium_paid) when the query was about a different metric
    (e.g. total_claim_amount) simply because it appeared first in the result.

    Falls back to positional selection (first non-count column) when nl_query
    is absent or no column scores above zero.

    Parameters
    ----------
    rows     : Full execution result (not truncated to 5 like result_preview).
    nl_query : Original natural-language query string (optional).  Used only
               for column preference scoring — never sent to the LLM.

    Returns
    -------
    dict with keys:
      value_col       : name of the aggregate column analysed
      group_col       : name of the primary grouping column (or None)
      n_groups        : number of result rows
      top_group       : value of group_col for the highest-value row
      top_value       : highest numeric value
      bottom_group    : value of group_col for the lowest-value row
      bottom_value    : lowest numeric value
      mean            : arithmetic mean across all rows
      ratio           : top_value / bottom_value (None if bottom_value ≤ 0)
      cv              : coefficient of variation (σ / μ), measure of spread
      variation_type  : "extreme" | "strong" | "moderate" | "uniform"
      ratio_sentence  : pre-formatted comparison string for the LLM to use
    """
    if len(rows) < 2:
        return {}

    sample = rows[0]

    # Partition columns into numeric and string buckets.
    # Exclude bool — isinstance(True, int) is True in Python.
    numeric_cols: list[str] = []
    string_cols: list[str] = []
    for k, v in sample.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float):
            numeric_cols.append(k)
        elif isinstance(v, str):
            string_cols.append(k)

    if not numeric_cols:
        return {}

    # Select primary value column.
    # Step 1: de-prioritise row-count columns.
    _count_pattern = re.compile(r"(^n_|_count$|^count_|^num_)", re.I)
    non_count_cols = [c for c in numeric_cols if not _count_pattern.search(c)]
    candidates = non_count_cols if non_count_cols else numeric_cols

    # Step 2: keyword-based preference using nl_query token overlap.
    # Score each candidate column by how many of its underscore-split tokens
    # appear in the NL query.  The column with the highest overlap wins.
    # Ties are broken by original column order (stable sort).
    # Example: nl_query = "top 5 customers by claim amount"
    #   total_claim_amount → tokens {total, claim, amount} → overlap 2 (claim, amount)
    #   total_premium_paid → tokens {total, premium, paid}  → overlap 0
    #   → value_col = total_claim_amount  ✓
    if nl_query and len(candidates) > 1:
        # TF-IDF-style column scoring to pick the metric the query is actually about.
        #
        # Problem with naive token overlap: "total premium paid" and
        # "total claim amount" both score identically on a query like
        # "top 5 customers by claim amount ... show total premium paid, total claim amount"
        # because both share 3 tokens with the query.
        #
        # Solution: down-weight tokens that appear in MULTIPLE candidate columns
        # (e.g. "total" in both total_premium_paid and total_claim_amount → CF=2 → weight 0.5)
        # and boost tokens unique to one column
        # (e.g. "claim" only in total_claim_amount → CF=1 → full weight).
        #
        # Score(col) = Σ  TF(token) / CF(token)
        #              for token in col_tokens
        # where TF = frequency of token in nl_query,
        #       CF = number of candidate columns containing that token.
        #
        # Example: "top 5 customers by claim amount ... total premium paid, total claim amount"
        #   total_claim_amount → total(2/2) + claim(2/1) + amount(2/1) = 1 + 2 + 2 = 5.0  ← wins
        #   total_premium_paid → total(2/2) + premium(1/1) + paid(1/1) = 1 + 1 + 1 = 3.0
        query_token_list = re.findall(r"\b\w+\b", nl_query.lower())
        _tf: dict[str, int] = {}
        for _t in query_token_list:
            _tf[_t] = _tf.get(_t, 0) + 1

        _cf: dict[str, int] = {}
        for _c in candidates:
            for _t in re.split(r"[_\s]+", _c.lower()):
                _cf[_t] = _cf.get(_t, 0) + 1

        def _col_score(col: str) -> float:
            return sum(_tf.get(t, 0) / _cf.get(t, 1) for t in re.split(r"[_\s]+", col.lower()))

        _scores = {c: _col_score(c) for c in candidates}
        _best = max(_scores.values())
        if _best > 0:
            # Stable tie-breaking: among equal-scoring candidates, prefer
            # the one with the lower original column index.
            value_col = max(candidates, key=lambda c: (_scores[c], -candidates.index(c)))
        else:
            value_col = candidates[0]
    else:
        value_col = candidates[0]

    group_col = string_cols[0] if string_cols else None

    # Top-N-per-group detection and aggregation.
    #
    # Problem: when the same group appears more than once (e.g. 5 rows per
    # policy_type in a top-5-per-group query), computing ratio/top/bottom on
    # the raw rows compares the single highest value (life rank-1, $4.36M)
    # against the single lowest value (auto rank-5, $219K) — inflating the
    # ratio and causing the LLM to say "averaging X times more" when it is
    # actually comparing two individual extremes.
    #
    # Fix: when any group appears more than once, aggregate value_col by mean
    # per group.  All subsequent calculations (values, sorted_rows, ratio, cv,
    # n_groups, ranked_groups) operate on these aggregated rows so they reflect
    # true group-level differences instead of individual record extremes.
    #
    # Example — top-5-per-group, policy_type result (20 rows → 4 agg rows):
    #   Before: ratio = life_rank1 / auto_rank5 = 4,362,964 / 219,910 = 19.8×  ← wrong
    #   After:  ratio = mean(life_5) / mean(auto_5) ≈ 3,859,564 / 237,750 = 16.2×  ← correct
    _analysis_rows: list[dict[str, Any]] = rows
    if group_col:
        _gcounts: dict[Any, int] = {}
        for _r in rows:
            _g = _r.get(group_col)
            if _g is not None:
                _gcounts[_g] = _gcounts.get(_g, 0) + 1

        if any(_c > 1 for _c in _gcounts.values()):
            # Aggregate: per-group mean of value_col, skipping null values.
            _gsums: dict[Any, float] = {}
            _gn: dict[Any, int] = {}
            for _r in rows:
                _g = _r.get(group_col)
                _v = _r.get(value_col)
                if _g is not None and _v is not None and not isinstance(_v, bool):
                    _gsums[_g] = _gsums.get(_g, 0.0) + float(_v)
                    _gn[_g] = _gn.get(_g, 0) + 1
            _analysis_rows = [
                {group_col: _g, value_col: _gsums[_g] / _gn[_g]} for _g in _gsums if _gn[_g] > 0
            ]

    # Extract values, skipping nulls and any non-numeric strings.
    values: list[float] = [
        _v
        for r in _analysis_rows
        if r.get(value_col) is not None and not isinstance(r[value_col], bool)
        for _v in (_safe_float(r[value_col]),)
        if _v is not None
    ]
    if len(values) < 2:
        return {}

    sorted_rows = sorted(
        _analysis_rows,
        key=lambda r: _safe_float(r[value_col] or 0) or 0.0,
        reverse=True,
    )
    top_row = sorted_rows[0]
    bottom_row = sorted_rows[-1]
    max_val: float = _safe_float(top_row[value_col]) or 0.0
    min_val: float = _safe_float(bottom_row[value_col]) or 0.0
    mean_val = sum(values) / len(values)

    # Coefficient of variation (σ / μ) — scale-invariant dispersion measure.
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    cv = round((variance**0.5) / mean_val, 3) if mean_val != 0 else 0.0

    # Ratio: only meaningful when bottom_value is strictly positive.
    ratio: float | None = None
    if min_val > 0:
        ratio = round(max_val / min_val, 1)

    # Variation classification — domain-agnostic thresholds.
    # Callers with domain knowledge (e.g. actuarial) should post-process
    # variation_type rather than relying on these generic cutoffs.
    if ratio is None:
        variation_type = "unknown"
    elif ratio >= 10:
        variation_type = "extreme"
    elif ratio >= 3:
        variation_type = "strong"
    elif ratio >= 1.5:
        variation_type = "moderate"
    else:
        variation_type = "uniform"

    top_group = top_row.get(group_col) if group_col else None
    bottom_group = bottom_row.get(group_col) if group_col else None

    metrics: dict[str, Any] = {
        "value_col": value_col,
        "group_col": group_col,
        "n_groups": len(_analysis_rows),  # unique groups, not total raw rows
        "top_group": top_group,
        "top_value": round(max_val, 2),
        "bottom_group": bottom_group,
        "bottom_value": round(min_val, 2),
        "mean": round(mean_val, 2),
        "ratio": ratio,
        "cv": cv,
        "variation_type": variation_type,
    }

    # Pre-formatted ratio sentence — gives the LLM a ready-to-use phrase so
    # it doesn't round differently or invent a different multiplier.
    if ratio is not None and ratio >= 1.5 and group_col:
        metrics["ratio_sentence"] = (
            f"{top_group} {value_col} is {ratio}× larger than " f"{bottom_group} {value_col}"
        )

    # Full ranked list — surfaces all groups in descending order.
    # For top-N-per-group results this list contains one entry per unique group
    # (the group mean), not one entry per raw row, so the LLM sees a clean
    # group-level ranking rather than 20 interleaved rows.
    if group_col:
        metrics["ranked_groups"] = [
            {
                "group": r.get(group_col),
                "value": round(_safe_float(r[value_col]) or 0.0, 2),
            }
            for r in sorted_rows
            if r.get(value_col) is not None and _safe_float(r[value_col]) is not None
        ]

    # Anomaly detection: scan every numeric column that is NOT the primary
    # value_col for statistically extreme values.
    #
    # Why this is needed:
    #   _compute_result_metrics selects one value_col based on NL query token
    #   overlap. Other numeric columns (e.g. loss_ratio when the query is
    #   "top customers by claim amount") are never examined, so the LLM has
    #   no signal to mention outliers that may be the most actionable finding.
    #
    # Threshold logic (domain-agnostic):
    #   - For ratio-like columns (name contains "ratio", "rate", "pct", "%"):
    #     flag max value when > 20 (loss_ratio > 20 is extreme in insurance).
    #   - For all other numeric columns: flag when max > 3× mean (CV-based).
    #
    # Output shape — appended to metrics as "anomalies": list[dict]:
    #   [{"col": "loss_ratio", "max_val": 102.49, "threshold": 20,
    #     "n_above_threshold": 3, "severity": "critical"}]
    #
    # Empty list when no anomalies detected — INSIGHT Rule 8 is a no-op then.
    _ratio_pattern = re.compile(r"(ratio|rate|pct|percent|%)", re.I)
    anomalies: list[dict[str, Any]] = []

    for col in numeric_cols:
        if col == value_col:
            continue  # already analysed as primary metric

        col_values: list[float] = [
            _v
            for r in _analysis_rows
            if r.get(col) is not None and not isinstance(r[col], bool)
            for _v in (_safe_float(r[col]),)
            if _v is not None
        ]
        if len(col_values) < 2:
            continue

        col_max = max(col_values)
        col_mean = sum(col_values) / len(col_values)

        if _ratio_pattern.search(col):
            threshold = 20.0
            is_anomalous = col_max > threshold
        else:
            # Generic: flag when max exceeds 3× mean (strong positive skew)
            threshold = round(col_mean * 3, 2)
            is_anomalous = col_mean > 0 and col_max > threshold

        if not is_anomalous:
            continue

        n_above = sum(1 for v in col_values if v > threshold)
        if col_max > 50:
            severity = "critical"
        elif col_max > 20:
            severity = "elevated"
        else:
            severity = "above_range"

        anomalies.append(
            {
                "col": col,
                "max_val": round(col_max, 2),
                "threshold": threshold,
                "n_above_threshold": n_above,
                "severity": severity,
            }
        )

    if anomalies:
        metrics["anomalies"] = anomalies

    return metrics
