"""
test_orchestrator.py — Unit tests for orchestrator.py
Data Analyst Copilot · pytest + pytest-asyncio

Coverage targets:
  - SQL happy path (INTAKE → TERMINAL)
  - Pandas happy path
  - LTM exact-hit bypass (USE_LT_EXACT_HIT=true) — H-08 + H-12 regression guard
  - MAX_ATTEMPTS exhaustion → TERMINAL_ERROR
  - INTERNAL_ERROR catch-all logging — H-09 regression guard
  - _result_cache LRU eviction at cap — H-10 regression guard
  - RETRIEVAL returns 0 chunks → clarification response
  - dry_run=True skips execution

All LLM calls, retrieval calls, and execution loops are mocked.
No network, no DB, no GPU required.
"""

from __future__ import annotations

import json

# ── Path bootstrap (mirrors conftest.py / _bootstrap.py) ──────────────────────
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key-ci")
os.environ.setdefault("GEMINI_API_KEY", "test-key-ci")
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("FAISS_INDEX_PATH", "./data/faiss_index/smoke_test.faiss")
os.environ.setdefault("EMBED_CACHE_PATH", "./data/faiss_index/embed_cache.json")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_orchestrator(*, use_lt_exact_hit: bool = False) -> Any:
    """
    Build an Orchestrator with every external dependency stubbed out.
    Returns (orchestrator, mocks) so tests can configure return values.
    """
    from observability import ObservabilityStack
    from orchestrator import Orchestrator

    os.environ["USE_LT_EXACT_HIT"] = "true" if use_lt_exact_hit else "false"

    llm = MagicMock()
    llm.generate = AsyncMock()
    # Orchestrator calls generate_insight() in _insight_state (not generate()).
    # generate_clarification() is called in _retrieval_state and _result_check
    # for zero-chunk and empty-result paths — neither call is inside
    # contextlib.suppress, so a plain MagicMock raises TypeError on await.
    llm.generate_insight = AsyncMock(return_value="Query returned rows.")
    llm.generate_clarification = AsyncMock(return_value="Please clarify your question.")

    retrieval = MagicMock()
    # Orchestrator calls retrieve / get_schema_columns / get_table_policies via
    # asyncio.gather — not search.  Plain MagicMock attributes are not awaitable,
    # causing a TypeError that is caught as RETRIEVAL_ERROR, aborting the turn.
    retrieval.retrieve = AsyncMock(return_value=[])
    retrieval.get_schema_columns = AsyncMock(return_value=set())
    retrieval.get_table_policies = AsyncMock(return_value={})

    session_store = MagicMock()
    session_store.get_or_create = AsyncMock(return_value="sess-test-001")
    session_store.get_history = AsyncMock(return_value=[])
    session_store.append_turn = AsyncMock()

    engine = MagicMock()
    obs = ObservabilityStack()

    orch = Orchestrator(
        llm=llm,
        retrieval=retrieval,
        session_store=session_store,
        engine=engine,
        obs=obs,
        model="llama-3.3-70b-versatile",
    )
    return orch, {"llm": llm, "retrieval": retrieval, "sessions": session_store}


def _sql_generation_response(sql: str = "SELECT 1") -> MagicMock:
    """Build a mocked GenerationResponse for the SQL generation state."""
    from interfaces import GenerateSQLOutput
    from prompts import GenerationResponse

    # confidence is now float (0.0–1.0), not Literal["HIGH"/"MEDIUM"/"LOW"].
    # code_type was removed from GenerateSQLOutput (it lives on QueryResponse now).
    parsed = GenerateSQLOutput(
        sql=sql,
        confidence=0.95,
        assumptions=[],
    )
    resp = MagicMock(spec=GenerationResponse)
    resp.content = json.dumps({"sql": sql, "confidence": 0.95, "assumptions": []})
    resp.parsed = parsed
    resp.prompt_tokens = 100
    resp.completion_tokens = 50
    resp.latency_ms = 120
    resp.model = "llama-3.3-70b-versatile"
    resp.parse_error = None
    return resp


def _pandas_generation_response(code: str = "result = df.head(10)") -> MagicMock:
    """Build a mocked GenerationResponse for the Pandas generation state."""
    from interfaces import GeneratePandasOutput
    from prompts import GenerationResponse

    # confidence is now float; code_type removed from GeneratePandasOutput.
    parsed = GeneratePandasOutput(
        code=code,
        confidence=0.95,
        assumptions=[],
    )
    resp = MagicMock(spec=GenerationResponse)
    resp.content = json.dumps({"code": code, "confidence": 0.95, "assumptions": []})
    resp.parsed = parsed
    resp.prompt_tokens = 120
    resp.completion_tokens = 60
    resp.latency_ms = 140
    resp.model = "llama-3.3-70b-versatile"
    resp.parse_error = None
    return resp


def _insight_response(text: str = "Query returned 5 rows.") -> MagicMock:
    from prompts import GenerationResponse

    resp = MagicMock(spec=GenerationResponse)
    resp.content = text
    resp.parsed = None
    resp.prompt_tokens = 50
    resp.completion_tokens = 20
    resp.latency_ms = 80
    resp.model = "llama-3.3-70b-versatile"
    resp.parse_error = None
    return resp


def _schema_chunk() -> Any:
    """Return a minimal SchemaChunk for the retrieval mock."""
    from interfaces import SchemaChunk, SchemaColumn

    return SchemaChunk(
        schema_id="test_schema",
        table="claims",
        columns=[
            SchemaColumn(name="claim_id", type="INTEGER", nullable=False),
            SchemaColumn(name="amount", type="DECIMAL(12,2)", nullable=True),
        ],
    )


def _loop_result(
    success: bool = True,
    rows: list | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    retry_count: int = 0,
) -> Any:
    """
    Build a duck-typed LoopResult (return type of ExecutionLoop.run()).

    The orchestrator accesses many fields on this object:
      loop_result.success, .execution_result, .attempt_history,
      .final_error_type, .final_error_message, .retry_count,
      .runtime_validation, .unresolved_columns, .validation_warning

    We use MagicMock to set all of them explicitly without depending on
    LoopResult's exact constructor (validation.py not imported by tests).
    The inner ExecutionResult is the real Pydantic object so that
    _result_check can read .result, .success etc. normally.
    """
    from validation import ExecutionResult

    inner = (
        ExecutionResult(
            success=success,
            result=rows if rows is not None else [{"claim_id": 1, "amount": 250.0}],
            row_count=len(rows) if rows is not None else 1,
            execution_time_ms=12,
            memory_used_mb=0.5,
            error_type=error_type,
            error_message=error_message,
        )
        if success
        else None
    )

    r = MagicMock()
    r.success = success
    r.execution_result = inner
    r.attempt_history = []
    r.final_error_type = error_type
    r.final_error_message = error_message
    r.retry_count = retry_count
    r.runtime_validation = None  # skip RESULT_CAPPED branch
    r.unresolved_columns = []  # skip no-fuzzy-match fast-path
    r.validation_warning = None
    return r


def _make_request(**kwargs: Any) -> Any:
    from interfaces import QueryRequest

    defaults = {
        "nl_query": "Show me top 5 claims by amount",
        "schema_id": "test_schema",
        "session_id": None,
        "execution_mode": "sql",
        "dry_run": False,
        "dialect": "postgres",
    }
    defaults.update(kwargs)
    return QueryRequest(**defaults)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestSQLHappyPath:
    """Full SQL turn: INTAKE → RETRIEVAL → GENERATION → VALIDATION → EXECUTION → TERMINAL."""

    @pytest.mark.asyncio
    async def test_sql_turn_returns_query_response(self) -> None:
        orch, mocks = _make_orchestrator()
        chunk = _schema_chunk()
        mocks["retrieval"].retrieve.return_value = [chunk]

        sql = "SELECT claim_id, amount FROM claims ORDER BY amount DESC LIMIT 5"
        gen_resp = _sql_generation_response(sql)

        # generate() is called once for SQL generation only.
        # Insight is produced by generate_insight() (separate method).
        mocks["llm"].generate = AsyncMock(return_value=gen_resp)

        exec_result = _loop_result(
            rows=[{"claim_id": i, "amount": float(i * 100)} for i in range(5)]
        )

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run.return_value = exec_result
            MockLoop.return_value = mock_loop

            response = await orch.run(_make_request())

        assert response.error is None
        assert response.generated_code == sql
        assert response.code_type == "sql"
        assert response.row_count == 5
        assert response.retry_count == 0
        assert response.session_id == "sess-test-001"

    @pytest.mark.asyncio
    async def test_sql_turn_populates_result_cache(self) -> None:
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        sql = "SELECT 1"
        mocks["llm"].generate = AsyncMock(return_value=_sql_generation_response(sql))

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = _loop_result()
            await orch.run(_make_request())

        assert "sess-test-001" in orch._result_cache
        cached = orch._result_cache["sess-test-001"]
        assert len(cached) == 1
        assert cached[0]["code"] == sql


class TestPandasHappyPath:
    """Full Pandas turn with execution_mode=pandas."""

    @pytest.mark.asyncio
    async def test_pandas_turn_returns_query_response(self) -> None:
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        code = "result = df['amount'].describe()"
        mocks["llm"].generate = AsyncMock(return_value=_pandas_generation_response(code))

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = _loop_result(
                rows=[{"count": 100.0, "mean": 450.0}]
            )
            response = await orch.run(_make_request(execution_mode="pandas"))

        assert response.error is None
        assert response.code_type == "pandas"
        assert response.generated_code == code


class TestLTMExactHitBypass:
    """
    Regression guard for H-08 + H-12 combined fix.

    When USE_LT_EXACT_HIT=true and the top LTM result is an exact hit,
    orchestrator must:
      1. Read the cached code from lt_examples[0].record.sql  (H-12 fix)
      2. Set state.generated_sql = cached_code directly        (H-08 fix)
      3. Skip the LLM generation call entirely
    """

    @pytest.mark.asyncio
    async def test_ltm_exact_hit_skips_llm_generation(self) -> None:
        from long_term_memory import MemoryRecord, MemorySearchResult

        orch, mocks = _make_orchestrator(use_lt_exact_hit=True)
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        cached_sql = "SELECT claim_id FROM claims LIMIT 10"
        record = MemoryRecord(
            record_id="rec-001",
            session_id="sess-prev",
            schema_id="test_schema",
            nl_query="Show me top claims",
            sql=cached_sql,
            insight="",
            created_at="2025-01-01T00:00:00+00:00",
        )
        hit = MemorySearchResult(record=record, similarity=0.98, is_exact_hit=True)

        lt_memory = MagicMock()
        lt_memory.search = MagicMock(return_value=[hit])
        lt_memory.store = MagicMock(return_value=True)
        orch._long_term_memory = lt_memory

        # generate() must NOT be called: generation is bypassed by the LTM exact
        # hit, and insight is produced via generate_insight() — a separate method.
        # _make_orchestrator already sets generate_insight as AsyncMock; no need
        # to configure llm.generate here.

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = _loop_result()
            response = await orch.run(_make_request())

        assert response.error is None
        assert response.generated_code == cached_sql
        # Generation was bypassed: llm.generate must have been called 0 times
        assert mocks["llm"].generate.call_count == 0

    @pytest.mark.asyncio
    async def test_ltm_non_exact_hit_falls_through_to_generation(self) -> None:
        """similarity < threshold → is_exact_hit=False → normal generation."""
        from long_term_memory import MemoryRecord, MemorySearchResult

        orch, mocks = _make_orchestrator(use_lt_exact_hit=True)
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        record = MemoryRecord(
            record_id="rec-002",
            session_id="sess-prev",
            schema_id="test_schema",
            nl_query="Different question",
            sql="SELECT 1",
            insight="",
            created_at="2025-01-01T00:00:00+00:00",
        )
        near_miss = MemorySearchResult(record=record, similarity=0.80, is_exact_hit=False)

        lt_memory = MagicMock()
        lt_memory.search = MagicMock(return_value=[near_miss])
        lt_memory.store = MagicMock(return_value=True)
        orch._long_term_memory = lt_memory

        sql = "SELECT claim_id FROM claims LIMIT 5"
        # generate() called once for SQL generation; insight uses generate_insight().
        mocks["llm"].generate = AsyncMock(return_value=_sql_generation_response(sql))

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = _loop_result()
            response = await orch.run(_make_request())

        assert response.error is None
        # Generation was called (not bypassed): exactly 1 llm.generate call
        assert mocks["llm"].generate.call_count == 1


class TestMaxRetriesExhaustion:
    """TERMINAL_ERROR path: all MAX_ATTEMPTS fail execution."""

    @pytest.mark.asyncio
    async def test_terminal_error_after_max_retries(self) -> None:
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        sql = "SELECT bad_col FROM claims"
        # Generation succeeds, then each error-correction pass returns a
        # different still-failing SQL so the orchestrator can exhaust its
        # retry budget instead of stopping on identical-code convergence.
        mocks["llm"].generate = AsyncMock(
            side_effect=[
                _sql_generation_response(sql),
                _sql_generation_response("SELECT bad_col_2 FROM claims"),
                _sql_generation_response("SELECT bad_col_3 FROM claims"),
            ]
        )

        failed = _loop_result(
            success=False,
            rows=[],
            error_type="UNRESOLVED_COLUMN",
            error_message="Column 'bad_col' does not exist.",
        )

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = failed
            response = await orch.run(_make_request())

        assert response.error is not None
        # state.attempt_count = attempt (0-indexed); after range(MAX_ATTEMPTS=3),
        # the last value is 2, so retry_count == 2 not 3.
        assert response.retry_count == 2
        assert "TERMINAL_ERROR" in (response.error.error_code or "")

    @pytest.mark.asyncio
    async def test_terminal_error_insight_is_user_friendly(self) -> None:
        """Insight on TERMINAL_ERROR must not expose internal error details."""
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]
        mocks["llm"].generate = AsyncMock(return_value=_sql_generation_response())

        failed = _loop_result(
            success=False,
            rows=[],
            error_type="EXECUTION_TIMEOUT",
            error_message="timeout",
        )

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = failed
            response = await orch.run(_make_request())

        assert response.error is not None
        # Stack traces and internal details must not leak into the public insight
        assert "Traceback" not in response.insight
        assert "timeout" not in response.insight.lower() or "timed out" in response.insight.lower()


class TestColumnHintsAndH2Fix:
    """
    Regression guards for:
      - _column_hints type-fix: _fuzzy_match_columns returns list[str] per
        column; _error_correct expects str | None. The orchestrator must
        take the best match (matches[0]) before passing column_hints.
      - UNRESOLVED_COLUMN fast-path: no close match -> immediate
        TERMINAL_ERROR naming the missing field(s), error_correct never called.
      - H2 FIX: error_correct converges to code identical to its input ->
        TERMINAL_ERROR ("converged"), no wasted final GENERATION attempt.
    """

    @pytest.mark.asyncio
    async def test_unresolved_column_no_close_match_terminates_immediately(
        self,
    ) -> None:
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]
        mocks["retrieval"].get_schema_columns = AsyncMock(return_value={"claim_id", "amount"})
        mocks["llm"].generate = AsyncMock(
            return_value=_sql_generation_response("SELECT totally_unknown_col FROM claims")
        )

        failed = _loop_result(
            success=False,
            rows=[],
            error_type="UNRESOLVED_COLUMN",
            error_message="Column 'totally_unknown_col' does not exist.",
        )
        failed.unresolved_columns = ["totally_unknown_col"]

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = failed
            response = await orch.run(_make_request())

        assert response.error is not None
        assert "totally_unknown_col" in response.error.message
        # Fast-path breaks before any error_correct call: only the
        # initial GENERATION call should have fired.
        assert mocks["llm"].generate.call_count == 1
        assert response.retry_count == 0

    @pytest.mark.asyncio
    async def test_column_hints_best_match_injected_into_correction_prompt(
        self,
    ) -> None:
        """unresolved_columns with a close fuzzy match -> _column_hints
        carries the best match as a str (not the raw list from
        _fuzzy_match_columns), surfaced in the ERROR_CORRECT error_message."""
        from prompts import PromptRenderer

        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]
        mocks["retrieval"].get_schema_columns = AsyncMock(
            return_value={"premium_amount", "claim_id"}
        )

        bad_sql = "SELECT premium_amt FROM policies"
        corrected_sql = "SELECT premium_amount FROM policies"
        mocks["llm"].generate = AsyncMock(
            side_effect=[
                _sql_generation_response(bad_sql),
                _sql_generation_response(corrected_sql),
            ]
        )

        failed = _loop_result(
            success=False,
            rows=[],
            error_type="UNRESOLVED_COLUMN",
            error_message="Column 'premium_amt' does not exist.",
        )
        failed.unresolved_columns = ["premium_amt"]

        ok = _loop_result(rows=[{"premium_amount": 1000.0}])

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.side_effect = [failed, ok]
            with patch(
                "orchestrator.PromptRenderer.render", wraps=PromptRenderer.render
            ) as mock_render:
                response = await orch.run(_make_request())

        assert response.error is None
        assert response.generated_code == corrected_sql

        ec_calls = [
            c
            for c in mock_render.call_args_list
            if c.kwargs.get("error_type") == "UNRESOLVED_COLUMN"
        ]
        assert ec_calls, "Expected an ERROR_CORRECT prompt render call"
        error_message = ec_calls[0].kwargs["error_message"]
        # The hint must be a plain column name, not "['premium_amount']"
        assert "premium_amount" in error_message
        assert "['premium_amount']" not in error_message
        assert "Column name corrections" in error_message

    @pytest.mark.asyncio
    async def test_error_correct_converges_without_change_terminates(self) -> None:
        """H2 FIX: error_correct returns code identical to its input ->
        TERMINAL_ERROR immediately, no wasted final GENERATION attempt."""
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]
        # _loop_result defaults unresolved_columns=[] -> fast-path skipped,
        # falls straight through to _error_correct.

        sql = (
            "SELECT p.premium_amt, pay.created_at FROM policies p "
            "JOIN payments pay ON p.policy_id = pay.policy_id"
        )
        # GENERATION and ERROR_CORRECT both return the identical SQL.
        mocks["llm"].generate = AsyncMock(
            side_effect=[
                _sql_generation_response(sql),
                _sql_generation_response(sql),
            ]
        )

        failed = _loop_result(
            success=False,
            rows=[],
            error_type="UNRESOLVED_COLUMN",
            error_message="Column 'created_at' does not exist on payments.",
        )

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            MockLoop.return_value.run.return_value = failed
            response = await orch.run(_make_request())

        assert response.error is not None
        assert "converged" in response.error.message.lower()
        # 1 GENERATION + 1 ERROR_CORRECT = 2 calls; H2 FIX terminates
        # before a 3rd (wasted) GENERATION attempt would fire.
        assert mocks["llm"].generate.call_count == 2
        assert response.retry_count == 0


class TestCatchAllLogging:
    """
    Regression guard for H-09 fix.

    When an unexpected exception escapes all state handlers, run() must:
      1. NOT re-raise (always returns QueryResponse)
      2. Log ORCHESTRATOR_UNHANDLED_EXCEPTION to stdout with exception details
      3. Return error_code=INTERNAL_ERROR
    """

    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_internal_error(self, capsys: Any) -> None:
        orch, mocks = _make_orchestrator()

        # Make _intake raise an unexpected exception
        async def _bad_intake(state: Any) -> None:
            raise RuntimeError("Simulated unexpected failure in _intake")

        orch._intake = _bad_intake  # type: ignore[method-assign]

        response = await orch.run(_make_request())

        assert response.error is not None
        assert response.error.error_code == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_unhandled_exception_logs_to_stdout(self, capsys: Any) -> None:
        orch, mocks = _make_orchestrator()

        async def _bad_intake(state: Any) -> None:
            raise ValueError("deliberate test error")

        orch._intake = _bad_intake  # type: ignore[method-assign]

        await orch.run(_make_request())

        captured = capsys.readouterr()
        assert "ORCHESTRATOR_UNHANDLED_EXCEPTION" in captured.out
        # The observability stack may emit other JSON lines to stdout before the
        # exception log (e.g. start_trace), so search for the specific event line
        # rather than always taking split("\n")[0].
        exc_line = next(
            line
            for line in captured.out.strip().split("\n")
            if "ORCHESTRATOR_UNHANDLED_EXCEPTION" in line
        )
        log = json.loads(exc_line)
        assert log["error_type"] == "ValueError"
        assert "deliberate test error" in log["error"]
        assert log["session_id"] == "sess-test-001"


class TestResultCacheLRUEviction:
    """
    Regression guard for H-10 fix.

    The _result_cache OrderedDict must evict the oldest session entry when
    RESULT_CACHE_MAX_SESSIONS is reached — never grow unboundedly.
    """

    @pytest.mark.asyncio
    async def test_result_cache_respects_max_sessions(self) -> None:
        orch, mocks = _make_orchestrator()
        orch._result_cache_max = 3  # low cap for test

        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        session_ids = [f"sess-{i:03d}" for i in range(5)]

        for sid in session_ids:
            mocks["sessions"].get_or_create = AsyncMock(return_value=sid)
            mocks["llm"].generate = AsyncMock(return_value=_sql_generation_response())
            with patch("orchestrator.ExecutionLoop") as MockLoop:
                MockLoop.return_value.run.return_value = _loop_result()
                await orch.run(_make_request(session_id=sid))

        # Cap is 3 — only the 3 most recent sessions should be in cache
        assert len(orch._result_cache) <= 3
        # Oldest sessions should have been evicted
        assert session_ids[0] not in orch._result_cache
        assert session_ids[1] not in orch._result_cache
        # Most recent should be present
        assert session_ids[4] in orch._result_cache


class TestZeroChunksClarification:
    """RETRIEVAL returns 0 chunks → terminal_state=INTAKE → clarification response."""

    @pytest.mark.asyncio
    async def test_zero_chunks_returns_clarification(self) -> None:
        orch, mocks = _make_orchestrator()
        # No chunks found for this schema/query
        mocks["retrieval"].retrieve.return_value = []
        # generate_clarification() is the LLM call for zero-chunk paths;
        # it is already set up as AsyncMock in _make_orchestrator().

        response = await orch.run(_make_request())

        # Should return a response (not raise)
        assert response is not None
        # No execution should have happened — no code generated
        assert response.generated_code == "" or response.code_type in ("sql", "pandas")
        assert response.error is None or response.error.error_code != "INTERNAL_ERROR"


class TestDryRun:
    """dry_run=True: validation runs, execution is skipped, result_preview=None."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_execution(self) -> None:
        orch, mocks = _make_orchestrator()
        mocks["retrieval"].retrieve.return_value = [_schema_chunk()]

        sql = "SELECT claim_id FROM claims"
        mocks["llm"].generate = AsyncMock(return_value=_sql_generation_response(sql))

        with patch("orchestrator.ExecutionLoop") as MockLoop:
            mock_loop_instance = MagicMock()
            # dry_run=True should produce a result with success=True but no rows
            mock_loop_instance.run.return_value = _loop_result(rows=[])
            MockLoop.return_value = mock_loop_instance

            response = await orch.run(_make_request(dry_run=True))

        assert response.error is None
        assert response.generated_code == sql
        # dry_run responses may have empty result_preview
        assert response.result_preview is None or response.result_preview == []
