"""
Section 3 — Interface Contracts
Data Analyst Copilot · Python 3.11+ · FastAPI 0.110+ · Pydantic v2

Covers:
  A. Function calling schemas (NL→SQL, NL→Pandas)
  B. API design (POST /query, POST /execute, GET /history/{session_id})
  C. Supporting models (ErrorDetail, TurnRecord, HistoryResponse)

No pseudocode. All imports included. All fields annotated with
validation rules and docstrings matching the spec.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

# fastapi imports intentionally removed — interfaces.py is a pure data-model
# module. Route definitions live in app.py; this file must not instantiate a
# FastAPI application (BUG-3 fix).
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Supporting primitives
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error object. Returned inside response bodies — never as HTTP 500."""

    error_code: str = Field(
        ...,
        description=(
            "Machine-readable code. One of: UNRESOLVED_REFERENCE, SYNTAX_ERROR, "
            "FORBIDDEN_IMPORT, UNRESOLVED_COLUMN, MUTATION_STATEMENT, "
            "POLICY_VIOLATION, EXECUTION_TIMEOUT, TERMINAL_ERROR."
        ),
    )
    message: str = Field(..., description="Human-readable explanation for primary persona.")
    attempted_code: str | None = Field(
        None,
        description="The last code string that was attempted before failure.",
    )
    attempt_history: list[str] | None = Field(
        None,
        description=(
            "Ordered list (oldest first) of all code strings attempted in the "
            "retry loop, for debuggability. Populated only when retry_count >= 1."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "UNRESOLVED_COLUMN",
                "message": (
                    "I couldn't find a column matching 'churn_rate'. "
                    "Did you mean one of: churn_flag, is_churned, churn_date?"
                ),
                "attempted_code": "SELECT churn_rate FROM customers GROUP BY segment",
                "attempt_history": None,
            }
        }
    }


# ---------------------------------------------------------------------------
# A. Function Calling Schemas
# ---------------------------------------------------------------------------


class SchemaColumn(BaseModel):
    """Single column descriptor inside a schema chunk."""

    name: str
    type: str = Field(..., description="SQL data type string, e.g. 'DECIMAL(12,2)'.")
    nullable: bool
    null_rate: float | None = Field(None, ge=0.0, le=1.0)
    cardinality: Literal["low", "medium", "high"] | None = None
    # P3-G FIX: Pydantic v2 does not enforce max_length on list[Any] when
    # passed directly to Field() — it applies that constraint only to str/bytes.
    # The correct Pydantic v2 pattern for capping list length is
    # Annotated[list[...], Field(max_length=N)].  The old form silently
    # accepted lists longer than 5 items.
    sample_values: Annotated[
        list[Any] | None,
        Field(
            default=None,
            description="Up to 5 sample values. PII-masked (hashed) for flagged columns.",
            max_length=5,
        ),
    ] = None
    description: str | None = Field(
        None,
        description="Optional analyst note about this column's semantics, nulls, or quirks.",
    )


class FKRelationship(BaseModel):
    column: str
    references: str = Field(..., description="Format: 'table.column'.")


class SchemaChunk(BaseModel):
    """
    A single retrieved schema chunk passed to the LLM as schema_context.
    Corresponds to one table in the hybrid chunking strategy (Section 2B).
    """

    table: str
    schema_id: str
    business_description: str | None = None
    columns: list[SchemaColumn] = Field(..., min_length=1)
    fk_relationships: list[FKRelationship] = Field(default_factory=list)
    row_count_estimate: int | None = Field(None, ge=0)
    pii_flagged: bool = False
    chunk_token_count: int | None = None


class GroundingCheck(BaseModel):
    all_columns_verified: bool
    unresolved_references: list[str] = Field(default_factory=list)


class GenerateSQLInput(BaseModel):
    """
    Function calling schema: NL → SQL.

    Contract: the LLM MUST NOT invent column names absent from schema_context.
    The `grounding_check` field in the response is the machine-verifiable
    assertion that this contract was upheld.
    """

    nl_query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's natural language question, unmodified.",
    )
    schema_context: list[SchemaChunk] = Field(
        ...,
        min_length=1,
        description=(
            "Retrieved schema chunks (top-K). Each chunk contains table name, "
            "column names with types, sample values, and FK relationships."
        ),
    )
    session_history: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=10,
        description="Last N turns of conversation for coreference resolution.",
    )
    dialect: Literal["postgres", "mysql", "sqlite", "bigquery", "snowflake"] = Field(
        ...,
        description="SQL dialect to generate. Required.",
    )


class GenerateSQLOutput(BaseModel):
    """
    Function calling response: NL → SQL.

    Failure mode: if the LLM returns error_code='UNRESOLVED_REFERENCE',
    the orchestrator transitions to ERROR_CORRECT without executing.
    """

    sql: str | None = Field(
        None,
        description="Generated SQL query. None if error_code is set.",
    )
    # Default 0.0 so that LLM shortcircuit responses (UNRESOLVED_REFERENCE,
    # MUTATION_REQUESTED) can omit this field without failing Pydantic validation.
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Interpretations made during generation. If confidence < 0.7, "
            "both interpretations are listed; the chosen one is marked '(selected)'."
        ),
    )
    # Optional for the same reason as confidence — shortcircuit responses do not
    # include a grounding check. Normal successful responses always populate it.
    grounding_check: GroundingCheck | None = None
    # MUTATION_REQUESTED added: NL_TO_SQL_SYSTEM_PROMPT Rule 4 instructs the LLM
    # to return this code when a write operation is detected. Without it, the
    # Pydantic Literal validation fails and the orchestrator retries with
    # error_code="LLM_PARSE_ERROR" instead of surfacing the correct block.
    # UNRECOVERABLE added: ERROR_CORRECT_SYSTEM_PROMPT Rule 3 instructs the LLM
    # to return this code when correction would require violating a grounding rule.
    error_code: Literal["UNRESOLVED_REFERENCE", "MUTATION_REQUESTED", "UNRECOVERABLE"] | None = None
    # Populated only when error_code="UNRECOVERABLE". Contains the LLM's
    # one-sentence explanation of why the error cannot be corrected.
    reason: str | None = None
    unresolved: list[str] = Field(
        default_factory=list,
        description="Terms from nl_query that could not be resolved to schema columns.",
    )

    @model_validator(mode="after")
    def sql_or_error_required(self) -> GenerateSQLOutput:
        if self.sql is None and self.error_code is None:
            raise ValueError("Either 'sql' or 'error_code' must be set.")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "sql": (
                    # NOTE: illustrative only — not tied to any registered schema.
                    "SELECT coverage_type, AVG(claim_amount) AS avg_claim "
                    "FROM claims "
                    "WHERE claim_date >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months') "
                    "AND claim_date < DATE_TRUNC('quarter', CURRENT_DATE) "
                    "GROUP BY coverage_type ORDER BY avg_claim DESC"
                ),
                "confidence": 0.91,
                "assumptions": [
                    "'Last quarter' resolved to the most recently completed calendar quarter."
                ],
                "grounding_check": {
                    "all_columns_verified": True,
                    "unresolved_references": [],
                },
                "error_code": None,
                "unresolved": [],
            }
        }
    }


class GeneratePandasInput(BaseModel):
    """
    Function calling schema: NL → Pandas.

    Same contract as GenerateSQLInput. Replaces `dialect` with
    `dataframe_refs` (names of DataFrames available in the execution scope).
    The LLM MUST assign its final result to a variable named `result`.
    """

    nl_query: str = Field(..., min_length=3, max_length=2000)
    schema_context: list[SchemaChunk] = Field(..., min_length=1)
    session_history: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    dataframe_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Names of DataFrames injected into the execution scope. "
            "Empty list is valid — orchestrator guards against generating Pandas "
            "code when no DataFrames have been uploaded for the session."
        ),
        # P2-07 FIX: removed min_length=1.
        # When the orchestrator selects Pandas mode via keyword detection but no
        # files have been uploaded, df_store.get(session_id) returns {} and
        # dataframe_refs=[].  Constructing GeneratePandasInput with min_length=1
        # caused a Pydantic ValidationError that surfaced as LLM_PARSE_ERROR
        # rather than a clear "please upload a file first" message.
        # The orchestrator now emits an explicit INTAKE clarification when
        # dataframe_refs is empty (see orchestrator._generation_state).
    )


class GeneratePandasOutput(BaseModel):
    """
    Function calling response: NL → Pandas.

    `code` must end with `result = <expression>`. Validated by the AST
    visitor in validate_python() before execution.
    """

    code: str | None = Field(
        None,
        description=(
            "Python code string. Last line must be `result = <expression>`. "
            "Uses \\n for newlines. None if error_code is set."
        ),
    )
    # Default 0.0 so LLM shortcircuit responses can omit this field.
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    # Optional — shortcircuit responses do not include a grounding check.
    grounding_check: GroundingCheck | None = None
    # UNRECOVERABLE added: ERROR_CORRECT_SYSTEM_PROMPT Rule 3 instructs the LLM
    # to return this code when correction would require violating a grounding rule.
    # MUTATION_REQUESTED added: mirrors GenerateSQLOutput — NL_TO_PANDAS_SYSTEM_PROMPT
    # Rule 4 instructs the LLM to return this when a write/delete operation is
    # detected. Without it, Pydantic validation fails and the mutation block
    # surfaces as LLM_PARSE_ERROR instead of MUTATION_STATEMENT.
    error_code: Literal["UNRESOLVED_REFERENCE", "MUTATION_REQUESTED", "UNRECOVERABLE"] | None = None
    # Populated only when error_code="UNRECOVERABLE".
    reason: str | None = None
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def code_or_error_required(self) -> GeneratePandasOutput:
        if self.code is None and self.error_code is None:
            raise ValueError("Either 'code' or 'error_code' must be set.")
        return self


# ---------------------------------------------------------------------------
# B. API Design — Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """
    POST /query — primary entry point.

    Contract with adjacent components:
      - Orchestration layer reads `schema_id` to scope retrieval and policy checks.
      - `session_id=None` instructs the session manager to create and return a
        new session_id in the response.
      - `execution_mode="auto"` applies the tool-selection policy from Section 2H:
        SQL default, Pandas only on explicit trigger conditions.
      - `dry_run=True` runs validation but not execution; useful for client-side
        syntax preview.

    Failure mode: schema_id that does not exist in the schema store → HTTP 422
    with a clear validation error before the orchestration layer is invoked.
    """

    nl_query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's natural language question.",
    )
    schema_id: str = Field(
        ...,
        min_length=1,
        description="ID of the ingested schema to query against.",
    )
    session_id: str | None = Field(
        None,
        description="Omit to start a new session. Include to continue an existing one.",
    )
    execution_mode: Literal["sql", "pandas", "auto"] = Field(
        "auto",
        description=(
            "Force a specific executor or let the tool-selection policy decide. "
            "'auto' is recommended for primary persona."
        ),
    )
    dry_run: bool = Field(
        False,
        description="Validate and generate code but do not execute. Returns generated_code only.",
    )
    dialect: Literal["postgres", "mysql", "sqlite", "bigquery", "snowflake"] = Field(
        "postgres",
        description=(
            "SQL dialect to generate and validate against. "
            "Must match the backend database engine. Defaults to 'postgres'."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "nl_query": "What was the average claim amount by policy type last quarter?",
                "schema_id": "ins_prod_v3",
                "session_id": None,
                "execution_mode": "auto",
                "dry_run": False,
                "dialect": "postgres",
            }
        }
    }


class QueryResponse(BaseModel):
    """
    Response for POST /query and POST /execute.

    Contract:
      - HTTP 200 when the query succeeds (error field is None).
      - HTTP 422 (UNRESOLVED_REFERENCE): column or table in the query could not
        be resolved against the registered schema.
      - HTTP 403 (POLICY_VIOLATION / MUTATION_STATEMENT): query targets a PII
        table or attempts a mutating statement (INSERT/UPDATE/DELETE/DROP).
      - HTTP 502 (TERMINAL_ERROR): LLM or execution failure after all retries.
      - HTTP 503 (server initialising or LLM API unreachable).
      - In all error cases the `error` field is populated with error_code and
        message; `result_preview` and `row_count` are None.
      - M-19 FIX: the previous docstring stated "HTTP status is always 200 for
        caught errors" which reflected an earlier design that was superseded.
        app.py _http_status_for() maps error codes to the status codes above.
      - `result_preview` is capped at 100 rows on the response; full result up to
        10,000 rows is available via a separate paginated endpoint (Phase 2+).
      - `insight` is always populated for primary persona, even on error
        (e.g., "I wasn't able to answer that — here is what went wrong.").

    Failure mode: LLM generation returns error_code='UNRESOLVED_REFERENCE' →
    `error` is populated, `generated_code` contains the last attempted code
    (or empty string if generation failed before producing code), `insight`
    contains a human-readable explanation.
    """

    session_id: str = Field(..., description="Session ID. Generated if not provided in request.")
    generated_code: str = Field(..., description="SQL query or Python code string generated.")
    code_type: Literal["sql", "pandas"]
    result_preview: list[dict[str, Any]] | None = Field(
        None,
        description="First 100 rows of the result. None on error or dry_run.",
    )
    row_count: int | None = Field(None, description="Total rows returned (capped at 10,000).")
    insight: str = Field(
        ...,
        description=(
            "Human-readable summary for the primary persona. Always present. "
            "On error, explains what went wrong without exposing stack traces."
        ),
    )
    execution_time_ms: int = Field(..., ge=0)
    retry_count: int = Field(..., ge=0, le=3)
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues, e.g. 'Result capped at 10,000 rows'.",
    )
    error: ErrorDetail | None = Field(
        None,
        description="Populated on any unrecoverable error. None on success.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "sess_a1b2c3",
                "generated_code": (
                    # NOTE: column names here are illustrative only.
                    # They are not asserted to exist in any registered schema.
                    "SELECT coverage_type, AVG(claim_amount) AS avg_claim "
                    "FROM claims WHERE claim_date >= DATE_TRUNC('quarter', "
                    "CURRENT_DATE - INTERVAL '3 months') "
                    "AND claim_date < DATE_TRUNC('quarter', CURRENT_DATE) "
                    "GROUP BY coverage_type ORDER BY avg_claim DESC LIMIT 10000"
                ),
                "code_type": "sql",
                "result_preview": [
                    {"coverage_type": "auto", "avg_claim": 4320.15},
                    {"coverage_type": "home", "avg_claim": 8910.44},
                ],
                "row_count": 3,
                "insight": "Home coverage had the highest average claim last quarter at $8,910.",
                "execution_time_ms": 412,
                "retry_count": 0,
                "warnings": [],
                "error": None,
            }
        }
    }


class ExecuteRequest(BaseModel):
    """
    POST /execute — used when the client has a pre-generated query and wants
    only execution + insight (no NL → code generation step).

    Contract: `code_type` must match the actual code provided. The executor
    applies the same validation, policy, and sandbox rules as /query.

    Failure mode: if `code_type='sql'` but `code` contains Python syntax,
    sqlglot parse fails → HTTP 200 with error='SYNTAX_ERROR' in response body.
    """

    code: str = Field(
        ...,
        min_length=1,
        description="SQL query or Python code string to execute.",
    )
    schema_id: str = Field(..., min_length=1)
    session_id: str | None = None
    code_type: Literal["sql", "pandas"] = "sql"
    dry_run: bool = False

    # Same structure as QueryResponse — see above


class TurnRecord(BaseModel):
    """
    Single turn in session history.

    Stored in the session store and returned by GET /history/{session_id}.
    The `generated_code` field is the final code that was executed (or the
    last attempted code on TERMINAL_ERROR).
    """

    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    nl_query: str
    generated_code: str
    code_type: Literal["sql", "pandas"]
    row_count: int | None
    insight: str
    retry_count: int = Field(..., ge=0, le=3)
    error: ErrorDetail | None


class HistoryResponse(BaseModel):
    """
    GET /history/{session_id} response.

    Turns are ordered oldest-first. The session is created on the first
    /query or /execute call that generates a new session_id.

    Failure mode: session_id not found in session store → HTTP 404.
    (This is the one endpoint that returns a non-200 error status, because
    a missing session is not a query execution failure — it is a bad reference.)
    """

    session_id: str
    turns: list[TurnRecord] = Field(
        ...,
        description="Ordered oldest-first. Empty list if session exists but has no turns.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "sess_a1b2c3",
                "turns": [
                    {
                        "turn_id": "7f3a1c2d-...",
                        "timestamp": "2024-11-01T14:23:01.123Z",
                        "nl_query": "What was the average claim amount by policy type last quarter?",
                        "generated_code": "SELECT policy_type, AVG(claim_amount) ...",
                        "code_type": "sql",
                        "row_count": 3,
                        "insight": "Home policies had the highest average claim at $8,910.",
                        "retry_count": 0,
                        "error": None,
                    }
                ],
            }
        }
    }
