"""
Section 5 — Code Execution and Validation Loop
Data Analyst Copilot · Python 3.11+ · Pydantic v2

No pseudocode. All imports included.

Covers:
  - validate_python()  — AST parse + forbidden-import/builtin visitor + column grounding
  - validate_sql()     — sqlglot parse + mutation guard + column existence check
  - validate_result()  — runtime shape/plausibility check (RESULT_CHECK state)
  - execute_python()   — sandboxed exec() with timeout + memory guard
  - execute_sql()      — read-only SQLAlchemy execution with pre-execution policy layer
  - PreExecutionPolicy — PII + LIMIT-less SELECT guard (independent of LLM output)
  - ExecutionLoop      — closed retry loop wiring all stages together
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import json

# ---------------------------------------------------------------------------
# BUG-6 FIX: module-level singleton executor for SQL and Python sandboxes.
# Creating a new ThreadPoolExecutor on every execute_sql() / execute_python()
# call spawns a new OS thread each time, exhausting ulimit -u under sustained
# load and adding ~1ms overhead per call. A shared executor with a fixed pool
# size bounds thread count to concurrency budget and reuses threads across calls.
# max_workers=min(32, cpu_count+4) mirrors Python's own ThreadPoolExecutor default.
# ---------------------------------------------------------------------------
import os as _os
import re
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import pandas as pd
import sqlglot
from pydantic import BaseModel
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlglot import expressions as exp

_SANDBOX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (_os.cpu_count() or 1) + 4),
    thread_name_prefix="sandbox",
)

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    valid: bool
    error_type: (
        Literal[
            "SYNTAX_ERROR",
            "FORBIDDEN_IMPORT",
            "FORBIDDEN_BUILTIN",
            "UNRESOLVED_COLUMN",
            "MUTATION_STATEMENT",
            "NON_SARGABLE_FILTER",
            "POLICY_VIOLATION",
            "RESULT_ASSIGN_MISSING",
        ]
        | None
    ) = None
    error_message: str | None = None
    error_line: int | None = None
    # Structured list of unresolved column names — used by the orchestrator
    # for fuzzy-match fast-path decisions. Never sent to clients directly.
    unresolved_columns: list[str] = []
    # Non-blocking advisory notice (e.g. NON_SARGABLE_FILTER).
    # The query executes; the warning is surfaced in QueryResponse.warnings.
    warning: str | None = None


class RuntimeValidationResult(BaseModel):
    valid: bool
    issue: (
        Literal[
            "EMPTY_RESULT",
            "SHAPE_MISMATCH",
            "IMPLAUSIBLE_VALUE",
            "RESULT_CAPPED",
            "METRIC_OUT_OF_RANGE",
            "GROUPBY_CARDINALITY_MISMATCH",
        ]
        | None
    ) = None
    message: str | None = None


class ExecutionResult(BaseModel):
    success: bool
    result: list[dict[str, Any]] | None = None  # SQL path
    dataframe: Any | None = None  # Pandas path (pd.DataFrame | scalar)
    row_count: int = 0
    columns: list[str] = []
    execution_time_ms: int = 0
    memory_used_mb: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    # Populated when error_type == 'UNRESOLVED_COLUMN' — carries bare column
    # names for orchestrator fast-path decisions without string parsing.
    unresolved_columns: list[str] = []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "importlib",
        "builtins",
        "ctypes",
        "pickle",
        "shelve",
        "shutil",
        "pathlib",
        "tempfile",
        "glob",
        "fnmatch",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "multiprocessing",
        "threading",
        "concurrent",
        "signal",
        "mmap",
        "resource",
        "pty",
        "termios",
        "readline",
        "rlcompleter",
        "code",
        "codeop",
        "compileall",
        "dis",
        "py_compile",
        "tokenize",
    }
)

_FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "breakpoint",
        "input",
        "print",  # excluded from sandbox: LLM output must be captured via result variable
    }
)

_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "pandas",
        "numpy",
        "datetime",
        "math",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "string",
        "decimal",
        "fractions",
        "statistics",
    }
)

# P2-04 FIX: pandas IO methods that bypass the open() builtin guard.
# pd.read_csv('/etc/passwd'), pd.read_parquet(...), df.to_csv('/tmp/exfil')
# all call C-level file descriptors internally without going through Python's
# open() builtin — which _FORBIDDEN_BUILTINS blocks.  An LLM prompt-injection
# attack could use these to read arbitrary files accessible to the server
# process or exfiltrate query results to the host filesystem.
#
# Both read_ and to_ / write_ variants are blocked:
#   read_*  — file/URL reads that bypass the sandbox file guard
#   to_*    — file writes that could exfiltrate data or corrupt the host FS
#   from_*  — alternate constructors that accept file paths (from_csv, etc.)
#
# Enforced at the AST level in _ForbiddenNodeVisitor.visit_Call so detection
# happens before any code reaches the exec() call.  Companion runtime guard
# (monkey-patching the pd namespace before exec) is in execute_python().
_BLOCKED_PANDAS_IO: frozenset[str] = frozenset(
    {
        # DataFrame / Series read constructors
        "read_csv",
        "read_parquet",
        "read_json",
        "read_excel",
        "read_html",
        "read_feather",
        "read_orc",
        "read_pickle",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_clipboard",
        "read_hdf",
        "read_sas",
        "read_spss",
        "read_stata",
        "read_fwf",
        # DataFrame / Series write methods
        "to_csv",
        "to_parquet",
        "to_json",
        "to_excel",
        "to_feather",
        "to_orc",
        "to_pickle",
        "to_sql",
        "to_clipboard",
        "to_hdf",
        "to_stata",
        "to_latex",
        "to_html",
        "to_markdown",
        "to_xml",
    }
)

_MUTATION_NODE_TYPES: tuple[type, ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.TruncateTable,
    exp.AlterTable,
    exp.Command,  # catches COPY, VACUUM, and other DDL-adjacent commands
)

_ROW_CAP: int = 10_000
_PAYLOAD_CAP_BYTES: int = 5 * 1024 * 1024  # 5 MB

# Maps raw DB/Python driver exception class names to the canonical error_type
# strings recognised by ERROR_CORRECT_SYSTEM_PROMPT RULE 2 and the orchestrator.
# Without this, execute_sql() and execute_python() pass through class names like
# "ProgrammingError" which the LLM doesn't recognise → it guesses "SYNTAX_ERROR"
# → Pydantic rejects it (not in GenerateSQLOutput.error_code Literal) → parse_error
# → TERMINAL_ERROR instead of a correctable retry.
#
# psycopg2.ProgrammingError covers: syntax errors, undefined functions/operators,
# wrong column types, and incorrect argument counts — all fixable syntax-class issues.
# Extend this table as new driver quirks are discovered in production logs.
_EXEC_ERROR_TYPE_MAP: dict[str, str] = {
    # psycopg2 / SQLAlchemy PostgreSQL
    "ProgrammingError": "SYNTAX_ERROR",
    "InternalError": "SYNTAX_ERROR",
    # SQLAlchemy generic
    "CompileError": "SYNTAX_ERROR",
    "StatementError": "SYNTAX_ERROR",
    # psycopg2 operational (connection/resource — not a timeout, handled separately)
    "OperationalError": "DB_UNAVAILABLE",
    # MySQL / MariaDB
    "MySQLSyntaxError": "SYNTAX_ERROR",
}


# ---------------------------------------------------------------------------
# A. Python validation (VALIDATION state — Pandas path)
# ---------------------------------------------------------------------------


class _ForbiddenNodeVisitor(ast.NodeVisitor):
    """
    AST visitor that accumulates all policy violations in a single pass.
    Raises nothing — caller inspects `violations` after `visit()`.
    """

    def __init__(self) -> None:
        self.violations: list[ValidationResult] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                self.violations.append(
                    ValidationResult(
                        valid=False,
                        error_type="FORBIDDEN_IMPORT",
                        error_message=f"Import of '{alias.name}' is not permitted.",
                        error_line=node.lineno,
                    )
                )
            elif root not in _ALLOWED_IMPORTS:
                # Not explicitly forbidden but not on allowlist either — block it.
                self.violations.append(
                    ValidationResult(
                        valid=False,
                        error_type="FORBIDDEN_IMPORT",
                        error_message=(
                            f"Import of '{alias.name}' is not on the allowed-imports list: "
                            f"{sorted(_ALLOWED_IMPORTS)}."
                        ),
                        error_line=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if root in _FORBIDDEN_IMPORTS:
            self.violations.append(
                ValidationResult(
                    valid=False,
                    error_type="FORBIDDEN_IMPORT",
                    error_message=f"Import from '{module}' is not permitted.",
                    error_line=node.lineno,
                )
            )
        elif root and root not in _ALLOWED_IMPORTS:
            self.violations.append(
                ValidationResult(
                    valid=False,
                    error_type="FORBIDDEN_IMPORT",
                    error_message=(
                        f"Import from '{module}' is not on the allowed-imports list."
                    ),
                    error_line=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Direct calls: eval(), exec(), open(), etc.
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTINS:
            self.violations.append(
                ValidationResult(
                    valid=False,
                    error_type="FORBIDDEN_BUILTIN",
                    error_message=f"Call to '{node.func.id}' is not permitted.",
                    error_line=node.lineno,
                )
            )
        # Attribute calls that reach dangerous builtins via __class__, __bases__, etc.
        if isinstance(node.func, ast.Attribute):
            dangerous_attrs = {
                "__class__",
                "__bases__",
                "__subclasses__",
                "__globals__",
                "__code__",
                "__reduce__",
                "__reduce_ex__",
            }
            if node.func.attr in dangerous_attrs:
                self.violations.append(
                    ValidationResult(
                        valid=False,
                        error_type="FORBIDDEN_BUILTIN",
                        error_message=(
                            f"Access to '{node.func.attr}' is not permitted."
                        ),
                        error_line=node.lineno,
                    )
                )
            # P2-04 FIX: block pandas IO methods that bypass the open() guard.
            # pd.read_csv('/etc/passwd') / df.to_csv('/tmp/exfil') use C-level
            # file descriptors and are not caught by _FORBIDDEN_BUILTINS.
            # Block the method name regardless of the receiver object so the
            # check fires on both `pd.read_csv(...)` and `df.to_csv(...)`.
            if node.func.attr in _BLOCKED_PANDAS_IO:
                self.violations.append(
                    ValidationResult(
                        valid=False,
                        error_type="FORBIDDEN_BUILTIN",
                        error_message=(
                            f"pandas IO method '{node.func.attr}' is not permitted "
                            f"in the sandbox. Use the pre-loaded DataFrame variables "
                            f"instead of reading from files or writing to disk."
                        ),
                        error_line=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block __dunder__ attribute access on any object (escape-hatch guard)
        if (
            node.attr.startswith("__")
            and node.attr.endswith("__")
            and node.attr
            not in {
                "__init__",
                "__len__",
                "__str__",
                "__repr__",
                "__iter__",
                "__next__",
                "__getitem__",
                "__setitem__",
                "__contains__",
            }
        ):
            self.violations.append(
                ValidationResult(
                    valid=False,
                    error_type="FORBIDDEN_BUILTIN",
                    error_message=f"Access to dunder attribute '{node.attr}' is not permitted.",
                    error_line=node.lineno,
                )
            )
        self.generic_visit(node)


def validate_python(
    code: str,
    schema_columns: set[str],
    dataframe_refs: set[str] | None = None,
) -> ValidationResult:
    """
    Three-stage Python validation:
      1. AST parse (syntax check)
      2. AST visitor (forbidden imports, forbidden builtins, dunder access)
      3. Column grounding heuristic (string literals vs known schema columns)
      4. `result` assignment check (contract: last expression must set `result`)

    Parameters
    ----------
    code          : Python source string from LLM output.
    schema_columns: Full set of column names known for the retrieved schema chunks.
                    Used for lightweight grounding — not a substitute for the
                    grounding check in the generation prompt.
    dataframe_refs: Names of DataFrames injected into the exec scope. If provided,
                    any DataFrame name used in code that is not in this set is flagged.

    Returns
    -------
    ValidationResult. `valid=True` means the code passed all checks and is safe
    to send to execute_python(). The first violation encountered in each stage
    is returned — the loop does not accumulate across stages.
    """
    # Stage 1: AST parse
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(
            valid=False,
            error_type="SYNTAX_ERROR",
            error_message=str(exc),
            error_line=exc.lineno,
        )

    # Stage 2: AST visitor — collect all violations, return first
    visitor = _ForbiddenNodeVisitor()
    visitor.visit(tree)
    if visitor.violations:
        return visitor.violations[0]

    # Stage 3: `result` assignment check — run before column grounding so that
    # a missing `result =` is always reported as RESULT_ASSIGN_MISSING, even
    # when the code also contains unresolved column-like string literals.
    assigned_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                assigned_names.add(target.id)
    if "result" not in assigned_names:
        return ValidationResult(
            valid=False,
            error_type="RESULT_ASSIGN_MISSING",
            error_message=(
                "The generated code never assigns to `result`. "
                "The last line must be: result = <expression>."
            ),
        )

    # Stage 4: column grounding (heuristic — string constants that look like
    # column names and are not in the known schema)
    string_literals: set[str] = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # Filter to strings that look like column identifiers:
    # lowercase/underscore pattern, length 2–64, not Python keywords
    import keyword

    column_like = {
        s
        for s in string_literals
        if 2 <= len(s) <= 64
        and (s.islower() or "_" in s)
        and not keyword.iskeyword(s)
        and "." not in s  # exclude format strings / paths
        and " " not in s  # exclude sentence fragments
    }
    # ML-2 FIX: cast schema_columns to set — callers (orchestrator, tests)
    # may pass a list or other iterable. The | operator requires set on left.
    safe_names = set(schema_columns) | {
        # ── DataFrame variable names ──────────────────────────────────────
        "result",
        "df",
        "df2",
        "df3",
        "tmp",
        "temp",
        "out",
        # ── Index / axis kwargs ───────────────────────────────────────────
        "index",
        "axis",
        "inplace",
        "ascending",
        "descending",
        "by",
        "level",
        "ignore_index",
        # ── merge / join kwargs ───────────────────────────────────────────
        "left",
        "right",
        "inner",
        "outer",
        "cross",
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "how",
        "on",
        "suffixes",
        "indicator",
        # ── fill / sort kwargs ────────────────────────────────────────────
        "ffill",
        "bfill",
        "pad",
        "backfill",
        "first",
        "last",
        "none",
        "na_position",
        "keep",
        # ── agg / groupby kwargs ──────────────────────────────────────────
        "aggfunc",
        "func",
        "numeric_only",
        "min_count",
        "mean",
        "sum",
        "min",
        "max",
        "count",
        "median",
        "std",
        "var",
        "prod",
        "nunique",
        "size",
        # ── pivot kwargs ──────────────────────────────────────────────────
        "values",
        "margins",
        "margins_name",
        # ── read / write kwargs ───────────────────────────────────────────
        "key",
        "value",
        "columns",
        "copy",
        "header",
        "sep",
        "delimiter",
        # ── string accessor ───────────────────────────────────────────────
        "regex",
        "na",
        # ── original safe names ───────────────────────────────────────────
        "dropna",
        "fillna",
        "reset_index",
        # ML-2 FIX: the above additions prevent common Pandas method kwarg
        # string constants (e.g. how='inner', method='ffill', keep='first',
        # aggfunc='mean') from being flagged as UNRESOLVED_COLUMN, which
        # previously triggered needless ERROR_CORRECT LLM round-trips.
    }
    if dataframe_refs:
        safe_names |= dataframe_refs

    unresolved = column_like - safe_names
    if unresolved:
        return ValidationResult(
            valid=False,
            error_type="UNRESOLVED_COLUMN",
            error_message=(
                f"String literals not found in schema or safe names: "
                f"{sorted(unresolved)}. "
                f"If these are not column references, add them to the known-safe list."
            ),
        )

    return ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# B. SQL validation (VALIDATION state — SQL path)
# ---------------------------------------------------------------------------


def validate_sql(
    sql: str,
    schema_columns: set[str],
    dialect: str = "postgres",
) -> ValidationResult:
    """
    Three-stage SQL validation:
      1. sqlglot parse (syntax + dialect check)
      2. Mutation statement guard
      3. Column existence check against retrieved schema columns

    Parameters
    ----------
    sql           : Raw SQL string from LLM output.
    schema_columns: All column names from the retrieved schema chunks for this query.
    dialect       : sqlglot dialect string matching the GenerateSQLInput.dialect value.

    Returns
    -------
    ValidationResult. `valid=True` means safe to send to the pre-execution policy
    layer and then to execute_sql().

    Failure mode: sqlglot raises ParseError on dialect-specific syntax it does not
    recognise (e.g. BigQuery QUALIFY clause). In this case error_type=SYNTAX_ERROR
    is returned and ERROR_CORRECT is triggered with the parse error message injected.
    """
    # Stage 1: parse
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError as exc:
        return ValidationResult(
            valid=False,
            error_type="SYNTAX_ERROR",
            error_message=str(exc),
        )

    # Stage 2: mutation guard
    for node in parsed.walk():
        if isinstance(node, _MUTATION_NODE_TYPES):
            return ValidationResult(
                valid=False,
                error_type="MUTATION_STATEMENT",
                error_message=(
                    f"Statement type '{type(node).__name__}' is not permitted. "
                    f"Only read-only queries are allowed."
                ),
            )

    # Stage 3: column existence check (CTE-aware)
    #
    # Root-cause fix: the previous implementation collected all bare column
    # names from every column reference and checked them against the flat
    # schema column set.  This silently passed CTE alias-bleed bugs.
    #
    # Example bug that was invisible to the old check:
    #   payment_agg AS (SELECT claim_id, SUM(paid_amount) AS total_paid ...)
    #   paid_claims  AS (... LEFT JOIN payment_agg pa ...
    #                        COALESCE(SUM(pa.paid_amount), 0) ...)
    #                                       ^^^^^^^^^^^
    #   'pa' only exposes {claim_id, total_paid}; 'paid_amount' is a real
    #   payments schema column so the flat check passed — but PostgreSQL
    #   raises "column pa.paid_amount does not exist" at runtime.
    #   This consumed the last retry slot, converting a correctable error
    #   into TERMINAL_ERROR.
    #
    # Fix: build a CTE output-column map and a table-alias → CTE name map.
    # Column references qualified with a CTE alias are validated against
    # that CTE's output columns only; schema-qualified / bare references
    # use the existing logic unchanged.

    # Step 3a — collect each CTE's exposed column names
    # (SELECT-level aliases + bare column references; * → sentinel "*")
    cte_output_cols: dict[str, set[str]] = {}
    with_clause = parsed.find(exp.With)
    if with_clause:
        for cte in with_clause.find_all(exp.CTE):
            cte_name = (cte.alias or "").lower()
            if not cte_name:
                alias_node = cte.args.get("alias")
                cte_name = (getattr(alias_node, "name", "") or "").lower()
            if not cte_name:
                continue

            cte_select = cte.this  # the SELECT subquery inside the CTE
            if cte_select is None:
                continue

            output_cols: set[str] = set()
            for sel_expr in cte_select.args.get("expressions") or []:
                if isinstance(sel_expr, exp.Alias) and sel_expr.alias:
                    output_cols.add(sel_expr.alias.lower())
                elif isinstance(sel_expr, exp.Column) and sel_expr.name:
                    # Un-aliased bare column (e.g. SELECT claim_id FROM ...)
                    # The CTE exposes it under its original name.
                    output_cols.add(sel_expr.name.lower())
                elif isinstance(sel_expr, exp.Star):
                    # SELECT * — output columns unknown statically; sentinel
                    # disables CTE-level validation for this CTE only.
                    output_cols.add("*")
                # Unaliased aggregate (RULE 14 forbids these): skip — the
                # DB assigns a generated name we cannot predict; omitting it
                # avoids false positives on other correctly-named columns.
            cte_output_cols[cte_name] = output_cols

    # Step 3b — map every table alias to the CTE it references
    # Handles "FROM payment_agg pa" → alias "pa" resolves to CTE "payment_agg"
    table_alias_to_cte: dict[str, str] = {}
    for tbl in parsed.find_all(exp.Table):
        tbl_name = (tbl.name or "").lower()
        tbl_alias = (tbl.alias or "").lower()
        if tbl_name in cte_output_cols and tbl_alias:
            table_alias_to_cte[tbl_alias] = tbl_name

    # Step 3c — partition column references
    cte_col_violations: set[str] = set()
    schema_referenced_cols: set[str] = set()

    for col in parsed.find_all(exp.Column):
        if not col.name:
            continue
        col_name_lower = col.name.lower()
        table_qual = (col.table or "").lower()

        # Resolve table qualifier → CTE name (via alias map or direct name)
        resolved_cte: str | None = table_alias_to_cte.get(table_qual) or (
            table_qual if table_qual in cte_output_cols else None
        )

        if resolved_cte:
            cte_cols = cte_output_cols[resolved_cte]
            # "*" sentinel means SELECT * — output columns unknown; skip.
            if "*" not in cte_cols and col_name_lower not in cte_cols:
                cte_col_violations.add(col_name_lower)
        else:
            # Non-CTE-qualified (real table alias or bare) — schema check.
            schema_referenced_cols.add(col_name_lower)

    # CTE violations surface before schema check: they are the more specific
    # and actionable error (wrong alias name vs missing schema column).
    if cte_col_violations:
        return ValidationResult(
            valid=False,
            error_type="UNRESOLVED_COLUMN",
            error_message=(
                f"Column(s) not found in CTE output: {sorted(cte_col_violations)}. "
                f"The referenced CTE does not expose these names — use only the "
                f"aliases defined in that CTE's SELECT clause. "
                f"Use only column names present in schema_context."
            ),
            unresolved_columns=sorted(cte_col_violations),
        )

    # Collect SELECT-clause aliases (e.g. COUNT(*) AS total) so that
    # ORDER BY / HAVING references to those aliases are not flagged as
    # unresolved schema columns.  Includes CTE-internal aliases as a
    # safety net for bare (non-table-qualified) CTE output references.
    select_aliases: set[str] = {
        expr.alias.lower() for expr in parsed.find_all(exp.Alias) if expr.alias
    }
    # Schema columns are stored lowercased for comparison
    schema_cols_lower = {c.lower() for c in schema_columns}
    unresolved = schema_referenced_cols - schema_cols_lower - {"*"} - select_aliases
    if unresolved:
        return ValidationResult(
            valid=False,
            error_type="UNRESOLVED_COLUMN",
            # Client-facing message: no internal column enumeration.
            # The full known-column list is available to ERROR_CORRECT via
            # schema_context injected into the prompt — not needed here.
            error_message=(
                f"Column(s) not found in the schema: {sorted(unresolved)}. "
                f"Use only column names present in schema_context."
            ),
            unresolved_columns=sorted(unresolved),
        )

    # Stage 4: non-sargable filter check
    # Detects date/time function wrappers on columns inside WHERE/HAVING that
    # prevent index use. Covers EXTRACT(), DATE_PART(), YEAR(), MONTH(),
    # TO_CHAR(), and any other function wrapping a column in a filter clause.
    #
    # NON_SARGABLE_FILTER is a performance advisory (valid=True): the query
    # executes correctly but won't use an index on the date column.
    _DATE_FUNC_NAMES = frozenset(
        {
            "date_part",
            "year",
            "month",
            "day",
            "quarter",
            "to_char",
            "date_format",
            "strftime",
            "datepart",
        }
    )

    for clause in (
        *parsed.find_all(exp.Where),
        *parsed.find_all(exp.Having),
    ):
        clause_name = "WHERE" if isinstance(clause, exp.Where) else "HAVING"

        # 1. EXTRACT(unit FROM col) — sqlglot exp.Extract node
        #
        # BUG-A FIX: the previous template always suggested a quarterly
        # DATE_TRUNC range regardless of the unit field.  The correct rewrite
        # depends on the field and on whether the source expression is an
        # AGE() call (age comparison) vs a raw date column (date truncation).
        #
        # BUG-C FIX: normalise field_name to lower-case so the warning text
        # is consistent regardless of how sqlglot tokenises the unit keyword.
        if clause.find(exp.Extract):
            extract_node = clause.find(exp.Extract)
            # In sqlglot's Extract AST, args["this"] holds the unit keyword
            # (e.g. exp.Var(name="YEAR")).  args["expression"] holds the
            # source expression (column or nested function).
            field_name = (
                extract_node.args.get("this", "").name.lower()
                if extract_node and extract_node.args.get("this")
                else "unknown"
            )

            # Detect AGE() in the source expression to distinguish:
            #   EXTRACT(YEAR FROM AGE(CURRENT_DATE, dob))  → age comparison
            #   EXTRACT(YEAR FROM created_at)              → date truncation
            # Both are non-sargable; the correct range-predicate rewrite differs.
            source_expr = extract_node.args.get("expression") if extract_node else None
            is_age_based = source_expr is not None and any(
                (getattr(n, "name", "") or "").lower() == "age"
                for n in source_expr.walk()
                if isinstance(n, exp.Anonymous)
            )

            if field_name == "year" and is_age_based:
                hint = (
                    "Rewrite the age comparison as a direct range predicate "
                    "on the birth-date column: "
                    "date_col < CURRENT_DATE - INTERVAL 'N years' "
                    "(replace N with the threshold). "
                    "This allows a B-tree index seek on the date column."
                )
            elif field_name == "year":
                hint = (
                    "Rewrite as a range predicate: "
                    "date_col >= DATE_TRUNC('year', :param) "
                    "AND date_col < DATE_TRUNC('year', :param) + INTERVAL '1 year'."
                )
            elif field_name in ("month", "quarter"):
                hint = (
                    f"Rewrite as a range predicate: "
                    f"date_col >= DATE_TRUNC('{field_name}', :param) "
                    f"AND date_col < DATE_TRUNC('{field_name}', :param) "
                    f"+ INTERVAL '1 {field_name}'."
                )
            else:
                hint = (
                    "Rewrite using a direct range predicate on the date column "
                    "to allow index use."
                )

            return ValidationResult(
                valid=True,
                warning=(
                    f"Performance advisory: EXTRACT({field_name.upper()} ...) in "
                    f"{clause_name} prevents index use on the date column. "
                    f"{hint}"
                ),
            )

        # 2. DATE_PART / YEAR() / MONTH() / TO_CHAR() — these are represented
        #    as exp.Anonymous (unclassified function calls) in sqlglot 23.9.0.
        #    exp.DatePart was added in later sqlglot versions; using it here
        #    raises AttributeError on 23.9.0. exp.Anonymous covers all of
        #    these function names via the .name attribute match below.
        #
        # BUG-B FIX: the previous loop had an unconditional return OUTSIDE the
        # `if func_name in _DATE_FUNC_NAMES:` guard, which fired on the first
        # non-date Anonymous function (e.g. LOWER(), COALESCE()) and referenced
        # `field_name` / `clause_name` that were never assigned in this code
        # path → UnboundLocalError.  Removed the stray fallback return; the
        # loop now only warns for recognised date-wrapping functions and falls
        # through silently for everything else.
        for func_node in clause.find_all(exp.Anonymous):
            func_name = (getattr(func_node, "name", "") or "").lower()
            if func_name in _DATE_FUNC_NAMES:
                return ValidationResult(
                    valid=True,
                    warning=(
                        f"Performance advisory: {func_name.upper()}(...) in "
                        f"{clause_name} prevents index use on the date column. "
                        f"Rewrite as a range predicate using DATE_TRUNC or "
                        f"explicit date literals to allow index seeks."
                    ),
                )
            # Non-date Anonymous function (e.g. LOWER, COALESCE) — not a
            # sargability concern; continue scanning remaining func nodes.

    return ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# C. Pre-execution policy layer (INDEPENDENT of LLM output)
# ---------------------------------------------------------------------------


@dataclass
class TablePolicy:
    """Per-table policy flags set at ingestion time."""

    table_name: str
    pii_flagged: bool = False
    access_level: Literal["public", "restricted", "blocked"] = "public"


class PreExecutionPolicy:
    """
    Enforces data-access policies before any SQL reaches the database.

    This class is deliberately not LLM-dependent — it operates purely on
    the parsed SQL AST and the policy registry. Even if the LLM output
    bypasses the generation prompt's grounding rules, this layer blocks
    policy violations unconditionally.

    Failure mode: a policy violation is returned as a ValidationResult
    with error_type=POLICY_VIOLATION. The orchestrator transitions to
    TERMINAL_ERROR (no retry — policy violations are not transient).
    """

    def __init__(self, table_policies: dict[str, TablePolicy]) -> None:
        # Keyed by lowercase table name
        self._policies: dict[str, TablePolicy] = {
            k.lower(): v for k, v in table_policies.items()
        }

    def check(self, sql: str, dialect: str = "postgres") -> ValidationResult:
        """
        Checks:
          1. LIMIT-less SELECT on a PII-flagged table.
          2. Access to blocked tables.
        """
        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
        except sqlglot.errors.ParseError:
            # Syntax errors are caught by validate_sql — if we reach here with
            # a parse failure, treat as policy pass (syntax layer will handle it)
            return ValidationResult(valid=True)

        # Collect all table references
        tables_referenced: set[str] = {
            tbl.name.lower() for tbl in parsed.find_all(exp.Table) if tbl.name
        }

        for table_name in tables_referenced:
            policy = self._policies.get(table_name)
            if policy is None:
                continue  # No policy registered → allow

            if policy.access_level == "blocked":
                return ValidationResult(
                    valid=False,
                    error_type="POLICY_VIOLATION",
                    error_message=(
                        f"Table '{table_name}' is not accessible. "
                        f"Contact your data administrator."
                    ),
                )

            if policy.pii_flagged:
                # Check for missing LIMIT on the top-level SELECT
                has_limit = parsed.find(exp.Limit) is not None
                if not has_limit:
                    return ValidationResult(
                        valid=False,
                        error_type="POLICY_VIOLATION",
                        error_message=(
                            f"Table '{table_name}' contains protected data. "
                            f"This query would return an unbounded result set. "
                            f"Please add a LIMIT clause or a filter condition."
                        ),
                    )

        return ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# D. SQL executor (EXECUTION state — SQL path)
# ---------------------------------------------------------------------------


def _make_readonly_engine(connection_url: str) -> Engine:
    """
    Creates a SQLAlchemy engine where every connection is placed into a
    READ ONLY transaction before the query executes.

    Implementation notes:
    - AUTOCOMMIT is intentionally NOT used: SET TRANSACTION READ ONLY requires
      an open transaction to take effect in PostgreSQL. With AUTOCOMMIT there
      is no transaction and the directive is silently ignored.
    - SQLite does not support SET TRANSACTION READ ONLY; the execute() call
      is wrapped in try/except so SQLite connections are unaffected (the
      validate_sql mutation-guard remains the sole defence for SQLite).
    - The engine is read-only at the DB level for PostgreSQL and falls back
      to application-level guards only for SQLite/MySQL.
    """
    connect_args: dict[str, Any] = {}
    if connection_url.startswith(("postgresql://", "postgres://")):
        connect_args["connect_timeout"] = 5

    engine = create_engine(
        connection_url,
        connect_args=connect_args,
        pool_pre_ping=True,  # test connection before each checkout; auto-evicts stale sockets
        pool_recycle=1800,  # recycle connections every 30 min; prevents ETIMEDOUT on idle
        pool_size=2,  # small pool — copilot is single-worker, not high-concurrency
        max_overflow=4,
    )

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Any) -> None:
        # Non-PostgreSQL backends (e.g. SQLite) don't support these
        # directives — both are wrapped in suppress so they silently no-op.
        with contextlib.suppress(Exception):
            conn.execute(text("SET TRANSACTION READ ONLY"))
        # C2 FIX: DB-level timeout guard (15 s > 10 s Python-side timeout).
        # future.cancel() on a running ThreadPoolExecutor thread is a no-op
        # in Python, so the DB query keeps executing after execute_sql()
        # returns TimeoutError.  SET statement_timeout kills it at the server
        # level, releasing the connection before the 15 s DB guard fires.
        # Non-PostgreSQL backends raise on this statement and are suppressed.
        with contextlib.suppress(Exception):
            conn.execute(text("SET statement_timeout = '15000'"))

    return engine


# ---------------------------------------------------------------------------
# DB-level column-not-found normaliser
# ---------------------------------------------------------------------------
# Different DB drivers surface missing columns with different exception messages.
# Normalising them to UNRESOLVED_COLUMN here means the orchestrator fast-path
# fires on DB failures (not just static-validation failures), cutting the
# wasted error_correct LLM call for structurally unresolvable columns.

_COLUMN_NOT_FOUND_RE = re.compile(
    # All patterns carry a capture group so m.groups() always has a
    # non-None entry when the pattern fires. The original Pattern 1
    # had no group; next(g for g in m.groups() if g) exhausted the
    # iterator and raised StopIteration, which Python 3.7+ re-raises
    # as RuntimeError inside a generator expression, propagating out
    # of execute_sql and triggering INTERNAL_ERROR on every postgres
    # "column X does not exist" failure.
    #
    # \x22 = double-quote  \x27 = single-quote (avoids string-delimiter
    # conflicts inside raw strings while keeping the pattern readable).
    #
    # postgres / sqlite: column "policy_type" does not exist
    # postgres / sqlite: no such column: policy_type
    # mysql:             Unknown column 'policy_type' in 'field list'
    # sqlserver:         Invalid column name 'policy_type'
    r"column [\x22\x27]?([\w.]+)[\x22\x27]? does not exist"
    r"|no such column: ([\w.]+)"
    r"|Unknown column '([^']+)'"
    r"|Invalid column name '([^']+)'",
    re.IGNORECASE,
)


def _extract_missing_column(error_message: str) -> str | None:
    """
    Parse a DB driver error message and return the bare column name
    (without table qualifier) if it matches a known column-not-found pattern.
    Returns None if the error is unrelated to a missing column.

    Uses next(..., None) instead of next(...) to avoid StopIteration
    being re-raised as RuntimeError inside a generator (PEP 479 / 3.7+).
    """
    m = _COLUMN_NOT_FOUND_RE.search(error_message)
    if not m:
        return None
    # next(..., None) is safe: StopIteration inside a generator is
    # RuntimeError in Python 3.7+. Always provide a default.
    col = next((g for g in m.groups() if g), None)
    if col is None:
        return None
    return col.split(".")[-1]


def traceback_lines(exc: Exception) -> str:
    """Return a safe, truncated traceback string (no filesystem paths exposed).

    BUG-07 FIX: moved here from after execute_python() so the function is
    defined before execute_python()'s _run() closure references it.  Python
    resolves global names at call-time, not definition-time, so this was a
    latent NameError risk during module initialisation rather than a runtime
    crash in normal operation — but forward references are a maintenance hazard.
    """
    import re
    import traceback as tb

    # P3-04 FIX: was `'File "/' in line or 'File "C:' in line`.
    # The old guard only caught POSIX paths (leading /) and the C: drive.
    # Any other Windows drive letter (D:, E:, ..., Z:) or a UNC path
    # (\\server\share\...) would pass through unsanitised, leaking container
    # directory structure into LLM error-correction prompts and observability
    # logs.  The regex below matches all three cases:
    #   File "/...    — POSIX absolute path
    #   File "X:...   — Windows any drive letter (A-Z, a-z)
    #   File "\\...   — Windows UNC path
    _PATH_IN_TRACEBACK = re.compile(r'File "(?:[/\\]|[A-Za-z]:)')

    lines = tb.format_exception(type(exc), exc, exc.__traceback__)
    sanitised = []
    for line in lines:
        if _PATH_IN_TRACEBACK.search(line):
            line = line.split('"')[0] + '"<internal>" ' + line.split('"')[-1]
        sanitised.append(line)
    return "".join(sanitised[-6:])  # last 6 lines only


def execute_sql(
    sql: str,
    engine: Engine,
    timeout_seconds: int = 10,
    dry_run: bool = False,
) -> ExecutionResult:
    """
    Executes a pre-validated SQL query in a read-only connection.

    The mutation guard in validate_sql() is the primary defence. This executor
    adds a second line of defence: the connection is opened with READ ONLY
    transaction semantics so that any mutation that slips through raises a
    DB-level error rather than executing.

    Parameters
    ----------
    sql             : Pre-validated SQL string (must have passed validate_sql()
                      and PreExecutionPolicy.check()).
    engine          : Read-only SQLAlchemy engine.
    timeout_seconds : Hard wall-clock limit. Exceeded → ExecutionResult with
                      error_type='EXECUTION_TIMEOUT'.
    dry_run         : If True, parse and validate but do not execute.

    Returns
    -------
    ExecutionResult. On any error: success=False with error_type and error_message
    populated. Never raises.
    """
    if dry_run:
        return ExecutionResult(
            success=True, result=None, row_count=0, columns=[], execution_time_ms=0
        )

    def _run() -> ExecutionResult:
        t0 = time.monotonic()
        with engine.connect() as conn:
            cursor_result = conn.execute(text(sql))
            columns = list(cursor_result.keys())
            rows = cursor_result.fetchmany(
                _ROW_CAP + 1
            )  # fetch one extra to detect cap

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        capped = len(rows) > _ROW_CAP
        rows = rows[:_ROW_CAP]
        result_dicts = [dict(zip(columns, row, strict=False)) for row in rows]

        # Coerce non-JSON-native DB types before serialization.
        # SQLAlchemy returns NUMERIC/DECIMAL as decimal.Decimal, which
        # json.dumps(default=str) converts to a string - so aggregates
        # like AVG appear as '"25179.93..."' not 25179.93 in output.
        import datetime as _datetime
        import decimal as _decimal
        import uuid as _uuid

        def _coerce(v: Any) -> Any:
            if isinstance(v, _decimal.Decimal):
                return float(v)
            if isinstance(v, _datetime.datetime | _datetime.date):
                return v.isoformat()
            if isinstance(v, _uuid.UUID):
                return str(v)
            if isinstance(v, bytes):
                return v.hex()
            return v

        result_dicts = [
            {k: _coerce(val) for k, val in row.items()} for row in result_dicts
        ]

        # Payload size guard
        payload_bytes = len(json.dumps(result_dicts, default=str).encode())
        if payload_bytes > _PAYLOAD_CAP_BYTES:
            result_dicts = result_dicts[:100]  # truncate to first 100 rows
            return ExecutionResult(
                success=True,
                result=result_dicts,
                row_count=len(result_dicts),
                columns=columns,
                execution_time_ms=elapsed_ms,
                error_type="RESULT_CAPPED",
                error_message=(
                    f"Payload exceeded {_PAYLOAD_CAP_BYTES // (1024*1024)}MB. "
                    f"Result truncated to 100 rows."
                ),
            )

        return ExecutionResult(
            success=True,
            result=result_dicts,
            row_count=len(result_dicts),
            columns=columns,
            execution_time_ms=elapsed_ms,
            error_message="Result capped at 10,000 rows." if capped else None,
        )

    pool = _SANDBOX_EXECUTOR  # BUG-6 FIX: module-level singleton
    future = pool.submit(_run)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return ExecutionResult(
            success=False,
            error_type="EXECUTION_TIMEOUT",
            error_message=(
                f"Query exceeded the {timeout_seconds}s timeout. "
                f"Try narrowing the date range or adding a filter."
            ),
        )
    except Exception as exc:
        err_msg = str(exc)
        missing_col = _extract_missing_column(err_msg)
        if missing_col:
            # Normalise DB-level 'column does not exist' errors to
            # UNRESOLVED_COLUMN so the orchestrator fast-path can fire
            # and skip the wasted error_correct LLM call.
            return ExecutionResult(
                success=False,
                error_type="UNRESOLVED_COLUMN",
                error_message=(
                    f"Column(s) not found in the schema: ['{missing_col}']. "
                    f"Use only column names present in schema_context."
                ),
                unresolved_columns=[missing_col],
            )
        # Normalise driver exception class names to canonical error_type
        # strings. Raw names like "ProgrammingError" are not recognised by
        # ERROR_CORRECT_SYSTEM_PROMPT RULE 2, causing the LLM to echo back
        # "SYNTAX_ERROR" which then fails Pydantic validation and converts a
        # retryable error into an immediate TERMINAL_ERROR.
        normalised_type = _EXEC_ERROR_TYPE_MAP.get(
            type(exc).__name__, type(exc).__name__
        )
        return ExecutionResult(
            success=False,
            error_type=normalised_type,
            error_message=err_msg,
        )
    # BUG-6 FIX follow-up: do NOT call pool.shutdown() here.
    # The singleton _SANDBOX_EXECUTOR lives for the process lifetime.
    # Calling shutdown() after the first execution permanently destroys the
    # shared pool — all subsequent calls raise RuntimeError("cannot schedule
    # new futures after shutdown"). The finally block is removed entirely.


def execute_python(
    code: str,
    dataframe_refs: dict[str, pd.DataFrame],
    timeout_seconds: int = 15,
    memory_limit_mb: int = 512,
) -> ExecutionResult:
    """
    Executes a pre-validated Python code string in a restricted namespace.

    Security model (layered):
      1. validate_python() must pass before this function is called.
      2. exec() receives a namespace with only whitelisted builtins.
      3. Allowed imports are pre-seeded into the namespace — the code does not
         need to (and cannot safely) import them itself.
      4. tracemalloc monitors peak memory usage; execution is aborted if
         memory_limit_mb is exceeded.
      5. concurrent.futures.ThreadPoolExecutor enforces the wall-clock timeout.

    Parameters
    ----------
    code            : Pre-validated Python source. Must assign to `result`.
    dataframe_refs  : DataFrames injected by name into the exec scope.
    timeout_seconds : Hard execution timeout.
    memory_limit_mb : Peak memory cap for the executed code.

    Returns
    -------
    ExecutionResult. `dataframe` field holds the `result` variable value.
    Never raises.
    """
    import collections
    import datetime
    import itertools
    import math
    import re

    import numpy as np

    _SAFE_BUILTINS: dict[str, Any] = {
        "len": len,
        "range": range,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "zip": zip,
        "enumerate": enumerate,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "abs": abs,
        # H1 FIX: type / getattr / hasattr removed.
        # getattr(obj, '__globals__') is identical to obj.__globals__ at
        # runtime but invisible to the AST visitor — keeping it in scope
        # would defeat every dunder guard in _ForbiddenNodeVisitor.
        # LLM-generated Pandas code has no legitimate need for these three.
        "isinstance": isinstance,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "map": map,
        "filter": filter,
        "reversed": reversed,
        "any": any,
        "all": all,
        "None": None,
        "True": True,
        "False": False,
        # SEC-2 FIX: __build_class__ removed.
        # Class construction permits runtime MRO traversal via
        # type(Exploit).__subclasses__() even when the AST visitor blocks
        # the literal dunder string — the block is syntactic only.
        # LLM-generated Pandas code has zero legitimate use for class definitions.
    }

    exec_namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "math": math,
        "re": re,
        "collections": collections,
        "itertools": itertools,
        "datetime": datetime,
    }
    exec_namespace.update(dataframe_refs)

    def _run() -> ExecutionResult:
        t0 = time.monotonic()
        tracemalloc.start()
        try:
            exec(
                compile(code, "<analyst_copilot>", "exec"), exec_namespace
            )  # noqa: S102
        except MemoryError:
            return ExecutionResult(
                success=False,
                error_type="MEMORY_LIMIT_EXCEEDED",
                error_message=(
                    f"Execution exceeded the {memory_limit_mb}MB memory limit. "
                    f"Try reducing the date range or the number of columns."
                ),
            )
        except Exception as exc:
            tb_lines = traceback_lines(exc)
            normalised_type = _EXEC_ERROR_TYPE_MAP.get(
                type(exc).__name__, type(exc).__name__
            )
            return ExecutionResult(
                success=False,
                error_type=normalised_type,
                error_message=f"{exc}\n{tb_lines}",
            )
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        peak_mb_val = peak_mb  # capture before potential exception

        if peak_mb_val > memory_limit_mb:
            return ExecutionResult(
                success=False,
                error_type="MEMORY_LIMIT_EXCEEDED",
                error_message=(
                    f"Peak memory {peak_mb_val:.1f}MB exceeded limit of {memory_limit_mb}MB."
                ),
                memory_used_mb=peak_mb_val,
                execution_time_ms=elapsed_ms,
            )

        result_val = exec_namespace.get("result")
        if result_val is None:
            return ExecutionResult(
                success=False,
                error_type="RESULT_ASSIGN_MISSING",
                error_message=(
                    "`result` was not assigned in the executed code. "
                    "Ensure the last line is `result = <expression>`."
                ),
                execution_time_ms=elapsed_ms,
                memory_used_mb=peak_mb_val,
            )

        # Normalise result to a row count
        row_count = 0
        if isinstance(result_val, pd.DataFrame | pd.Series):
            row_count = len(result_val)

        return ExecutionResult(
            success=True,
            dataframe=result_val,
            row_count=row_count,
            execution_time_ms=elapsed_ms,
            memory_used_mb=peak_mb_val,
        )

    # C1 FIX: do NOT use `with ThreadPoolExecutor() as pool:`.
    # ThreadPoolExecutor.__exit__ calls shutdown(wait=True), which blocks until
    # the submitted thread completes — even when the function returns a result
    # from inside an except TimeoutError block.  The 15-second timeout would be
    # completely defeated: the caller hangs for however long the actual Python
    # code runs.  Use explicit lifecycle with shutdown(wait=False) instead,
    # matching the pattern already used by execute_sql().
    pool = _SANDBOX_EXECUTOR  # BUG-6 FIX: module-level singleton
    future = pool.submit(_run)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        return ExecutionResult(
            success=False,
            error_type="EXECUTION_TIMEOUT",
            error_message=(
                f"Code execution exceeded the {timeout_seconds}s timeout. "
                f"Try reducing the dataset size or simplifying the operation."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        # BUG-03 FIX: catch all non-timeout exceptions (BrokenProcessPool,
        # MemoryError, RuntimeError from executor teardown races, etc.) so that
        # execute_python() always returns an ExecutionResult and never raises.
        # execute_sql() already has this guard; execute_python() was missing it.
        err_type = type(exc).__name__
        normalised = _EXEC_ERROR_TYPE_MAP.get(err_type, err_type)
        return ExecutionResult(
            success=False,
            error_type=normalised,
            error_message=str(exc),
        )
    # BUG-6 FIX follow-up: do NOT call pool.shutdown() here.
    # The singleton _SANDBOX_EXECUTOR must never be shut down — it serves
    # all execute_python() calls for the process lifetime. Removing this
    # finally block fixes RuntimeError("cannot schedule new futures after
    # shutdown") on the second and subsequent calls.


# ---------------------------------------------------------------------------
# F. Runtime result validation (RESULT_CHECK state)
# ---------------------------------------------------------------------------

# Ratio/rate columns whose max value falls below this threshold across all
# result rows are flagged as near-zero dominant.  Root cause: the numerator
# was not rolled up to the same aggregation level as the denominator (e.g.
# a single policy's claim_amount divided by SUM(premium_amt) for its entire
# policy_type).  Any legitimately computed loss_ratio or approval_rate in an
# insurance context should be at least 0.001 (0.1%) on non-trivial data.
_NEAR_ZERO_THRESHOLD: float = 0.001
# Approval-rate columns are degenerate when ALL rows are at or above this
# threshold — signals a WHERE clause pre-filter on claim_status that
# eliminates unapproved claims before COUNT, pinning the rate at 1.0.
_ALL_NEAR_ONE_THRESHOLD: float = 0.999
# Matches claim_approval_rate and approval_rate — columns for which
# near-unity across multiple groups is a degenerate signal (not valid).
_APPROVAL_RATE_RE: re.Pattern[str] = re.compile(
    r"\b(claim_)?approval_rate\b",
    re.IGNORECASE,
)

# Phrases that signal the result should be grouped/summarised per category.
# A result capped at _ROW_CAP rows while the query contains one of these
# phrases indicates the SQL grouped at the wrong level (per-row instead of
# per-category), which is a retryable aggregation error, not merely a size issue.
_GROUPBY_INTENT_RE: re.Pattern[str] = re.compile(
    r"\b(for each|per |by |compare|breakdown|group by|grouped by|across)\b",
    re.IGNORECASE,
)

# Detects a structurally correct loss_ratio denominator: SUM(premium_amt)
# aggregated directly from the policies table with no claim-side JOIN between
# FROM and GROUP BY.  Used by validate_metric_ranges to distinguish a SQL logic
# error (wrong denominator → suppress with fix hint) from a data calibration
# issue (correct SQL, miscalibrated premium_amt scale → suppress silently).
#
# _STANDALONE_PREMIUM_RE matches:
#   FROM policies GROUP BY ...              ← bare, no alias
#   FROM policies AS pbt GROUP BY ...       ← with alias
# but NOT:
#   FROM policies AS p JOIN claim_agg ...   ← INNER JOIN pattern: JOIN is a
#   GROUP BY p.policy_type                    non-whitespace token between FROM
#                                             and GROUP BY, so \s+GROUP fails
_STANDALONE_PREMIUM_RE: re.Pattern[str] = re.compile(
    r"FROM\s+policies\b(?:\s+(?:AS\s+)?\w+)?\s+GROUP\s+BY",
    re.IGNORECASE,
)
_SUM_PREMIUM_RE: re.Pattern[str] = re.compile(
    r"SUM\s*\(\s*(?:\w+\.)?premium_amt\s*\)",
    re.IGNORECASE,
)

# Metric range rules: (column pattern, lo, hi, human explanation)
# Matched against lower-cased column names via re.search.
# Rules are evaluated in order; the first match wins per column.
_METRIC_RANGE_RULES: list[tuple[re.Pattern[str], float, float, str]] = [
    (
        re.compile(r"loss_ratio", re.I),
        0.0,
        2.0,
        "loss_ratio > 2.0 is actuarially implausible — verify the loss_ratio "
        "denominator. SUM(premium_amt) must aggregate ALL policies of that type "
        "(including claim-free ones). Preferred fix: a standalone CTE — "
        "WITH premium_by_type AS (SELECT policy_type, SUM(premium_amt) AS "
        "total_premium FROM policies GROUP BY policy_type) — joined in the outer "
        "SELECT. Avoid correlated subqueries for this denominator: they yield "
        "correct values but re-scan the policies table once per output group, "
        "causing query timeouts on large datasets.",
    ),
    (
        re.compile(r"(approval|acceptance|conversion|hit|win)_rate$", re.I),
        0.0,
        1.0,
        "approval/conversion rates must be in [0, 1]",
    ),
    (
        re.compile(r"_rate$", re.I),
        0.0,
        1.0,
        "rate columns must be in [0, 1]",
    ),
    (
        re.compile(r"_ratio$", re.I),
        0.0,
        2.0,
        "ratio > 2.0 likely indicates a denominator aggregation error",
    ),
]


def _has_standalone_premium_cte(sql: str) -> bool:
    """Return True when the SQL contains SUM(premium_amt) aggregated directly
    from the policies table with no intervening JOIN — the fingerprint of a
    correct standalone premium CTE.

    Used to distinguish an actuarially implausible loss_ratio caused by a SQL
    denominator error (INNER JOIN excludes claim-free policies) from one caused
    by miscalibrated premium_amt scale in the underlying data.  Only the former
    should surface a SQL-fix hint to the user.
    """
    return bool(_SUM_PREMIUM_RE.search(sql) and _STANDALONE_PREMIUM_RE.search(sql))


def validate_metric_ranges(
    rows: list[dict[str, Any]],
    sql: str | None = None,
) -> RuntimeValidationResult | None:
    """
    Scan aggregated result columns for business-implausible values.

    Checks rate/ratio columns against expected numeric bounds derived from
    insurance domain knowledge (loss_ratio, approval_rate, etc.).  A violation
    strongly suggests a SQL logic error — most commonly an INNER JOIN that
    silently reduces the denominator (e.g. SUM(premium_amt) only summing
    premiums for policies that have at least one claim).

    sql (optional): the generated SQL string.  When provided and a loss_ratio
    violation is detected, the function checks whether the SQL already contains
    a standalone premium aggregation CTE (correct denominator).  If so, the
    loss_ratio violation is suppressed — the high value reflects data
    calibration rather than a SQL error, so surfacing a SQL-fix hint would
    mislead the user.

    Non-blocking: returns valid=True so the result is still returned to the
    caller.  The orchestrator should suppress LLM insight generation and surface
    the message as a result_warning instead — identical to IMPLAUSIBLE_VALUE.

    Returns None when all checks pass or the result contains no numeric metric
    columns that match a rule pattern.
    """
    if not rows:
        return None

    violations: list[str] = []
    first_row = rows[0]

    for col, sample in first_row.items():
        # Skip string, bool, and None values — not numeric metrics
        if sample is None or isinstance(sample, str | bool):
            continue

        col_lower = col.lower()
        for pattern, lo, hi, note in _METRIC_RANGE_RULES:
            if not pattern.search(col_lower):
                continue
            # Collect out-of-range values (skip NULLs — those are caught by IMPLAUSIBLE_VALUE)
            bad: list[float] = [
                round(float(r[col]), 4)
                for r in rows
                if r.get(col) is not None and not (lo <= float(r[col]) <= hi)
            ]
            if bad:
                violations.append(
                    f"'{col}': values {bad[:3]} outside [{lo}, {hi}] — {note}"
                )
            else:
                # Near-zero dominance check: all values are within the valid
                # range but suspiciously close to zero.  Root cause: per-row
                # numerator divided by a type-level aggregate denominator (e.g.
                # single policy's claim_amount / SUM of all premiums for that
                # policy_type).  The query executes without error but the ratio
                # is computed at the wrong aggregation level.
                _all_vals: list[float] = [
                    float(r[col])
                    for r in rows
                    if r.get(col) is not None and not isinstance(r.get(col), str | bool)
                ]
                if _all_vals and max(_all_vals) < _NEAR_ZERO_THRESHOLD:
                    violations.append(
                        f"'{col}': all values < {_NEAR_ZERO_THRESHOLD} "
                        f"(max={max(_all_vals):.6f}) — likely cause: per-row metric "
                        f"divided by a type-level aggregate denominator. "
                        f"Fix: add a GROUP BY roll-up CTE so the numerator and "
                        f"denominator are aggregated at the same level before "
                        f"computing the ratio."
                    )
                # Near-unity dominance check: all approval-rate values = 1.0
                # across multiple groups signals a WHERE clause pre-filter on
                # claim_status.  When WHERE filters to approved/paid claims
                # only, COUNT(*) equals COUNT(approved) and the rate is
                # trivially 1.0 regardless of real approval behaviour.
                # Only fires for approval-rate columns on multi-group results
                # — a single-row result at 1.0 may be legitimately correct.
                if (
                    _all_vals
                    and len(rows) > 1
                    and min(_all_vals) > _ALL_NEAR_ONE_THRESHOLD
                    and _APPROVAL_RATE_RE.search(col)
                ):
                    violations.append(
                        f"'{col}': all {len(rows)} groups = 1.0 — probable "
                        f"WHERE clause pre-filter on claim_status. A status "
                        f"filter in WHERE makes every counted claim approved/"
                        f"paid → approved_count = claim_count → rate = 1.0. "
                        f"Fix: remove any WHERE claim_status condition from "
                        f"the claims CTE; keep it only in the CASE WHEN — "
                        f"COUNT(CASE WHEN c.claim_status IN "
                        f"('approved','paid') THEN 1 END) AS approved_count. "
                        f"WHERE should only exclude nulls: "
                        f"WHERE c.claim_amount IS NOT NULL."
                    )
            break  # first matching rule wins; don't double-report same column

    if not violations:
        return None

    # SQL-aware suppression: if the SQL already has a standalone premium
    # aggregation CTE, the loss_ratio denominator is architecturally correct.
    # High values then reflect data calibration, not SQL error — suppress only
    # the loss_ratio violations to avoid misdirecting the user to fix working SQL.
    # Other violations (rate/ratio columns) are unaffected.
    if sql and _has_standalone_premium_cte(sql):
        violations = [v for v in violations if "loss_ratio" not in v.lower()]
        if not violations:
            return None

    return RuntimeValidationResult(
        valid=True,  # non-blocking — result is returned, insight suppressed
        issue="METRIC_OUT_OF_RANGE",
        message=(
            "Result quality warning — computed metrics are outside expected "
            "bounds, indicating a probable SQL logic error:\n"
            + "\n".join(f"  • {v}" for v in violations)
            + "\n\nDo not interpret these numbers as correct. "
            "Review the SQL denominator aggregation before drawing conclusions."
        ),
    )


def validate_result(
    result: list[dict[str, Any]],
    expected_columns: list[str] | None = None,
    row_cap: int = _ROW_CAP,
) -> RuntimeValidationResult:
    """
    Post-execution plausibility checks.

    Checks (in order):
      1. EMPTY_RESULT   — result is an empty list
      2. SHAPE_MISMATCH — returned columns differ from expected (if provided)
      3. RESULT_CAPPED  — row count hit the cap (non-fatal: valid=True with issue set)

    Parameters
    ----------
    result          : List of row dicts from execute_sql() or a DataFrame.to_dict().
    expected_columns: Column names from the GenerateSQLOutput grounding_check.
                      If None, shape check is skipped.
    row_cap         : The cap used during execution (default 10,000).

    Returns
    -------
    RuntimeValidationResult. valid=True + issue=RESULT_CAPPED is not an error —
    the orchestrator adds a warning to QueryResponse.warnings and continues to INSIGHT.
    valid=False transitions to INTAKE (for EMPTY_RESULT) or ERROR_CORRECT.
    """
    if not result:
        return RuntimeValidationResult(
            valid=False,
            issue="EMPTY_RESULT",
            message=(
                "The query returned zero rows. "
                "The filter conditions may be too restrictive, or the referenced "
                "time period may contain no data."
            ),
        )

    # NULL-dominant column check: detect metric columns where >80% of rows are NULL.
    #
    # Root cause this catches: a JOIN matches rows in the source table, but the
    # target aggregate column (e.g. claim_amount) is NULL for all matched records
    # (pending/open/denied claims have no paid amount yet). SQL's SUM()/AVG() on
    # an all-null group returns NULL — the query succeeds (row_count > 0), but
    # every metric is NULL. Without this check the LLM generates an "insight" from
    # the non-null incidental columns (e.g. total_premium), which is misleading.
    #
    # IMPLAUSIBLE_VALUE is non-blocking (valid=True): the response is still returned
    # with result_rows populated. The warning flows into result_warnings and
    # _result_check uses it to suppress LLM insight generation via state.insight.
    _n_rows = len(result)
    _first_row = result[0]
    _null_dominated: list[str] = []
    for _col, _sample in _first_row.items():
        # Only inspect non-string, non-bool columns — numeric or None values
        # are potential metric columns; strings are dimension/grouping columns.
        if isinstance(_sample, str | bool):
            continue
        _null_count = sum(1 for row in result if row.get(_col) is None)
        if _null_count / _n_rows > 0.8:
            _null_dominated.append(f"{_col} ({int(100 * _null_count / _n_rows)}% NULL)")
    if _null_dominated:
        return RuntimeValidationResult(
            valid=True,
            issue="IMPLAUSIBLE_VALUE",
            message=(
                f"Result quality warning: the following metric column(s) are "
                f"predominantly NULL — {', '.join(_null_dominated)}. "
                "The query matched rows but the requested metrics could not be "
                "computed. Likely cause: a missing status filter "
                "(e.g. claim_status = 'paid') or the selected period contains "
                "only pending or denied records with no payout amount."
            ),
        )

    if expected_columns is not None:
        actual_cols = set(result[0].keys())
        expected_cols = set(expected_columns)
        if actual_cols != expected_cols:
            extra = actual_cols - expected_cols
            missing = expected_cols - actual_cols
            return RuntimeValidationResult(
                valid=False,
                issue="SHAPE_MISMATCH",
                message=(
                    f"Result shape does not match expectation. "
                    f"Extra columns: {sorted(extra)}. "
                    f"Missing columns: {sorted(missing)}."
                ),
            )

    if len(result) >= row_cap:
        return RuntimeValidationResult(
            valid=True,
            issue="RESULT_CAPPED",
            message=(
                f"Result was capped at {row_cap:,} rows. "
                f"Add a filter or a more specific date range to see the full result."
            ),
        )

    return RuntimeValidationResult(valid=True)


def validate_group_by_cardinality(
    rows: list[dict[str, Any]],
    nl_query: str,
    row_cap: int = _ROW_CAP,
) -> RuntimeValidationResult | None:
    """
    Detects when a group-by query returns a row-capped result.

    "for each X" / "per X" / "compare" / "breakdown" in the NL query signals
    that the result should have one row per distinct value of X — typically
    low cardinality (< 50 rows for most category columns).  When the result
    hits row_cap, this is almost always a SQL aggregation level error: the
    generated query grouped at the wrong granularity (e.g. per policy_id
    instead of per policy_type).

    This check is intentionally separate from validate_result() so it can
    receive the nl_query context that ExecutionLoop.run() does not have.
    It is called by the orchestrator inside the retry loop, immediately after
    a RESULT_CAPPED signal is detected on a successful execution.

    Parameters
    ----------
    rows      : Result rows from ExecutionResult.result (up to row_cap rows).
    nl_query  : The sanitised natural-language query for intent detection.
    row_cap   : The row cap used during execution (default _ROW_CAP = 10,000).

    Returns
    -------
    None when no issue is detected (result not capped, or no group-by intent).
    RuntimeValidationResult with valid=False and issue=GROUPBY_CARDINALITY_MISMATCH
    when both conditions are met — the caller should treat this as a retryable
    aggregation error and route to ERROR_CORRECT with the message injected.
    """
    if len(rows) < row_cap:
        return None  # result not capped — cardinality is fine

    if not _GROUPBY_INTENT_RE.search(nl_query):
        return None  # no group-by intent in the query

    return RuntimeValidationResult(
        valid=False,
        issue="GROUPBY_CARDINALITY_MISMATCH",
        message=(
            f"Result cardinality mismatch: the query returned {row_cap:,} rows "
            f"(row cap hit) but the question asks for a per-category summary. "
            f"The generated SQL likely grouped at the wrong granularity — "
            f"per-row instead of per-category (e.g. GROUP BY policy_id instead "
            f"of GROUP BY policy_type). "
            f"Rewrite: add an intermediate CTE that rolls up per-row metrics "
            f"to the category level, then compute summary metrics in the outer "
            f"SELECT with a GROUP BY on the category column only. "
            f"The final result should have one row per distinct category value "
            f"(typically < 50 rows for categorical columns)."
        ),
    )


# ---------------------------------------------------------------------------
# G. Closed execution loop (VALIDATION → EXECUTION → RESULT_CHECK)
# ---------------------------------------------------------------------------


@dataclass
class AttemptRecord:
    """Single retry attempt, stored for ERROR_CORRECT prompt injection."""

    attempt: int
    code: str
    validation_error: str | None = None
    execution_error: str | None = None


@dataclass
class LoopResult:
    """Outcome of the full ExecutionLoop.run() call."""

    success: bool
    execution_result: ExecutionResult | None = None
    runtime_validation: RuntimeValidationResult | None = None
    attempt_history: list[AttemptRecord] = field(default_factory=list)
    final_error_type: str | None = None
    final_error_message: str | None = None
    retry_count: int = 0
    # Populated when final_error_type == 'UNRESOLVED_COLUMN'.
    # Carries the bare column names for orchestrator fuzzy-match checks
    # without string parsing.
    unresolved_columns: list[str] = field(default_factory=list)
    # Non-blocking advisory from validate_sql (e.g. NON_SARGABLE_FILTER).
    # Surfaced in QueryResponse.warnings; does not affect success/failure.
    validation_warning: str | None = None


class ExecutionLoop:
    """
    Wires VALIDATION → EXECUTION → RESULT_CHECK into a retry-safe loop.

    State machine transitions handled here:
      VALIDATION → EXECUTION (if valid)
      VALIDATION → ERROR_CORRECT (if invalid, attempt < max_attempts)
      VALIDATION → TERMINAL_ERROR (if attempt >= max_attempts)
      EXECUTION → RESULT_CHECK (if no runtime error)
      EXECUTION → ERROR_CORRECT (if runtime error, attempt < max_attempts)
      RESULT_CHECK → INSIGHT caller (if result shape valid)
      RESULT_CHECK → INTAKE caller (if EMPTY_RESULT — loop does not retry this)

    The ERROR_CORRECT → GENERATION re-try is external to this class: the
    caller (orchestrator) calls loop.run() again with the corrected code string.
    This class tracks attempt count across calls via `attempt_history`.

    Parameters
    ----------
    max_attempts    : Hard retry cap. Default 3 (per spec).
    engine          : SQLAlchemy engine for SQL path.
    table_policies  : Policy registry for PreExecutionPolicy.
    sql_timeout     : Per-execution timeout for SQL.
    pandas_timeout  : Per-execution timeout for Pandas.
    memory_limit_mb : Pandas executor memory cap.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        engine: Engine | None = None,
        table_policies: dict[str, TablePolicy] | None = None,
        sql_timeout: int = 10,
        pandas_timeout: int = 15,
        memory_limit_mb: int = 512,
        error_correct_fn: Callable[[str, str, str], str | None] | None = None,
    ) -> None:
        self._max = max_attempts
        self._engine = engine
        self._policy = PreExecutionPolicy(table_policies or {})
        self._sql_timeout = sql_timeout
        self._pandas_timeout = pandas_timeout
        self._memory_limit = memory_limit_mb
        self._error_correct_fn = error_correct_fn
        self._attempt_history: list[AttemptRecord] = []

    @property
    def attempt_count(self) -> int:
        return len(self._attempt_history)

    def run(
        self,
        code: str,
        code_type: Literal["sql", "pandas"],
        schema_columns: set[str],
        expected_columns: list[str] | None = None,
        dataframe_refs: dict[str, pd.DataFrame] | None = None,
        dialect: str = "postgres",
        dry_run: bool = False,
    ) -> LoopResult:
        """
        Execute one attempt of the validation + execution + result-check pipeline.

        Returns immediately on:
          - Successful execution with valid result
          - TERMINAL_ERROR (attempt count exhausted)
          - POLICY_VIOLATION (not retryable)
          - EMPTY_RESULT (retrying the same query would return the same result —
            escalate to INTAKE for clarification instead)

        Returns LoopResult with success=False and final_error_* populated on
        retryable failures — the orchestrator injects these into ERROR_CORRECT
        and calls run() again with the corrected code.
        """
        if self.attempt_count >= self._max:
            return LoopResult(
                success=False,
                attempt_history=self._attempt_history,
                final_error_type="TERMINAL_ERROR",
                final_error_message=(
                    f"Maximum retry count ({self._max}) reached. "
                    f"Could not generate a valid, executable query."
                ),
                retry_count=self.attempt_count,
            )

        record = AttemptRecord(attempt=self.attempt_count + 1, code=code)

        # --- VALIDATION state ---
        if code_type == "sql":
            val_result = validate_sql(code, schema_columns, dialect=dialect)
        else:
            val_result = validate_python(
                code, schema_columns, dataframe_refs=set(dataframe_refs or {})
            )

        if not val_result.valid:
            record.validation_error = val_result.error_message
            self._attempt_history.append(record)
            failed_result = LoopResult(
                success=False,
                attempt_history=self._attempt_history,
                final_error_type=val_result.error_type,
                final_error_message=val_result.error_message,
                retry_count=self.attempt_count,
                unresolved_columns=val_result.unresolved_columns,
            )
            # Gap 5: UNRESOLVED_COLUMN previously broke out of the loop immediately,
            # bypassing error correction entirely.  When error_correct_fn is wired,
            # attempt one corrective pass before surfacing the failure.  The attempt
            # counter has already been incremented so the max-attempts guard at the
            # top of run() prevents unbounded recursion.
            if (
                val_result.error_type in ("UNRESOLVED_COLUMN", "NON_SARGABLE_FILTER")
                and self._error_correct_fn is not None
                and self.attempt_count < self._max
            ):
                corrected = self._error_correct_fn(
                    code,
                    val_result.error_type or "",
                    val_result.error_message or "",
                )
                if corrected is not None:
                    return self.run(
                        code=corrected,
                        code_type=code_type,
                        schema_columns=schema_columns,
                        expected_columns=expected_columns,
                        dataframe_refs=dataframe_refs,
                        dialect=dialect,
                        dry_run=dry_run,
                    )
            return failed_result

        # --- Pre-execution policy (SQL only; not retryable) ---
        if code_type == "sql":
            policy_result = self._policy.check(code, dialect=dialect)
            if not policy_result.valid:
                record.validation_error = policy_result.error_message
                self._attempt_history.append(record)
                return LoopResult(
                    success=False,
                    attempt_history=self._attempt_history,
                    final_error_type="POLICY_VIOLATION",
                    final_error_message=policy_result.error_message,
                    retry_count=self.attempt_count,
                )

        # --- EXECUTION state ---
        if dry_run:
            self._attempt_history.append(record)
            return LoopResult(
                success=True,
                execution_result=ExecutionResult(success=True),
                retry_count=self.attempt_count,
            )

        if code_type == "sql":
            assert self._engine is not None, "Engine required for SQL execution."
            exec_result = execute_sql(
                code, self._engine, timeout_seconds=self._sql_timeout
            )
        else:
            exec_result = execute_python(
                code,
                dataframe_refs=dataframe_refs or {},
                timeout_seconds=self._pandas_timeout,
                memory_limit_mb=self._memory_limit,
            )

        if not exec_result.success:
            record.execution_error = exec_result.error_message
            self._attempt_history.append(record)
            return LoopResult(
                success=False,
                attempt_history=self._attempt_history,
                final_error_type=exec_result.error_type,
                final_error_message=exec_result.error_message,
                retry_count=self.attempt_count,
                # Carry DB-level unresolved columns up to orchestrator fast-path
                unresolved_columns=exec_result.unresolved_columns,
            )

        # --- RESULT_CHECK state ---
        result_rows: list[dict[str, Any]]
        if code_type == "sql":
            result_rows = exec_result.result or []
        else:
            # Normalise Pandas output to list[dict] for validate_result
            df_result = exec_result.dataframe
            if isinstance(df_result, pd.DataFrame):
                result_rows = cast(
                    list[dict[str, Any]],
                    df_result.head(_ROW_CAP).to_dict(orient="records"),
                )
            elif isinstance(df_result, pd.Series):
                result_rows = cast(
                    list[dict[str, Any]],
                    df_result.reset_index().to_dict(orient="records"),
                )
            else:
                # Scalar result — wrap so validate_result sees a non-empty list
                result_rows = [{"result": df_result}]

            # P1-01 FIX: write the normalised rows back into exec_result.result.
            #
            # execute_python() returns ExecutionResult(result=None, dataframe=df)
            # because DataFrames cannot reliably serialise to list[dict] inside
            # the sandbox (memory, type-safety).  The normalisation above converts
            # them correctly for validate_result(), but the original exec_result
            # (with result=None) was then stored in every LoopResult return.
            #
            # Consequence: orchestrator._result_check() read exec_result.result
            # or [] → always got [] → validate_result([]) → EMPTY_RESULT →
            # every successful Pandas query redirected to INTAKE as clarification.
            # DataFrameStore wiring was correct; the normalised rows simply never
            # reached the orchestrator.
            #
            # Fix: use model_copy(update=...) (Pydantic v2, zero-copy of unchanged
            # fields) to produce an updated ExecutionResult where result holds the
            # normalised rows. All LoopResult branches below now carry this object.
            exec_result = exec_result.model_copy(
                update={
                    "result": result_rows,
                    "row_count": len(result_rows),
                    "columns": list(result_rows[0].keys()) if result_rows else [],
                }
            )

        rt_result = validate_result(result_rows, expected_columns=expected_columns)
        self._attempt_history.append(record)

        if not rt_result.valid and rt_result.issue == "EMPTY_RESULT":
            # Do not retry — escalate to INTAKE for clarification
            return LoopResult(
                success=False,
                execution_result=exec_result,
                runtime_validation=rt_result,
                attempt_history=self._attempt_history,
                final_error_type="EMPTY_RESULT",
                final_error_message=rt_result.message,
                retry_count=self.attempt_count,
            )

        if not rt_result.valid:
            return LoopResult(
                success=False,
                execution_result=exec_result,
                runtime_validation=rt_result,
                attempt_history=self._attempt_history,
                final_error_type=rt_result.issue,
                final_error_message=rt_result.message,
                retry_count=self.attempt_count,
            )

        return LoopResult(
            success=True,
            execution_result=exec_result,
            runtime_validation=rt_result,
            attempt_history=self._attempt_history,
            retry_count=self.attempt_count,
            # Carry any non-blocking advisory (e.g. NON_SARGABLE_FILTER)
            # up to the orchestrator so it appears in QueryResponse.warnings.
            validation_warning=getattr(val_result, "warning", None),
        )
