"""
tests/test_validation.py
Unit tests for validation.py — written against the actual module.

Key differences from sandbox assumptions:
  - _ForbiddenNodeVisitor class (not a simple loop)
  - ALLOWED_IMPORTS allowlist blocks unlisted imports
  - RESULT_ASSIGN_MISSING is a distinct error_type
  - select_aliases excluded from UNRESOLVED_COLUMN check
  - validate_result(expected_columns=None) skips shape check
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import pytest
from validation import (
    PreExecutionPolicy,
    TablePolicy,
    validate_python,
    validate_result,
    validate_sql,
)

SCHEMA_COLUMNS = {
    "claim_id",
    "policy_type",
    "claim_amount",
    "claim_date",
    "customer_id",
}


class TestValidatePython:
    def test_valid_code_passes(self) -> None:
        r = validate_python("result = df['claim_amount'].mean()", SCHEMA_COLUMNS)
        assert r.valid is True

    def test_syntax_error(self) -> None:
        r = validate_python("result = df['claim_amount'..mean()", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "SYNTAX_ERROR"
        assert r.error_line is not None

    def test_forbidden_import_os(self) -> None:
        r = validate_python("import os\nresult = os.getcwd()", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_non_allowlisted_import_blocked(self) -> None:
        # requests is NOT in _ALLOWED_IMPORTS
        r = validate_python(
            "import requests\nresult = requests.get('http://x.com').text",
            SCHEMA_COLUMNS,
        )
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_allowed_pandas_import_passes(self) -> None:
        r = validate_python(
            "import pandas as pd\nresult = pd.DataFrame({'a': [1]})", SCHEMA_COLUMNS
        )
        assert r.valid is True

    def test_forbidden_builtin_eval(self) -> None:
        r = validate_python("result = eval('1+1')", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_forbidden_builtin_open(self) -> None:
        r = validate_python("f = open('/etc/passwd')\nresult = f.read()", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_dunder_subclasses_blocked(self) -> None:
        r = validate_python("result = ().__class__.__bases__[0].__subclasses__()", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_missing_result_assignment(self) -> None:
        r = validate_python("x = df['claim_amount'].sum()", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "RESULT_ASSIGN_MISSING"

    def test_annotated_result_assign_counts(self) -> None:
        r = validate_python("result: float = df['claim_amount'].mean()", SCHEMA_COLUMNS)
        assert r.valid is True

    def test_unknown_column_flagged(self) -> None:
        r = validate_python("result = df['completely_unknown_col'].sum()", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "UNRESOLVED_COLUMN"

    def test_known_column_passes(self) -> None:
        r = validate_python("result = df['claim_amount'].sum()", SCHEMA_COLUMNS)
        assert r.valid is True


class TestValidateSQL:
    def test_valid_aggregation(self) -> None:
        sql = (
            "SELECT policy_type, AVG(claim_amount) AS avg_claim "
            "FROM claims WHERE claim_date >= '2024-01-01' "
            "GROUP BY policy_type LIMIT 1000"
        )
        assert validate_sql(sql, SCHEMA_COLUMNS).valid is True

    def test_syntax_error(self) -> None:
        r = validate_sql("SELEKT policy_type FORM claims", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "SYNTAX_ERROR"

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO claims (policy_type) VALUES ('auto')",
            "UPDATE claims SET claim_amount = 0 WHERE claim_id = '1'",
            "DELETE FROM claims WHERE claim_date < '2020-01-01'",
            "DROP TABLE claims",
        ],
    )
    def test_mutation_blocked(self, sql: str) -> None:
        r = validate_sql(sql, SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "MUTATION_STATEMENT"

    def test_unresolved_column(self) -> None:
        r = validate_sql("SELECT hallucinated_col FROM claims LIMIT 10", SCHEMA_COLUMNS)
        assert r.valid is False
        assert r.error_type == "UNRESOLVED_COLUMN"
        assert "hallucinated_col" in (r.error_message or "")

    def test_select_alias_not_flagged(self) -> None:
        sql = (
            "SELECT policy_type, COUNT(*) AS total_claims "
            "FROM claims GROUP BY policy_type "
            "ORDER BY total_claims DESC LIMIT 100"
        )
        assert validate_sql(sql, SCHEMA_COLUMNS).valid is True

    def test_empty_schema_runs_column_check(self) -> None:
        # validate_sql() has no guard on empty schema_columns — it always
        # checks referenced columns against the (empty) set, so any column
        # reference is flagged UNRESOLVED_COLUMN. This is intentional:
        # callers must pass a non-empty schema_columns or skip calling validate_sql.
        r = validate_sql("SELECT anything FROM anywhere LIMIT 10", set())
        assert r.valid is False
        assert r.error_type == "UNRESOLVED_COLUMN"


class TestPreExecutionPolicy:
    def test_passes_clean_query(self) -> None:
        policy = PreExecutionPolicy({"claims": TablePolicy("claims", pii_flagged=False)})
        assert policy.check("SELECT policy_type FROM claims LIMIT 100").valid is True

    def test_blocks_limitless_pii_select(self) -> None:
        policy = PreExecutionPolicy({"customers": TablePolicy("customers", pii_flagged=True)})
        r = policy.check("SELECT customer_id FROM customers")
        assert r.valid is False
        assert r.error_type == "POLICY_VIOLATION"

    def test_allows_limited_pii_select(self) -> None:
        policy = PreExecutionPolicy({"customers": TablePolicy("customers", pii_flagged=True)})
        assert policy.check("SELECT customer_id FROM customers LIMIT 100").valid is True

    def test_blocks_blocked_table(self) -> None:
        policy = PreExecutionPolicy(
            {"audit_log": TablePolicy("audit_log", pii_flagged=False, access_level="blocked")}
        )
        r = policy.check("SELECT * FROM audit_log LIMIT 10")
        assert r.valid is False
        assert r.error_type == "POLICY_VIOLATION"

    def test_no_policy_passes(self) -> None:
        policy = PreExecutionPolicy({})
        assert policy.check("SELECT policy_type FROM claims LIMIT 10").valid is True


class TestValidateResult:
    def test_valid(self) -> None:
        r = validate_result([{"policy_type": "auto", "avg_claim": 1500.0}])
        assert r.valid is True

    def test_empty(self) -> None:
        r = validate_result([])
        assert r.valid is False
        assert r.issue == "EMPTY_RESULT"

    def test_shape_mismatch(self) -> None:
        rows = [{"policy_type": "auto", "wrong_col": 1500.0}]
        r = validate_result(rows, expected_columns=["policy_type", "avg_claim"])
        assert r.valid is False
        assert r.issue == "SHAPE_MISMATCH"

    def test_no_shape_check_when_none(self) -> None:
        assert validate_result([{"anything": 1}], expected_columns=None).valid is True

    def test_result_capped(self) -> None:
        rows = [{"id": i} for i in range(10_001)]
        r = validate_result(rows, row_cap=10_000)
        assert r.valid is True
        assert r.issue == "RESULT_CAPPED"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
