"""
tests/test_pandas_executor.py
Tests for the sandboxed Python executor.

Exact function (validation.py line 720):
  execute_python(
      code: str,
      dataframe_refs: dict[str, pd.DataFrame],
      timeout_seconds: int = 15,
      memory_limit_mb: int = 512,
  ) -> ExecutionResult

ExecutionResult fields (validation.py line 73):
  success:          bool
  dataframe:        Any   — the `result` variable from exec scope
  error_type:       str | None
  error_message:    str | None
  execution_time_ms: int

Security model (four layers, validation.py lines 723-730):
  1. validate_python() must pass before execute_python() is called.
  2. exec() receives only whitelisted builtins.
  3. Allowed modules are pre-seeded — code cannot import new ones.
  4. tracemalloc monitors peak memory; ThreadPoolExecutor enforces timeout.

Test strategy: validate_python() is called first in every security test,
because the real execution path always calls it before execute_python().
Tests where validate_python() blocks the code verify the first line of
defence. Tests marked "sandbox" exercise execute_python() directly to
verify the runtime namespace restrictions even if validation is bypassed.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import pandas as pd
import pytest
from validation import (
    execute_python,
    validate_python,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def claims_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_type": ["auto", "home", "life", "auto", "home"],
            "claim_amount": [1200.0, 3500.0, 800.0, 2100.0, 4200.0],
            "claim_date": pd.to_datetime(
                ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-18"]
            ),
            "customer_id": [101, 102, 103, 101, 104],
        }
    )


SCHEMA = {"policy_type", "claim_amount", "claim_date", "customer_id"}


# ── Validation layer blocks forbidden code BEFORE execution ──────────────────


class TestValidationBlocksForbiddenCode:
    """
    These tests confirm that validate_python() — the first line of defence —
    catches forbidden patterns before execute_python() is ever called.
    """

    def test_import_os_blocked_at_validation(self) -> None:
        code = "import os\nresult = os.listdir('.')"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_import_subprocess_blocked_at_validation(self) -> None:
        code = "import subprocess\nresult = subprocess.check_output(['whoami'])"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_from_os_import_blocked(self) -> None:
        code = "from os.path import join\nresult = join('a', 'b')"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_import_socket_blocked(self) -> None:
        code = "import socket\nresult = socket.gethostname()"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_import_requests_blocked(self) -> None:
        code = "import requests\nresult = requests.get('http://evil.com').text"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_IMPORT"

    def test_eval_blocked_at_validation(self) -> None:
        code = 'result = eval(\'__import__("os").system("ls")\')'
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_exec_blocked_at_validation(self) -> None:
        code = "exec('import os')\nresult = 1"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_open_blocked_at_validation(self) -> None:
        code = "f = open('/etc/passwd')\nresult = f.read()"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_dunder_subclasses_blocked(self) -> None:
        code = "result = ().__class__.__bases__[0].__subclasses__()"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "FORBIDDEN_BUILTIN"

    def test_missing_result_assignment_caught(self) -> None:
        code = "x = claims_df['claim_amount'].mean()"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "RESULT_ASSIGN_MISSING"

    def test_syntax_error_caught(self) -> None:
        code = "result = claims_df['claim_amount'..mean()"
        r = validate_python(code, SCHEMA)
        assert r.valid is False
        assert r.error_type == "SYNTAX_ERROR"
        assert r.error_line is not None


# ── Sandbox runtime blocks imports not in the pre-seeded namespace ────────────


class TestSandboxRuntime:
    """
    These tests call execute_python() directly (bypassing validate_python())
    to verify the exec() namespace restrictions are enforced at runtime.
    This is the second layer of defence.
    """

    def test_os_not_in_namespace(self, claims_df: pd.DataFrame) -> None:
        # os is not pre-seeded — NameError expected
        code = "result = os.listdir('.')"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is False
        assert r.error_type is not None

    def test_open_not_in_builtins(self, claims_df: pd.DataFrame) -> None:
        code = "result = open('/etc/passwd').read()"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is False

    def test_import_inside_exec_blocked(self, claims_df: pd.DataFrame) -> None:
        # __import__ is not in _SAFE_BUILTINS — the call fails at runtime
        code = "__import__('os').system('ls')\nresult = 'done'"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is False


# ── Correct execution — pre-seeded modules work as expected ──────────────────


class TestCorrectExecution:
    def test_simple_mean(self, claims_df: pd.DataFrame) -> None:
        code = "result = claims_df['claim_amount'].mean()"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert abs(r.dataframe - 2360.0) < 0.01

    def test_groupby_aggregation(self, claims_df: pd.DataFrame) -> None:
        code = "result = claims_df.groupby('policy_type')['claim_amount']" ".mean().reset_index()"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert isinstance(r.dataframe, pd.DataFrame)
        assert len(r.dataframe) == 3  # auto, home, life

    def test_numpy_available(self, claims_df: pd.DataFrame) -> None:
        code = "result = np.percentile(claims_df['claim_amount'], 75)"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert isinstance(r.dataframe, float)

    def test_math_module_available(self, claims_df: pd.DataFrame) -> None:
        code = "result = math.floor(claims_df['claim_amount'].mean())"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert r.dataframe == 2360

    def test_multiple_dataframe_refs(self, claims_df: pd.DataFrame) -> None:
        policies_df = pd.DataFrame(
            {
                "policy_type": ["auto", "home", "life"],
                "max_payout": [50000, 200000, 500000],
            }
        )
        code = (
            "merged = claims_df.merge(policies_df, on='policy_type')\n"
            "result = merged[['policy_type', 'claim_amount', 'max_payout']]"
        )
        r = execute_python(code, {"claims_df": claims_df, "policies_df": policies_df})
        assert r.success is True
        assert "max_payout" in r.dataframe.columns

    def test_result_type_dataframe(self, claims_df: pd.DataFrame) -> None:
        code = "result = claims_df[claims_df['claim_amount'] > 2000]"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert isinstance(r.dataframe, pd.DataFrame)

    def test_result_type_series(self, claims_df: pd.DataFrame) -> None:
        code = "result = claims_df['claim_amount']"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert isinstance(r.dataframe, pd.Series)

    def test_result_type_scalar_int(self, claims_df: pd.DataFrame) -> None:
        code = "result = len(claims_df)"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert r.dataframe == 5

    def test_execution_time_is_recorded(self, claims_df: pd.DataFrame) -> None:
        code = "result = claims_df['claim_amount'].sum()"
        r = execute_python(code, {"claims_df": claims_df})
        assert r.execution_time_ms >= 0

    def test_unknown_dataframe_ref_raises_name_error(self) -> None:
        code = "result = nonexistent_df['col'].mean()"
        r = execute_python(code, {})
        assert r.success is False
        # NameError because 'nonexistent_df' is not in the exec namespace
        assert r.error_type is not None

    def test_empty_dataframe_ref_dict_works(self) -> None:
        code = "result = pd.DataFrame({'a': [1, 2, 3]})"
        r = execute_python(code, {})
        assert r.success is True
        assert list(r.dataframe.columns) == ["a"]

    def test_re_module_available(self, claims_df: pd.DataFrame) -> None:
        code = (
            "mask = claims_df['policy_type'].str.match(re.compile(r'^auto$'))\n"
            "result = claims_df[mask]"
        )
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert len(r.dataframe) == 2  # two 'auto' rows

    def test_collections_module_available(self, claims_df: pd.DataFrame) -> None:
        code = (
            "counts = collections.Counter(claims_df['policy_type'].tolist())\n"
            "result = dict(counts)"
        )
        r = execute_python(code, {"claims_df": claims_df})
        assert r.success is True
        assert r.dataframe["auto"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
