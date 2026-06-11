"""
MERGE INSTRUCTIONS
──────────────────
Append BOTH classes below to tests/test_validation.py.
  • TestSandboxExecutorSingleton replaces tests/test_singleton_executor.py (deleted).
  • TestPandasValidationML2    replaces tests/test_pandas_validation_ml2.py (deleted).

Additional imports needed at the top of test_validation.py (if not already there):
  import concurrent.futures
  from analyst_copilot.validation import (
      _SANDBOX_EXECUTOR,
      validate_python,
      ValidationResult,
  )
"""

from __future__ import annotations

import concurrent.futures

from analyst_copilot.validation import (
    _SANDBOX_EXECUTOR,
    validate_python,
)

# ---------------------------------------------------------------------------
# Merge from test_singleton_executor.py — BUG-6 fix verification
# ---------------------------------------------------------------------------


class TestSandboxExecutorSingleton:
    """
    _SANDBOX_EXECUTOR must be a module-level ThreadPoolExecutor singleton.

    BUG-6 fix: previously `execute_sql` / `execute_python` created a new
    ThreadPoolExecutor on every call, leaking threads on every query.
    The module-level singleton prevents that.
    """

    def test_is_thread_pool_executor(self):
        assert isinstance(_SANDBOX_EXECUTOR, concurrent.futures.ThreadPoolExecutor)

    def test_is_module_level_singleton(self):
        """Re-importing the module must return the identical object."""
        from analyst_copilot import validation as v1
        from analyst_copilot import validation as v2

        assert v1._SANDBOX_EXECUTOR is v2._SANDBOX_EXECUTOR


# ---------------------------------------------------------------------------
# Merge from test_pandas_validation_ml2.py — validate_python stage coverage
# ---------------------------------------------------------------------------


class TestPandasValidationML2:
    """
    validate_python() three-stage validation:
      Stage 1 — AST parse (syntax check)
      Stage 2 — AST visitor (forbidden imports, forbidden builtins, dunder)
      Stage 3 — result= assignment contract
    """

    _COLS: set[str] = {"policy_id", "premium_amt", "claim_amount"}

    # ── Stage 1: syntax errors ───────────────────────────────────────────────

    def test_syntax_error_returns_invalid(self):
        vr = validate_python("def broken(:\n    pass\n", self._COLS)
        assert vr.valid is False
        assert vr.error_type == "SYNTAX_ERROR"

    # ── Stage 2: forbidden imports ───────────────────────────────────────────

    def test_os_import_blocked(self):
        code = "import os\nresult = os.listdir('.')\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False
        assert vr.error_type == "FORBIDDEN_IMPORT"

    def test_subprocess_import_blocked(self):
        code = "import subprocess\nresult = subprocess.run(['ls'])\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False
        assert vr.error_type == "FORBIDDEN_IMPORT"

    def test_sys_import_blocked(self):
        code = "import sys\nresult = sys.version\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False
        assert vr.error_type == "FORBIDDEN_IMPORT"

    def test_socket_import_blocked(self):
        code = "import socket\nresult = socket.gethostname()\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False
        assert vr.error_type == "FORBIDDEN_IMPORT"

    def test_allowed_pandas_import_passes(self):
        code = "import pandas as pd\nresult = pd.DataFrame()\n"
        vr = validate_python(code, self._COLS)
        # Pandas is allowed — stage 2 must not block it
        assert vr.error_type != "FORBIDDEN_IMPORT"

    def test_allowed_numpy_import_passes(self):
        code = "import numpy as np\nresult = np.array([1, 2, 3])\n"
        vr = validate_python(code, self._COLS)
        assert vr.error_type != "FORBIDDEN_IMPORT"

    # ── Stage 2: forbidden builtins ──────────────────────────────────────────

    def test_eval_call_blocked(self):
        code = "result = eval('1+1')\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False

    def test_exec_call_blocked(self):
        code = "exec('x=1')\nresult = 1\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False

    def test_open_call_blocked(self):
        code = "f = open('/etc/passwd')\nresult = f.read()\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False

    # ── Stage 3: result= assignment contract ─────────────────────────────────

    def test_missing_result_assignment_blocked(self):
        code = "x = df['policy_id'].sum()\n"
        vr = validate_python(code, self._COLS)
        assert vr.valid is False
        assert vr.error_type == "RESULT_ASSIGN_MISSING"

    def test_result_assignment_present_passes_stage3(self):
        code = "result = df['premium_amt'].sum()\n"
        vr = validate_python(code, self._COLS, dataframe_refs={"df"})
        # May fail column grounding for unknown df contents, but not stage 3
        assert vr.error_type != "RESULT_ASSIGN_MISSING"

    # ── Valid code ────────────────────────────────────────────────────────────

    def test_valid_pandas_aggregation_passes_all_stages(self):
        code = "total = df['premium_amt'].sum()\n" "result = total\n"
        vr = validate_python(code, self._COLS, dataframe_refs={"df"})
        # All three stages pass for clean Pandas code
        assert vr.error_type not in (
            "SYNTAX_ERROR",
            "FORBIDDEN_IMPORT",
            "RESULT_ASSIGN_MISSING",
        )
