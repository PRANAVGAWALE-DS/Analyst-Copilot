"""
MERGE INSTRUCTIONS
──────────────────
Append this class to tests/test_prompts.py.
Replaces tests/test_budget_warning_ml3.py (deleted).

Fixes applied vs the original:
  • Removed module-level sys.exit(1) — used pytest.fail() instead.
  • Changed stderr assertion to stdout — prompts.py:1181 emits to sys.stdout.
  • Wrapped in a proper pytest class.

Additional imports needed at the top of test_prompts.py (if not already there):
  import json
  from analyst_copilot.interfaces import SchemaChunk, SchemaColumn
  from analyst_copilot.prompts import enforce_token_budget, TOKEN_BUDGET
"""

import json

from analyst_copilot.interfaces import SchemaChunk, SchemaColumn
from analyst_copilot.prompts import TOKEN_BUDGET, enforce_token_budget


class TestBudgetWarningML3:
    """ML-3: BUDGET_WARNING emitted to stdout when column descriptions are stripped."""

    @staticmethod
    def _large_chunk(schema_id: str = "test_schema", n_cols: int = 60) -> SchemaChunk:
        """Build a single chunk whose column descriptions push it over TOKEN_BUDGET."""
        return SchemaChunk(
            table="wide_table",
            schema_id=schema_id,
            columns=[
                SchemaColumn(
                    name=f"col_{i}",
                    type="TEXT",
                    nullable=True,
                    description=f"Column {i}: a moderately long annotation. " * 4,
                )
                for i in range(n_cols)
            ],
        )

    @staticmethod
    def _fill_history(n: int = 10) -> list[dict]:
        return [
            {
                "nl_query": "query",
                "generated_code": "SELECT 1",
                "insight": "insight text " * 50,
            }
        ] * n

    def test_budget_warning_goes_to_stdout_not_stderr(self, capsys):
        """BUDGET_WARNING must be on stdout — ML-3 fix (was previously missing)."""
        chunks = [self._large_chunk(n_cols=60)] * 6
        history = self._fill_history(10)

        enforce_token_budget(chunks, history)

        out, err = capsys.readouterr()
        assert "BUDGET_WARNING" in out, (
            "BUDGET_WARNING not found in stdout. "
            "Check prompts.py enforce_token_budget Step 1b — event must use file=sys.stdout."
        )
        assert (
            "BUDGET_WARNING" not in err
        ), "BUDGET_WARNING found in stderr — it must go to stdout for log-aggregator visibility."

    def test_budget_warning_event_structure(self, capsys):
        """BUDGET_WARNING JSON must carry stripped_columns_by_schema and budget fields."""
        chunks = [self._large_chunk("test_schema", n_cols=60)] * 6
        history = self._fill_history(10)

        enforce_token_budget(chunks, history)

        out, _ = capsys.readouterr()
        warning_lines = [line for line in out.splitlines() if "BUDGET_WARNING" in line]
        assert warning_lines, "No BUDGET_WARNING line found in stdout"

        event = json.loads(warning_lines[0])
        assert event["event"] == "BUDGET_WARNING"
        assert "stripped_columns_by_schema" in event
        assert "budget" in event
        assert event["budget"] == TOKEN_BUDGET["total"]

    def test_stripped_columns_schema_key_present(self, capsys):
        chunks = [self._large_chunk("test_schema", n_cols=60)] * 6
        history = self._fill_history(10)

        enforce_token_budget(chunks, history)

        out, _ = capsys.readouterr()
        warning_lines = [line for line in out.splitlines() if "BUDGET_WARNING" in line]
        assert warning_lines
        event = json.loads(warning_lines[0])
        assert "test_schema" in event["stripped_columns_by_schema"]
        assert len(event["stripped_columns_by_schema"]["test_schema"]) > 0

    def test_column_names_preserved_after_description_strip(self):
        """Column names must survive after descriptions are stripped."""
        chunks = [self._large_chunk("test_schema", n_cols=60)] * 6
        history = self._fill_history(10)

        trimmed_chunks, _, _ = enforce_token_budget(chunks, history)

        all_names = {col.name for c in trimmed_chunks for col in c.columns}
        for i in range(60):
            assert f"col_{i}" in all_names, f"col_{i} missing after budget trim"

    def test_no_budget_warning_for_small_context(self, capsys):
        """A context well within the 10k-token budget must not emit BUDGET_WARNING."""
        small = SchemaChunk(
            table="policies",
            schema_id="s",
            columns=[SchemaColumn(name="id", type="INT", nullable=False)],
        )
        enforce_token_budget([small], [])
        out, _ = capsys.readouterr()
        assert "BUDGET_WARNING" not in out
