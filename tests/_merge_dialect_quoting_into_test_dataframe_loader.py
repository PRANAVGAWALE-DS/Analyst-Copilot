"""
MERGE INSTRUCTIONS
──────────────────
Append the class below to tests/test_dataframe_loader.py.
Replaces tests/test_dialect_quoting.py (deleted).

Tests the BUG-2 fix in DataFrameLoader._load_from_db():
  MySQL / MariaDB  → backtick quoting  (`table`)
  All others       → ANSI double-quote ("table")

Additional imports needed at the top of test_dataframe_loader.py (if not already there):
  from unittest.mock import MagicMock, patch
  import pandas as pd
  from analyst_copilot.dataframe_loader import DataFrameLoader
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from analyst_copilot.dataframe_loader import DataFrameLoader


class TestDialectQuotingBug2:
    """
    BUG-2 fix: _load_from_db() must use dialect-aware identifier quoting.

    Before the fix, hardcoded double-quotes caused MySQL to interpret the
    identifier as a string alias, silently returning wrong results.
    """

    def _make_engine(self, dialect_name: str) -> MagicMock:
        """Return a minimal mock SQLAlchemy engine with the given dialect name."""
        engine = MagicMock()
        engine.dialect.name = dialect_name
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        return engine

    def _captured_sql(self, dialect_name: str, table: str = "policies") -> str:
        """Run _load_from_db() with a mocked engine and return the SQL string used."""
        engine = self._make_engine(dialect_name)
        empty_df = pd.DataFrame(columns=["id"])
        with patch(
            "analyst_copilot.dataframe_loader.pd.read_sql",
            return_value=empty_df,
        ) as mock_read:
            loader = DataFrameLoader(engine)
            loader._load_from_db(table, limit=100)
            return mock_read.call_args[0][0]

    # ── Double-quote dialects ─────────────────────────────────────────────────

    def test_postgres_uses_double_quotes(self):
        sql = self._captured_sql("postgresql")
        assert '"policies"' in sql
        assert "`policies`" not in sql

    def test_sqlite_uses_double_quotes(self):
        sql = self._captured_sql("sqlite")
        assert '"policies"' in sql

    def test_unknown_dialect_fallback_double_quotes(self):
        sql = self._captured_sql("oracle")
        assert '"policies"' in sql
        assert "`policies`" not in sql

    # ── Backtick dialects ────────────────────────────────────────────────────

    def test_mysql_uses_backticks(self):
        sql = self._captured_sql("mysql")
        assert "`policies`" in sql
        assert '"policies"' not in sql

    def test_mariadb_uses_backticks(self):
        sql = self._captured_sql("mariadb")
        assert "`policies`" in sql
        assert '"policies"' not in sql

    # ── LIMIT clause always present ──────────────────────────────────────────

    def test_limit_clause_present_postgres(self):
        sql = self._captured_sql("postgresql", "claims")
        assert "LIMIT" in sql.upper()

    def test_limit_clause_present_mysql(self):
        sql = self._captured_sql("mysql", "claims")
        assert "LIMIT" in sql.upper()

    # ── Table name injection check (regression) ──────────────────────────────

    def test_table_name_embedded_in_sql(self):
        """The correct table name must appear in the generated SQL."""
        sql = self._captured_sql("postgresql", "claim_events")
        assert "claim_events" in sql
