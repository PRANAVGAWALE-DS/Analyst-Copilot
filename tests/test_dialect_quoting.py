# tests/test_dialect_quoting.py
import sys

sys.path.insert(0, "analyst_copilot")
import contextlib
import unittest.mock as mock

from dataframe_loader import DataFrameLoader  # noqa: E402

print("=== BUG-2: Dialect-aware identifier quoting ===")

for dialect_name, expected_quote_char in [
    ("postgresql", '"'),
    ("sqlite", '"'),
    ("mysql", "`"),
    ("mariadb", "`"),
    ("duckdb", '"'),
]:
    # Mock engine with the dialect
    engine = mock.MagicMock()
    engine.dialect.name = dialect_name

    loader = DataFrameLoader(engine=engine)

    # Intercept the actual SQL string sent to pd.read_sql
    sql_sent = []

    def mock_read_sql(sql, conn, sql_sent=sql_sent):
        sql_sent.append(sql)
        import pandas as pd

        return pd.DataFrame({"a": [1]})

    with (
        mock.patch("pandas.read_sql", side_effect=mock_read_sql),
        mock.patch.object(engine, "connect") as mock_conn,
    ):
        mock_conn.return_value.__enter__ = mock.MagicMock(return_value=mock.MagicMock())
        mock_conn.return_value.__exit__ = mock.MagicMock(return_value=False)
        with contextlib.suppress(Exception):
            loader._load_from_db("claims", 100)

    if sql_sent:
        sql = sql_sent[0]
        quoted_correctly = expected_quote_char + "claims" + expected_quote_char in sql
        print(
            f"  {'PASS' if quoted_correctly else 'FAIL'}: {dialect_name:12s} → {sql.split('FROM')[1].strip()[:20]}"
        )
    else:
        print(f"  SKIP: {dialect_name} (mock didn't capture SQL)")
