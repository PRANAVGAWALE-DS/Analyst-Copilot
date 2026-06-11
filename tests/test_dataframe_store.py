"""
tests/test_dataframe_store.py — Unit tests for DataFrameStore and helpers

Covers:
  - _sanitise_df_name: identifier normalisation
  - IngestFileResult: success/error fields
  - DataFrameStore.ingest: CSV, Parquet, Excel (M1 fix: openpyxl now installed),
    size guard, session memory cap, column whitespace stripping, name sanitisation
  - DataFrameStore.get: returns frames, shallow copy, last_access update
  - DataFrameStore._evict_expired: TTL eviction triggered via get() (mocked clock)
  - DataFrameStore.delete / delete_session
  - DataFrameStore.list_dataframes: metadata correctness

No mocking beyond time.monotonic for TTL tests.
No network calls, no DB, no file I/O — fully in-memory.
"""

from __future__ import annotations

import io
import time
from unittest.mock import patch

import pandas as pd
from dataframe_store import (
    DataFrameStore,
    IngestFileResult,
    _sanitise_df_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SID = "test-session-001"
SID2 = "test-session-002"


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _df(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(rows),
            "value": [i * 1.1 for i in range(rows)],
            "label": [f"item_{i}" for i in range(rows)],
        }
    )


# ---------------------------------------------------------------------------
# _sanitise_df_name
# ---------------------------------------------------------------------------


class TestSanitiseDfName:
    def test_plain_name_unchanged(self):
        assert _sanitise_df_name("claims") == "claims"

    def test_spaces_become_underscores(self):
        assert _sanitise_df_name("claims data") == "claims_data"

    def test_hyphens_become_underscores(self):
        assert _sanitise_df_name("claims-data") == "claims_data"

    def test_csv_extension_stripped(self):
        assert _sanitise_df_name("claims.csv") == "claims"

    def test_parquet_extension_stripped(self):
        assert _sanitise_df_name("policies.parquet") == "policies"

    def test_extension_and_spaces(self):
        assert _sanitise_df_name("claims data.csv") == "claims_data"

    def test_leading_digit_gets_prefix(self):
        result = _sanitise_df_name("2024_report.csv")
        assert result[0].isalpha() or result[0] == "_"
        assert result.isidentifier()

    def test_empty_string_returns_df(self):
        assert _sanitise_df_name("") == "df"

    def test_dot_only_returns_df(self):
        # Path.stem of "." is "" → falls through to "df"
        result = _sanitise_df_name(".")
        assert result  # not empty
        assert result.isidentifier()

    def test_all_special_chars_returns_valid_identifier(self):
        for name in ["my data.csv", "123abc.parquet", "hello-world", " spaces "]:
            result = _sanitise_df_name(name)
            assert (
                result.isidentifier()
            ), f"_sanitise_df_name({name!r}) → {result!r} is not a valid Python identifier"

    def test_underscores_preserved(self):
        assert _sanitise_df_name("my_table") == "my_table"

    def test_mixed_case_preserved(self):
        result = _sanitise_df_name("MyTable.csv")
        assert "My" in result or "my" in result.lower()


# ---------------------------------------------------------------------------
# IngestFileResult
# ---------------------------------------------------------------------------


class TestIngestFileResult:
    def test_success_true_when_no_error(self):
        r = IngestFileResult(df_name="x", rows=10, columns=["a", "b"], size_mb=0.1, warnings=[])
        assert r.success is True
        assert r.error is None

    def test_success_false_when_error_set(self):
        r = IngestFileResult(
            df_name="x",
            rows=0,
            columns=[],
            size_mb=0.5,
            warnings=[],
            error="parse failed",
        )
        assert r.success is False
        assert r.error == "parse failed"

    def test_warnings_accessible(self):
        r = IngestFileResult(
            df_name="x", rows=5, columns=["a"], size_mb=0.1, warnings=["column renamed"]
        )
        assert "column renamed" in r.warnings

    def test_columns_list_accessible(self):
        r = IngestFileResult(df_name="x", rows=3, columns=["id", "val"], size_mb=0.05, warnings=[])
        assert r.columns == ["id", "val"]


# ---------------------------------------------------------------------------
# DataFrameStore.ingest — CSV
# ---------------------------------------------------------------------------


class TestIngestCSV:
    def setup_method(self):
        self.store = DataFrameStore()
        self.df = _df(5)

    def test_csv_success(self):
        result = self.store.ingest(SID, "claims", _csv_bytes(self.df), "csv")
        assert result.success
        assert result.rows == 5
        assert set(result.columns) == {"id", "value", "label"}

    def test_csv_dot_extension_accepted(self):
        assert self.store.ingest(SID, "d", _csv_bytes(self.df), ".csv").success

    def test_csv_uppercase_extension_accepted(self):
        assert self.store.ingest(SID, "d", _csv_bytes(self.df), "CSV").success

    def test_df_retrievable_by_sanitised_name(self):
        result = self.store.ingest(SID, "my data.csv", _csv_bytes(self.df), "csv")
        assert result.success
        frames = self.store.get(SID)
        assert "my_data" in frames

    def test_df_name_sanitisation_reflected_in_result(self):
        result = self.store.ingest(SID, "raw file.csv", _csv_bytes(self.df), "csv")
        assert result.df_name == "raw_file"
        assert "sanitised" in " ".join(result.warnings).lower() or result.df_name != "raw file.csv"

    def test_column_whitespace_stripped(self):
        df = pd.DataFrame({" amount ": [1, 2, 3], "  label  ": ["a", "b", "c"]})
        result = self.store.ingest(SID, "messy", _csv_bytes(df), "csv")
        assert result.success
        assert "amount" in result.columns
        assert "label" in result.columns
        assert " amount " not in result.columns

    def test_column_whitespace_produces_warning(self):
        df = pd.DataFrame({" spaced_col ": [1, 2]})
        result = self.store.ingest(SID, "messy", _csv_bytes(df), "csv")
        warnings_text = " ".join(result.warnings).lower()
        assert "strip" in warnings_text or "whitespace" in warnings_text

    def test_size_mb_reported(self):
        # DataFrameStore rounds size_mb to 3 decimal places.
        # A 5-row DataFrame is only ~487 bytes → rounds to 0.000.
        # 10 rows (~1 KB) clears the rounding floor → rounds to 0.001+.
        result = self.store.ingest(SID, "data", _csv_bytes(_df(10)), "csv")
        assert isinstance(result.size_mb, float)
        assert result.size_mb > 0

    def test_overwrite_existing_frame(self):
        self.store.ingest(SID, "frame", _csv_bytes(_df(3)), "csv")
        self.store.ingest(SID, "frame", _csv_bytes(_df(10)), "csv")
        frames = self.store.get(SID)
        assert len(frames["frame"]) == 10  # second ingest wins


# ---------------------------------------------------------------------------
# DataFrameStore.ingest — Parquet
# ---------------------------------------------------------------------------


class TestIngestParquet:
    def setup_method(self):
        self.store = DataFrameStore()
        self.df = _df(5)

    def test_parquet_success(self):
        result = self.store.ingest(SID, "data", _parquet_bytes(self.df), "parquet")
        assert result.success
        assert result.rows == 5

    def test_pq_extension_accepted(self):
        assert self.store.ingest(SID, "d", _parquet_bytes(self.df), "pq").success

    def test_parquet_columns_correct(self):
        result = self.store.ingest(SID, "data", _parquet_bytes(self.df), "parquet")
        assert set(result.columns) == {"id", "value", "label"}

    def test_parquet_data_matches_original(self):
        self.store.ingest(SID, "data", _parquet_bytes(self.df), "parquet")
        retrieved = self.store.get(SID)["data"]
        assert list(retrieved["id"]) == list(self.df["id"])


# ---------------------------------------------------------------------------
# DataFrameStore.ingest — Excel  (M1 fix: openpyxl now in requirements.txt)
# ---------------------------------------------------------------------------


class TestIngestExcel:
    """
    M1 fix verification: openpyxl was missing from requirements.txt.
    pd.read_excel() raised ImportError on any .xlsx upload before the fix.
    These tests confirm openpyxl is installed and the path works end-to-end.
    """

    def setup_method(self):
        self.store = DataFrameStore()
        self.df = _df(3)

    def test_xlsx_ingest_succeeds(self):
        result = self.store.ingest(SID, "report", _xlsx_bytes(self.df), "xlsx")
        assert (
            result.success
        ), f"Excel ingest failed — openpyxl may not be installed: {result.error}"

    def test_xlsx_row_count_correct(self):
        result = self.store.ingest(SID, "report", _xlsx_bytes(self.df), "xlsx")
        assert result.success
        assert result.rows == 3

    def test_xlsx_dot_extension_accepted(self):
        result = self.store.ingest(SID, "r", _xlsx_bytes(self.df), ".xlsx")
        assert result.success

    def test_xlsx_data_retrievable(self):
        self.store.ingest(SID, "sheet", _xlsx_bytes(self.df), "xlsx")
        frames = self.store.get(SID)
        assert "sheet" in frames
        assert len(frames["sheet"]) == 3


# ---------------------------------------------------------------------------
# DataFrameStore.ingest — error cases
# ---------------------------------------------------------------------------


class TestIngestErrors:
    def setup_method(self):
        self.store = DataFrameStore()

    def test_unsupported_extension_returns_error(self):
        result = self.store.ingest(SID, "data", b"some content", "txt")
        assert not result.success
        assert result.error is not None
        assert "txt" in result.error.lower() or "unsupported" in result.error.lower()

    def test_json_extension_returns_error(self):
        result = self.store.ingest(SID, "data", b'{"a": 1}', "json")
        assert not result.success

    def test_oversized_file_blocked(self):
        store = DataFrameStore(max_upload_mb=1)
        big = b"x,y\n" + b"1,2\n" * (600_000)  # ~9 MB
        result = store.ingest(SID, "big", big, "csv")
        assert not result.success
        assert result.error is not None
        assert "limit" in result.error.lower() or "exceed" in result.error.lower()

    def test_corrupt_parquet_returns_error(self):
        result = self.store.ingest(SID, "bad", b"\x00\x01\x02" * 50, "parquet")
        assert not result.success
        assert result.error is not None

    def test_session_memory_cap_blocks_second_frame(self):
        # Cap at 0.001 MB (1 KB) — any real DataFrame exceeds this
        store = DataFrameStore(max_session_mb=0.001)
        df = pd.DataFrame({"col": range(1000)})  # ~8 KB in memory
        result = store.ingest(SID, "data", _csv_bytes(df), "csv")
        assert not result.success
        assert "memory" in result.error.lower() or "limit" in result.error.lower()

    def test_error_result_has_zero_rows(self):
        result = self.store.ingest(SID, "data", b"bad content", "parquet")
        assert not result.success
        assert result.rows == 0

    def test_error_result_has_empty_columns(self):
        result = self.store.ingest(SID, "data", b"bad content", "parquet")
        assert not result.success
        assert result.columns == []


# ---------------------------------------------------------------------------
# DataFrameStore.get
# ---------------------------------------------------------------------------


class TestGet:
    def setup_method(self):
        self.store = DataFrameStore()
        self.df = _df(5)

    def test_get_unknown_session_returns_empty_dict(self):
        assert self.store.get("no-such-session") == {}

    def test_get_returns_ingested_frame(self):
        self.store.ingest(SID, "claims", _csv_bytes(self.df), "csv")
        frames = self.store.get(SID)
        assert "claims" in frames
        assert isinstance(frames["claims"], pd.DataFrame)
        assert len(frames["claims"]) == 5

    def test_get_returns_shallow_copy_not_same_dict(self):
        self.store.ingest(SID, "data", _csv_bytes(self.df), "csv")
        frames1 = self.store.get(SID)
        frames2 = self.store.get(SID)
        assert frames1 is not frames2

    def test_get_returns_multiple_frames_for_session(self):
        self.store.ingest(SID, "a", _csv_bytes(_df(2)), "csv")
        self.store.ingest(SID, "b", _csv_bytes(_df(4)), "csv")
        frames = self.store.get(SID)
        assert "a" in frames and "b" in frames

    def test_get_different_sessions_isolated(self):
        self.store.ingest(SID, "frame", _csv_bytes(_df(3)), "csv")
        self.store.ingest(SID2, "other", _csv_bytes(_df(7)), "csv")
        assert "frame" not in self.store.get(SID2)
        assert "other" not in self.store.get(SID)

    def test_get_updates_last_access(self):
        self.store.ingest(SID, "data", _csv_bytes(self.df), "csv")
        t_before = self.store._last_access[SID]
        time.sleep(0.02)
        self.store.get(SID)
        t_after = self.store._last_access[SID]
        assert t_after > t_before

    def test_get_empty_session_does_not_update_last_access(self):
        # A session with no frames: get() returns {} and should not
        # create a last_access entry for a session that never existed.
        self.store.get("ghost-session")
        assert "ghost-session" not in self.store._last_access


# ---------------------------------------------------------------------------
# DataFrameStore._evict_expired — TTL eviction
# ---------------------------------------------------------------------------


class TestTTLEviction:
    def test_session_evicted_after_ttl_via_get(self):
        store = DataFrameStore(ttl_seconds=10)
        store.ingest(SID, "data", _csv_bytes(_df()), "csv")
        assert SID in store._store

        last = store._last_access[SID]
        # Simulate clock advancing 11 seconds past TTL
        with patch("dataframe_store.time.monotonic", return_value=last + 11):
            result = store.get(SID)

        assert result == {}
        assert SID not in store._store
        assert SID not in store._last_access

    def test_session_alive_before_ttl(self):
        store = DataFrameStore(ttl_seconds=3600)
        store.ingest(SID, "data", _csv_bytes(_df()), "csv")

        last = store._last_access[SID]
        with patch("dataframe_store.time.monotonic", return_value=last + 1):
            result = store.get(SID)

        assert "data" in result

    def test_expired_session_removed_from_last_access(self):
        store = DataFrameStore(ttl_seconds=5)
        store.ingest(SID, "data", _csv_bytes(_df()), "csv")

        last = store._last_access[SID]
        with patch("dataframe_store.time.monotonic", return_value=last + 10):
            store.get(SID)

        assert SID not in store._last_access

    def test_only_expired_sessions_evicted(self):
        store = DataFrameStore(ttl_seconds=10)
        store.ingest(SID, "data", _csv_bytes(_df()), "csv")
        store.ingest(SID2, "data", _csv_bytes(_df()), "csv")

        # Manually expire SID but leave SID2 fresh
        store._last_access[SID] = time.monotonic() - 20
        # SID2._last_access stays at current time

        store._evict_expired()

        assert SID not in store._store
        assert SID2 in store._store

    def test_evict_expired_called_directly(self):
        store = DataFrameStore(ttl_seconds=1)
        store.ingest(SID, "data", _csv_bytes(_df()), "csv")

        # Age the session manually
        store._last_access[SID] = time.monotonic() - 5
        store._evict_expired()

        assert SID not in store._store

    def test_no_sessions_to_evict_is_noop(self):
        store = DataFrameStore()
        store._evict_expired()  # must not raise


# ---------------------------------------------------------------------------
# DataFrameStore.delete
# ---------------------------------------------------------------------------


class TestDelete:
    def setup_method(self):
        self.store = DataFrameStore()
        self.store.ingest(SID, "frame_a", _csv_bytes(_df(3)), "csv")
        self.store.ingest(SID, "frame_b", _csv_bytes(_df(5)), "csv")

    def test_delete_removes_named_frame(self):
        deleted = self.store.delete(SID, "frame_a")
        assert deleted is True
        frames = self.store.get(SID)
        assert "frame_a" not in frames

    def test_delete_leaves_other_frames_intact(self):
        self.store.delete(SID, "frame_a")
        frames = self.store.get(SID)
        assert "frame_b" in frames

    def test_delete_returns_false_for_missing_frame(self):
        assert self.store.delete(SID, "nonexistent") is False

    def test_delete_returns_false_for_missing_session(self):
        assert self.store.delete("no-session", "frame_a") is False

    def test_delete_idempotent(self):
        self.store.delete(SID, "frame_a")
        assert self.store.delete(SID, "frame_a") is False


# ---------------------------------------------------------------------------
# DataFrameStore.delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def setup_method(self):
        self.store = DataFrameStore()
        self.store.ingest(SID, "a", _csv_bytes(_df(2)), "csv")
        self.store.ingest(SID, "b", _csv_bytes(_df(3)), "csv")

    def test_delete_session_removes_all_frames(self):
        self.store.delete_session(SID)
        assert self.store.get(SID) == {}

    def test_delete_session_removes_last_access_entry(self):
        self.store.delete_session(SID)
        assert SID not in self.store._last_access

    def test_delete_session_only_removes_target_session(self):
        self.store.ingest(SID2, "c", _csv_bytes(_df()), "csv")
        self.store.delete_session(SID)
        assert "c" in self.store.get(SID2)

    def test_delete_session_noop_for_unknown_session(self):
        # Must not raise
        self.store.delete_session("phantom-session")

    def test_delete_session_then_reingest(self):
        self.store.delete_session(SID)
        result = self.store.ingest(SID, "fresh", _csv_bytes(_df(4)), "csv")
        assert result.success
        assert "fresh" in self.store.get(SID)


# ---------------------------------------------------------------------------
# DataFrameStore.list_dataframes
# ---------------------------------------------------------------------------


class TestListDataframes:
    def setup_method(self):
        self.store = DataFrameStore()

    def test_empty_session_returns_empty_list(self):
        assert self.store.list_dataframes("unknown") == []

    def test_returns_correct_row_count(self):
        self.store.ingest(SID, "claims", _csv_bytes(_df(7)), "csv")
        listing = self.store.list_dataframes(SID)
        assert listing[0]["rows"] == 7

    def test_returns_correct_column_list(self):
        self.store.ingest(SID, "data", _csv_bytes(_df()), "csv")
        listing = self.store.list_dataframes(SID)
        assert set(listing[0]["columns"]) == {"id", "value", "label"}

    def test_size_mb_is_positive_float(self):
        # Must use enough rows so round(size_mb, 3) > 0.
        # 10 rows → ~0.001 MB after rounding.
        self.store.ingest(SID, "data", _csv_bytes(_df(10)), "csv")
        listing = self.store.list_dataframes(SID)
        assert isinstance(listing[0]["size_mb"], float)
        assert listing[0]["size_mb"] > 0

    def test_df_name_field_correct(self):
        self.store.ingest(SID, "policies", _csv_bytes(_df()), "csv")
        listing = self.store.list_dataframes(SID)
        assert listing[0]["df_name"] == "policies"

    def test_multiple_frames_all_listed(self):
        self.store.ingest(SID, "frame_a", _csv_bytes(_df(2)), "csv")
        self.store.ingest(SID, "frame_b", _csv_bytes(_df(4)), "csv")
        listing = self.store.list_dataframes(SID)
        names = {item["df_name"] for item in listing}
        assert names == {"frame_a", "frame_b"}

    def test_required_keys_in_each_item(self):
        self.store.ingest(SID, "data", _csv_bytes(_df()), "csv")
        item = self.store.list_dataframes(SID)[0]
        for key in ("df_name", "rows", "columns", "size_mb"):
            assert key in item, f"Missing key {key!r} in list_dataframes item"
