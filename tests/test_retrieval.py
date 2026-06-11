"""
tests/test_retrieval.py
Tests for analyst_copilot.retrieval

Coverage matrix
───────────────
_quote()                     — dialect-specific identifier quoting
_count_chunk_tokens()        — tiktoken cl100k_base token counting
_hash_sample()               — SHA-256 PII value hashing (16-char hex)
SchemaRegistry               — put/get/list_ids/all_columns/table_policies
build_chunks()               — one chunk per table, Gap-9 token-cap logic,
                               PII sample hashing, FK wiring
_chunk_dict_to_schema_chunk()— new-format (columns_meta) and legacy
                               (column_names / fk_targets) conversion
RetrievalLayer               — get_schema_columns, get_table_policies,
                               _bootstrap_registry guard (mocked deps)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from analyst_copilot.retrieval import (
    ColumnMeta,
    RetrievalLayer,
    SchemaProfile,
    SchemaRegistry,
    TableMeta,
    _chunk_dict_to_schema_chunk,
    _count_chunk_tokens,
    _hash_sample,
    _quote,
    build_chunks,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _col(name: str = "col", dtype: str = "INTEGER", **kw) -> ColumnMeta:
    # Pop nullable from **kw first — callers like _col("id", nullable=False) pass
    # it via **kw, which would collide with a hardcoded positional default and raise
    # "multiple values for keyword argument 'nullable'" (TEST-E FIX).
    nullable = kw.pop("nullable", True)
    return ColumnMeta(name=name, data_type=dtype, nullable=nullable, **kw)


def _tbl(name: str = "policies", cols: list | None = None, **kw) -> TableMeta:
    return TableMeta(
        table_name=name,
        columns=cols or [_col("id", nullable=False)],
        **kw,
    )


def _profile(
    schema_id: str = "test_schema",
    tables: list | None = None,
    dialect: str = "postgres",
) -> SchemaProfile:
    return SchemaProfile(
        schema_id=schema_id,
        dialect=dialect,
        tables=tables or [_tbl()],
    )


def _mock_layer(registry: SchemaRegistry | None = None) -> RetrievalLayer:
    embedder = MagicMock()
    indexer = MagicMock()
    indexer.list_indexed_schemas.return_value = []
    return RetrievalLayer(
        embedder=embedder,
        indexer=indexer,
        registry=registry or SchemaRegistry(),
    )


# ---------------------------------------------------------------------------
# _quote — dialect-aware identifier quoting
# ---------------------------------------------------------------------------


class TestQuote:
    def test_postgres_double_quote(self):
        assert _quote("my_table", "postgres") == '"my_table"'

    def test_sqlite_double_quote(self):
        assert _quote("my_table", "sqlite") == '"my_table"'

    def test_bigquery_double_quote(self):
        assert _quote("my_table", "bigquery") == '"my_table"'

    def test_snowflake_double_quote(self):
        assert _quote("my_table", "snowflake") == '"my_table"'

    def test_mysql_backtick(self):
        assert _quote("my_table", "mysql") == "`my_table`"

    def test_unknown_dialect_ansi_fallback(self):
        assert _quote("my_table", "oracle") == '"my_table"'

    def test_name_with_spaces_preserved(self):
        # Quoting does not modify the name itself
        assert _quote("Order Details", "postgres") == '"Order Details"'

    def test_empty_name_still_quoted(self):
        result = _quote("", "postgres")
        assert result.startswith('"') and result.endswith('"')


# ---------------------------------------------------------------------------
# _count_chunk_tokens — tiktoken cl100k_base
# ---------------------------------------------------------------------------


class TestCountChunkTokens:
    def test_returns_int(self):
        assert isinstance(_count_chunk_tokens("hello"), int)

    def test_non_negative(self):
        assert _count_chunk_tokens("") >= 0
        assert _count_chunk_tokens("hello") >= 0

    def test_longer_text_more_tokens(self):
        short = _count_chunk_tokens("hi")
        long = _count_chunk_tokens(
            "hello world this is a considerably longer sentence with more tokens"
        )
        assert long > short

    def test_realistic_schema_text_in_range(self):
        text = (
            "Table: policies\nSchema: ins_prod_v3\nRow count (estimate): 50000\n"
            "Columns:\n  - policy_id: INTEGER NOT NULL\n  - premium_amt: DECIMAL(12,2)"
        )
        n = _count_chunk_tokens(text)
        assert 10 < n < 150


# ---------------------------------------------------------------------------
# _hash_sample — SHA-256 PII hashing
# ---------------------------------------------------------------------------


class TestHashSample:
    def test_returns_16_char_hex(self):
        result = _hash_sample("test@example.com")
        assert isinstance(result, str)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        assert _hash_sample("foo") == _hash_sample("foo")

    def test_different_inputs_different_hashes(self):
        assert _hash_sample("a") != _hash_sample("b")

    def test_handles_integer(self):
        result = _hash_sample(12345)
        assert len(result) == 16

    def test_handles_none(self):
        result = _hash_sample(None)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# SchemaRegistry — in-memory profile store
# ---------------------------------------------------------------------------


class TestSchemaRegistry:
    def test_put_then_get_returns_same_object(self):
        reg = SchemaRegistry()
        p = _profile("s")
        reg.put("s", p)
        assert reg.get("s") is p

    def test_get_missing_returns_none(self):
        assert SchemaRegistry().get("nonexistent") is None

    def test_list_ids_empty_on_init(self):
        assert SchemaRegistry().list_ids() == []

    def test_list_ids_all_keys_present(self):
        reg = SchemaRegistry()
        reg.put("a", _profile("a"))
        reg.put("b", _profile("b"))
        assert set(reg.list_ids()) == {"a", "b"}

    def test_put_overwrites_existing(self):
        reg = SchemaRegistry()
        p1, p2 = _profile("s"), _profile("s")
        reg.put("s", p1)
        reg.put("s", p2)
        assert reg.get("s") is p2

    def test_all_columns_lowercases_names(self):
        reg = SchemaRegistry()
        cols = [_col("Policy_ID"), _col("Premium_Amt")]
        reg.put("s", _profile(tables=[_tbl(cols=cols)]))
        cols_out = reg.all_columns("s")
        assert "policy_id" in cols_out
        assert "premium_amt" in cols_out
        assert "Policy_ID" not in cols_out

    def test_all_columns_aggregates_across_tables(self):
        reg = SchemaRegistry()
        t1 = _tbl("policies", cols=[_col("policy_id")])
        t2 = _tbl("claims", cols=[_col("claim_id")])
        reg.put("s", _profile(tables=[t1, t2]))
        cols = reg.all_columns("s")
        assert "policy_id" in cols
        assert "claim_id" in cols

    def test_all_columns_missing_schema_returns_empty_set(self):
        assert SchemaRegistry().all_columns("missing") == set()

    def test_table_policies_pii_flag_propagated(self):
        reg = SchemaRegistry()
        tbl = _tbl("claims")
        tbl.is_pii_flagged = True
        reg.put("s", _profile(tables=[tbl]))
        policies = reg.table_policies("s")
        assert "claims" in policies
        assert policies["claims"].pii_flagged is True

    def test_table_policies_non_pii_flag(self):
        reg = SchemaRegistry()
        reg.put("s", _profile(tables=[_tbl("policies")]))
        assert reg.table_policies("s")["policies"].pii_flagged is False

    def test_table_policies_missing_schema_returns_empty(self):
        assert SchemaRegistry().table_policies("missing") == {}


# ---------------------------------------------------------------------------
# build_chunks — one chunk per table, token-cap logic
# ---------------------------------------------------------------------------


class TestBuildChunks:
    def test_one_chunk_per_table(self):
        p = _profile(tables=[_tbl("policies"), _tbl("claims")])
        assert len(build_chunks(p)) == 2

    def test_chunk_id_format(self):
        p = _profile("ins_prod_v3", tables=[_tbl("policies")])
        assert build_chunks(p)[0]["chunk_id"] == "ins_prod_v3::policies"

    def test_required_keys_present(self):
        chunks = build_chunks(_profile())
        required = {
            "chunk_id",
            "schema_id",
            "table_name",
            "text",
            "column_names",
            "columns_meta",
        }
        assert required.issubset(chunks[0].keys())

    def test_column_names_match_table(self):
        cols = [_col("policy_id"), _col("premium_amt")]
        p = _profile(tables=[_tbl(cols=cols)])
        assert build_chunks(p)[0]["column_names"] == ["policy_id", "premium_amt"]

    def test_text_contains_table_name(self):
        p = _profile(tables=[_tbl("claims")])
        assert "claims" in build_chunks(p)[0]["text"]

    def test_text_contains_schema_id(self):
        p = _profile("ins_prod_v3")
        assert "ins_prod_v3" in build_chunks(p)[0]["text"]

    def test_pii_samples_are_hashed_not_verbatim(self):
        col = ColumnMeta(
            name="email",
            data_type="TEXT",
            nullable=True,
            is_pii=True,
            sample_values=["user@example.com"],
        )
        p = _profile(tables=[_tbl(cols=[col])])
        text = build_chunks(p)[0]["text"]
        assert "user@example.com" not in text

    def test_fk_targets_populated(self):
        col = _col("policy_id")
        tbl = _tbl(
            "claims",
            cols=[col],
            foreign_keys=[
                {
                    "from_column": "policy_id",
                    "to_table": "policies",
                    "to_column": "policy_id",
                }
            ],
        )
        chunk = build_chunks(_profile(tables=[tbl]))[0]
        assert "policies" in chunk["fk_targets"]

    def test_fk_relationships_from_column_populated(self):
        col = _col("policy_id")
        tbl = _tbl(
            "claims",
            cols=[col],
            foreign_keys=[
                {
                    "from_column": "policy_id",
                    "to_table": "policies",
                    "to_column": "policy_id",
                }
            ],
        )
        chunk = build_chunks(_profile(tables=[tbl]))[0]
        assert chunk["fk_relationships"][0]["from_column"] == "policy_id"

    def test_business_description_propagated(self):
        tbl = _tbl()
        tbl.business_description = "Core policy table"
        chunk = build_chunks(_profile(tables=[tbl]))[0]
        assert chunk["business_description"] == "Core policy table"

    def test_row_count_estimate_propagated(self):
        tbl = _tbl()
        tbl.row_count_estimate = 99_000
        chunk = build_chunks(_profile(tables=[tbl]))[0]
        assert chunk["row_count_estimate"] == 99_000

    def test_token_cap_strips_descriptions_column_names_preserved(self):
        """Gap-9: after description strip, every column name must survive."""
        long_desc = "This is a very long description. " * 30
        cols = [
            ColumnMeta(
                name=f"col_{i}",
                data_type="TEXT",
                nullable=True,
                column_description=long_desc,
            )
            for i in range(50)
        ]
        p = _profile(tables=[_tbl(cols=cols)])
        chunk = build_chunks(p)[0]
        # All column names must still be in the text after truncation passes
        for i in range(50):
            assert f"col_{i}" in chunk["text"]

    def test_pii_flagged_propagated_to_chunk(self):
        tbl = _tbl()
        tbl.is_pii_flagged = True
        chunk = build_chunks(_profile(tables=[tbl]))[0]
        assert chunk["is_pii_flagged"] is True

    def test_columns_meta_contains_name_and_description(self):
        col = ColumnMeta(
            name="premium_amt",
            data_type="DECIMAL(12,2)",
            nullable=True,
            column_description="Annual premium",
        )
        chunk = build_chunks(_profile(tables=[_tbl(cols=[col])]))[0]
        meta = chunk["columns_meta"]
        assert meta[0]["name"] == "premium_amt"
        assert meta[0]["description"] == "Annual premium"


# ---------------------------------------------------------------------------
# _chunk_dict_to_schema_chunk — dict → SchemaChunk conversion
# ---------------------------------------------------------------------------


class TestChunkDictToSchemaChunk:
    def _new_chunk(self, **overrides):
        base = {
            "table_name": "claims",
            "schema_id": "ins_prod_v3",
            "columns_meta": [
                {"name": "claim_id", "description": "Primary key"},
                {"name": "claim_amount", "description": None},
            ],
            "fk_relationships": [
                {
                    "from_column": "policy_id",
                    "to_table": "policies",
                    "to_column": "policy_id",
                }
            ],
            "is_pii_flagged": False,
            "business_description": "Insurance claims",
            "row_count_estimate": 50_000,
        }
        base.update(overrides)
        return base

    def _legacy_chunk(self):
        return {
            "table_name": "claims",
            "schema_id": "ins_prod_v3",
            "column_names": ["claim_id", "claim_amount"],
            "fk_targets": ["policies"],
            "is_pii_flagged": False,
        }

    # ── new-format ───────────────────────────────────────────────────────────

    def test_new_format_table_name(self):
        assert _chunk_dict_to_schema_chunk(self._new_chunk()).table == "claims"

    def test_new_format_schema_id(self):
        assert _chunk_dict_to_schema_chunk(self._new_chunk()).schema_id == "ins_prod_v3"

    def test_new_format_column_count(self):
        assert len(_chunk_dict_to_schema_chunk(self._new_chunk()).columns) == 2

    def test_new_format_column_name(self):
        chunk = _chunk_dict_to_schema_chunk(self._new_chunk())
        assert chunk.columns[0].name == "claim_id"

    def test_new_format_column_description_populated(self):
        chunk = _chunk_dict_to_schema_chunk(self._new_chunk())
        assert chunk.columns[0].description == "Primary key"

    def test_new_format_column_null_description(self):
        chunk = _chunk_dict_to_schema_chunk(self._new_chunk())
        assert chunk.columns[1].description is None

    def test_new_format_fk_count(self):
        assert len(_chunk_dict_to_schema_chunk(self._new_chunk()).fk_relationships) == 1

    def test_new_format_fk_references_target(self):
        chunk = _chunk_dict_to_schema_chunk(self._new_chunk())
        assert "policies" in chunk.fk_relationships[0].references

    def test_new_format_fk_from_column(self):
        chunk = _chunk_dict_to_schema_chunk(self._new_chunk())
        assert chunk.fk_relationships[0].column == "policy_id"

    def test_new_format_business_description(self):
        assert (
            _chunk_dict_to_schema_chunk(self._new_chunk()).business_description
            == "Insurance claims"
        )

    def test_new_format_row_count(self):
        assert _chunk_dict_to_schema_chunk(self._new_chunk()).row_count_estimate == 50_000

    def test_new_format_pii_true(self):
        chunk = _chunk_dict_to_schema_chunk(self._new_chunk(is_pii_flagged=True))
        assert chunk.pii_flagged is True

    # ── legacy format ────────────────────────────────────────────────────────

    def test_legacy_column_names_fallback(self):
        chunk = _chunk_dict_to_schema_chunk(self._legacy_chunk())
        names = [c.name for c in chunk.columns]
        assert names == ["claim_id", "claim_amount"]

    def test_legacy_fk_targets_converted(self):
        chunk = _chunk_dict_to_schema_chunk(self._legacy_chunk())
        assert len(chunk.fk_relationships) == 1
        assert "policies" in chunk.fk_relationships[0].references

    def test_legacy_no_business_description(self):
        assert _chunk_dict_to_schema_chunk(self._legacy_chunk()).business_description is None


# ---------------------------------------------------------------------------
# RetrievalLayer — unit tests with mocked embedder / indexer
# ---------------------------------------------------------------------------


class TestRetrievalLayer:
    # ── get_schema_columns ───────────────────────────────────────────────────

    def test_get_schema_columns_returns_registered(self):
        reg = SchemaRegistry()
        cols = [_col("policy_id"), _col("premium_amt")]
        reg.put("s", _profile("s", tables=[_tbl(cols=cols)]))
        layer = _mock_layer(registry=reg)
        result = asyncio.run(layer.get_schema_columns("s"))
        assert {"policy_id", "premium_amt"}.issubset(result)

    def test_get_schema_columns_missing_schema_empty(self):
        result = asyncio.run(_mock_layer().get_schema_columns("nonexistent"))
        assert result == set()

    # ── get_table_policies ───────────────────────────────────────────────────

    def test_get_table_policies_pii_flag_propagated(self):
        reg = SchemaRegistry()
        tbl = _tbl("claims")
        tbl.is_pii_flagged = True
        reg.put("s", _profile("s", tables=[tbl]))
        policies = asyncio.run(_mock_layer(registry=reg).get_table_policies("s"))
        assert "claims" in policies
        assert policies["claims"].pii_flagged is True

    def test_get_table_policies_missing_schema_empty(self):
        result = asyncio.run(_mock_layer().get_table_policies("nonexistent"))
        assert result == {}

    # ── _bootstrap_registry guard ────────────────────────────────────────────

    def test_bootstrap_runs_only_once(self):
        layer = _mock_layer()
        assert layer._bootstrapped is False
        asyncio.run(layer.get_schema_columns("any"))
        assert layer._bootstrapped is True
        # Second call must NOT re-run list_indexed_schemas
        asyncio.run(layer.get_schema_columns("any"))
        layer._indexer.list_indexed_schemas.assert_called_once()

    # ── retrieve (empty FAISS result) ────────────────────────────────────────

    def test_retrieve_returns_empty_for_unindexed_schema(self):
        layer = _mock_layer()
        layer._embedder.embed_query.return_value = [0.0] * 1024
        layer._indexer.search.return_value = []

        async def _go():
            return await layer.retrieve("query", "nonexistent", k=5)

        assert asyncio.run(_go()) == []
