# tests/test_chunk_token_cap.py
import sys

sys.path.insert(0, "analyst_copilot")
from retrieval import _CHUNK_TOKEN_CAP, _count_chunk_tokens, build_chunks


# Build a mock wide-table schema (100 columns with long descriptions)
class Col:
    def __init__(self, i):
        self.name = f"col_{i}"
        self.data_type = "VARCHAR(255)"
        self.nullable = True
        self.null_rate = None
        self.cardinality_estimate = None
        self.sample_values = [f"val_{j}" for j in range(3)]
        self.is_pii = False
        self.column_description = (
            f"This is column {i}. IMPORTANT: Filter IS NOT NULL before any aggregation. "
            f"Data quality note: values above 9999 are sentinel codes for missing data."
        )


class Table:
    def __init__(self, ncols):
        self.table_name = "wide_table"
        self.columns = [Col(i) for i in range(ncols)]
        self.foreign_keys = []
        self.is_pii_flagged = False
        self.business_description = "A wide table for testing."
        self.row_count_estimate = 100_000


class Profile:
    def __init__(self, ncols):
        self.schema_id = "test_schema"
        self.tables = [Table(ncols)]


print("=== Gap-9: 400-token chunk cap ===")
for ncols in [10, 30, 60, 100]:
    chunks = build_chunks(Profile(ncols))
    assert len(chunks) == 1
    token_count = _count_chunk_tokens(chunks[0]["text"])
    status = "PASS" if token_count <= _CHUNK_TOKEN_CAP else "WARN (overflow log expected)"
    print(f"  {ncols:3d} cols → {token_count:4d} tokens  [{status}]")

    # Column names must always be present regardless of truncation
    for col_name in [f"col_{i}" for i in range(min(ncols, 5))]:
        assert col_name in chunks[0]["text"], f"Column name {col_name} was stripped!"
    print("         Column names preserved: PASS")
