# tests/test_pandas_validation_ml2.py
import sys

sys.path.insert(0, "analyst_copilot")
from validation import validate_python

# These are real Pandas patterns that were incorrectly flagged before the fix
LEGIT_PATTERNS = [
    ("merge_inner", "result = df.merge(df2, how='inner', on='policy_id')"),
    ("fillna_ffill", "result = df.fillna(method='ffill')"),
    ("sort_na_position", "result = df.sort_values(by='amount', na_position='last')"),
    ("pivot_mean", "result = df.pivot_table(values='amount', aggfunc='mean')"),
    ("drop_duplicates_first", "result = df.drop_duplicates(keep='first')"),
    ("groupby_sum", "result = df.groupby('region').agg({'amount': 'sum'})"),
    (
        "merge_left",
        "result = df.merge(df2, how='left', left_on='id', right_on='policy_id')",
    ),
]

schema_cols = ["policy_id", "amount", "region", "id"]
print("=== ML-2: Pandas kwargs must not be flagged as UNRESOLVED_COLUMN ===")
for name, code in LEGIT_PATTERNS:
    vr = validate_python(code, schema_columns=schema_cols)
    # Check: must not have UNRESOLVED_COLUMN error
    is_fp = not vr.valid and vr.error_type == "UNRESOLVED_COLUMN"
    print(f"  {'FAIL (false positive!)' if is_fp else 'PASS'}: {name}")
    if is_fp:
        print(f"    Error: {vr.error_message}")
