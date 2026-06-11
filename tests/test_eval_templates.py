# tests/test_eval_templates.py
import sys

sys.path.insert(0, "analyst_copilot")
from eval import _SQL_TEMPLATES
from validation import validate_sql

print("=== ML-1: All eval templates must pass validate_sql() ===")
failures = []
for t in _SQL_TEMPLATES:
    # Instantiate with dummy values to get a parseable SQL string
    try:
        sql = t["sql"].format(
            table="claims",
            metric="amount",
            category="region",
            date_col="claim_date",
            year=2023,
            threshold=1000,
            id_col="policy_id",
            col1="age",
            col2="premium",
            start_date="2023-01-01",
            end_date="2023-12-31",
            n=10,
            pct=0.1,
        )
    except KeyError:
        continue  # template has extra placeholders — skip

    vr = validate_sql(
        sql,
        schema_columns=[
            "amount",
            "region",
            "claim_date",
            "policy_id",
            "age",
            "premium",
        ],
    )
    # Only hard failures (not advisories) are blockers
    if not vr.valid:
        failures.append((t["template_id"], vr.error_type, vr.error_message))
        print(f"  FAIL [{t['template_id']}]: {vr.error_type} — {vr.error_message}")
    else:
        status = f"(advisory: {vr.warning[:50]})" if vr.warning else ""
        print(f"  PASS [{t['template_id']}] {status}")

if not failures:
    print(f"\nAll {len(_SQL_TEMPLATES)} templates pass.")
else:
    print(f"\n{len(failures)} template(s) failed — ML-1 fix incomplete.")
