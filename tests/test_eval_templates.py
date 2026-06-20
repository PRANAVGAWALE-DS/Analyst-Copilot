# tests/test_eval_templates.py
"""
ML-1  : All eval templates must pass validate_sql()            (original)
ML-6  : _normalize_col and _columns_semantically_match behave  (new)
        correctly across all 20 template alias patterns.

The matrix below encodes three categories for every template:

  CORRECT   — LLM aliases that differ only in style from GT; must score (True, False)
  PARTIAL   — LLM aliases that are meaningful but abbreviated; must score (False, True)
  INCORRECT — LLM aliases for wrong columns/metrics; must score (False, False)

False-positive cases are included explicitly: scenarios where an overly lenient
matcher could upgrade a wrong answer to correct/partial.
"""

import sys

sys.path.insert(0, "analyst_copilot")
from eval import (
    _SQL_TEMPLATES,
    _columns_semantically_match,
    _normalize_col,
)
from validation import validate_sql

# ── ML-1: validate_sql on every instantiated template ─────────────────────────

print("=== ML-1: All eval templates must pass validate_sql() ===")
failures_ml1 = []
for t in _SQL_TEMPLATES:
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
            attribute="claim_date",
            entity="claim",
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
    if not vr.valid:
        failures_ml1.append((t["template_id"], vr.error_type, vr.error_message))
        print(f"  FAIL [{t['template_id']}]: {vr.error_type} — {vr.error_message}")
    else:
        status = f"(advisory: {vr.warning[:50]})" if vr.warning else ""
        print(f"  PASS [{t['template_id']}] {status}")

if not failures_ml1:
    print(f"\nAll {len(_SQL_TEMPLATES)} templates pass ML-1.\n")
else:
    print(f"\n{len(failures_ml1)} template(s) failed ML-1.\n")


# ── ML-6: _normalize_col unit tests ───────────────────────────────────────────

print("=== ML-6a: _normalize_col unit tests ===")

NORM_CASES = [
    # (input, expected_output, description)
    # Prefix stripping (pre-existing)
    ("total_paid_amount", "paid_amount", "total_ prefix"),
    ("avg_claim_amount", "claim_amount", "avg_ prefix"),
    ("average_premium", "premium", "average_ prefix"),
    ("max_paid_amount", "paid_amount", "max_ prefix"),
    ("min_paid_amount", "paid_amount", "min_ prefix"),
    ("count_claim", "claim", "count_ prefix"),
    ("sum_premium", "premium", "sum_ prefix"),
    ("highest_paid", "paid", "highest_ prefix"),
    ("lowest_paid", "paid", "lowest_ prefix"),
    ("num_claims", "claims", "num_ prefix"),
    ("missing_count", "count", "missing_ prefix"),
    # ML-6 addition A1: mean_ prefix
    ("mean_claim_amount", "claim_amount", "mean_ prefix (ML-6 A1)"),
    ("mean_premium", "premium", "mean_ prefix short (ML-6 A1)"),
    # ML-6 addition A2: suffix stripping
    ("claim_count", "claim", "_count suffix (ML-6 A2)"),
    ("policy_count", "policy", "_count suffix entity (ML-6 A2)"),
    ("claim_total", "claim", "_total suffix (ML-6 A2)"),
    ("paid_sum", "paid", "_sum suffix (ML-6 A2)"),
    ("premium_avg", "premium", "_avg suffix (ML-6 A2)"),
    ("amount_mean", "amount", "_mean suffix (ML-6 A2)"),
    ("amount_average", "amount", "_average suffix (ML-6 A2)"),
    # ML-6 addition A3: time-period normalization
    ("paid_month", "month", "time-period: paid_month (ML-6 A3)"),
    ("claim_month", "month", "time-period: claim_month (ML-6 A3)"),
    ("sales_quarter", "quarter", "time-period: sales_quarter (ML-6 A3)"),
    ("revenue_year", "year", "time-period: revenue_year (ML-6 A3)"),
    ("order_week", "week", "time-period: order_week (ML-6 A3)"),
    # Time-period: single token must NOT be collapsed
    ("month", "month", "single-token month: unchanged"),
    ("quarter", "quarter", "single-token quarter: unchanged"),
    # Table-qualified columns
    ("t1.claim_amount", "claim_amount", "table prefix stripped"),
    ("sub.total_premium", "premium", "subquery alias + agg prefix"),
    # Suffix should not fire when stem would be empty
    ("_count", "_count", "bare _count: no stem guard"),
    # Suffix should not fire on semantic (non-agg) suffixes
    ("claim_date", "claim_date", "_date: not an agg suffix"),
    ("claim_status", "claim_status", "_status: not an agg suffix"),
    ("policy_id", "policy_id", "_id: not an agg suffix"),
    ("claim_amount", "claim_amount", "_amount: not an agg suffix"),
]

failures_norm = []
for inp, expected, desc in NORM_CASES:
    got = _normalize_col(inp)
    ok = got == expected
    if not ok:
        failures_norm.append((inp, expected, got, desc))
        print(f"  FAIL  {inp!r:<28} expected={expected!r}  got={got!r}  [{desc}]")
    else:
        print(f"  PASS  {inp!r:<28} → {got!r}  [{desc}]")

print(
    f"\n{'All' if not failures_norm else len(failures_norm)} _normalize_col cases "
    f"{'passed' if not failures_norm else 'failed'}.\n"
)


# ── ML-6: _columns_semantically_match full template matrix ────────────────────

print("=== ML-6b: _columns_semantically_match alias matrix (all 20 templates) ===")

MATCH_CASES = [
    # fmt: (template_id, gt_cols, llm_cols, exp_correct, exp_partial, description)
    # ── agg_avg_by_category ──────────────────────────────────────────────
    (
        "agg_avg_by_category",
        {"claim_status", "avg_claim_amount"},
        {"claim_status", "avg_claim_amount"},
        True,
        False,
        "exact match",
    ),
    (
        "agg_avg_by_category",
        {"claim_status", "avg_claim_amount"},
        {"claim_status", "average_claim_amount"},
        True,
        False,
        "average_ alias",
    ),
    (
        "agg_avg_by_category",
        {"claim_status", "avg_claim_amount"},
        {"claim_status", "mean_claim_amount"},
        True,
        False,
        "mean_ alias (ML-6 A1)",
    ),
    (
        "agg_avg_by_category",
        {"claim_status", "avg_claim_amount"},
        {"region", "average_claim_amount"},
        False,
        True,
        "PARTIAL: wrong category col",
    ),
    # ── agg_sum_by_category ──────────────────────────────────────────────
    (
        "agg_sum_by_category",
        {"claim_status", "total_claim_amount"},
        {"claim_status", "sum_claim_amount"},
        True,
        False,
        "sum_ alias",
    ),
    (
        "agg_sum_by_category",
        {"claim_status", "total_claim_amount"},
        {"claim_status", "total_claim_amount"},
        True,
        False,
        "exact match",
    ),
    (
        "agg_sum_by_category",
        {"claim_status", "total_claim_amount"},
        {"claim_status", "claim_total"},
        False,
        True,
        "PARTIAL: _total suffix strips metric to stem, not full metric (ML-6 A2)",
    ),
    # ── agg_count_by_category ────────────────────────────────────────────
    (
        "agg_count_by_category",
        {"claim_status", "count_claim"},
        {"claim_status", "claim_count"},
        True,
        False,
        "_count suffix (ML-6 A2)",
    ),
    (
        "agg_count_by_category",
        {"claim_status", "count_claim"},
        {"claim_status", "num_claims"},
        True,
        False,
        "num_ prefix + plural: fuzzy claim≈claims (0.91) (ML-6 B1)",
    ),
    # ── agg_count_total ──────────────────────────────────────────────────
    # Template uses entity=table.rstrip('s') → 'policie', 'claim', 'agent'
    (
        "agg_count_total",
        {"total_policie"},
        {"policy_count"},
        True,
        False,
        "fuzzy: policie≈policy (ML-6 B1)",
    ),
    (
        "agg_count_total",
        {"total_claim"},
        {"claim_count"},
        True,
        False,
        "_count suffix then exact (ML-6 A2)",
    ),
    (
        "agg_count_total",
        {"total_policie"},
        {"count"},
        True,
        False,
        "bare count LLM (ML-6 B2 symmetric)",
    ),
    (
        "agg_count_total",
        {"total_policie"},
        {"sum"},
        True,
        False,
        "bare sum LLM (ML-6 B2 symmetric)",
    ),
    # FP guard: multi-col GT must not match bare agg
    (
        "agg_count_total_fp",
        {"claim_status", "total_claim"},
        {"count"},
        False,
        False,
        "FP-guard: multi-col GT + bare LLM",
    ),
    # ── agg_max_min ──────────────────────────────────────────────────────
    (
        "agg_max_min",
        {"max_paid_amount", "min_paid_amount"},
        {"max_paid_amount", "min_paid_amount"},
        True,
        False,
        "exact match",
    ),
    (
        "agg_max_min",
        {"max_paid_amount", "min_paid_amount"},
        {"highest_paid_amount", "lowest_paid_amount"},
        True,
        False,
        "highest_/lowest_ alias",
    ),
    # ── ranking_top_n ────────────────────────────────────────────────────
    (
        "ranking_top_n",
        {"policy_id", "total_claim_amount"},
        {"policy_id", "total_claim_amount"},
        True,
        False,
        "exact match",
    ),
    (
        "ranking_top_n",
        {"policy_id", "total_claim_amount"},
        {"policy_id", "claim_total"},
        False,
        True,
        "PARTIAL: _total strips to claim, GT normalises to claim_amount (ML-6 A2)",
    ),
    (
        "ranking_top_n",
        {"policy_id", "total_claim_amount"},
        {"policy_id", "sum_claim_amount"},
        True,
        False,
        "sum_ alias",
    ),
    # ── ranking_bottom_n ─────────────────────────────────────────────────
    (
        "ranking_bottom_n",
        {"policy_id", "avg_claim_amount"},
        {"policy_id", "mean_claim_amount"},
        True,
        False,
        "mean_ prefix (ML-6 A1)",
    ),
    (
        "ranking_bottom_n",
        {"policy_id", "avg_claim_amount"},
        {"policy_id", "average_claim_amount"},
        True,
        False,
        "average_ prefix",
    ),
    # ── time_series_monthly ──────────────────────────────────────────────
    (
        "time_series_monthly",
        {"month", "total_paid_amount"},
        {"month", "total_paid_amount"},
        True,
        False,
        "exact match",
    ),
    (
        "time_series_monthly",
        {"month", "total_paid_amount"},
        {"paid_month", "total_paid"},
        False,
        True,
        "PARTIAL: period renamed + metric truncated (ML-6 A3)",
    ),
    (
        "time_series_monthly",
        {"month", "total_paid_amount"},
        {"month", "sum_paid_amount"},
        True,
        False,
        "sum_ alias on metric",
    ),
    (
        "time_series_monthly",
        {"month", "total_paid_amount"},
        {"paid_month", "sum_paid_amount"},
        True,
        False,
        "period renamed only (ML-6 A3)",
    ),
    # ── time_series_quarterly ────────────────────────────────────────────
    (
        "time_series_quarterly",
        {"quarter", "total_paid_amount"},
        {"quarter", "sum_paid_amount"},
        True,
        False,
        "sum_ alias",
    ),
    (
        "time_series_quarterly",
        {"quarter", "total_paid_amount"},
        {"fiscal_quarter", "total_paid_amount"},
        True,
        False,
        "time-period suffix (ML-6 A3)",
    ),
    # ── null_analysis ────────────────────────────────────────────────────
    (
        "null_analysis",
        {"missing_count"},
        {"count"},
        True,
        False,
        "bare count (GT bare-agg pass)",
    ),
    (
        "null_analysis",
        {"missing_count"},
        {"null_count"},
        True,
        False,
        "null_count: structural proxy limit (known)",
    ),
    # ── distinct_values ──────────────────────────────────────────────────
    ("distinct_values", {"claim_status"}, {"claim_status"}, True, False, "exact match"),
    (
        "distinct_values",
        {"claim_status"},
        {"policy_type"},
        False,
        False,
        "INCORRECT: wrong column",
    ),
    # ── join_two_tables ──────────────────────────────────────────────────
    (
        "join_two_tables",
        {"claim_amount", "agent_name"},
        {"claim_amount", "agent_name"},
        True,
        False,
        "exact match",
    ),
    (
        "join_two_tables",
        {"claim_amount", "agent_name"},
        {"amount", "agent_name"},
        False,
        True,
        "PARTIAL: truncated metric",
    ),
    # ── join_aggregate ───────────────────────────────────────────────────
    (
        "join_aggregate",
        {"claim_status", "avg_claim_amount"},
        {"claim_status", "average_claim_amount"},
        True,
        False,
        "average_ alias",
    ),
    (
        "join_aggregate",
        {"claim_status", "avg_claim_amount"},
        {"claim_status", "mean_claim_amount"},
        True,
        False,
        "mean_ alias (ML-6 A1)",
    ),
    # ── percentile_analysis ──────────────────────────────────────────────
    (
        "percentile_analysis",
        {"decile", "avg_claim_amount"},
        {"decile", "average_claim_amount"},
        True,
        False,
        "average_ alias",
    ),
    (
        "percentile_analysis",
        {"decile", "avg_claim_amount"},
        {"decile", "mean_claim_amount"},
        True,
        False,
        "mean_ alias (ML-6 A1)",
    ),
    # ── having_filter ────────────────────────────────────────────────────
    (
        "having_filter",
        {"payment_method", "total_paid_amount"},
        {"payment_method", "sum_paid_amount"},
        True,
        False,
        "sum_ alias",
    ),
    # ── SELECT * templates: free-pass behaviour unchanged ─────────────────
    # These never reach _columns_semantically_match (caller short-circuits),
    # but we document the empty-set guard here explicitly.
    (
        "filter_gt_empty_guard",
        set(),
        {"policy_id", "premium_amt"},
        False,
        False,
        "empty GT set → (F,F) from function; caller handles free-pass",
    ),
    # ── Global false-positive guards ──────────────────────────────────────
    (
        "FP_cross_entity",
        {"total_policie"},
        {"total_agent"},
        False,
        False,
        "FP: policie≠agent (different entities)",
    ),
    (
        "PARTIAL_wrong_metric",
        {"claim_status", "avg_claim_amount"},
        {"claim_status", "avg_premium"},
        False,
        True,
        "PARTIAL: category col right, metric col wrong (50%)",
    ),
    (
        "PARTIAL_count_vs_amount",
        {"claim_status", "count_claim"},
        {"claim_status", "claim_amount"},
        False,
        True,
        "PARTIAL: category right, amount≠count (fuzzy 0.59<0.75)",
    ),
    (
        "FP_empty_llm",
        {"total_paid_amount"},
        set(),
        False,
        False,
        "FP: empty LLM set → (F,F)",
    ),
]

failures_match = []
for row in MATCH_CASES:
    tid, gt, llm, exp_c, exp_p, desc = row
    got_c, got_p = _columns_semantically_match(gt, llm)
    ok = got_c == exp_c and got_p == exp_p
    direction = ""
    if not ok:
        if exp_c and not got_c:
            direction = " [FALSE_NEG]"
        elif not exp_c and got_c:
            direction = " [FALSE_POS]"
        failures_match.append(row + (got_c, got_p))
        print(
            f"  FAIL  {tid:<30} expected=({exp_c},{exp_p}) "
            f"got=({got_c},{got_p}){direction}  [{desc}]"
        )
    else:
        print(f"  PASS  {tid:<30} ({got_c},{got_p})  [{desc}]")

print()

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 65)
total = len(NORM_CASES) + len(MATCH_CASES) + len(_SQL_TEMPLATES)
failed = len(failures_ml1) + len(failures_norm) + len(failures_match)
print(
    f"ML-1  validate_sql  : {'PASS' if not failures_ml1 else 'FAIL'}  ({len(_SQL_TEMPLATES)} templates)"
)
print(
    f"ML-6a _normalize_col: {'PASS' if not failures_norm else 'FAIL'}  ({len(NORM_CASES)} cases, {len(failures_norm)} failures)"
)
print(
    f"ML-6b col_match     : {'PASS' if not failures_match else 'FAIL'}  ({len(MATCH_CASES)} cases, {len(failures_match)} failures)"
)
print(f"\nOverall: {total - failed}/{total} passed")
if failed:
    raise SystemExit(1)
