"""
tests/test_sql_postprocessor.py
Tests for analyst_copilot.sql_postprocessor

Coverage matrix
───────────────
F1  Integer division    — COUNT/SUM numerator CAST
F2  Division-by-zero    — NULLIF denominator guard
W1  Fan-out detection   — payments/claims/policies join
    • BUG-05 fix: CTE pre-agg suppresses false-positive
    • P3-D  fix: no SUM → no warning
PostProcessResult       — was_modified, needs_retry, retry_hint properties
Parse failure           — graceful degradation, original SQL preserved
Combined fixes          — F1 + F2 applied together; result is re-parseable
"""

from __future__ import annotations

import pytest
import sqlglot

from analyst_copilot.sql_postprocessor import PostProcessResult, postprocess_sql

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f1(r: PostProcessResult) -> bool:
    return any(s.startswith("F1") for s in r.fixes_applied)


def _f2(r: PostProcessResult) -> bool:
    return any(s.startswith("F2") for s in r.fixes_applied)


def _w1(r: PostProcessResult) -> bool:
    return any(s.startswith("W1") for s in r.warnings)


# ---------------------------------------------------------------------------
# F1 — Integer division: COUNT/SUM numerators cast to NUMERIC
# ---------------------------------------------------------------------------


class TestF1IntegerDivision:
    def test_count_over_count_receives_cast(self):
        r = postprocess_sql("SELECT COUNT(approved) / COUNT(total) AS rate FROM claims")
        assert _f1(r)
        assert "NUMERIC" in r.sql.upper() or "CAST" in r.sql.upper()

    def test_sum_over_sum_receives_cast(self):
        r = postprocess_sql("SELECT SUM(paid) / SUM(claimed) AS ratio FROM payments")
        assert _f1(r)

    def test_f1_fix_label_names_the_aggregate(self):
        r = postprocess_sql("SELECT COUNT(a) / COUNT(b) FROM t")
        assert any("COUNT" in fix for fix in r.fixes_applied)

    def test_already_cast_to_numeric_is_skipped(self):
        r = postprocess_sql("SELECT CAST(COUNT(a) AS NUMERIC) / COUNT(b) FROM t")
        assert not _f1(r)

    def test_already_cast_to_float_is_skipped(self):
        r = postprocess_sql("SELECT CAST(COUNT(a) AS FLOAT) / COUNT(b) FROM t")
        assert not _f1(r)

    def test_already_cast_to_decimal_is_skipped(self):
        r = postprocess_sql("SELECT CAST(SUM(x) AS DECIMAL) / SUM(y) FROM t")
        assert not _f1(r)

    def test_plain_column_numerator_no_f1(self):
        # Non-aggregate numerator — F1 only targets COUNT/SUM
        r = postprocess_sql("SELECT revenue / total_count FROM summary")
        assert not _f1(r)

    def test_constant_divisor_no_f1(self):
        # Plain column (no aggregate) over a constant literal — F1 only targets
        # COUNT/SUM numerators, so it must not fire here (TEST-F FIX: original SQL
        # had SUM(premium) as numerator which correctly triggers F1; replaced with
        # a plain column reference so the "no F1 on constant divisor" intent holds).
        r = postprocess_sql("SELECT premium / 100 AS pct FROM policies")
        assert not _f1(r)


# ---------------------------------------------------------------------------
# F2 — Division-by-zero: denominator wrapped in NULLIF(..., 0)
# ---------------------------------------------------------------------------


class TestF2NullifGuard:
    def test_aggregate_denominator_gets_nullif(self):
        r = postprocess_sql("SELECT SUM(a) / SUM(b) FROM t")
        assert _f2(r)
        assert "NULLIF" in r.sql.upper()

    def test_column_denominator_gets_nullif(self):
        r = postprocess_sql("SELECT revenue / total FROM summary")
        assert _f2(r)

    def test_constant_divisor_skips_f2(self):
        """/100 for percentage scaling is intentional — must not be guarded."""
        r = postprocess_sql("SELECT SUM(premium) / 100 AS pct FROM policies")
        assert not _f2(r)

    def test_constant_1000_skips_f2(self):
        r = postprocess_sql("SELECT val / 1000 FROM t")
        assert not _f2(r)

    def test_already_nullif_wrapped_skips_f2(self):
        r = postprocess_sql("SELECT a / NULLIF(b, 0) FROM t")
        assert not _f2(r)

    def test_f2_fix_label_mentions_nullif(self):
        r = postprocess_sql("SELECT x / y FROM t")
        f2_labels = [f for f in r.fixes_applied if f.startswith("F2")]
        assert f2_labels
        assert "NULLIF" in f2_labels[0]


# ---------------------------------------------------------------------------
# W1 — Fan-out detection
# ---------------------------------------------------------------------------

_THREE_TABLE_FANOUT = """
    SELECT p.policy_type,
           SUM(c.claim_amount)  AS total_claims,
           SUM(pay.paid_amount) AS total_paid
    FROM policies p
    JOIN claims  c   ON p.policy_id  = c.policy_id
    JOIN payments pay ON c.claim_id  = pay.claim_id
    GROUP BY p.policy_type
"""

_CLAIMS_POLICIES_FANOUT = """
    SELECT p.policy_type, SUM(c.claim_amount)
    FROM policies p
    JOIN claims c ON p.policy_id = c.policy_id
    GROUP BY p.policy_type
"""

_PAYMENTS_CLAIMS_FANOUT = """
    SELECT c.claim_id, SUM(pay.paid_amount)
    FROM claims c
    JOIN payments pay ON c.claim_id = pay.claim_id
    GROUP BY c.claim_id
"""

_CTE_PRE_AGG = """
    WITH payment_agg AS (
        SELECT claim_id, SUM(paid_amount) AS total_paid
        FROM payments GROUP BY claim_id
    ),
    claim_agg AS (
        SELECT policy_id, SUM(claim_amount) AS total_claim
        FROM claims GROUP BY policy_id
    )
    SELECT p.policy_id, ca.total_claim, pa.total_paid
    FROM policies p
    JOIN claim_agg   ca ON p.policy_id = ca.policy_id
    JOIN payment_agg pa ON p.policy_id = pa.policy_id
"""


class TestW1FanoutDetection:
    def test_three_table_join_with_sum_triggers_two_w1s(self):
        r = postprocess_sql(_THREE_TABLE_FANOUT)
        w1s = [w for w in r.warnings if w.startswith("W1")]
        assert len(w1s) == 2  # payments→claims AND claims→policies

    def test_three_table_fanout_sets_needs_retry(self):
        assert postprocess_sql(_THREE_TABLE_FANOUT).needs_retry is True

    def test_claims_policies_join_emits_w1(self):
        r = postprocess_sql(_CLAIMS_POLICIES_FANOUT)
        assert _w1(r)

    def test_payments_claims_join_emits_w1(self):
        r = postprocess_sql(_PAYMENTS_CLAIMS_FANOUT)
        assert _w1(r)

    def test_no_w1_without_sum_aggregate(self):
        """P3-D fix: pure row-level join on fan-out tables must not trigger W1."""
        sql = """
            SELECT p.policy_id, c.claim_id, pay.paid_amount
            FROM policies p
            JOIN claims   c   ON p.policy_id = c.policy_id
            JOIN payments pay ON c.claim_id  = pay.claim_id
        """
        r = postprocess_sql(sql)
        assert not _w1(r)
        assert r.needs_retry is False

    def test_cte_pre_aggregation_suppresses_w1(self):
        """BUG-05 fix: tables only in CTE bodies must not count as outer tables."""
        r = postprocess_sql(_CTE_PRE_AGG)
        assert not _w1(r)
        assert r.needs_retry is False

    def test_no_w1_for_single_table_sum(self):
        r = postprocess_sql("SELECT SUM(premium_amt) FROM policies")
        assert not _w1(r)


# ---------------------------------------------------------------------------
# PostProcessResult properties
# ---------------------------------------------------------------------------


class TestPostProcessResultProperties:
    def test_was_modified_true_when_fixes_applied(self):
        r = postprocess_sql("SELECT COUNT(a) / COUNT(b) FROM t")
        assert r.was_modified is True

    def test_was_modified_false_for_clean_query(self):
        r = postprocess_sql("SELECT policy_id, policy_type FROM policies LIMIT 10")
        assert r.was_modified is False

    def test_needs_retry_false_without_w1(self):
        # F1/F2 applied but no fan-out → no retry needed
        r = postprocess_sql("SELECT COUNT(a) / COUNT(b) FROM t")
        assert r.needs_retry is False

    def test_needs_retry_true_with_w1(self):
        assert postprocess_sql(_CLAIMS_POLICIES_FANOUT).needs_retry is True

    def test_retry_hint_empty_when_no_w1(self):
        r = postprocess_sql("SELECT policy_id FROM policies")
        assert r.retry_hint == ""

    def test_retry_hint_contains_cte_guidance_when_w1(self):
        r = postprocess_sql(_CLAIMS_POLICIES_FANOUT)
        assert r.needs_retry is True
        assert "CTE" in r.retry_hint
        assert len(r.retry_hint) > 50

    def test_retry_hint_covers_all_w1_warnings(self):
        r = postprocess_sql(_THREE_TABLE_FANOUT)
        # Both payments and claims are fan-out sources; hint must mention both
        assert "payment" in r.retry_hint.lower()
        assert "claim" in r.retry_hint.lower()

    def test_fixes_applied_is_list(self):
        assert isinstance(postprocess_sql("SELECT 1").fixes_applied, list)

    def test_warnings_is_list(self):
        assert isinstance(postprocess_sql("SELECT 1").warnings, list)

    def test_sql_field_present_unchanged_for_clean_query(self):
        sql = "SELECT id FROM policies LIMIT 5"
        r = postprocess_sql(sql)
        # For a no-fix query the sql field is set (may be re-formatted or same)
        assert r.sql  # non-empty


# ---------------------------------------------------------------------------
# Parse failure — graceful degradation
# ---------------------------------------------------------------------------


class TestParseFailure:
    def test_invalid_sql_returns_original_unchanged(self):
        bad = "NOT VALID SQL $$$$ ???"
        r = postprocess_sql(bad)
        assert r.sql == bad

    def test_invalid_sql_adds_parse_warning(self):
        r = postprocess_sql("NOT VALID SQL $$$$ ???")
        assert any("parse failed" in w.lower() for w in r.warnings)

    def test_invalid_sql_was_modified_false(self):
        r = postprocess_sql("NOT VALID SQL $$$$ ???")
        assert r.was_modified is False

    def test_invalid_sql_needs_retry_false(self):
        r = postprocess_sql("NOT VALID SQL $$$$ ???")
        assert r.needs_retry is False

    def test_empty_string_does_not_raise(self):
        # Empty SQL should not raise — it either parses as nothing or fails gracefully
        try:
            r = postprocess_sql("")
            assert isinstance(r, PostProcessResult)
        except Exception as exc:
            pytest.fail(f"postprocess_sql('') raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Combined F1 + F2 applied together
# ---------------------------------------------------------------------------


class TestCombinedFixes:
    def test_f1_and_f2_both_applied(self):
        # COUNT numerator (F1) over column denominator (F2)
        r = postprocess_sql("SELECT COUNT(approved) / COUNT(total) FROM claims")
        assert _f1(r)
        assert _f2(r)
        assert len(r.fixes_applied) >= 2

    def test_fixed_sql_is_re_parseable(self):
        """The fixed SQL must be valid PostgreSQL — sqlglot must parse it cleanly."""
        r = postprocess_sql("SELECT SUM(paid) / SUM(claim) FROM payments")
        # Must not raise
        tree = sqlglot.parse_one(r.sql, dialect="postgres")
        assert tree is not None

    def test_f1_f2_and_w1_all_fire_on_canonical_query(self):
        """The self-test SQL from the module docstring hits all three paths."""
        sql = """
            SELECT
                p.policy_type,
                SUM(c.claim_amount) / SUM(p.premium_amt) AS loss_ratio,
                COUNT(CASE WHEN c.claim_status = 'approved' THEN 1 END) / COUNT(c.claim_id)
                    AS approval_rate
            FROM policies p
            JOIN claims   c   ON p.policy_id = c.policy_id
            JOIN payments pay ON c.claim_id  = pay.claim_id
            GROUP BY p.policy_type
        """
        r = postprocess_sql(sql)
        assert _f1(r), "Expected F1 fix on COUNT/COUNT"
        assert _f2(r), "Expected F2 NULLIF on denominator"
        assert _w1(r), "Expected W1 fan-out warning"
        assert r.needs_retry is True
