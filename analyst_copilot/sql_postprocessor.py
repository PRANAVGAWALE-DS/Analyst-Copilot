"""
sql_postprocessor.py
--------------------
Post-process LLM-generated SQL *before* DB execution in analyst_copilot.

Automatic fixes applied every query (no human needed):
  F1  Integer division    — COUNT/SUM numerators cast to DECIMAL/NUMERIC
  F2  Division-by-zero    — aggregate denominators wrapped in NULLIF(..., 0)

Warnings emitted → trigger LLM self-correction retry (see needs_retry):
  W1  Fan-out risk        — payments/claims/policies joined without CTE pre-aggregation

Schema context (analyst_copilot insurance DB):
  policies (1) ──< claims (many) ──< payments (many)

Usage:
    from sql_postprocessor import postprocess_sql

    pp = postprocess_sql(llm_sql)

    if pp.needs_retry:
        # Re-prompt LLM; fanout cannot be fixed by AST rewrite alone
        llm_sql = regenerate_with_hint(original_query, pp.retry_hint)
        pp = postprocess_sql(llm_sql)          # apply F1/F2 to retry result too

    result["generated_code"]   = pp.sql
    result["fixes_applied"]    = pp.fixes_applied   # surfaced in API response
    result["warnings"]         = pp.warnings

Install dependency:
    pip install sqlglot
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import ParseError, SqlglotError

logger = logging.getLogger(__name__)

_DIALECT = "postgres"

# Cast types that already satisfy float semantics — skip F1 if present
_FLOAT_TYPES = {"NUMERIC", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "FLOAT4", "FLOAT8"}

# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PostProcessResult:
    """Output of postprocess_sql().  All fields are safe to log / return in API."""

    sql: str  # Fixed SQL (or original if parse failed)
    fixes_applied: list[str] = field(default_factory=list)  # Audit trail of F1/F2
    warnings: list[str] = field(default_factory=list)  # W1 fanout warnings

    @property
    def was_modified(self) -> bool:
        """True if any automatic fix was applied."""
        return bool(self.fixes_applied)

    @property
    def needs_retry(self) -> bool:
        """
        True when a warning requires the LLM to regenerate the query.
        (W1 fan-out cannot be fixed by AST rewrite; needs schema-aware CTEs.)
        """
        return any(w.startswith("W1") for w in self.warnings)

    @property
    def retry_hint(self) -> str:
        """
        Formatted hint to append to the LLM retry prompt when needs_retry is True.
        Explains exactly what went wrong and what pattern to use.
        """
        w1_warnings = [w for w in self.warnings if w.startswith("W1")]
        if not w1_warnings:
            return ""
        hint_lines = [
            "⚠ Your previous SQL had a JOIN fan-out problem that inflates SUM results.",
            "Details:",
        ]
        for w in w1_warnings:
            # Strip the 'W1 FANOUT RISK: ' prefix for cleaner LLM prompt
            hint_lines.append(f"  - {w.split(':', 1)[-1].strip()}")
        hint_lines.append(
            "\nFix: use CTEs to pre-aggregate before joining. "
            "First CTE: aggregate payments by claim_id. "
            "Second CTE: aggregate claims (with payment totals) by policy_id. "
            "Final SELECT: join CTEs to policies. "
            "See the CTE pattern in the SQL rules."
        )
        return "\n".join(hint_lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def postprocess_sql(raw_sql: str) -> PostProcessResult:
    """
    Apply systematic fixes to LLM-generated SQL before execution.

    Safe to call on every query.  Returns original SQL unchanged on parse failure
    (with a warning) so execution can still proceed.

    Parameters
    ----------
    raw_sql : str
        SQL as returned by the LLM, before any execution.

    Returns
    -------
    PostProcessResult
        .sql           — fixed SQL ready for execution
        .fixes_applied — list of transformations applied (for response audit)
        .warnings      — issues detected that require LLM retry
        .needs_retry   — bool shortcut: True when W1 warnings present
        .retry_hint    — formatted hint string for LLM re-prompt
    """
    result = PostProcessResult(sql=raw_sql)

    try:
        tree = sqlglot.parse_one(
            raw_sql,
            dialect=_DIALECT,
            error_level=sqlglot.ErrorLevel.RAISE,
        )
    except (ParseError, SqlglotError) as exc:
        result.warnings.append(f"SQL parse failed — post-processing skipped: {exc}")
        return result

    tree, f1 = _fix_integer_division(tree)
    tree, f2 = _fix_missing_nullif(tree)
    warnings = _warn_fanout(tree)

    if f1 or f2:
        result.sql = tree.sql(dialect=_DIALECT, pretty=True)

    result.fixes_applied = f1 + f2
    result.warnings = warnings

    if result.was_modified:
        logger.info(
            "[sql_postprocessor] %d fix(es) applied: %s",
            len(result.fixes_applied),
            result.fixes_applied,
        )
    if warnings:
        logger.warning(
            "[sql_postprocessor] %d warning(s): %s",
            len(warnings),
            warnings,
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# F1 — Integer division: cast COUNT/SUM numerators to DECIMAL
# ──────────────────────────────────────────────────────────────────────────────


def _fix_integer_division(
    tree: exp.Expression,
) -> tuple[exp.Expression, list[str]]:
    """
    Wraps COUNT/SUM expressions that are direct numerators in CAST(... AS DECIMAL).

    Why needed:
        PostgreSQL COUNT() returns BIGINT.
        BIGINT / BIGINT = BIGINT (truncated toward zero).
        COUNT(approved) / COUNT(total) = 0 when approved < total.

    Skips:
        - Numerators already wrapped in a float-compatible CAST
          (NUMERIC, FLOAT, DOUBLE, DECIMAL, REAL, FLOAT4, FLOAT8)
        - Non-aggregate numerators
    """
    fixes: list[str] = []

    for div_node in tree.find_all(exp.Div):
        numerator = div_node.this

        # Already cast to a float-compatible type → skip
        if isinstance(numerator, exp.Cast):
            dtype_upper = numerator.to.sql(dialect=_DIALECT).upper()
            if any(ft in dtype_upper for ft in _FLOAT_TYPES):
                continue

        if isinstance(numerator, exp.Count | exp.Sum):
            cast_node = exp.Cast(
                this=numerator.copy(),  # type: ignore[no-untyped-call]
                to=exp.DataType.build("NUMERIC"),
            )
            div_node.set("this", cast_node)
            agg_name = type(numerator).__name__.upper()
            fixes.append(
                f"F1: CAST({agg_name}(...) AS NUMERIC) — prevents integer division truncation"
            )

    return tree, fixes


# ──────────────────────────────────────────────────────────────────────────────
# F2 — Division-by-zero: wrap aggregate denominators in NULLIF(..., 0)
# ──────────────────────────────────────────────────────────────────────────────


def _fix_missing_nullif(
    tree: exp.Expression,
) -> tuple[exp.Expression, list[str]]:
    """
    Wraps non-constant division denominators in NULLIF(..., 0).

    Skips:
        - Denominators already wrapped in NULLIF
        - Plain numeric literals (e.g. / 100 for percentage scaling — intentional)
    """
    fixes: list[str] = []

    for div_node in tree.find_all(exp.Div):
        denominator = div_node.expression

        if isinstance(denominator, exp.Nullif):
            continue  # Already guarded

        # Plain constant divisor (/ 100, / 1000, etc.) — intentional, skip
        if isinstance(denominator, exp.Literal) and denominator.is_number:
            continue

        nullif_node = exp.Nullif(
            this=denominator.copy(),
            expression=exp.Literal.number(0),
        )
        div_node.set("expression", nullif_node)
        fixes.append("F2: NULLIF(denominator, 0) — prevents ZeroDivisionError")

    return tree, fixes


# ──────────────────────────────────────────────────────────────────────────────
# W1 — Fan-out detection (warning only; auto-fix requires schema-aware CTEs)
# ──────────────────────────────────────────────────────────────────────────────

# Schema cardinalities:
#   policies (1) ──< claims (many)    — joining inflates SUM(premium_amt)
#   claims   (1) ──< payments (many)  — joining inflates SUM(claim_amount)


def _warn_fanout(tree: exp.Expression) -> list[str]:
    """
    Detects fan-out joins that will inflate SUM aggregates.

    Logic:
        1. Collect table names referenced in the *outer* query (exclude tables
           that appear only inside CTE bodies — BUG-05 FIX: the old code used
           find_all(exp.Table) on the whole AST, so a CTE body's FROM clause
           added fan-out table names to all_tables even when the CTE correctly
           pre-aggregated them, causing a false-positive W1 warning).
        2. Collect CTE aliases — a CTE with 'payment' or 'claim' in its name
           signals that pre-aggregation is already in place (heuristic, works
           for LLM-generated CTEs which follow standard naming).
        3. P3-D FIX: only emit W1 when at least one SUM aggregate is present.
           Fan-out inflates SUM results; pure row-level SELECTs against these
           tables without any SUM are unaffected and should not trigger a retry.
        4. Emit W1 when fan-out tables are joined without matching pre-agg CTEs.
    """
    warnings: list[str] = []

    # BUG-05 FIX: subtract tables that appear only inside CTE body expressions.
    # find_all() is depth-first on the full tree; CTE bodies contain their own
    # FROM clauses whose table references must not pollute the outer table set.
    cte_body_tables: set[str] = {
        t.name.lower() for cte in tree.find_all(exp.CTE) for t in cte.find_all(exp.Table) if t.name
    }
    all_tables = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    outer_tables = all_tables - cte_body_tables

    cte_aliases = {c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias}

    has_payments = "payments" in outer_tables
    has_claims = "claims" in outer_tables
    has_policies = "policies" in outer_tables

    payments_pre_agg = any("payment" in alias for alias in cte_aliases)
    claims_pre_agg = any("claim" in alias for alias in cte_aliases)

    # P3-D FIX: fan-out only inflates SUM results.  Skip the warning entirely
    # for row-level queries that contain no SUM aggregate — they are unaffected
    # by the fan-out and would otherwise receive a spurious retry round-trip.
    has_sum = bool(tree.find(exp.Sum))
    if not has_sum:
        return warnings

    # payments → claims fan-out
    if has_payments and has_claims and not payments_pre_agg:
        warnings.append(
            "W1 FANOUT RISK: `payments` joined to `claims` without a CTE pre-aggregating "
            "payments by claim_id first. SUM(paid_amount) and SUM(claim_amount) will both be "
            "inflated by the number of payments per claim. "
            "Aggregate payments per claim_id in a CTE, then join."
        )

    # claims → policies fan-out (warn independently — both may apply)
    if has_claims and has_policies and not claims_pre_agg:
        warnings.append(
            "W1 FANOUT RISK: `claims` joined to `policies` without a CTE pre-aggregating "
            "claims by policy_id first. SUM(premium_amt) will be inflated by the number of "
            "claims per policy. "
            "Aggregate claims per policy_id in a CTE, then join."
        )

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test  (python sql_postprocessor.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _TEST_SQL = """
    SELECT
        p.policy_type,
        SUM(c.claim_amount) AS total_claim_amount,
        SUM(pay.paid_amount) AS total_paid_amount,
        COUNT(CASE WHEN c.claim_status IN ('approved', 'paid') THEN 1 END) / COUNT(c.claim_id) AS claim_approval_rate,
        SUM(c.claim_amount) / SUM(p.premium_amt) AS loss_ratio
    FROM policies p
    JOIN claims c ON p.policy_id = c.policy_id
    LEFT JOIN payments pay ON c.claim_id = pay.claim_id
    WHERE c.claim_amount IS NOT NULL
    GROUP BY p.policy_type
    ORDER BY p.policy_type
    """

    pp = postprocess_sql(_TEST_SQL)

    print("=" * 60)
    print("FIXES APPLIED")
    print("=" * 60)
    for f in pp.fixes_applied:
        print(f"  ✓ {f}")
    if not pp.fixes_applied:
        print("  (none)")

    print("\nWARNINGS")
    print("=" * 60)
    for w in pp.warnings:
        print(f"  ⚠ {w}")
    if not pp.warnings:
        print("  (none)")

    print(f"\nneeds_retry = {pp.needs_retry}")

    if pp.needs_retry:
        print("\nRETRY HINT FOR LLM")
        print("=" * 60)
        print(pp.retry_hint)

    print("\nFIXED SQL")
    print("=" * 60)
    print(pp.sql)
