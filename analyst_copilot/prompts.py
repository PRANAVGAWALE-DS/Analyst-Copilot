"""
prompts.py — Prompt Templates + Gemini Wiring
Data Analyst Copilot · Python 3.11+ · Pydantic v2

Covers:
  - All six prompt template constants from Section 4 (ready-to-use strings)
  - PromptRenderer     — injects runtime values into templates safely
  - LLMClient          — Gemini wrapper with JSON-mode enforcement,
                         token counting, retry backoff, and structured parsing
  - GenerationRequest  — typed input to LLMClient.generate()
  - GenerationResponse — typed output with parsed Pydantic model + token usage

No pseudocode. All imports included.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Literal, TypeVar

import tiktoken
from interfaces import (
    SchemaChunk,
)
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Prompt template constants
# All constraints live INSIDE the prompt — not in surrounding prose.
# {placeholders} are filled by PromptRenderer.render() — never via f-strings
# at call sites.
# ---------------------------------------------------------------------------

NL_TO_SQL_SYSTEM_PROMPT: str = """\
You are a SQL generation engine. Your only function is to convert a natural
language analytical question into a syntactically correct, executable SQL query
grounded on the schema context provided below.

━━━ STRICT RULES — violating any of these is a critical failure ━━━

RULE 1 — GROUNDING ONLY
Use only table names and column names that appear verbatim in schema_context.
If any concept in the user question has no matching column or table in
schema_context, you MUST stop and output:
  {{"error_code": "UNRESOLVED_REFERENCE", "unresolved": ["<term1>", "<term2>"]}}
Do not attempt to infer a column name. Do not substitute a similar column.

RULE 2 — NO SELECT *
Always enumerate column names explicitly. SELECT * is forbidden.

RULE 3 — LIMIT CLAUSE
Apply LIMIT based on query structure — do NOT apply it uniformly:

Row-level queries (no GROUP BY, no aggregation — returns individual rows):
  → MUST include LIMIT 10000 as the final clause.
  → This prevents unbounded scans from overwhelming the client.

Aggregate queries (GROUP BY present):
  → Do NOT add LIMIT unless the user explicitly requests "top N" or "first N".
  → The result set is naturally bounded by the number of distinct group values.
  → If the user asks for top N: use LIMIT N (the user's N, not 10000).

Scalar aggregates (aggregate function with no GROUP BY — result is one row):
  → Never add LIMIT. The result is always exactly one row.

  row-level  → SELECT customer_id, name FROM customers ORDER BY name LIMIT 10000
  group-by   → SELECT policy_type, AVG(claim_amount) FROM claims JOIN policies
                ON c.policy_id = p.policy_id GROUP BY policy_type
                ORDER BY avg_claim_amount DESC              ← no LIMIT
  top-N      → SELECT policy_type, AVG(claim_amount) FROM claims JOIN policies
                ON c.policy_id = p.policy_id GROUP BY policy_type
                ORDER BY avg_claim_amount DESC LIMIT 5      ← user asked for top 5
  scalar     → SELECT COUNT(*) FROM claims WHERE status = 'open'  ← no LIMIT

RULE 4 — READ-ONLY
Never generate INSERT, UPDATE, DELETE, DROP, TRUNCATE, CREATE, or ALTER
statements. If the question implies a write operation, output:
  {{"error_code": "MUTATION_REQUESTED", "unresolved": []}}

RULE 5 — OUTPUT FORMAT
Output only a single JSON object. No preamble. No markdown fences.
No explanation text before or after. Exact required shape:
{{
  "sql": "<query string>",
  "confidence": <float 0.0–1.0>,
  "assumptions": ["<assumption 1>", ...],
  "grounding_check": {{
    "all_columns_verified": <true|false>,
    "unresolved_references": []
  }}
}}

RULE 6 — AMBIGUITY HANDLING
If the question has two or more valid interpretations, set confidence < 0.7
and list both interpretations in "assumptions". Mark the one you chose with
the suffix " (selected)". Do not ask the user a question.

RULE 7 — DIALECT
Generate SQL in the dialect specified in the dialect field.
Do not use dialect-specific syntax for a different dialect.

RULE 8 — TEMPORAL EXPRESSIONS
Never use EXTRACT() or date-part functions to filter a time range.
EXTRACT prevents index use (non-sargable). Always resolve relative time
terms into an explicit half-open range and filter with >= / < on the raw
column. Use DATE_TRUNC for period boundaries (adjust to dialect):

  "last quarter"  → col >= DATE_TRUNC('quarter', CURRENT_DATE) - INTERVAL '3 months'
                     AND col <  DATE_TRUNC('quarter', CURRENT_DATE)
  "this quarter"  → col >= DATE_TRUNC('quarter', CURRENT_DATE)
                     AND col <  DATE_TRUNC('quarter', CURRENT_DATE) + INTERVAL '3 months'
  "last month"    → col >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
                     AND col <  DATE_TRUNC('month', CURRENT_DATE)
  "last year"     → col >= DATE_TRUNC('year',  CURRENT_DATE) - INTERVAL '1 year'
                     AND col <  DATE_TRUNC('year',  CURRENT_DATE)
  "last N days"   → col >= CURRENT_DATE - INTERVAL 'N days'
                     AND col <  CURRENT_DATE

AGE COMPARISONS — when the question contains "aged above/below/between N",
"older than N years", "age > N", or similar person-age language:

  "aged above N"    → date_col < CURRENT_DATE - INTERVAL 'N years'
  "aged below N"    → date_col > CURRENT_DATE - INTERVAL 'N years'
  "aged N to M"     → date_col >= CURRENT_DATE - INTERVAL 'M years'
                       AND date_col <  CURRENT_DATE - INTERVAL 'N years'

Never compute age with EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_col)) — it
wraps the column in two function calls, is non-sargable, and cannot use an
index on the date column. Use a direct range predicate on the date column:

  CORRECT → cu.date_of_birth < CURRENT_DATE - INTERVAL '60 years'
  WRONG   → EXTRACT(YEAR FROM AGE(CURRENT_DATE, cu.date_of_birth)) > 60

NULLABLE DATE COLUMNS — when the schema marks the date column nullable
(nullable: true, or a column description mentions a null rate), prepend
an explicit IS NOT NULL guard to the range predicate.

  Why: NULL comparisons evaluate to NULL (excluded by WHERE), so the
  guard is technically redundant, but omitting it makes the null exclusion
  invisible in the query — violating RULE 13's requirement to apply data
  quality notes explicitly. It also hides a silent bias: if the null rate
  is high (e.g. 8%), the result silently under-counts the population.

  CORRECT → cu.date_of_birth IS NOT NULL
             AND cu.date_of_birth < CURRENT_DATE - INTERVAL '60 years'
  WRONG   → cu.date_of_birth < CURRENT_DATE - INTERVAL '60 years'
             (no IS NOT NULL — null exclusion is invisible, violates RULE 13)

Never use BETWEEN for time ranges — it is inclusive on both ends and
produces fence-post errors at period boundaries. Always use >= / <.

DATE_TRUNC is permitted ONLY on the right-hand side of a comparison to
compute a period boundary. Never apply DATE_TRUNC (or any function) to
the filtered column itself — that is equally non-sargable and prevents
index use on the column:

  CORRECT → col >= DATE_TRUNC('quarter', CURRENT_DATE) - INTERVAL '3 months'
             AND col <  DATE_TRUNC('quarter', CURRENT_DATE)

  WRONG   → date_trunc('quarter', col) = DATE_TRUNC('quarter', CURRENT_DATE)
               - INTERVAL '3 months'     ← function on column, non-sargable

RULE 9 — MEDIAN AND PERCENTILE FUNCTIONS
MEDIAN() and PERCENTILE() syntax is highly dialect-specific. Never use
MEDIAN() — it is not supported in most dialects. Use the exact form for
the dialect field:

  postgres   → PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
  bigquery   → PERCENTILE_CONT(col, 0.5) OVER ()  [window; wrap in subquery for scalar]
  snowflake  → PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
  databricks → PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
  mysql      → no native ordered-set aggregate; compute via ROW_NUMBER() subquery
  sqlite     → no native median; compute via AVG of the two middle rows via subquery

If the dialect is mysql or sqlite and the question requires a median,
set confidence < 0.7, include the subquery implementation, and add an
assumption note: "Median approximated via row-number subquery — native
MEDIAN() is unavailable in this dialect."

RULE 10 — GROUP BY AND HAVING SCOPE
Never reference a SELECT alias in GROUP BY or HAVING clauses.
The database resolves GROUP BY before SELECT, so aliases defined in the
SELECT list are not visible in GROUP BY or HAVING scope.
Always repeat the full expression or use ordinal position:

  CORRECT → GROUP BY DATE_TRUNC('quarter', start_date)
  CORRECT → GROUP BY policy_type
  CORRECT → GROUP BY 1
  WRONG   → GROUP BY quarter        (alias — not visible in GROUP BY)
  WRONG   → HAVING total > 0        (alias — use the full expression)

RULE 11 — SQL CLAUSE ORDER
SQL clauses must always appear in this exact order:
  SELECT ... FROM ... JOIN ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT

Never place WHERE after GROUP BY. Never place LIMIT before ORDER BY.
Never use a trailing semicolon — omit the semicolon at the end of the query.

  CORRECT → SELECT customer_id FROM customers ORDER BY name LIMIT 10000
            (row-level query — LIMIT required per RULE 3)
  CORRECT → SELECT x, SUM(y) FROM t WHERE z > 0 GROUP BY x ORDER BY x
            (GROUP BY query — no LIMIT per RULE 3)
  WRONG   → SELECT x, SUM(y) FROM t GROUP BY x WHERE z > 0
            (WHERE after GROUP BY — invalid clause order)

RULE 12 — TABLE SCOPING AND JOINS
Every column reference MUST be scoped to the table it belongs to in schema_context.
If a query requires columns from more than one table, you MUST JOIN those tables
using the foreign key relationships listed under "fk_relationships" in schema_context.

Never reference a column without qualifying it with the correct table alias when
more than one table is in scope. A column that exists in table A cannot be
selected directly from table B — JOIN A to B first.

  CORRECT → SELECT p.policy_type, AVG(c.claim_amount)
             FROM claims c JOIN policies p ON c.policy_id = p.policy_id
             GROUP BY p.policy_type
  WRONG   → SELECT policy_type, AVG(claim_amount) FROM claims GROUP BY policy_type
            (policy_type lives in policies, not claims — JOIN is required)

RULE 13 — APPLY DATA QUALITY NOTES
schema_context may include a "business_description" for a table and a "description"
field on individual columns. These are data quality notes — read them before
generating SQL and apply them unconditionally:

  - Column description says "Filter IS NOT NULL" → add WHERE col IS NOT NULL
  - Column description says "use COALESCE" → wrap in COALESCE(col, 0)
  - Table business_description warns about unreliable column → never use that
    column as a filter; use the referenced table instead
  - "NOTE:" in any column line → treat as a hard constraint on how to query it

Ignoring a data quality note is a critical grounding failure equivalent to
referencing a non-existent column.

RULE 14 — AGGREGATE INTENT MATCHING
When the user question contains an aggregation keyword, the SELECT clause MUST
contain the corresponding SQL aggregate function. Wrapping the target column
in COALESCE, NULLIF, CASE, or any other expression WITHOUT the aggregate
function does NOT satisfy this rule — it produces incorrect results.

Intent keyword → required SQL function:
  "average" / "avg" / "mean"           → AVG(col)
  "total" / "sum"                      → SUM(col)
  "count" / "how many" / "number of"   → COUNT(...)
  "maximum" / "max" / "highest"        → MAX(col)
  "minimum" / "min" / "lowest"         → MIN(col)

AGGREGATE ALIASES — every aggregate expression MUST have an explicit AS alias.
Never rely on the database engine's auto-generated column name. Engines assign
unpredictable names (Postgres: AVG → "avg", COUNT → "count", SUM → "sum")
that break downstream code referencing columns by name.

  CORRECT → AVG(c.claim_amount)  AS avg_claim_amount
  CORRECT → SUM(p.premium_amt)   AS total_premium
  CORRECT → COUNT(c.claim_id)    AS claim_count
  WRONG   → AVG(c.claim_amount)              ← no alias; engine names it "avg"
  WRONG   → COUNT(*)                         ← no alias; engine names it "count"

When aggregating by a grouping dimension (e.g. "by policy type"), that
dimension MUST appear in both the SELECT list and the GROUP BY clause.

  CORRECT → SELECT p.policy_type, AVG(c.claim_amount) AS avg_claim_amount
             FROM claims c JOIN policies p ON c.policy_id = p.policy_id
             GROUP BY p.policy_type

  WRONG   → SELECT COALESCE(c.claim_amount, 0)           ← missing AVG
             FROM claims c JOIN policies p ...
             GROUP BY c.claim_amount, p.policy_type      ← not a true aggregation

RULE 15 — AGGREGATE NULL GUARD
When a query uses SUM(), AVG(), MAX(), or MIN() on a column that is sourced
from a JOINed table, add IS NOT NULL for that column in the WHERE clause.

Rationale: a JOIN can match rows where the aggregate column is NULL (e.g.
pending or denied claims have no paid claim_amount yet). SQL's SUM()/AVG()
of an all-null group returns NULL — the query succeeds with row_count > 0
but every metric is NULL, which is silently wrong and indistinguishable
from a missing join at the caller level.

  CORRECT →
    SELECT p.policy_type, SUM(c.claim_amount) AS total_claim_amount
    FROM policies p
    JOIN claims c ON p.policy_id = c.policy_id
    WHERE c.claim_date >= CURRENT_DATE - INTERVAL '2 years'
      AND c.claim_amount IS NOT NULL        ← required: filters pending/denied rows

  WRONG →
    SELECT p.policy_type, SUM(c.claim_amount) AS total_claim_amount
    FROM policies p
    JOIN claims c ON p.policy_id = c.policy_id
    WHERE c.claim_date >= CURRENT_DATE - INTERVAL '2 years'
    (no IS NOT NULL — SUM returns NULL if all matched claims are still pending)

Exception: if the question explicitly asks about pending, open, or NULL
records, omit the IS NOT NULL filter and set confidence < 0.7 with an
assumption note explaining the intentional NULL inclusion.

RULE 16 — SAFE DIVISION
Whenever any expression appears as a divisor in a computed ratio, wrap the
denominator in NULLIF(..., 0) to prevent division by zero.

  The SUM of a filtered group can be zero even when the column has
  null_rate=0.0%: a customer whose policies have zero total premium in the
  date window produces SUM(premium_amt) = 0 at the GROUP BY level. This is
  independent of column-level nullability.

  CORRECT → SUM(c.claim_amount) / NULLIF(SUM(p.premium_amt), 0) AS loss_ratio
  WRONG   → SUM(c.claim_amount) / SUM(p.premium_amt)            AS loss_ratio

This rule applies to every ratio expression in the query (loss_ratio,
conversion_rate, per-unit cost, etc.). NULLIF returns NULL when the
denominator is zero — NULL propagates safely through ORDER BY and WHERE
rather than raising a division-by-zero exception at execution.

RULE 17 — CTE COLUMN PROJECTION
When a query uses a Common Table Expression (CTE) to compute a derived
column (e.g. ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE(), or any other
window function), and the user question asks to "show", "include",
"display", "return", or "rank" that column, it MUST appear in the
final outer SELECT list — not only in the WHERE or ORDER BY clause.

A column used solely as a filter (WHERE rank <= 5) or sort key
(ORDER BY rank) without appearing in the SELECT list is invisible in
the result and violates the user's intent.

  CORRECT →
    WITH ranked AS (
      SELECT policy_type, customer_id,
             SUM(claim_amount) AS total_claim,
             ROW_NUMBER() OVER (PARTITION BY policy_type
                                ORDER BY SUM(claim_amount) DESC) AS rank
      FROM ...
      GROUP BY policy_type, customer_id
    )
    SELECT policy_type, customer_id, rank, total_claim   ← rank included
    FROM ranked
    WHERE rank <= 5
    ORDER BY policy_type, rank

  WRONG →
    SELECT policy_type, customer_id, total_claim         ← rank omitted
    FROM ranked
    WHERE rank <= 5
    ORDER BY policy_type, rank

This rule also applies to any other CTE-computed column the user
explicitly requests in the question (e.g. "show the percentile",
"include the running total", "display the dense rank").

RULE 18 — INTEGER DIVISION GUARD
PostgreSQL COUNT() returns BIGINT. BIGINT / BIGINT truncates toward zero —
COUNT(approved) / COUNT(total) = 0 when approved < total.
Always cast the numerator to NUMERIC before dividing a count ratio:

  CORRECT → COUNT(CASE WHEN claim_status IN ('approved','paid') THEN 1 END)::numeric
               / NULLIF(COUNT(claim_id), 0)  AS claim_approval_rate
  WRONG   → COUNT(CASE WHEN claim_status IN ('approved','paid') THEN 1 END)
               / COUNT(claim_id)             AS claim_approval_rate  -- always 0

This complements RULE 16: RULE 16 guards the denominator from zero;
RULE 18 guards the quotient from integer truncation.

RULE 19 — JOIN FAN-OUT: MULTI-TABLE AGGREGATION
The schema has a one-to-many chain: policies (1) → claims (many) → payments (many).
Joining all three in a single FROM / JOIN block inflates every SUM by the number
of rows on the many side of each join — producing wrong results without an error.

  WRONG — totals inflated (premium multiplied by # claims, claim_amount by # payments):
  SELECT p.policy_type,
         SUM(p.premium_amt)   AS total_premium,
         SUM(c.claim_amount)  AS total_claims,
         SUM(pay.paid_amount) AS total_paid
  FROM   policies p
  JOIN   claims c ON p.policy_id = c.policy_id
  LEFT JOIN payments pay ON c.claim_id = pay.claim_id
  GROUP  BY p.policy_type

  CORRECT — pre-aggregate each fan-out level in CTEs first:
  WITH payment_agg AS (
      SELECT claim_id,
             SUM(paid_amount) AS total_paid
      FROM   payments
      GROUP  BY claim_id
  ),
  claim_agg AS (
      SELECT c.policy_id,
             SUM(c.claim_amount)                                                  AS total_claim_amount,
             COUNT(c.claim_id)                                                    AS claim_count,
             COUNT(CASE WHEN c.claim_status IN ('approved','paid') THEN 1 END)   AS approved_count,
             COALESCE(SUM(pa.total_paid), 0)                                      AS total_paid
      FROM   claims c
      LEFT JOIN payment_agg pa ON c.claim_id = pa.claim_id
      GROUP  BY c.policy_id
  )
  SELECT p.policy_type,
         SUM(p.premium_amt)                                                       AS total_premium,
         SUM(ca.total_claim_amount)                                               AS total_claim_amount,
         SUM(ca.total_paid)                                                       AS total_paid_amount,
         ROUND(SUM(ca.approved_count)::numeric
               / NULLIF(SUM(ca.claim_count), 0), 4)                              AS claim_approval_rate,
         ROUND(SUM(ca.total_claim_amount)
               / NULLIF(SUM(p.premium_amt), 0), 4)                               AS loss_ratio
  FROM   policies p
  JOIN   claim_agg ca ON p.policy_id = ca.policy_id
  GROUP  BY p.policy_type
  ORDER  BY p.policy_type

GROUPBY CARDINALITY — additional constraint for the outer SELECT:

claim_agg groups at per-policy (policy_id) level — one row per policy_id.
The outer SELECT MUST NOT pass those rows through without rolling them up to
the category level first.  Two failure modes produce this (both WRONG):

WRONG A — no outer aggregation:
  SELECT ca.policy_type, ca.total_claim_amount, ca.total_paid, ...
  FROM   claim_agg ca
  JOIN   premium_by_type pbt ON ca.policy_type = pbt.policy_type
  ORDER  BY ca.policy_type               ← no GROUP BY at all
  -- Result: one row per policy_id (up to 50,000 rows) → RESULT_CAPPED
  --         → GROUPBY_CARDINALITY_MISMATCH

WRONG B — GROUP BY includes non-category columns (unique per policy):
  SELECT ca.policy_type, ca.total_claim_amount, ...
  FROM   claim_agg ca JOIN premium_by_type pbt ON ca.policy_type = pbt.policy_type
  GROUP  BY ca.policy_type, ca.total_claim_amount, ca.total_paid,
            ca.approved_count, ca.claim_count, pbt.total_premium
  -- Same result: one row per unique combination → 50,000 rows → RESULT_CAPPED

CORRECT — add a category_rollup CTE that collapses claim_agg to policy_type:
  category_rollup AS (
      SELECT policy_type,
             SUM(total_claim_amount)  AS total_claim_amount,
             SUM(total_paid)          AS total_paid_amount,
             SUM(claim_count)         AS total_claim_count,
             SUM(approved_count)      AS total_approved_count
      FROM   claim_agg
      GROUP  BY policy_type           ← one row per distinct policy_type
  )
  SELECT cr.policy_type, cr.total_claim_amount, ...
  FROM   category_rollup cr
  JOIN   premium_by_type pbt ON cr.policy_type = pbt.policy_type
  ORDER  BY cr.policy_type            ← exactly 4 rows (one per policy_type)

TRIGGER: apply this CTE pattern whenever the query aggregates across:
  claims + payments   (paid amounts alongside claim amounts)
  claims + policies   (premiums alongside claim metrics)
  all three tables    (always use both CTEs above)

PAYMENT JOIN — additional constraint for the payment_agg CTE:

payment_agg MUST be joined to claims with LEFT JOIN, never INNER JOIN or a
direct JOIN on the payments table. An INNER JOIN silently discards every claim
that has no payment record (pending, rejected, recently filed). Consequence:
claim_approval_rate inflates to 1.0 and total_claim_amount is understated by
the fraction of unapproved claims — roughly 50% in a typical dataset.

WRONG — direct INNER JOIN on payments (common failure mode):
  claim_agg AS (
      SELECT c.policy_id,
             SUM(c.claim_amount)                    AS total_claim_amount,
             COALESCE(SUM(pay.paid_amount), 0)      AS total_paid
      FROM   claims c
      JOIN   payments pay ON pay.claim_id = c.claim_id   ← INNER JOIN
      GROUP  BY c.policy_id
  )
  -- Only claims with a matching payment row survive the join.
  -- Symptom: claim_approval_rate = 1.0 for every group;
  --          total_claim_amount ≈ 50% of the correct value.

CORRECT — pre-aggregate payments into a CTE, LEFT JOIN to claims:
  payment_agg AS (
      SELECT claim_id, SUM(paid_amount) AS total_paid
      FROM   payments
      GROUP  BY claim_id                                   -- one row per claim
  ),
  claim_agg AS (
      SELECT c.policy_id,
             SUM(c.claim_amount)                                                AS total_claim_amount,
             COUNT(c.claim_id)                                                  AS claim_count,
             COUNT(CASE WHEN c.claim_status IN ('approved','paid') THEN 1 END) AS approved_count,
             COALESCE(SUM(pa.total_paid), 0)                                   AS total_paid
      FROM   claims c
      LEFT JOIN payment_agg pa ON c.claim_id = pa.claim_id   ← LEFT JOIN
      GROUP  BY c.policy_id
  )

CLAIM STATUS FILTER — additional constraint when claim_approval_rate is requested:

Never add WHERE c.claim_status IN ('approved','paid') to any CTE that also
computes claim_count or claim_approval_rate.  Filtering by status in WHERE
eliminates unapproved claims before COUNT — every claim that survives into
claim_count is already approved/paid, so claim_approval_rate = 1.0 for every
group regardless of real approval behaviour.  total_claim_amount is also
understated to ~50% of the correct value.

WRONG — WHERE pre-filters claims by status (common failure mode):
  claim_metrics AS (
      SELECT p.policy_type,
             SUM(c.claim_amount)                                                AS total_claim_amount,
             COUNT(c.claim_id)                                                  AS claim_count,
             COUNT(CASE WHEN c.claim_status IN ('approved','paid') THEN 1 END) AS approved_count
      FROM   claims c
      JOIN   policies p  ON c.policy_id = p.policy_id
      LEFT JOIN payment_agg pa ON c.claim_id = pa.claim_id
      WHERE  c.claim_status IN ('approved','paid')   ← wrong: pre-filter
        AND  c.claim_amount IS NOT NULL
      GROUP  BY p.policy_type
  )
  -- approved_count = claim_count for every group → claim_approval_rate = 1.0
  -- total_claim_amount ≈ 50% of correct value (approved/paid claims only)

CORRECT — status filter belongs only in the CASE WHEN expression:
  claim_metrics AS (
      SELECT p.policy_type,
             SUM(c.claim_amount)                                                AS total_claim_amount,
             COUNT(c.claim_id)                                                  AS claim_count,
             COUNT(CASE WHEN c.claim_status IN ('approved','paid') THEN 1 END) AS approved_count
      FROM   claims c
      JOIN   policies p  ON c.policy_id = p.policy_id
      LEFT JOIN payment_agg pa ON c.claim_id = pa.claim_id
      WHERE  c.claim_amount IS NOT NULL              ← only exclude null amounts
      GROUP  BY p.policy_type
  )
  -- claim_count = all non-null claims; approved_count = approved/paid subset
  -- → claim_approval_rate ≈ 0.496 (not 1.0)

LOSS RATIO DENOMINATOR — additional constraint when loss_ratio is requested:

The final JOIN above (`FROM policies p JOIN claim_agg ca`) is an INNER JOIN.
This is correct for claim-level metrics (total_claim_amount, approval_rate).
But for loss_ratio the denominator (total_premium) MUST include ALL policies of
that type — including policies that have never filed a claim.  An INNER JOIN
silently excludes claim-free policies from SUM(premium_amt), understating the
denominator and inflating loss_ratio by 5–20× in typical insurance datasets.

Fix: add a separate premium aggregation CTE and join it independently:

  WITH premium_by_type AS (
      SELECT policy_type,
             SUM(premium_amt) AS total_premium      -- ALL policies, no filter
      FROM   policies
      GROUP  BY policy_type
  ),
  payment_agg AS ( ... ),     -- same as above
  claim_metrics AS (
      SELECT p.policy_type,
             SUM(c.claim_amount)                                               AS total_claim_amount,
             COUNT(c.claim_id)                                                 AS claim_count,
             COUNT(CASE WHEN c.claim_status IN ('approved','paid') THEN 1 END) AS approved_count,
             COALESCE(SUM(pa.total_paid), 0)                                   AS total_paid
      FROM   claims c
      JOIN   policies p ON c.policy_id = p.policy_id
      LEFT JOIN payment_agg pa ON c.claim_id = pa.claim_id
      WHERE  c.claim_amount IS NOT NULL
      GROUP  BY p.policy_type
  )
  SELECT cm.policy_type,
         cm.total_claim_amount,
         cm.total_paid                                                          AS total_paid_amount,
         ROUND(cm.approved_count::numeric / NULLIF(cm.claim_count, 0), 4)     AS claim_approval_rate,
         ROUND(cm.total_claim_amount / NULLIF(pbt.total_premium, 0), 4)       AS loss_ratio
  FROM   claim_metrics cm
  JOIN   premium_by_type pbt ON cm.policy_type = pbt.policy_type
  ORDER  BY cm.policy_type

RULE: whenever loss_ratio appears in the SELECT list, the denominator MUST
come from a standalone premium aggregation CTE (or subquery) that queries the
policies table without any claim-side join condition.

WRONG — do not compute loss_ratio this way (common failure mode):
  SELECT p.policy_type,
         ROUND(CAST(SUM(ca.total_claim_amount) AS DECIMAL)
               / NULLIF(SUM(p.premium_amt), 0), 4) AS loss_ratio
  FROM policies AS p
  JOIN claim_agg AS ca ON p.policy_id = ca.policy_id   ← INNER JOIN
  GROUP BY p.policy_type
  -- SUM(p.premium_amt) only covers policies present in claim_agg.
  -- Claim-free policies are silently excluded from the denominator.
  -- This is wrong even when claim_agg itself is correctly pre-aggregated.
  -- Result: loss_ratio inflated 5–20× (observed: 11–174 on synthetic data).

CORRECT — loss_ratio denominator must come from premium_by_type CTE only:
  ROUND(cm.total_claim_amount / NULLIF(pbt.total_premium, 0), 4) AS loss_ratio
  FROM claim_metrics cm
  JOIN premium_by_type pbt ON cm.policy_type = pbt.policy_type
  -- pbt.total_premium = SUM(ALL premiums for that type, no claim-side filter)
  -- See full CTE example above.

━━━ SCHEMA-DERIVED HARD CONSTRAINTS — apply before writing any SQL ━━━

The following constraints are extracted from the column and table metadata
of the schema retrieved for this query. Each constraint is a grounding
requirement with the same authority as a STRICT RULE above. Violating any
constraint below is a critical failure identical to referencing a
non-existent column.

{hard_constraints}

━━━ INPUTS ━━━

schema_context: {schema_context}
dialect: {dialect}
session_history: {session_history}
user question: {nl_query}
lt_examples (past similar queries — use as few-shot reference only, do not copy verbatim):
{lt_examples}
"""

NL_TO_PANDAS_SYSTEM_PROMPT: str = """\
You are a Pandas code generation engine. Your only function is to convert a
natural language analytical question into a Python code block that operates on
one or more named DataFrames and assigns its final result to a variable named
`result`.

━━━ STRICT RULES ━━━

RULE 1 — DATAFRAME SCOPE
Use only DataFrame names listed in dataframe_refs. Never reference a DataFrame
not in that list. Never create a DataFrame from raw data literals.

RULE 2 — COLUMN GROUNDING
Use only column names present in schema_context for each referenced DataFrame.
String literals used as column identifiers MUST match a known column name exactly
(case-sensitive as stored in schema_context).

RULE 3 — RESULT ASSIGNMENT
The final line of the code MUST be:
  result = <expression>
If the computation requires multiple steps, use intermediate variables freely,
but the last assignment must always be `result`.

RULE 4 — ALLOWED IMPORTS
You may use: pandas, numpy, datetime, math, re, collections, itertools.
All other imports are forbidden. Do not import anything else.
Do not use: os, sys, subprocess, open(), eval(), exec(), __import__().

RULE 5 — OUTPUT FORMAT
Output only a single JSON object. No preamble. No markdown fences. Exact shape:
{{
  "code": "<python code string — use \\n for newlines>",
  "confidence": <float 0.0–1.0>,
  "assumptions": ["<assumption>", ...],
  "grounding_check": {{
    "all_columns_verified": <true|false>,
    "unresolved_references": []
  }}
}}
If any column or DataFrame cannot be resolved, output:
  {{"error_code": "UNRESOLVED_REFERENCE", "unresolved": ["<term>"]}}

RULE 6 — AMBIGUITY
Set confidence < 0.7 and list both interpretations in "assumptions" with the
chosen one marked "(selected)".

━━━ INPUTS ━━━

dataframe_refs: {dataframe_refs}
schema_context: {schema_context}
session_history: {session_history}
user question: {nl_query}
"""

ERROR_CORRECT_SYSTEM_PROMPT: str = """\
A previous code generation attempt failed during validation or execution.
Your job is to produce a corrected version of the code that fixes the reported
error without violating any of the original generation rules.

━━━ STRICT RULES ━━━

RULE 1 — READ THE HISTORY
The attempt_history field contains every code string and error generated in
previous attempts, oldest first. You MUST read all of them before generating.
Do not reproduce a fix that was already attempted.

RULE 2 — ROOT CAUSE FIRST
Identify the root cause from error_type and error_message before writing code.
If the root cause is UNRESOLVED_COLUMN: use only columns from schema_context.
If the root cause is SYNTAX_ERROR: fix only the syntax at error_line.
If the root cause is EXECUTION_TIMEOUT: add a more restrictive filter or reduce
the column set. Do not remove the LIMIT clause.
If the root cause is SEMANTIC_AGG_MISSING: the error_message names the required
aggregate function (e.g. AVG, SUM, COUNT). Rewrite ONLY the SELECT clause to
wrap the target column with that function. Do not alter JOINs, WHERE, or LIMIT.
If the root cause is GROUPBY_CARDINALITY_MISMATCH: the SQL grouped at the wrong
granularity (per-row instead of per-category). The error_message identifies the
target category column. Rewrite using a CTE roll-up:
  (a) CRITICAL — verify which table owns the category column BEFORE touching
      any CTE. Consult schema_context. If the category column (e.g. policy_type)
      is NOT a column of the CTE's own FROM table (e.g. claims has no
      policy_type), you MUST join the table that owns it:
        WRONG  — SELECT c.policy_type ... FROM claims c  ← claims has no such col
        CORRECT— SELECT p.policy_type ... FROM claims c
                 JOIN policies p ON c.policy_id = p.policy_id
      Never add a column to a CTE's GROUP BY or SELECT that does not exist on
      that CTE's FROM table — that is an UNRESOLVED_COLUMN error.
  (b) Add a roll-up CTE that pre-aggregates per-row metrics grouped by the
      category column (e.g. policy_type — never policy_id or claim_id).
  (c) Compute all derived metrics (ratios, rates) in the outer SELECT from the
      CTE aggregates. Never compute a ratio inside a correlated subquery.
  (d) The outer SELECT must GROUP BY the category column only. The final result
      must have one row per distinct category value (typically < 50 rows).

RULE 3 — UNRECOVERABLE CONDITION
If fixing the error would require inventing a column name, mutating data, or
violating any grounding rule, output exactly:
  {{"error_code": "UNRECOVERABLE", "reason": "<one sentence explanation>"}}

RULE 4 — OUTPUT FORMAT
Same JSON shape as the original generation prompt for this code_type.
No preamble. No markdown fences.

RULE 5 — NO CORRELATED SUBQUERIES IN DENOMINATORS
Never place a correlated subquery inside a ratio denominator. A correlated
subquery re-scans the source table once per output group, causing execution
timeouts on tables with more than 10,000 rows.

  WRONG — correlated subquery in loss_ratio denominator:
    ROUND(SUM(cm.total_claim_amount) / NULLIF(
        (SELECT SUM(premium_amt) FROM policies p2
         WHERE p2.policy_type = pr.policy_type), 0), 4) AS loss_ratio

  CORRECT — standalone CTE joined once before the outer SELECT:
    WITH premium_by_type AS (
        SELECT policy_type, SUM(premium_amt) AS total_premium
        FROM   policies
        GROUP  BY policy_type
    )
    ...
    ROUND(cm.total_claim_amount / NULLIF(pbt.total_premium, 0), 4) AS loss_ratio
    FROM ... JOIN premium_by_type pbt ON ....policy_type = pbt.policy_type

This applies to ALL aggregation denominators, not only loss_ratio.

RULE 6 — PAYMENT JOIN MUST USE PRE-AGGREGATED LEFT JOIN
When any rewrite aggregates paid_amount, the payments table MUST be accessed
through a pre-aggregated payment_agg CTE that is LEFT JOINed to claims.
Never JOIN the payments table directly into claim_agg.

  WRONG — direct JOIN on payments discards unapproved claims:
    FROM claims c
    JOIN payments pay ON pay.claim_id = c.claim_id   ← direct INNER JOIN
    -- Claims with no payment record are silently excluded.
    -- Symptom: claim_approval_rate = 1.0; total_claim_amount ≈ 50% correct.

  CORRECT — pre-aggregate then LEFT JOIN:
    payment_agg AS (
        SELECT claim_id, SUM(paid_amount) AS total_paid
        FROM   payments GROUP BY claim_id
    ),
    claim_agg AS (
        ...
        FROM claims c
        LEFT JOIN payment_agg pa ON c.claim_id = pa.claim_id   ← LEFT JOIN
        ...
    )

RULE 7 — CLAIM STATUS MUST NOT APPEAR IN THE WHERE CLAUSE
Never filter by claim_status in the WHERE clause of any CTE that also computes
claim_count or claim_approval_rate. A WHERE status filter makes every surviving
claim already approved — claim_approval_rate is trivially 1.0.

  WRONG — WHERE filters denominator to approved/paid only:
    FROM claims c JOIN policies p ON c.policy_id = p.policy_id
    WHERE c.claim_status IN ('approved','paid')   ← eliminates unapproved claims
      AND c.claim_amount IS NOT NULL
    -- Symptom: claim_approval_rate = 1.0; total_claim_amount ≈ 50% correct.

  CORRECT — status filter in CASE WHEN only; WHERE only excludes nulls:
    FROM claims c JOIN policies p ON c.policy_id = p.policy_id
    WHERE c.claim_amount IS NOT NULL
    ...
    COUNT(CASE WHEN c.claim_status IN ('approved','paid') THEN 1 END) AS approved_count

━━━ SCHEMA-DERIVED HARD CONSTRAINTS — apply before writing any SQL ━━━

The following constraints are extracted from the schema metadata.
They carry the same authority as the STRICT RULES above.
Violating any constraint is a critical failure equivalent to referencing
a non-existent column.

{hard_constraints}

━━━ INPUTS ━━━

original question: {nl_query}
code_type: {code_type}
schema_context: {schema_context}
attempt_history (oldest first):
{attempt_history}

latest error:
  type: {error_type}
  message: {error_message}
  line: {error_line}
"""

SCHEMA_GROUNDING_SYSTEM_PROMPT: str = """\
You are a schema resolution engine. Your only function is to map terms from a
natural language question to exact table and column names in the provided
schema context.

━━━ STRICT RULES ━━━

RULE 1 — NO INFERENCE
Only resolve a term if it matches a schema element verbatim or is an
unambiguous semantic synonym with confidence >= 0.9.
"Revenue" → "revenue" (same word): resolve.
"Revenue" → "gross_income" (indirect synonym): mark UNRESOLVED.

RULE 2 — TABLE.COLUMN FORMAT
All resolved references must be in the format "table_name.column_name".

RULE 3 — OUTPUT FORMAT
Output only a single JSON object. No preamble. No markdown fences:
{{
  "resolved": {{"<user_term>": "table_name.column_name"}},
  "unresolved": ["<term1>", ...]
}}

RULE 4 — EMPTY RESULT
If no terms can be resolved: {{"resolved": {{}}, "unresolved": [<all terms>]}}

━━━ INPUTS ━━━

schema_context: {schema_context}
user question: {nl_query}
"""

INSIGHT_SYSTEM_PROMPT: str = """\
You are a business analyst writing a summary for a non-technical audience.
You have been given the result of a data query. Write a concise, plain-English
summary of the key finding.

━━━ STRICT RULES ━━━

RULE 1 — INTERPRETIVE FRAMING
Do NOT merely report what the numbers are. Explain what they mean and why
the pattern exists — move from descriptive to interpretive.

When schema_context is provided, use the table and column descriptions as
domain vocabulary to explain the business significance of the pattern.
Apply the descriptions to answer the implicit question: "Why does this gap
or pattern exist?"

GROUNDING BOUND: schema_context descriptions are the only permitted source
for interpretive claims. Do NOT assert product-specific domain facts
(benefit structures, coverage mechanics, actuarial terms, regulatory
reasons) that are not explicitly stated in the business_description or
column description text. If the description does not explain the cause,
state the pattern as observed without inventing a cause.

  DESCRIPTIVE (weak):
    "Life insurance claims are 15× larger than auto claims."

  INTERPRETIVE — schema-grounded (correct):
    Schema column descriptions are the ONLY permitted source for interpretive
    claims. Quote or closely paraphrase the description text; do not extend it.
    If the schema says claim_amount is "the total invoiced amount", write:
    "Life insurance claims average 15× more than auto claims. Per the schema,
     claim_amount is the total invoiced amount — for life policies this
     typically exceeds auto claim invoices by an order of magnitude."
    If the schema description is silent on the cause, do NOT invent one.

  INTERPRETIVE — no schema explanation available (correct):
    "Life insurance claims average 15× more than auto claims. The data does not
     establish why — this likely reflects structural differences between product
     lines, though the query result does not confirm the underlying cause."

  INTERPRETIVE — ungrounded (wrong — schema description does not say this):
    "Per the schema, claim_amount represents the amount requested by the
     claimant, which for life policies results in structurally larger figures
     than the loss-based reimbursements typical of auto and home policies."
    This is wrong when the schema's claim_amount column description does not
    contain the phrase "requested by the claimant". Never assert what a column
    measures if the description is silent on it.

If schema_context is empty, use the question and column names as vocabulary
to reason about business significance rather than only restating the numbers.

RULE 2 — LENGTH
2–4 sentences maximum. Do not pad. If the result has only one row, one sentence
is sufficient.

RULE 3 — CAVEATS
If result_warnings is non-empty, append exactly one sentence acknowledging the
caveat (e.g. "Note: results were capped at 10,000 rows — add a filter for the
full picture.").
If result_warnings is empty, do NOT add any caveat or hedge — not about sample
size, row count, data freshness, or completeness. n_groups being small is not
a caveat. Absence of a warning means no caveat is warranted.

RULE 4 — ERROR STATE
If result is null and error is provided, explain what went wrong in plain
English without exposing SQL, error codes, or stack traces.
Start with: "I wasn't able to answer that because…"

RULE 5 — OUTPUT FORMAT
Plain text only. No JSON. No markdown. No bullet points.

RULE 6 — PRE-COMPUTED METRICS
When result_metrics is non-empty, use the provided values directly — do NOT
re-derive ratios or spreads from result_preview, which may only contain 5 rows.
The metrics are computed over the full result set for precision.

  Use them to quantify comparisons and characterise distribution:
  - Lead with the top group and its value, then contrast with the bottom group.
  - State the ratio explicitly: "X is {{ratio}}× larger than Y."
  - If variation_type is "extreme" (ratio ≥ 10): emphasise the magnitude gap.
  - If variation_type is "strong"  (ratio ≥ 3):  call out the spread clearly.
  - If variation_type is "moderate" (ratio ≥ 1.5): note meaningful differences.
  - If variation_type is "uniform" (ratio < 1.5):  state that values are similar.
  - If ranked_groups is present and n_groups ≥ 3: acknowledge ALL groups, not
    just top and bottom. Mention middle groups by name and value in rank order
    so the reader sees the full range — e.g. "Health and home policies sit
    between these extremes, averaging $30,391 and $17,839 respectively."
    Do not silently omit any group that appears in ranked_groups.
  - Do NOT invent metrics not present in result_metrics.
  - If result_metrics is empty, derive observations from result_preview as normal.

  CROSS-METRIC CONTRAST — surface the more surprising finding:
  When a result contains multiple metrics, the most analytically valuable
  observation is often a CONTRAST between metrics — not the largest number.

  Pattern: primary metric shows extreme/strong variation, but a secondary
  metric on the same groups shows uniform variation.
  This contrast IS the finding and should be stated explicitly:

    Lead sentence:  state the primary metric's magnitude gap.
    Contrast pivot: "However, [secondary metric] is remarkably consistent
                     across all groups, ranging only from X to Y."
    Interpretation: "This suggests [hedged conclusion], though the query
                     does not establish the underlying cause."

  Example (claim amount extreme, KPIs uniform):
    "Life insurance accounts for the largest claim volume ($5.46B),
     approximately 14.7× higher than auto ($372M). However, portfolio
     performance metrics are remarkably consistent across products: approval
     rates vary by less than 0.4 percentage points (49.5%–49.9%), while loss
     ratios remain tightly clustered between 0.84 and 0.86. This suggests
     that claims outcomes and premium-to-claim relationships are highly
     uniform across the portfolio, despite large differences in claim volume."

  Do NOT lead with the obvious magnitude ranking when the tighter KPI cluster
  is the more surprising finding — both must appear, with the contrast framed
  as the analytical payoff.

RULE 7 — DATA QUALITY GUARD
Before generating any insight, inspect result_warnings.
If result_warnings contains a message with "predominantly NULL" or
"result quality warning", the primary metric columns are unavailable.
In this case output ONLY a plain statement of the data quality issue:

  "The query returned {row_count} rows but the requested metrics
   (e.g. total_claim_amount, loss_ratio) could not be computed —
   all matched records appear to have no payout amount, likely because
   they are pending or denied claims. Re-run with a status filter
   (e.g. claim_status = 'paid') to get valid numbers."

Do NOT generate ratio, ranking, comparison, or variation statements
from null data. Do NOT narrate incidental non-null columns (e.g.
total_premium) as if they were the requested metrics.

RULE 8 — ANOMALY DETECTION
When result_metrics contains an "anomalies" key, you MUST mention the
anomalies in the insight — they are the most actionable finding for the
business audience.

  - Lead with the primary pattern (Rule 6), then add one sentence flagging
    the anomaly column, the extreme value(s), and a plain-English
    interpretation of what they signal.
  - Do NOT skip anomalies because the primary value_col is different.
    A loss_ratio of 100 is business-critical even when the query was
    "ranked by claim amount".
  - Use the anomaly threshold from result_metrics to characterise severity:
    threshold > 50  → "exceptionally high / critical"
    threshold > 20  → "elevated / worth investigating"
    threshold > 10  → "above typical range"
  - Do NOT invent anomaly thresholds not present in result_metrics.

  Example:
    "Life insurance customers dominate by claim amount, averaging $3.86M —
     16.2× higher than auto ($238K). Notably, several customers show
     exceptionally high loss ratios (up to 102×), meaning claims far
     exceed the premiums paid; these accounts warrant further review."

━━━ INPUTS ━━━

original question: {nl_query}
result_preview (first 5 rows): {result_preview}
row_count: {row_count}
result_warnings: {result_warnings}
result_metrics: {result_metrics}
schema_context: {schema_context}
error: {error}
"""

CLARIFICATION_SYSTEM_PROMPT: str = """\
You are a data analyst assistant. A user asked a question that you could not
answer because either the schema could not be found, the query returned no
results, or the question was ambiguous.

━━━ STRICT RULES ━━━

RULE 1 — ONE QUESTION ONLY
Ask exactly one clarifying question. Do not ask multiple questions in one turn.

RULE 2 — OFFER CONCRETE OPTIONS
Where possible, offer 2–3 specific options derived from the schema context or
error context rather than an open-ended question.

RULE 3 — PRIMARY PERSONA LANGUAGE
No SQL terminology. No mention of schemas, tables, JOIN, NULL, or index.
Speak in business terms.

RULE 4 — TRIGGER CONTEXT
Use clarification_trigger to shape the question:
  "NO_SCHEMA_MATCH"   → ask the user to describe the data they're looking for.
  "EMPTY_RESULT"      → suggest the filter may be too restrictive; offer to relax it.
  "LOW_CONFIDENCE"    → present interpretation A and B; ask which they meant.
  "UNRESOLVED_COLUMN" → list top-3 fuzzy column matches; ask which they meant.

RULE 5 — OUTPUT FORMAT
Plain text only. No JSON. No markdown.

━━━ INPUTS ━━━

original question: {nl_query}
clarification_trigger: {clarification_trigger}
context: {context}
"""

# ---------------------------------------------------------------------------
# PromptRenderer — safe template rendering
# ---------------------------------------------------------------------------

# Matches {placeholder} but NOT {{escaped}} double-brace literals
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


class PromptRenderer:
    """
    Fills {placeholder} slots in prompt template strings.

    Uses str.format_map() with a SafeDict that leaves unknown keys intact
    rather than raising KeyError. This prevents partial renders from silently
    producing malformed prompts.

    Security note: values are serialised to JSON strings before injection.
    This ensures that a user value containing curly braces cannot be
    interpreted as a nested placeholder by a second render pass.
    """

    class _SafeDict(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return f"{{{key}}}"  # leave unfilled placeholders intact

    @staticmethod
    def render(template: str, **kwargs: Any) -> str:
        """
        Render a prompt template.

        Complex types (lists, dicts, Pydantic models) are serialised to
        compact JSON. Strings are injected as-is (already serialised at
        the call site or plain text).
        """
        serialised: dict[str, str] = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                serialised[key] = value
            elif isinstance(value, BaseModel):
                serialised[key] = value.model_dump_json()
            elif isinstance(value, list | dict):
                serialised[key] = json.dumps(value, default=str)
            else:
                serialised[key] = str(value)

        return template.format_map(PromptRenderer._SafeDict(serialised))

    @staticmethod
    def missing_placeholders(template: str, **kwargs: Any) -> list[str]:
        """Return a list of placeholder names that were not supplied."""
        required = set(_PLACEHOLDER_RE.findall(template))
        provided = set(kwargs.keys())
        return sorted(required - provided)


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding(
    "cl100k_base"
)  # cl100k_base; approximate token count for Gemini

TOKEN_BUDGET = {
    "schema_context": 6_000,  # increased: business_description + column notes need room
    "session_history": 2_000,
    "prompt_overhead": 500,
    "safety_margin": 500,
    "total": 10_000,  # llama-3.3-70b-versatile supports 32k; 5k was stripping descriptions
}


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def enforce_token_budget(
    schema_context: list[SchemaChunk],
    session_history: list[dict[str, Any]],
    system_prompt_tokens: int = 500,
) -> tuple[list[SchemaChunk], list[dict[str, Any]], int]:
    """
    Trims schema_context and session_history to stay within TOKEN_BUDGET.

    Trim order (least-important first):
      1. Truncate business_description in schema chunks
      2. Reduce session history from 10 → 5 → 3 turns
      3. Reduce K from 5 → 3 chunks

    Returns (trimmed_chunks, trimmed_history, estimated_total_tokens).
    """
    chunks = list(schema_context)
    history = list(session_history)

    def _estimate() -> int:
        chunk_tokens = count_tokens(
            json.dumps([c.model_dump() for c in chunks], default=str)
        )
        history_tokens = count_tokens(json.dumps(history, default=str))
        return chunk_tokens + history_tokens + system_prompt_tokens

    # Step 1: strip business descriptions if over budget
    if _estimate() > TOKEN_BUDGET["total"]:
        chunks = [c.model_copy(update={"business_description": None}) for c in chunks]

    # Step 1b: strip column-level descriptions if still over budget.
    # M6 FIX: business_description trimming alone was insufficient when schemas
    # have many annotated columns — column descriptions (applied as hard LLM
    # constraints by Rule 13) are not trimmed, so the prompt could still exceed
    # the token budget after step 1, silently truncating mid-JSON.
    # ML-3 FIX: emit BUDGET_WARNING so operators can detect when Rule 13
    # grounding constraints are stripped. Without this log event, column-level
    # data quality notes (e.g. "Filter IS NOT NULL", "use COALESCE") are dropped
    # silently and generated SQL may violate them without any signal.
    if _estimate() > TOKEN_BUDGET["total"]:
        stripped_cols_by_schema: dict[str, list[str]] = {}
        stripped_chunks = []
        for c in chunks:
            cols_with_desc = [col.name for col in c.columns if col.description]
            if cols_with_desc:
                stripped_cols_by_schema.setdefault(
                    getattr(c, "schema_id", "unknown"), []
                ).extend(cols_with_desc)
            new_cols = [
                col.model_copy(update={"description": None}) for col in c.columns
            ]
            stripped_chunks.append(c.model_copy(update={"columns": new_cols}))
        chunks = stripped_chunks

        import sys
        from datetime import UTC, datetime

        print(
            json.dumps(
                {
                    "event": "BUDGET_WARNING",
                    "message": (
                        "Token budget exceeded after Step 1: column-level descriptions "
                        "stripped. Rule 13 grounding constraints disabled for this turn."
                    ),
                    "stripped_columns_by_schema": stripped_cols_by_schema,
                    "estimated_tokens_before_strip": _estimate(),
                    "budget": TOKEN_BUDGET["total"],
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
            ),
            file=sys.stdout,
            flush=True,
        )

    # Step 2: reduce history window
    for max_turns in (5, 3):
        if _estimate() > TOKEN_BUDGET["total"] and len(history) > max_turns:
            history = history[-max_turns:]

    # Step 3: reduce K
    if _estimate() > TOKEN_BUDGET["total"] and len(chunks) > 3:
        chunks = chunks[:3]

    return chunks, history, _estimate()


# ---------------------------------------------------------------------------
# LLMClient — Gemini wrapper
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)

_BACKOFF_BASE_S = 1.0
_MAX_LLM_RETRIES = 3
# Status codes on GenAI ClientError that are transient and warrant a retry.
# 400 (bad request), 401 (invalid key), 403 (quota exceeded at account level),
# and 422 (validation error) are non-retryable — propagate immediately.
_RETRYABLE_CLIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Alias map for error_code values emitted by small models that deviate from the
# Literal contract in GenerateSQLOutput / GeneratePandasOutput.
# Key   → the raw string the model returns
# Value → the canonical Literal value to substitute, or None to pop the key
#         (None means "no error — model included the field but left it empty")
#
# Extend this table as new model quirks are discovered in production logs.
# Do NOT add aliases for values that should trigger a real parse error
# (e.g. completely unrecognised strings with no clear mapping).
_ERROR_CODE_ALIASES: dict[str, str | None] = {
    "": None,  # llama-3.1-8b-instant: successful correction,
    # model echoes field but empties it (2026-05-11)
    "UNRESOLVABLE": "UNRECOVERABLE",  # llama-3.1-8b-instant: misspelling of the
    # UNRECOVERABLE sentinel (2026-05-12)
    "RESOLVED": None,  # llama-3.1-8b-instant: signals successful
    # correction with "RESOLVED" instead of
    # omitting the field (2026-05-12)
    "SYNTAX_ERROR": None,  # llama-3.1-8b-instant: during ERROR_CORRECT the model
    # echoes the error type it was asked to fix as error_code
    # instead of omitting the field (2026-05-12). If sql/code
    # is also present this is a successful correction — pop the
    # field so Pydantic sees error_code=None and sql_or_error_required
    # passes. If sql is also absent, model_validate raises on the
    # sql_or_error_required check, which is the correct behaviour.
    "EXECUTION_TIMEOUT": "UNRECOVERABLE",
}


class GenerationRequest(BaseModel):
    """Typed input to LLMClient.generate()."""

    prompt_type: Literal[
        "nl_to_sql",
        "nl_to_pandas",
        "error_correct",
        "schema_grounding",
        "insight",
        "clarification",
    ]
    system_prompt: str
    # For JSON-mode calls: the Pydantic model class to parse the response into.
    # For plain-text calls (insight, clarification): set to None.
    response_model: type[BaseModel] | None = None
    model: str = ""  # "" → LLMClient falls back to self._default_model
    max_tokens: int = 1_000
    session_id: str = ""
    turn_id: str = ""


class GenerationResponse(BaseModel):
    """Typed output from LLMClient.generate()."""

    content: str  # raw string from the LLM
    parsed: BaseModel | None = None  # Pydantic-parsed result (if response_model set)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    parse_error: str | None = (
        None  # populated if JSON parse or Pydantic validation fails
    )


class LLMClient:
    """
    Google Gemini wrapper with:
      - JSON mode enforcement for structured generation calls
      - Exponential backoff on rate-limit and network errors
      - Token usage extraction
      - Pydantic parsing with graceful error capture
      - tiktoken-based pre-call token count (cl100k_base; approximate for Gemini)

    Parameters
    ----------
    api_key      : Google AI Studio API key. Obtain free at aistudio.google.com.
                   If None, reads from GEMINI_API_KEY env var.
    default_model: Model string used when GenerationRequest.model is not overridden.
                   Free-tier models: see build_llm_client() for provider defaults.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-2.0-flash",
    ) -> None:
        # Deferred imports — only run when LLMClient is instantiated, not when
        # prompts.py is first imported.  Groq-only deployments that don't have
        # google-generativeai installed can still import this module and use
        # GroqLLMClient, mirroring the pattern in GroqLLMClient.__init__.
        try:
            from google import genai as _genai
            from google.genai import errors as _genai_errors
            from google.genai import types as _genai_types
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for LLMClient. "
                "Install with: pip install google-generativeai"
            ) from exc
        self._genai = _genai
        self._genai_errors = _genai_errors
        self._genai_types = _genai_types
        self._client = _genai.Client(api_key=api_key)
        self._default_model = default_model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Make one Gemini call with exponential backoff on retryable errors.

        JSON mode is active when request.response_model is not None.
        Gemini's response_mime_type="application/json" enforces JSON output.
        The response is parsed into request.response_model via Pydantic.
        If parsing fails, GenerationResponse.parse_error is populated and
        GenerationResponse.parsed is None — the caller decides whether to
        retry or escalate to ERROR_CORRECT.

        Backoff: 1s, 2s, 4s (base=1, max_retries=3, exponential).
        After 3 retries: raises the last exception to the orchestrator,
        which maps it to a TERMINAL_ERROR with error_type=LLM_UNAVAILABLE.
        """
        use_json_mode = request.response_model is not None
        model_name = request.model or self._default_model

        gen_config = self._genai_types.GenerateContentConfig(
            max_output_tokens=request.max_tokens,
            response_mime_type="application/json" if use_json_mode else None,
        )

        last_exc: Exception | None = None
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                t0 = time.monotonic()
                response = await self._client.aio.models.generate_content(
                    model=model_name,
                    contents=request.system_prompt,
                    config=gen_config,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)

                content = response.text or ""
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                completion_tokens = (
                    getattr(usage, "candidates_token_count", 0) if usage else 0
                )

                parsed, parse_error = self._parse_response(
                    content, request.response_model
                )

                return GenerationResponse(
                    content=content,
                    parsed=parsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    model=model_name,
                    parse_error=parse_error,
                )

            except self._genai_errors.ServerError as exc:
                # 5xx — always transient; back off and retry.
                last_exc = exc
                backoff = _BACKOFF_BASE_S * (2**attempt)
                await asyncio.sleep(backoff)
            except self._genai_errors.ClientError as exc:
                # 4xx — only a subset are transient.
                # 401/400/403/422 are non-retryable: propagate immediately so
                # the orchestrator surfaces the failure without burning retry
                # slots on a 7-second backoff that cannot succeed.
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status not in _RETRYABLE_CLIENT_STATUS_CODES:
                    raise
                last_exc = exc
                backoff = _BACKOFF_BASE_S * (2**attempt)
                await asyncio.sleep(backoff)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _parse_response(
        content: str,
        model_cls: type[BaseModel] | None,
    ) -> tuple[BaseModel | None, str | None]:
        """
        Parse content into model_cls. Returns (parsed, None) on success,
        (None, error_message) on failure.

        Strips markdown fences defensively in case the model ignores JSON mode.
        """
        if model_cls is None:
            return None, None

        cleaned = content.strip()
        # Strip ```json ... ``` or ``` ... ``` fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"JSON parse failed: {exc}. Raw content length: {len(content)}"

        # Key normalisation for small models (llama-3.1-8b-instant).
        # These models produce two classes of mis-keyed responses:
        #
        # Class 1 — wrong top-level key:
        #   {"query": "SELECT ..."}   → rename to {"sql": "SELECT ..."}
        #   {"code":  "SELECT ..."}   → rename to {"sql": "SELECT ..."} for SQL only
        #
        # Class 2 — double-wrapped value (seen in logs 2026-05-12):
        #   {"query": {"query": "SELECT ..."}}  → rename + unwrap
        #   {"sql":   {"query": "SELECT ..."}}  → unwrap only
        #
        # The normalisation runs before Pydantic so the model_validator
        # `sql_or_error_required` never sees a dict where a str is expected.
        if isinstance(data, dict):
            # Step 1: rename top-level SQL aliases → "sql" if "sql" is absent.
            if "sql" not in data and "query" in data:
                data["sql"] = data.pop("query")
            if (
                model_cls.__name__ == "GenerateSQLOutput"
                and "sql" not in data
                and isinstance(data.get("code"), str)
            ):
                data["sql"] = data.pop("code")

            # Step 2: unwrap if the "sql" value is itself a dict
            # (model returned {"sql": {"query": "SELECT ..."}} or similar)
            sql_val = data.get("sql")
            if isinstance(sql_val, dict):
                for _nested_key in ("sql", "query", "statement", "code"):
                    if _nested_key in sql_val and isinstance(sql_val[_nested_key], str):
                        data["sql"] = sql_val[_nested_key]
                        break
                else:
                    # No recognised nested key — leave as-is so Pydantic
                    # produces a clear string_type error rather than a silent
                    # truncation or wrong-value assignment.
                    pass

        # Normalise error_code deviations emitted by small models (llama-3.1-8b-instant).
        # See _ERROR_CODE_ALIASES for the full table and rationale per entry.
        if isinstance(data, dict) and "error_code" in data:
            raw_code = data["error_code"]
            if raw_code in _ERROR_CODE_ALIASES:
                canonical = _ERROR_CODE_ALIASES[raw_code]
                if canonical is None:
                    data.pop("error_code")  # no error — let Pydantic default to None
                else:
                    data["error_code"] = canonical

        # Handle error shortcircuit: {error_code: "...", unresolved: [...]}
        # These are valid structured responses — parse them into the model.
        try:
            return model_cls.model_validate(data), None
        except ValidationError as exc:
            return None, f"Pydantic validation failed: {exc}"

    async def generate_insight(
        self,
        nl_query: str,
        result_preview: list[dict[str, Any]],
        row_count: int,
        result_warnings: list[str],
        error: str | None,
        model: str = "",  # "" → falls back to self._default_model
        result_metrics: dict[str, Any] | None = None,
        schema_descriptions: dict[str, str] | None = None,
    ) -> str:
        """
        Convenience method for the INSIGHT state. Returns plain text.
        Falls back to a safe default string on any LLM error.

        result_metrics: pre-computed analytical metrics dict from
        _compute_result_metrics() in orchestrator.py. When non-empty, these
        are injected into the INSIGHT_SYSTEM_PROMPT so the LLM uses exact
        values (ratio, variation_type, top/bottom group) rather than inferring
        them from the 5-row result_preview — which may be a partial view.

        schema_descriptions: {table_name: business_description} extracted from
        the schema chunks retrieved for this query. Used by RULE 1 to explain
        WHY patterns exist (interpretive framing) rather than only reporting
        WHAT the numbers are (descriptive framing).
        """
        system_prompt = PromptRenderer.render(
            INSIGHT_SYSTEM_PROMPT,
            nl_query=nl_query,
            result_preview=json.dumps(result_preview[:5], default=str),
            row_count=str(row_count),
            result_warnings=json.dumps(result_warnings),
            result_metrics=json.dumps(result_metrics or {}, default=str),
            schema_context=json.dumps(schema_descriptions or {}, default=str),
            error=error or "",
        )
        req = GenerationRequest(
            prompt_type="insight",
            system_prompt=system_prompt,
            response_model=None,
            model=model,
            max_tokens=500,
        )
        try:
            resp = await self.generate(req)
            return resp.content.strip()
        except Exception:
            return (
                f"I wasn't able to summarise the result because of an internal error. "
                f"The query returned {row_count:,} rows."
            )

    async def generate_clarification(
        self,
        nl_query: str,
        trigger: str,
        context: str,
        model: str = "",  # "" → falls back to self._default_model
    ) -> str:
        """Convenience method for the INTAKE clarification path."""
        system_prompt = PromptRenderer.render(
            CLARIFICATION_SYSTEM_PROMPT,
            nl_query=nl_query,
            clarification_trigger=trigger,
            context=context,
        )
        req = GenerationRequest(
            prompt_type="clarification",
            system_prompt=system_prompt,
            response_model=None,
            model=model,
            max_tokens=200,
        )
        try:
            resp = await self.generate(req)
            return resp.content.strip()
        except Exception:
            return (
                "I wasn't able to understand the question. "
                "Could you rephrase or describe the data you're looking for?"
            )


# ---------------------------------------------------------------------------
# GroqLLMClient — drop-in replacement using Groq's free API
# ---------------------------------------------------------------------------


class GroqLLMClient(LLMClient):
    """
    Groq API wrapper implementing the same interface as LLMClient.

    Inherits generate_insight() and generate_clarification() from LLMClient
    since both delegate to self.generate(). Only __init__ and generate()
    are overridden.

    Parameters
    ----------
    api_key      : Groq API key. If None, reads from GROQ_API_KEY env var.
    default_model: Model string. Recommended: "llama-3.3-70b-versatile"
                   for best SQL generation quality on the free tier.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "llama-3.3-70b-versatile",
    ) -> None:
        try:
            import groq as _groq
        except ImportError as exc:
            raise ImportError(
                "groq package is required for GroqLLMClient. "
                "Install it with: pip install groq"
            ) from exc
        self._groq_client = _groq.AsyncGroq(api_key=api_key)
        self._default_model = default_model
        # Stash module ref so generate() can catch groq-specific errors
        self._groq_mod = _groq

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Make one Groq chat completion call with exponential backoff.

        JSON mode is active when request.response_model is not None —
        passed as response_format={"type": "json_object"} to the API.
        """
        use_json_mode = request.response_model is not None
        model_name = request.model or self._default_model

        call_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": request.system_prompt}],
            "max_tokens": request.max_tokens,
        }
        if use_json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}

        # Build retryable tuple with a getattr guard on InternalServerError:
        # the class has existed in the Groq SDK since v0.3 but a strict
        # attribute access would raise AttributeError on unexpected old/custom
        # builds, crashing the generation call before any retry fires.
        _internal_server_err = getattr(self._groq_mod, "InternalServerError", None)
        _retryable = (
            self._groq_mod.RateLimitError,  # 429 — transient
            self._groq_mod.APIConnectionError,  # network error — transient
            *(
                [_internal_server_err] if _internal_server_err is not None else []
            ),  # 500
        )

        last_exc: Exception | None = None
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                t0 = time.monotonic()
                response = await self._groq_client.chat.completions.create(
                    **call_kwargs
                )
                latency_ms = int((time.monotonic() - t0) * 1000)

                content = response.choices[0].message.content or ""
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                parsed, parse_error = self._parse_response(
                    content, request.response_model
                )

                return GenerationResponse(
                    content=content,
                    parsed=parsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    model=model_name,
                    parse_error=parse_error,
                )

            except _retryable as exc:
                last_exc = exc
                backoff = _BACKOFF_BASE_S * (2**attempt)
                await asyncio.sleep(backoff)

        assert last_exc is not None
        # The exception is re-raised immediately; the orchestrator's TraceLogger
        # captures it as "exception" in the STATE_TRANSITION log event.
        raise last_exc


# ---------------------------------------------------------------------------
# Factory — resolves provider from env, returns correct LLMClient subtype
# ---------------------------------------------------------------------------

_GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def build_llm_client(
    provider: str = "gemini",
    api_key: str | None = None,
    default_model: str | None = None,
) -> LLMClient:
    """
    Instantiate the correct LLMClient subtype for the given provider.

    Parameters
    ----------
    provider     : "gemini" or "groq". Reads from LLM_PROVIDER env var.
    api_key      : Provider API key. None → reads from env inside each client.
    default_model: Override model string. None → provider's recommended default.
    """
    if provider == "groq":
        return GroqLLMClient(
            api_key=api_key,
            default_model=default_model or _GROQ_DEFAULT_MODEL,
        )
    # Default: gemini
    return LLMClient(
        api_key=api_key,
        default_model=default_model or _GEMINI_DEFAULT_MODEL,
    )
