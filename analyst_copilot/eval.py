"""
eval.py — Evaluation pipeline
Data Analyst Copilot · Phase 3 · Section 8 · Python 3.11+

Implements the full evaluation framework from Section 8:

  Metrics (all operationally defined):
    schema_recall_at_5   — correct table in top-5 retrieved chunks
    executable_rate      — code runs without SyntaxError / NameError
    correctness_rate     — result matches ground-truth SQL (row-level equality)
    error_recovery_rate  — correct result within 3 retry attempts
    p50/p95/p99_latency  — end-to-end latency percentiles

  Ground truth: synthetically generated from a known reference SQLite DB.
  Minimum test set: 200 NL→SQL pairs split 60/20/20 train/val/test.

Usage:
    # Generate synthetic test pairs from the insurance DB
    python eval.py generate --db data/insurance.db --output data/eval_pairs.json

    # Run evaluation against the live server
    python eval.py run --pairs data/eval_pairs.json --url http://localhost:8000

    # Quick smoke test (10 pairs, no server required)
    python eval.py smoke --db data/insurance.db
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EvalPair:
    pair_id: str
    schema_id: str
    nl_query: str
    ground_truth_sql: str
    expected_table: str  # table that MUST appear in top-5 retrieved chunks
    semantic_template: str  # e.g. "aggregation", "filter", "join", "time_series"
    split: str = "test"  # "train" | "val" | "test"
    # Optional path to a local SQLite DB for row-level correctness comparison.
    # When set, _evaluate_pair executes both ground_truth_sql and generated_sql
    # locally and compares result sets. When None, column-name matching is used.
    db_path: str | None = None


@dataclass
class EvalResult:
    pair_id: str
    nl_query: str
    generated_code: str
    executable: bool
    correct: bool  # >=75% column overlap (or exact scalar match)
    # A-01 FIX: partial_correct had no default value.  Two error-path
    # instantiation sites (asyncio.gather exception handler and the HTTP
    # exception handler in _evaluate_pair) omitted it, causing TypeError on
    # every network error or server exception — exactly the paths that must
    # succeed gracefully to produce a meaningful eval report.
    partial_correct: bool = False  # ML-5: 50–74% overlap — tracked separately
    retry_count: int = 0
    latency_ms: int = 0
    schema_recall: bool = False
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class EvalReport:
    total_pairs: int
    executable_rate: float
    executable_ci_95: tuple[float, float]
    correctness_rate: float
    correctness_ci_95: tuple[float, float]
    partial_correct_rate: float  # ML-5: 50–74% column overlap; distinct from full correct
    error_recovery_rate: float
    schema_recall_at_5: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    results: list[EvalResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "── Evaluation Report ─────────────────────────────────────",
            f"  Pairs evaluated:     {self.total_pairs}",
            f"  Executable rate:     {self.executable_rate:.1%}  "
            f"(95% CI: {self.executable_ci_95[0]:.1%}–{self.executable_ci_95[1]:.1%})",
            f"  Correctness rate:    {self.correctness_rate:.1%}  "
            f"(95% CI: {self.correctness_ci_95[0]:.1%}–{self.correctness_ci_95[1]:.1%})",
            f"  Partial correct:     {self.partial_correct_rate:.1%}  "
            f"(50–74% column overlap — not counted as correct)",
            f"  Error recovery rate: {self.error_recovery_rate:.1%}",
            f"  Schema recall@5:     {self.schema_recall_at_5:.1%}",
            f"  Latency p50/p95/p99: "
            f"{self.p50_latency_ms:.0f} / {self.p95_latency_ms:.0f} / {self.p99_latency_ms:.0f} ms",
            "──────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic pair generator
# ---------------------------------------------------------------------------

# 20 semantic templates covering aggregation, filter, join, time-series, ranking.
# Each is instantiated with column names from the insurance schema at generation time.
_SQL_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "agg_avg_by_category",
        "semantic": "aggregation",
        "nl": "What is the average {metric} by {category}?",
        "sql": "SELECT {category}, AVG({metric}) AS avg_{metric} FROM {table} GROUP BY {category} ORDER BY avg_{metric} DESC LIMIT 10000",
    },
    {
        "template_id": "agg_sum_by_category",
        "semantic": "aggregation",
        "nl": "What is the total {metric} for each {category}?",
        "sql": "SELECT {category}, SUM({metric}) AS total_{metric} FROM {table} GROUP BY {category} ORDER BY total_{metric} DESC LIMIT 10000",
    },
    {
        "template_id": "agg_count_by_category",
        "semantic": "aggregation",
        "nl": "How many {entity} are there per {category}?",
        "sql": "SELECT {category}, COUNT(*) AS count_{entity} FROM {table} GROUP BY {category} ORDER BY count_{entity} DESC LIMIT 10000",
    },
    {
        "template_id": "filter_gt",
        "semantic": "filter",
        "nl": "Show all {entity} where {metric} is greater than {threshold}",
        "sql": "SELECT * FROM {table} WHERE {metric} > {threshold} LIMIT 10000",
    },
    {
        "template_id": "filter_eq",
        "semantic": "filter",
        "nl": "Show all {entity} where {category} is '{value}'",
        "sql": "SELECT * FROM {table} WHERE {category} = '{value}' LIMIT 10000",
    },
    {
        "template_id": "filter_date_range",
        "semantic": "filter",
        "nl": "Show all {entity} between {start_date} and {end_date}",
        "sql": "SELECT * FROM {table} WHERE {date_col} BETWEEN '{start_date}' AND '{end_date}' LIMIT 10000",
    },
    {
        "template_id": "join_two_tables",
        "semantic": "join",
        "nl": "Show the {metric} from {table1} along with the {attr} from {table2}",
        "sql": "SELECT t1.{metric}, t2.{attr} FROM {table1} t1 JOIN {table2} t2 ON t1.{join_key} = t2.{join_key} LIMIT 10000",
    },
    {
        "template_id": "time_series_monthly",
        "semantic": "time_series",
        "nl": "What is the monthly total {metric} for the past year?",
        "sql": "SELECT TO_CHAR(DATE_TRUNC('month', {date_col}::date), 'YYYY-MM') AS month, SUM({metric}) AS total_{metric} FROM {table} WHERE {date_col}::date >= CURRENT_DATE - INTERVAL '1 year' GROUP BY month ORDER BY month LIMIT 10000",
    },
    {
        "template_id": "time_series_quarterly",
        "semantic": "time_series",
        "nl": "What was the total {metric} per quarter last year?",
        # ML-1 FIX: previous version used EXTRACT(YEAR FROM ...) = {year} in
        # WHERE, violating NL_TO_SQL Rule 8 ("Never use EXTRACT() to filter a
        # time range"). Replaced with a range predicate using DATE_TRUNC so the
        # ground truth SQL is policy-compliant and matches what a correctly
        # behaving system would generate.
        "sql": "SELECT DATE_TRUNC('quarter', {date_col}::date) AS quarter, SUM({metric}) AS total_{metric} FROM {table} WHERE {date_col}::date >= DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 year') AND {date_col}::date < DATE_TRUNC('year', CURRENT_DATE) GROUP BY quarter ORDER BY quarter LIMIT 10000",
    },
    {
        "template_id": "ranking_top_n",
        "semantic": "ranking",
        "nl": "What are the top 10 {entity} by {metric}?",
        "sql": "SELECT {id_col}, SUM({metric}) AS total_{metric} FROM {table} GROUP BY {id_col} ORDER BY total_{metric} DESC LIMIT 10",
    },
    {
        "template_id": "ranking_bottom_n",
        "semantic": "ranking",
        "nl": "Which 5 {entity} have the lowest {metric}?",
        "sql": "SELECT {id_col}, AVG({metric}) AS avg_{metric} FROM {table} GROUP BY {id_col} ORDER BY avg_{metric} ASC LIMIT 5",
    },
    {
        "template_id": "null_analysis",
        "semantic": "data_quality",
        "nl": "How many {entity} are missing a {attribute}?",
        "sql": "SELECT COUNT(*) AS missing_count FROM {table} WHERE {attribute} IS NULL",
    },
    {
        "template_id": "distinct_values",
        "semantic": "exploration",
        "nl": "What are the different types of {category} in the data?",
        "sql": "SELECT DISTINCT {category} FROM {table} ORDER BY {category} LIMIT 10000",
    },
    {
        "template_id": "agg_count_total",
        "semantic": "aggregation",
        "nl": "How many {entity} are in the database?",
        "sql": "SELECT COUNT(*) AS total_{entity} FROM {table}",
    },
    {
        "template_id": "filter_between",
        "semantic": "filter",
        "nl": "Show {entity} where {metric} is between {low} and {high}",
        "sql": "SELECT * FROM {table} WHERE {metric} BETWEEN {low} AND {high} LIMIT 10000",
    },
    {
        "template_id": "agg_max_min",
        "semantic": "aggregation",
        "nl": "What is the highest and lowest {metric} on record?",
        "sql": "SELECT MAX({metric}) AS max_{metric}, MIN({metric}) AS min_{metric} FROM {table}",
    },
    {
        "template_id": "join_aggregate",
        "semantic": "join",
        "nl": "What is the average {metric} per {category} across {table1} and {table2}?",
        "sql": "SELECT t2.{category}, AVG(t1.{metric}) AS avg_{metric} FROM {table1} t1 JOIN {table2} t2 ON t1.{join_key} = t2.{join_key} GROUP BY t2.{category} ORDER BY avg_{metric} DESC LIMIT 10000",
    },
    {
        "template_id": "time_latest",
        "semantic": "time_series",
        "nl": "What were the most recent 20 {entity}?",
        "sql": "SELECT * FROM {table} ORDER BY {date_col} DESC LIMIT 20",
    },
    {
        "template_id": "percentile_analysis",
        "semantic": "statistics",
        "nl": "What is the distribution of {metric} by decile?",
        "sql": "SELECT decile, AVG({metric}) AS avg_{metric} FROM (SELECT {metric}, NTILE(10) OVER (ORDER BY {metric}) AS decile FROM {table}) sub GROUP BY decile ORDER BY decile LIMIT 10000",
    },
    {
        "template_id": "having_filter",
        "semantic": "aggregation",
        "nl": "Which {category} have a total {metric} above {threshold}?",
        # ML-1 FIX: previous version used SELECT alias `total_{metric}` in both
        # HAVING and ORDER BY, violating NL_TO_SQL Rule 10 ("Never reference a
        # SELECT alias in GROUP BY or HAVING"). Replaced with the full aggregate
        # expression SUM({metric}) in both clauses so ground truth SQL is
        # policy-compliant. A correctly behaving system will generate this form
        # and now scores correctly against it.
        "sql": "SELECT {category}, SUM({metric}) AS total_{metric} FROM {table} GROUP BY {category} HAVING SUM({metric}) > {threshold} ORDER BY SUM({metric}) DESC LIMIT 10000",
    },
]


def generate_pairs(
    db_path: str,
    schema_id: str = "insurance_v1",
    output_path: str | None = None,
    n_variations: int = 10,
    seed: int = 42,
) -> list[EvalPair]:
    """
    Generate synthetic NL→SQL test pairs from a SQLite database.

    Each semantic template is instantiated n_variations times with different
    column/table combinations from the actual schema.

    Parameters
    ----------
    db_path      : Path to the SQLite database.
    schema_id    : Schema identifier to embed in each pair.
    output_path  : If set, save pairs to JSON at this path.
    n_variations : How many NL variants to generate per template.
    seed         : Random seed for reproducibility.
    """
    import random

    random.seed(seed)

    # A-07 FIX: bare conn.close() after _introspect_sqlite(conn) left the
    # connection open if _introspect_sqlite raised (corrupt DB, permission
    # error, keyword-named table).  SQLite handles are OS file descriptors;
    # leaking them accumulates until ulimit -n is hit.
    with sqlite3.connect(db_path) as conn:
        schema = _introspect_sqlite(conn)

    pairs: list[EvalPair] = []
    pair_idx = 0

    for template in _SQL_TEMPLATES:
        for var_idx in range(n_variations):
            try:
                pair = _instantiate_template(
                    template, schema, schema_id, pair_idx=pair_idx, seed=seed + var_idx
                )
                if pair:
                    pairs.append(pair)
                    pair_idx += 1
            except Exception as exc:  # noqa: BLE001
                # A-09 FIX: log the exception type and template id so coding
                # bugs (KeyError, AttributeError in _instantiate_template) are
                # visible rather than silently reducing the generated pair count.
                import sys as _sys

                print(
                    f"[eval] WARNING: template '{template.get('template_id', '?')}' "
                    f"var {var_idx} skipped — {type(exc).__name__}: {exc}",
                    file=_sys.stderr,
                )
                continue

    # Assign splits: 60% train, 20% val, 20% test
    random.shuffle(pairs)
    n = len(pairs)
    for i, pair in enumerate(pairs):
        if i < int(n * 0.6):
            pair.split = "train"
        elif i < int(n * 0.8):
            pair.split = "val"
        else:
            pair.split = "test"

    if output_path:
        with open(output_path, "w") as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)
        print(f"Generated {len(pairs)} pairs → {output_path}")

    return pairs


_NON_CATEGORICAL_FRAGMENTS: frozenset[str] = frozenset(
    {
        "id",
        "email",
        "name",
        "code",
        "url",
        "hash",
        "token",
        "address",
        "phone",
        "zip",
        "description",
        "note",
        "comment",
        "text",
        "body",
        "content",
        "uuid",
        "password",
        "secret",
        "key",
        "full_name",
    }
)


def _is_categorical(col_name: str) -> bool:
    low = col_name.lower()
    return not any(frag in low for frag in _NON_CATEGORICAL_FRAGMENTS)


def _introspect_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema: dict[str, Any] = {}
    for table in tables:
        # L1 FIX: quote the table identifier so names that are SQL keywords
        # (e.g. "order", "group") or contain spaces don't cause syntax errors.
        # SQLite uses double-quote for ANSI identifier quoting; PRAGMA also
        # accepts a quoted name.
        q_table = f'"{table}"'
        cursor.execute(f"PRAGMA table_info({q_table})")
        cols = cursor.fetchall()

        text_cols = [
            c[1]
            for c in cols
            if "TEXT" in c[2].upper() or "CHAR" in c[2].upper() or "CLOB" in c[2].upper()
        ]
        categorical_cols = [c for c in text_cols if _is_categorical(c)]

        cat_values: dict[str, list[str]] = {}
        for col in categorical_cols:
            try:
                # L1 FIX: quote both col and table identifiers.
                # Bare identifiers fail for SQL-keyword column names
                # (e.g. "select", "order", "group") and names with spaces.
                q_col = f'"{col}"'
                rows = cursor.execute(
                    f"SELECT DISTINCT {q_col} FROM {q_table} " f"WHERE {q_col} IS NOT NULL LIMIT 20"
                ).fetchall()
                vals = [str(r[0]) for r in rows if r[0] is not None]
                if vals:
                    cat_values[col] = vals
            except Exception:
                pass

        schema[table] = {
            "columns": [c[1] for c in cols],
            "types": {c[1]: c[2] for c in cols},
            "numeric_cols": [
                c[1]
                for c in cols
                if "INT" in c[2].upper()
                or "REAL" in c[2].upper()
                or "FLOAT" in c[2].upper()
                or "NUM" in c[2].upper()
                or "DEC" in c[2].upper()
            ],
            "text_cols": text_cols,
            "categorical_cols": categorical_cols,
            "cat_values": cat_values,
            "date_cols": [c[1] for c in cols if "DATE" in c[2].upper() or "TIME" in c[2].upper()],
        }
    return schema


def _instantiate_template(
    template: dict[str, Any],
    schema: dict[str, Any],
    schema_id: str,
    pair_idx: int,
    seed: int,
) -> EvalPair | None:
    import random

    random.seed(seed)

    tables = list(schema.keys())
    if not tables:
        return None

    table = random.choice(tables)
    t = schema[table]

    numeric_cols = t["numeric_cols"]
    date_cols = t["date_cols"]
    all_cols = t["columns"]

    if not all_cols:
        return None

    categorical_cols = t.get("categorical_cols") or []
    cat_values = t.get("cat_values", {})

    # Only use categoricals that have real sampled values
    usable_cat_cols = [c for c in categorical_cols if c in cat_values and cat_values[c]]

    # Templates needing {metric} require at least one numeric column
    needs_metric = "{metric}" in template["nl"] or "{metric}" in template["sql"]
    if needs_metric and not numeric_cols:
        return None

    # Templates needing {category}/{value} require at least one real categorical
    needs_category = "{category}" in template["nl"] or "{category}" in template["sql"]
    if needs_category and not usable_cat_cols:
        return None

    chosen_cat = random.choice(usable_cat_cols) if usable_cat_cols else ""
    cat_val_pool = cat_values.get(chosen_cat, [])
    chosen_val = random.choice(cat_val_pool) if cat_val_pool else ""

    placeholders: dict[str, str] = {
        "table": table,
        "table1": table,
        "table2": random.choice(tables),
        "entity": table.rstrip("s"),
        "metric": random.choice(numeric_cols) if numeric_cols else "",
        "category": chosen_cat,
        "attribute": random.choice(all_cols),
        "id_col": all_cols[0],
        "date_col": random.choice(date_cols) if date_cols else "created_at",
        "attr": random.choice(all_cols),
        "join_key": all_cols[0],
        "threshold": str(random.choice([100, 500, 1000, 5000])),
        "low": "100",
        "high": "5000",
        "value": chosen_val,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "year": "2023",
    }

    try:
        nl = template["nl"].format(**placeholders)
        sql = template["sql"].format(**placeholders)
    except KeyError:
        return None

    # TEST-3 FIX: validate ground truth SQL against the system's own policy
    # rules before saving it as a reference answer. A policy-violating ground
    # truth (e.g. EXTRACT() in WHERE, SELECT alias in HAVING) causes a
    # correctly-behaving system to score lower than a rule-violating one.
    # validate_sql() returns valid=True for advisory warnings (NON_SARGABLE)
    # so only hard errors (MUTATION_STATEMENT, SYNTAX_ERROR) are rejected here.
    try:
        from validation import validate_sql

        schema_columns = list({col for cols in (t.get("columns") or []) for col in [cols]})
        vr = validate_sql(sql, schema_columns=schema_columns)
        if not vr.valid:
            import sys

            print(
                f"[eval] WARNING: ground truth SQL for template "
                f"'{template['template_id']}' failed validate_sql() "
                f"({vr.error_type}: {vr.error_message}). Pair skipped.",
                file=sys.stderr,
            )
            return None
    except Exception:  # noqa: BLE001
        # validate_sql unavailable (e.g. running outside the package) — skip check
        pass

    return EvalPair(
        pair_id=f"pair_{pair_idx:04d}",
        schema_id=schema_id,
        nl_query=nl,
        ground_truth_sql=sql,
        expected_table=table,
        semantic_template=template["semantic"],
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


async def run_evaluation(
    pairs: list[EvalPair],
    base_url: str = "http://localhost:8000",
    split: str = "test",
    concurrency: int = 5,
    timeout_seconds: int = 60,
    rate_limit_sleep_s: float = 50.0,
    api_key: str = "",
    limit: int | None = None,
    offset: int = 0,
) -> EvalReport:
    """
    Run the evaluation pipeline against a live server.

    Parameters
    ----------
    pairs               : List of EvalPair objects (usually loaded from JSON).
    base_url            : Server URL.
    split               : Which split to evaluate ("test", "val", or "all").
    concurrency         : Max concurrent requests.
    rate_limit_sleep_s  : Seconds to sleep between semaphore acquisitions to
                          pace requests within the LLM provider's TPM limit.
                          Default 50s (Groq 70b: 2 calls × ~2500 tok). Set
                          lower for providers with higher limits.
                          A-08 FIX: sleep is now OUTSIDE the semaphore so
                          slots are not held during the wait period, enabling
                          a true steady-rate stream instead of burst batches.
    api_key             : A-03 FIX: X-API-Key header value for production
                          deployments where H-07 auth middleware is active.
                          Without this, all /query calls return HTTP 401 and
                          every pair scores executable=False with no indication
                          that auth is the cause.
    """
    test_pairs = pairs if split == "all" else [p for p in pairs if p.split == split]
    if offset:
        test_pairs = test_pairs[offset:]
    if limit is not None:
        test_pairs = test_pairs[:limit]

    semaphore = asyncio.Semaphore(concurrency)
    results: list[EvalResult] = []

    # A-03 FIX: inject API key header when provided.
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout_seconds, headers=headers
    ) as client:
        tasks = [
            _evaluate_pair(client, pair, semaphore, rate_limit_sleep_s=rate_limit_sleep_s)
            for pair in test_pairs
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for pair, raw in zip(test_pairs, raw_results, strict=False):
        if isinstance(raw, BaseException):
            results.append(
                EvalResult(
                    pair_id=pair.pair_id,
                    nl_query=pair.nl_query,
                    generated_code="",
                    executable=False,
                    correct=False,
                    error_code="REQUEST_FAILED",
                    error_message=str(raw),
                )
            )
        else:
            results.append(raw)

    return _compute_report(results)


async def _evaluate_pair(
    client: httpx.AsyncClient,
    pair: EvalPair,
    semaphore: asyncio.Semaphore,
    *,
    rate_limit_sleep_s: float = 50.0,
) -> EvalResult:
    # SLEEP FIX: sleep AFTER the request, inside the semaphore.
    #
    # The previous A-08 FIX moved sleep BEFORE the semaphore to avoid holding
    # the slot during the wait.  That logic is correct for concurrency > 1, but
    # it breaks rate limiting entirely for concurrency=1:
    #
    #   With sleep BEFORE semaphore, concurrency=1:
    #     All N tasks are created and immediately start sleeping simultaneously.
    #     After sleep_s seconds ALL N tasks wake up and queue for the one slot.
    #     The semaphore serialises them with ZERO gap between requests — the sleep
    #     provided no inter-request delay whatsoever.  This is why 100K TPD was
    #     exhausted in under 3 minutes despite rate_limit_sleep_s=50.
    #
    #   With sleep AFTER request, inside semaphore, concurrency=1:
    #     task1: acquire → request → sleep → release
    #     task2: acquire → request → sleep → release
    #     Genuine sleep_s gap between end of one request and start of next.
    #
    #   For concurrency=N (N>1):
    #     Each of the N concurrent tasks sleeps after its own request before
    #     yielding the slot.  Still some burst potential (N tasks fire together)
    #     but each burst is followed by a sleep_s pause — acceptable for typical
    #     eval concurrency of 1–3.
    #
    # The right fix for production-grade rate limiting is a shared token-bucket
    # across all workers.  For eval purposes, sleep-after-request is correct
    # and simple.
    async with semaphore:
        t0 = time.monotonic()
        try:
            resp = await client.post(
                "/query",
                json={
                    "nl_query": pair.nl_query,
                    "schema_id": pair.schema_id,
                    "execution_mode": "auto",
                    "dry_run": False,
                },
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return EvalResult(
                pair_id=pair.pair_id,
                nl_query=pair.nl_query,
                generated_code="",
                executable=False,
                correct=False,
                latency_ms=int((time.monotonic() - t0) * 1000),
                error_code="REQUEST_FAILED",
                error_message=str(exc),
            )
        finally:
            # Always sleep after releasing this slot — even on failure — so a
            # burst of errors doesn't immediately exhaust the retry budget on
            # a rate-limited API.
            await asyncio.sleep(rate_limit_sleep_s)

        error = data.get("error")
        executable = error is None and data.get("generated_code", "") != ""

        # F10: schema_recall — check whether the response used the correct
        # table rather than hardcoding True.
        schema_recall = _check_schema_recall(data, pair)

        # F11: correctness — row-level comparison against the local SQLite DB
        # when pair.db_path is set; otherwise fall back to column-name proxy.
        correct = False
        partial_correct = False
        if executable:
            generated_sql = data.get("generated_code", "")
            if pair.db_path and generated_sql:
                correct = _check_correctness_local(
                    pair.ground_truth_sql, generated_sql, pair.db_path
                )
            elif data.get("result_preview"):
                # Column-name proxy (fallback when no local DB path is available)
                generated_cols = set(
                    data["result_preview"][0].keys() if data["result_preview"] else []
                )
                gt_cols = _expected_columns(pair.ground_truth_sql)
                if not gt_cols:
                    correct = bool(generated_cols)
                else:
                    if generated_cols == gt_cols:
                        correct = True
                    elif generated_cols:
                        correct, partial_correct = _columns_semantically_match(
                            generated_cols, gt_cols
                        )

        return EvalResult(
            pair_id=pair.pair_id,
            nl_query=pair.nl_query,
            generated_code=data.get("generated_code", ""),
            executable=executable,
            correct=correct,
            partial_correct=partial_correct,
            retry_count=data.get("retry_count", 0),
            latency_ms=latency_ms,
            schema_recall=schema_recall,
            error_code=error.get("error_code") if error else None,
            error_message=error.get("message") if error else None,
        )


def _check_schema_recall(response_data: dict[str, Any], pair: EvalPair) -> bool:
    """
    Return True if pair.expected_table appears in the response's retrieved
    context chunks or in the generated SQL. Falls back to False when neither
    signal is available.

    Priority:
      1. response_data["chunks_used"] list (if the server exposes it)
      2. Check whether expected_table is referenced in generated_code (SQL)
    """
    chunks_used: list[str] = response_data.get("chunks_used", [])
    if chunks_used:
        return any(pair.expected_table.lower() in c.lower() for c in chunks_used)

    # Fallback: check if the expected table appears in the generated SQL
    generated_sql = response_data.get("generated_code", "")
    if generated_sql:
        return pair.expected_table.lower() in generated_sql.lower()

    return False


def _check_correctness_local(
    ground_truth_sql: str,
    generated_sql: str,
    db_path: str,
) -> bool:
    """
    Execute both SQLs against a local SQLite DB and compare result sets.

    Comparison is order-insensitive (frozensets of row tuples).
    Returns False on any execution error in either query.

    P2-08 FIX: the generated SQL is LLM output — it may contain destructive
    statements (DROP TABLE, DELETE FROM ... WHERE 1=1) that would corrupt the
    evaluation DB if executed in autocommit mode.  Two-layer defence:

      1. Open the DB file in read-only mode (?mode=ro URI) so SQLite refuses
         any write at the OS level — the fastest and most reliable barrier.
      2. Wrap generated_sql in an explicit transaction that is always rolled
         back.  This is a belt-and-suspenders fallback for edge cases where
         the SQLite build does not honour the URI flag.

    Ground-truth SQL is trusted (authored by the eval writer) so it runs
    after the read-only connection is opened — a read-only SELECT is fine;
    if it somehow isn't, the OS-level guard catches it.
    """
    import sqlite3

    try:
        # mode=ro: opens DB file read-only at the VFS level.
        # immutable=1 would be stronger but prevents concurrent reads on some
        # platforms. check_same_thread=False is safe here (single-threaded eval).
        ro_uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True)
        try:
            gt_rows = set(map(tuple, conn.execute(ground_truth_sql).fetchall()))
            # Run generated SQL inside a savepoint so any implicit side-effect
            # (in case the read-only flag is bypassed) is always discarded.
            conn.execute("SAVEPOINT correctness_check")
            try:
                gen_rows = set(map(tuple, conn.execute(generated_sql).fetchall()))
            finally:
                conn.execute("ROLLBACK TO SAVEPOINT correctness_check")
                conn.execute("RELEASE SAVEPOINT correctness_check")
            return gt_rows == gen_rows
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return False


def _split_select_expressions(select_part: str) -> list[str]:
    """Split SELECT clause on commas, respecting parentheses nesting."""
    exprs, depth, current = [], 0, []
    for ch in select_part:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            exprs.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        exprs.append("".join(current).strip())
    return [e for e in exprs if e]


def _expected_columns(sql: str) -> set[str]:
    """
    Extract aliased column names from a SELECT statement.
    Handles nested parentheses correctly. Returns empty set for SELECT *.
    """
    import re as _re

    sql_upper = sql.upper()
    if "SELECT" not in sql_upper:
        return set()
    select_part = sql[sql_upper.index("SELECT") + 6 :]
    if "FROM" in select_part.upper():
        select_part = select_part[: select_part.upper().index("FROM")]
    if select_part.strip() == "*":
        return set()
    cols: set[str] = set()
    for col_expr in _split_select_expressions(select_part):
        col_upper = col_expr.upper()
        if " AS " in col_upper:
            alias = col_expr[col_upper.rfind(" AS ") + 4 :].strip().strip('"').strip("'")
            if alias:
                cols.add(alias.lower())
        else:
            parts = _re.split(r"\s+", col_expr.strip())
            if parts:
                raw = parts[-1].lower().strip("()")
                if "." in raw:
                    raw = raw.split(".")[-1]
                if raw and raw != "*" and not raw.startswith("("):
                    cols.add(raw)
    return cols


def _normalize_col(col: str) -> str:
    """Strip table prefix and aggregate prefix for semantic comparison."""
    if "." in col:
        col = col.split(".")[-1]
    for prefix in (
        "total_",
        "avg_",
        "average_",
        "max_",
        "min_",
        "count_",
        "sum_",
        "highest_",
        "lowest_",
        "num_",
        "number_of_",
        "missing_",
    ):
        if col.startswith(prefix):
            col = col[len(prefix) :]
            break
    return col


_BARE_AGGREGATES = frozenset({"count", "sum", "avg", "average", "max", "min"})


def _columns_semantically_match(a: set[str], b: set[str]) -> tuple[bool, bool]:
    """
    Return (correct, partial_correct) after normalising aggregate prefixes.

    ML-5 FIX: the previous 50% threshold meant a 4-column result was "correct"
    with only 2 matching columns — too lenient for analytical queries where every
    column is semantically significant.

    Thresholds:
      correct         — overlap >= 75%  (or exact scalar match)
      partial_correct — overlap >= 50% and < 75%  (tracked separately)

    Scalar result (single aggregation): exact column match required for correct;
    bare aggregate names (count/sum/avg) match any single-column GT result.

    Returns (False, False) when either set is empty.
    """
    if not a or not b:
        return False, False

    a_norm = {_normalize_col(c) for c in a}
    b_norm = {_normalize_col(c) for c in b}

    # Scalar: bare aggregate matches any single-value ground truth
    if a_norm <= _BARE_AGGREGATES and len(b_norm) == 1:
        return True, False

    overlap = len(a_norm & b_norm) / max(len(a_norm), len(b_norm))
    correct = overlap >= 0.75
    partial = (not correct) and overlap >= 0.50
    return correct, partial


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_report(results: list[EvalResult]) -> EvalReport:
    n = len(results)
    if n == 0:
        return EvalReport(
            total_pairs=0,
            executable_rate=0.0,
            executable_ci_95=(0.0, 0.0),
            correctness_rate=0.0,
            correctness_ci_95=(0.0, 0.0),
            partial_correct_rate=0.0,
            error_recovery_rate=0.0,
            schema_recall_at_5=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
        )

    executable = [r for r in results if r.executable]
    correct = [r for r in results if r.correct]
    partial = [r for r in results if r.partial_correct and not r.correct]
    recovered = [r for r in results if r.retry_count > 0 and r.correct]
    retried = [r for r in results if r.retry_count > 0]
    recalled = [r for r in results if r.schema_recall]
    latencies = sorted(float(r.latency_ms) for r in results)

    exec_rate = len(executable) / n
    corr_rate = len(correct) / max(len(executable), 1)
    partial_rate = len(partial) / max(len(executable), 1)
    recovery_rate = len(recovered) / max(len(retried), 1)
    recall_rate = len(recalled) / n

    return EvalReport(
        total_pairs=n,
        executable_rate=exec_rate,
        executable_ci_95=_binomial_ci(exec_rate, n),
        correctness_rate=corr_rate,
        correctness_ci_95=_binomial_ci(corr_rate, len(executable) or 1),
        partial_correct_rate=partial_rate,
        error_recovery_rate=recovery_rate,
        schema_recall_at_5=recall_rate,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        results=results,
    )


def _binomial_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _percentile(sorted_data: list[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    f, c = int(k), math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ---------------------------------------------------------------------------
# Canonical regression suite
# ---------------------------------------------------------------------------
# Fixed, hand-verified test cases for query patterns that have been debugged
# and confirmed correct.  Unlike the synthetic EvalPair suite, canonical cases:
#   - are deterministic (fixed nl_query text → same LTM cache key every run)
#   - check METRIC VALUE BOUNDS, not just column-name overlap
#   - verify system-level invariants: warnings, retry_count, row_count
#
# Run:  python eval.py canonical --url http://localhost:8000
# CI:   exits 0 on full pass, 1 on any failure — safe for GitHub Actions gate
# ---------------------------------------------------------------------------


@dataclass
class MetricBound:
    """Assert that every value in `column` falls within [min_val, max_val]."""

    column: str
    min_val: float | None = None  # inclusive lower bound; None = unchecked
    max_val: float | None = None  # inclusive upper bound; None = unchecked


@dataclass
class CanonicalCase:
    case_id: str
    nl_query: str
    schema_id: str
    description: str = ""
    expected_row_count: int | None = None
    expected_max_retry_count: int = 0
    expected_warnings_empty: bool = True
    # list (not set) so dataclasses.asdict → json.dump works without a custom encoder
    required_columns: list[str] = field(default_factory=list)
    metric_assertions: list[MetricBound] = field(default_factory=list)


@dataclass
class CanonicalResult:
    case_id: str
    nl_query: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    retry_count: int = 0
    latency_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None


@dataclass
class CanonicalReport:
    total: int
    passed: int
    failed: int
    results: list[CanonicalResult] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.failed == 0 else "FAIL"
        lines = [
            f"── Canonical Regression ── {status} ───────────────────────────",
            f"  {self.passed}/{self.total} cases passed",
        ]
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(
                f"  {icon} [{r.case_id}] {r.nl_query[:65]}"
                f"  ({r.latency_ms}ms, retry={r.retry_count})"
            )
            for failure in r.failures:
                lines.append(f"      → {failure}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)


# Add new cases here as you verify additional query patterns.
# expected_row_count and metric_assertions are the primary regression gates.
# expected_max_retry_count=0 asserts that LTM serves the result without
# ERROR_CORRECT (after the first cold-cache run).
_CANONICAL_CASES: list[CanonicalCase] = [
    CanonicalCase(
        case_id="canon-001",
        description=(
            "Multi-metric policy comparison — exercises payment pre-aggregation "
            "(LEFT JOIN on payment_agg CTE), standalone premium CTE, "
            "::numeric cast for approval rate, calibrated loss ratio 0.83–0.86."
        ),
        nl_query=(
            "Compare total claim amount, total paid amount, claim approval rate, "
            "and loss ratio for each policy type."
        ),
        schema_id="ins_prod_v3",
        expected_row_count=4,
        expected_max_retry_count=0,
        expected_warnings_empty=True,
        required_columns=[
            "policy_type",
            "total_claim_amount",
            "claim_approval_rate",
            "loss_ratio",
        ],
        metric_assertions=[
            # Approval rate: ~49.6% in calibrated data; >0.99 signals WHERE
            # claim_status pre-filter bug; <0.1 signals integer-division bug.
            MetricBound(column="claim_approval_rate", min_val=0.490, max_val=0.510),
            # Loss ratio: ~0.85 with calibrated premiums; >2.0 signals
            # denominator understatement (INNER JOIN or correlated subquery).
            MetricBound(column="loss_ratio", min_val=0.80, max_val=0.92),
            # Sanity floor: each type should have >$100M in claims.
            MetricBound(column="total_claim_amount", min_val=1e8),
        ],
    ),
    CanonicalCase(
        case_id="canon-002",
        description=(
            "Top-N per group with window function — ROW_NUMBER() OVER "
            "(PARTITION BY region), agents→policies→claims join path, "
            "top-3 filter per region yields 5×3=15 rows."
        ),
        nl_query=(
            "For each region, rank policy types by total claim amount " "and return the top 3."
        ),
        schema_id="ins_prod_v3",
        expected_row_count=15,  # 5 regions × top-3 policy types
        expected_max_retry_count=0,
        expected_warnings_empty=True,
        required_columns=["region", "policy_type", "total_claim_amount"],
        metric_assertions=[
            # Each top-3 entry should have >$10M (life rows are >$900M).
            MetricBound(column="total_claim_amount", min_val=1e7),
        ],
    ),
]


async def run_canonical(
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    timeout_seconds: float = 60.0,
    rate_limit_sleep_s: float = 2.0,
) -> CanonicalReport:
    """
    Run the canonical regression suite against a live server.

    LTM-cached queries complete in ~2s each (no generation LLM call), so
    rate_limit_sleep_s=2.0 is sufficient for standard Groq rate limits.
    The first run after an LTM clear triggers GENERATION (~6–10s); subsequent
    runs use the cache.

    Returns a CanonicalReport.  The CLI wrapper exits with code 1 when
    report.failed > 0, making this safe as a CI gate.
    """
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout_seconds, headers=headers
    ) as client:
        tasks = [
            _evaluate_canonical_case(client, case, rate_limit_sleep_s=rate_limit_sleep_s)
            for case in _CANONICAL_CASES
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[CanonicalResult] = []
    for case, raw in zip(_CANONICAL_CASES, raw_results, strict=False):
        if isinstance(raw, BaseException):
            results.append(
                CanonicalResult(
                    case_id=case.case_id,
                    nl_query=case.nl_query,
                    passed=False,
                    failures=[f"Unhandled exception: {raw}"],
                    error=str(raw),
                )
            )
        else:
            results.append(raw)

    passed = sum(1 for r in results if r.passed)
    return CanonicalReport(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )


async def _evaluate_canonical_case(
    client: httpx.AsyncClient,
    case: CanonicalCase,
    *,
    rate_limit_sleep_s: float = 2.0,
) -> CanonicalResult:
    await asyncio.sleep(rate_limit_sleep_s)
    t0 = time.monotonic()

    try:
        resp = await client.post(
            "/query",
            json={
                "nl_query": case.nl_query,
                "schema_id": case.schema_id,
                "execution_mode": "auto",
                "dry_run": False,
            },
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return CanonicalResult(
            case_id=case.case_id,
            nl_query=case.nl_query,
            passed=False,
            failures=[f"HTTP error: {exc}"],
            latency_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc),
        )

    failures: list[str] = []

    # --- system-level invariants ---
    if data.get("error"):
        err = data["error"]
        failures.append(
            f"error present: {err.get('error_code','?')} — {err.get('message','')[:100]}"
        )

    retry_count = data.get("retry_count", 0)
    if retry_count > case.expected_max_retry_count:
        failures.append(
            f"retry_count={retry_count} exceeds expected max {case.expected_max_retry_count}"
        )

    warnings = data.get("warnings", [])
    if case.expected_warnings_empty and warnings:
        failures.append(f"unexpected warning: {warnings[0][:120]}")

    row_count = data.get("row_count", 0)
    if case.expected_row_count is not None and row_count != case.expected_row_count:
        failures.append(f"row_count={row_count} != expected {case.expected_row_count}")

    # --- required column presence ---
    preview = data.get("result_preview", [])
    if preview and case.required_columns:
        actual = {k.lower() for k in preview[0]}
        missing = {c.lower() for c in case.required_columns} - actual
        if missing:
            failures.append(f"missing columns: {sorted(missing)}")

    # --- metric value bounds ---
    if preview:
        for assertion in case.metric_assertions:
            col_lower = assertion.column.lower()
            matched = next((k for k in preview[0] if k.lower() == col_lower), None)
            if matched is None:
                continue  # already reported as missing column above
            bad: list[str] = []
            for row in preview:
                val = row.get(matched)
                if not isinstance(val, int | float):
                    continue
                if assertion.min_val is not None and val < assertion.min_val:
                    bad.append(f"{val} < min {assertion.min_val}")
                if assertion.max_val is not None and val > assertion.max_val:
                    bad.append(f"{val} > max {assertion.max_val}")
            if bad:
                failures.append(
                    f"{assertion.column} bound violation: {bad[0]}"
                    + (f" (+{len(bad)-1} more)" if len(bad) > 1 else "")
                )

    return CanonicalResult(
        case_id=case.case_id,
        nl_query=case.nl_query,
        passed=len(failures) == 0,
        failures=failures,
        retry_count=retry_count,
        latency_ms=latency_ms,
        warnings=warnings,
        row_count=row_count,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Data Analyst Copilot — evaluation CLI")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate synthetic test pairs")
    gen.add_argument("--db", required=True, help="SQLite DB path")
    gen.add_argument("--output", default="data/eval_pairs.json")
    gen.add_argument("--schema-id", default="insurance_v1")
    gen.add_argument("--variations", type=int, default=10)

    run_p = sub.add_parser("run", help="Run evaluation against live server")
    run_p.add_argument("--pairs", required=True, help="Path to eval_pairs.json")
    run_p.add_argument("--url", default="http://localhost:8000")
    run_p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    run_p.add_argument("--concurrency", type=int, default=5)
    run_p.add_argument("--output", default=None, help="Save report JSON to this path")
    # A-03 FIX: production deployments require X-API-Key (H-07 auth middleware).
    # Without this flag, all /query calls return HTTP 401 and every pair scores
    # executable=False with REQUEST_FAILED — making the eval report useless.
    run_p.add_argument(
        "--api-key",
        default="",
        dest="api_key",
        help="X-API-Key header value for authenticated production servers.",
    )
    run_p.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=50.0,
        dest="rate_limit_sleep",
        help=(
            "Seconds to sleep between requests (default 50.0). "
            "With --concurrency 1, set to 0 — processing time alone "
            "keeps you under the app 10 RPM gate."
        ),
    )
    run_p.add_argument(
        "--limit",
        type=int,
        default=None,
        dest="limit",
        help=(
            "Maximum number of pairs to evaluate after split filtering "
            "(default: all). Use to stay within provider daily token budget; "
            "e.g. --limit 9 for Groq free-tier llama-3.3-70b at ~11k tok/req "
            "(floor(100_000 / 11_000) = 9 requests before hitting TPD)."
        ),
    )
    run_p.add_argument(
        "--offset",
        type=int,
        default=0,
        dest="offset",
        help=(
            "Number of pairs to skip after split filtering and before "
            "applying --limit (default: 0). Use with --limit to walk "
            "through the dataset across multiple days: "
            "--offset 0 --limit 10 on day 1, "
            "--offset 10 --limit 10 on day 2, etc."
        ),
    )

    smoke_p = sub.add_parser("smoke", help="Quick 10-pair smoke test (no server)")
    smoke_p.add_argument("--db", required=True)
    smoke_p.add_argument("--schema-id", default="insurance_v1", dest="schema_id")

    canon_p = sub.add_parser(
        "canonical",
        help="Run fixed regression suite with metric-bound assertions (CI-safe)",
    )
    canon_p.add_argument("--url", default="http://localhost:8000")
    canon_p.add_argument(
        "--api-key",
        default="",
        dest="api_key",
        help="X-API-Key header for authenticated servers",
    )
    canon_p.add_argument(
        "--output",
        default=None,
        help="Save full JSON report to this path",
    )
    canon_p.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=2.0,
        dest="rate_limit_sleep",
        help="Seconds between requests (default 2.0; increase if hitting Groq rate limits)",
    )

    args = parser.parse_args()

    if args.command == "generate":
        pairs = generate_pairs(args.db, args.schema_id, args.output, args.variations)
        print(f"Generated {len(pairs)} pairs.")

    elif args.command == "run":
        with open(args.pairs) as f:
            raw = json.load(f)
        pairs = [EvalPair(**p) for p in raw]
        eval_report = asyncio.run(
            run_evaluation(
                pairs,
                args.url,
                args.split,
                args.concurrency,
                rate_limit_sleep_s=args.rate_limit_sleep,
                api_key=args.api_key,
                limit=args.limit,
                offset=args.offset,
            )
        )
        print(eval_report.summary())
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(eval_report), f, indent=2)
            print(f"Report saved → {args.output}")

    elif args.command == "smoke":
        pairs = generate_pairs(args.db, schema_id=args.schema_id, n_variations=1)[:10]
        print(f"Smoke test: {len(pairs)} pairs generated from {args.db}")
        for p in pairs:
            print(f"  [{p.semantic_template:15s}] {p.nl_query[:60]}")

    elif args.command == "canonical":
        import sys as _sys

        canonical_report = asyncio.run(
            run_canonical(
                base_url=args.url,
                api_key=args.api_key,
                rate_limit_sleep_s=args.rate_limit_sleep,
            )
        )
        print(canonical_report.summary())
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(canonical_report), f, indent=2)
            print(f"Report saved → {args.output}")
        # Non-zero exit for CI gate: any failure fails the pipeline.
        _sys.exit(0 if canonical_report.failed == 0 else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
