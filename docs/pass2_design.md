# Data Analyst Copilot — Phase 2 & 3 Design Document

**Version:** 1.0  
**Status:** Implemented  
**Scope:** Execution loop, Pandas DataFrame injection, long-term memory,
evaluation pipeline, load testing, observability alerts

---

## 1. What Phase 1 Left Incomplete

Phase 1 delivered a working NL→SQL pipeline: schema ingestion, retrieval,
generation, validation, and SQL execution. Three stubs were left deliberately
unfilled:

| Stub | Location | Impact |
|---|---|---|
| `dataframe_refs=None` | `orchestrator.py` line ~310 | Pandas executor always ran in an empty namespace — no DataFrames injected |
| No long-term memory | `orchestrator.py` `_retrieval_state` | Every query generated from scratch; no few-shot learning from past successes |
| Empty `eval.py` | `analyst_copilot/eval.py` | No way to measure correctness, executable rate, or latency against a ground truth |

Phases 2 and 3 fill these gaps.

---

## 2. Phase 2 — DataFrame Injection Architecture

### 2.1 The Problem

The Pandas execution path in `validation.py` (`execute_python`) accepts a
`dataframe_refs: dict[str, pd.DataFrame]` argument — a namespace of named
DataFrames injected into the sandboxed `exec()` scope. Without this, generated
Pandas code like `result = df['claim_amount'].mean()` fails immediately with
`NameError: name 'df' is not defined`.

Phase 1 passed `None`, making the entire Pandas path non-functional.

### 2.2 Two Sources of DataFrames

Two distinct components fill `dataframe_refs`, merged before passing to the
executor:

```
User uploads a file          →  DataFrameStore  →  per-session dict
                                                         ↘
                                                    merged dataframe_refs
                                                         ↗
DB table requested by query  →  DataFrameLoader  →  per-schema cache
```

**`dataframe_store.py`** — handles user-uploaded files (CSV, Parquet, Excel).
DataFrames are keyed by a sanitised Python identifier derived from the filename
and stored per session. TTL: 1 hour of inactivity. Max per-session: 256 MB.

**`dataframe_loader.py`** — handles DB-sourced tables. When the orchestrator
detects the Pandas path is needed, it identifies which tables are in the
retrieved schema chunks and loads them via `pd.read_sql`. Results are cached
in-process by `(schema_id, table_name)` with a 5-minute TTL. Max cache: 512 MB.

### 2.3 Merge Logic (orchestrator.py patch)

```python
# Inside orchestrator._execution_state(), before loop.run():

# Source 1: user-uploaded DataFrames for this session
uploaded_refs: dict = {}
if self._df_store is not None:
    uploaded_refs = self._df_store.get(state.session_id)

# Source 2: DB-loaded DataFrames for tables in the retrieved schema chunks
db_refs: dict = {}
if state.code_type == "pandas" and self._df_loader is not None:
    tables = [chunk.table for chunk in state.chunks]
    db_refs = await self._df_loader.load(tables, schema_id=state.schema_id)

# Merge — uploaded files take precedence (explicit > inferred)
dataframe_refs = {**db_refs, **uploaded_refs}

loop_result = loop.run(
    code=state.active_code,
    code_type=state.code_type,
    schema_columns=state.schema_columns,
    dataframe_refs=dataframe_refs,   # ← filled
    dialect=_dialect_for_request(request),
    dry_run=request.dry_run,
)
```

### 2.4 Security Properties

Both sources enforce the same sandbox constraints defined in `validation.py`:

- AST validation (`_ForbiddenNodeVisitor`) runs before `exec()`.
- The exec namespace contains only whitelisted builtins and pre-seeded modules
  (`pd`, `np`, `math`, `re`, `collections`).
- `os`, `sys`, `subprocess`, `open()`, `eval()`, `exec()` are blocked at AST
  level before the code reaches the runtime.
- DataFrames are passed by reference — the sandbox cannot reassign them to
  affect the outer scope (Python's exec scoping).

### 2.5 DataFrame Path Selection Policy

The orchestrator selects the execution path (SQL vs Pandas) in
`_select_code_type()`. Pandas is selected when:

1. `execution_mode == "pandas"` (explicit request), or
2. The schema source is a file (CSV/Parquet) — detected from `schema_id`
   prefix or from `DataFrameStore.list_dataframes()` returning non-empty, or
3. The SQL path has failed twice and the query involves transformation logic
   that SQL cannot express (multi-step groupby, pivot, custom functions).

SQL is always the default (`execution_mode == "auto"`).

---

## 3. Phase 3 — Long-Term Memory

### 3.1 Why Long-Term Memory Improves Generation Quality

Without memory, every query is generated cold. The LLM receives only the
current schema context and session history (last 10 turns). For recurring
query patterns — e.g. "show me X by policy type last quarter" — the model
regenerates the same SQL every time, including any errors it previously made.

Long-term memory provides up to 3 semantically similar past (nl_query, sql)
pairs as few-shot examples in the generation prompt. This:

- Reduces hallucinated column names (the model sees what actually worked).
- Improves dialect-specific syntax (it sees real DATE_TRUNC / DATEADD usage).
- Enables an optional fast path: if similarity ≥ 0.97 (`EXACT_HIT_THRESHOLD`),
  the cached SQL is used directly, skipping the LLM call entirely.

### 3.2 Storage Architecture

```
data/lt_memory/
  lt_memory.faiss        — FAISS IndexHNSWFlat (global, all schemas)
  lt_memory.meta.json    — parallel list of MemoryRecord dicts
```

One global FAISS index (not per-schema) is used for the vector store.
Schema isolation is enforced at search time by filtering `active_indices`
to records whose `schema_id` matches the query. This avoids the overhead of
maintaining N separate FAISS indices while preserving isolation.

**Index type:** `IndexHNSWFlat` with `METRIC_INNER_PRODUCT`.
- M=32 (neighbours per node), efConstruction=200, efSearch=40.
- Vectors are L2-normalised before insertion, so inner product = cosine similarity.
- HNSW chosen over Flat because at 10,000 records (the cap) it reduces
  query time from O(N) to O(log N) with <1% recall loss at ef=40.

### 3.3 Write Path

```
_persist_turn()
  └─ if response.error is None and active_code is not None:
       LongTermMemory.store(session_id, schema_id, nl_query, sql, insight)
         ├─ PII guard: _contains_pii(nl_query) → skip if detected
         ├─ Dedup: sha256(schema_id + nl_query) → skip if already stored
         ├─ Capacity check: len(records) >= MAX_RECORDS (10,000) → skip
         ├─ embed_query(nl_query) → vector
         ├─ index.add(vector)
         ├─ records.append(MemoryRecord)
         └─ auto-save every 50 new records
```

Only successful turns are stored (`response.error is None`). Failed queries
are never stored — injecting a bad SQL example would degrade generation.

### 3.4 Read Path

```
_retrieval_state()
  └─ after fetching schema chunks:
       LongTermMemory.search(nl_query, schema_id, k=2)
         ├─ active_indices = [i for i in records if schema_id matches and not stale]
         ├─ embed_query(nl_query) → query vector
         ├─ index.search(query_vector, k * 5) → candidates
         ├─ filter: similarity >= SIMILARITY_THRESHOLD (0.85)
         ├─ sort by similarity desc
         └─ return top-k MemorySearchResult
       state.lt_memory_hits = hits  → injected into generation prompt
```

### 3.5 PII Guard

Four regex patterns block storage of queries containing detected PII:

| Pattern | Matches |
|---|---|
| Email | `user@domain.tld` |
| US phone | `555-555-5555`, `(555) 555 5555` |
| SSN | `123-45-6789` |
| Credit card | `4111 1111 1111 1111` |

This is a heuristic filter, not a complete PII solution. For production
deployments handling regulated data (HIPAA, PCI-DSS), replace with a dedicated
PII detection service before enabling long-term memory.

### 3.6 Staleness and Rebuild Policy

- Records older than `RECORD_TTL_DAYS` (30) are excluded from search results
  by filtering `active_indices` to records with `created_at >= cutoff`.
- The FAISS index is not pruned in place (HNSW does not support deletion).
  Instead, `rebuild_if_stale()` reconstructs the index from scratch when
  >20% of records are stale. Triggered manually or on server restart.
- `save()` is called every 50 writes and on server shutdown via the `_shutdown`
  hook in `app.py`.

### 3.7 Exact-Hit Fast Path

When a search result has `similarity >= EXACT_HIT_THRESHOLD` (0.97), the
orchestrator may use the cached SQL directly without an LLM call. This is an
optional optimisation gated on the similarity threshold:

```python
if lt_hits and lt_hits[0].is_exact_hit:
    state.active_code = lt_hits[0].record.sql
    state.code_type = "sql"
    # skip GENERATION state → go directly to VALIDATION
```

This path is disabled by default. Enable it by setting `USE_LT_EXACT_HIT=true`
in `.env`. Use with caution: a 0.97 threshold still allows false positives when
two queries are phrased very similarly but have different semantics
(e.g. "claims this quarter" vs "claims last quarter").

---

## 4. Evaluation Pipeline (eval.py)

### 4.1 Metrics (operationally defined per Section 8)

| Metric | Definition | Measurement method |
|---|---|---|
| `executable_rate` | Fraction of generated code that runs without SyntaxError, NameError, or UNRESOLVED_COLUMN | Automated: run every generated query against a test DB |
| `correctness_rate` | Fraction of executable results matching ground-truth SQL (row-level equality, order-insensitive, column-subset) | Automated: `pd.testing.assert_frame_equal(check_like=True)` |
| `error_recovery_rate` | Fraction of initially-failing queries that succeed within 3 retry attempts | Automated: `retry_count > 0 and executable` |
| `schema_recall_at_5` | Fraction of queries where the correct table appears in the generated code | Proxy: `correct_table.lower() in generated_code.lower()` |
| `p50/p95/p99 latency` | Wall-clock ms from `POST /query` to response | Instrumented at HTTP client |

All accuracy metrics report 95% Wilson score confidence intervals.

### 4.2 Ground Truth Generation Strategy

Ground truth is generated synthetically from a known reference database where
the correct SQL is deterministic:

1. Write correct SQL queries by hand for each semantic template (aggregation,
   filter, join, time-series grouping, ranking).
2. Parameterise templates: generate NL variations by substituting column names,
   time ranges, and filter values.
3. Human annotation is only required for ambiguous NL queries where multiple
   SQL statements are semantically valid.

Minimum: 5 semantic templates × 4 NL variations = 20 pairs for smoke testing.
Production target: 200 pairs (statistically valid at 95% CI with ±7% margin).

### 4.3 Running the Evaluation

```bash
# Quick smoke test (20 synthetic pairs, no ground truth DB required)
python -m analyst_copilot.eval \
    --schema-id ins_prod_v3 \
    --base-url http://localhost:8000 \
    --synthetic \
    --n-synthetic 20

# Full evaluation with ground truth correctness checking
python -m analyst_copilot.eval \
    --test-file data/eval/test_pairs.json \
    --schema-id ins_prod_v3 \
    --base-url http://localhost:8000 \
    --concurrency 2 \
    --output data/eval/report.json
```

### 4.4 Test Pair JSON Schema

```json
[
  {
    "nl_query": "What was the average claim amount by policy type last quarter?",
    "ground_truth_sql": "SELECT policy_type, AVG(claim_amount) AS avg_claim FROM claims WHERE ...",
    "expected_columns": ["policy_type", "avg_claim"],
    "expected_row_count_min": 1,
    "correct_table": "claims"
  }
]
```

---

## 5. Load Testing (scripts/load_test.py)

### 5.1 SLA Target

p99 end-to-end latency ≤ 5,000 ms at 50 concurrent users.
This is the production-grade requirement from Section 1 of the system design.

### 5.2 Test Design

15 representative NL queries covering all query complexity tiers:

- Simple aggregation (avg, sum, count)
- Filter queries (WHERE clauses, date ranges)
- Ranking queries (TOP N, ORDER BY)
- Time-series grouping (monthly, quarterly)
- Cross-table implicit joins (via schema context)

Queries are randomised per request to avoid cache bias. Each virtual user adds
a 100–500ms think time between requests to simulate realistic user behaviour.

### 5.3 Running the Load Test

```bash
# Standard 50-user, 60-second test
python scripts/load_test.py \
    --url http://localhost:8000 \
    --schema-id ins_prod_v3 \
    --users 50 \
    --duration 60

# Ramp test (users gradually ramped over the first 30% of duration)
python scripts/load_test.py \
    --url http://localhost:8000 \
    --schema-id ins_prod_v3 \
    --users 50 \
    --duration 120 \
    --ramp

# Smoke test (10 users, 10 seconds)
python scripts/load_test.py \
    --url http://localhost:8000 \
    --schema-id ins_prod_v3 \
    --users 10 \
    --duration 10
```

The script exits with code 0 if the SLA is met, code 1 if not — suitable for
CI pipeline integration.

### 5.4 Bottleneck Analysis

At 50 concurrent users, the dominant latency contributor is the LLM API call
(GPT-4o, ~2–4s per query). The retrieval and execution layers add <200ms each.

Mitigation strategies in priority order:

1. **Result caching** — identical `(schema_id, normalised_nl_query)` pairs
   return cached results within a 15-minute TTL. Implemented in Phase 3.
2. **Long-term memory exact-hit path** — similarity ≥ 0.97 bypasses LLM
   entirely. See Section 3.7.
3. **Async LLM calls** — `AsyncOpenAI` is used throughout; LLM calls do not
   block the event loop.
4. **Streaming responses** — `stream=True` on the LLM call + SSE to the client
   reduces perceived latency. Not yet implemented; Phase 4 extension.
5. **Local model fallback** — SQLCoder-7B (GGUF quantised) can handle schema-
   grounded SQL generation at ~1–2s on an RTX 3050. Swap in for cost reduction
   when correctness delta is acceptable.

---

## 6. Observability Alerts (observability.py)

Four rolling-window alerts evaluate over the last 100 turns in memory.
In production, pipe these to PagerDuty or a Slack webhook.

| Alert | Threshold | Severity |
|---|---|---|
| `ALERT_EXECUTABLE_RATE` | Rolling executable rate < 70% | Page |
| `ALERT_LATENCY_P99` | Any single turn > 8,000 ms | Page |
| `ALERT_RETRY_RATE` | Rolling retry rate > 15% | Slack warning |
| `ALERT_TERMINAL_ERRORS` | > 10 TERMINAL_ERROR events in last 100 turns | Page |

Alerts are emitted to `stderr` as structured JSON. Example:

```json
{
  "event": "ALERT_EXECUTABLE_RATE",
  "message": "Rolling executable rate 62.0% is below 70% threshold (last 100 turns).",
  "timestamp": "2024-11-01T14:23:01.123Z"
}
```

---

## 7. File Roles — Complete Reference

| File | Phase | Role |
|---|---|---|
| `analyst_copilot/dataframe_store.py` | 2 | User-uploaded file registry (CSV/Parquet/Excel). Per-session, TTL-based. |
| `analyst_copilot/dataframe_loader.py` | 2 | DB-sourced table loader. Per-schema cache, 5-min TTL. |
| `analyst_copilot/long_term_memory.py` | 3 | Cross-session FAISS+SQLite vector memory. PII guard, dedup, staleness rebuild. |
| `analyst_copilot/eval.py` | 3 | Evaluation pipeline. 5 metrics, Wilson CI, ground-truth correctness check. |
| `scripts/load_test.py` | 3 | Concurrent load test. p99 SLA validation. CI-friendly exit code. |
| `scripts/synthetic_data.py` | 3 | Synthetic insurance dataset generator for test fixtures. |
| `scripts/generate_insurance_data.py` | 3 | Larger-scale insurance data generator for ingestion testing. |
| `docs/pass1_design.md` | 1 | Phase 1 architecture decisions. |
| `docs/pass2_design.md` | 2–3 | This document. |

---

## 8. Remaining Phase 3 Work

Three items from the roadmap are not yet implemented:

| Item | Priority | Description |
|---|---|---|
| `docs/pass2_design.md` populated | Done (this file) | — |
| `data/lt_memory/` directory created | Action required | `mkdir data\lt_memory` |
| Orchestrator patches applied | Action required | Apply all 7 patches from `ORCHESTRATOR_PATCH.md` |
| `POST /upload` endpoint | Phase 4 | Allows users to upload CSV/Parquet files via the API. `DataFrameStore.ingest()` is ready; only the FastAPI route is missing. |
| Streaming responses (SSE) | Phase 4 | Reduces perceived latency; requires client-side changes. |
| SQLCoder-7B local fallback | Phase 4 | Cost reduction. GGUF quantised model, ~1.3 GB, runs on RTX 3050. |