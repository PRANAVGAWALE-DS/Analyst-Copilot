# Data Analyst Copilot — Pass 1 Design Document
> Sections 1 · 2 · 6 · 11

---

## Section 1 — Problem Framing

### Why Existing Tools Fall Short

Business intelligence platforms like Tableau and Power BI solve a specific, bounded problem well: visualising pre-modelled data through drag-and-drop interactions. They break down along three axes that matter for this system.

**Ad-hoc flexibility.** BI tools require a semantic layer — a manually curated set of dimensions, measures, and hierarchies — to exist before any question can be answered. When an analyst asks a net-new question that crosses two tables that were never joined in the semantic model, the answer is "build a new dashboard" (days) not "here is the result" (seconds). The combinatorial space of analytical questions grows faster than any modelling team can pre-anticipate.

**Schema coverage.** Both platforms impose a modelling contract: a developer must explicitly register every table and relationship. In a 200-table schema, perhaps 30 tables get modelled for the dashboards that exist today. The other 170 are invisible to analysts — not because the data is missing, but because the tool has no mechanism for dynamic, query-time schema discovery.

**Multi-turn reasoning.** BI tools are stateless per query. When an analyst says "now break that down by region, but exclude the refund rows we discussed," the tool has no referent for "we discussed" or "that." Each interaction starts from scratch. Complex analytical workflows that take a human data analyst three back-and-forth clarifications to nail cannot be expressed in a single drag-and-drop interaction.

### Why a Pure LLM Without Grounding Fails

A raw LLM given only a natural language question and no schema context fails in at least three concrete, production-critical ways.

1. **Hallucinated column names.** LLMs trained on code and SQL will confidently generate `SELECT churn_rate FROM customers` even if `churn_rate` is not a stored column. The query appears syntactically valid, fails at execution time, and produces a confusing error rather than an actionable answer.

2. **Stale schema knowledge.** Even if a model was fine-tuned on a snapshot of a company's schema, that snapshot ages. Tables are renamed, columns are added, data types change. Without live schema grounding, every inference is against a schema that may no longer match production reality.

3. **No execution feedback loop.** A pure LLM produces code. Without an execution sandbox that runs the code and returns actual errors, there is no mechanism to distinguish a query that will work from one that will fail. The model cannot self-correct because it never observes the runtime outcome. Retry with error context — the core of this system's resilience — is architecturally impossible without tool use.

### Why RAG + Tool Use + Memory + Agent Orchestration Is Correct

Each architectural primitive solves a distinct failure mode: RAG solves schema staleness by retrieving live metadata at query time instead of relying on training-time knowledge. Tool use solves the feedback gap by giving the LLM a real execution environment whose outputs become inputs for the next reasoning step. Memory solves the statelesness problem by threading entity and result context across turns so that coreferences resolve correctly. Agent orchestration ties these together into a state machine that can detect failure (validation error, empty retrieval, runtime crash), route to the appropriate recovery path, and cap retry depth to prevent infinite loops.

No subset of these four primitives is sufficient. RAG without execution produces unvalidated code. Execution without RAG produces hallucinated column references. Memory without orchestration has no principled trigger for when to consult it. The combination is not an ensemble heuristic — it is the minimum architecture that closes every identified failure loop while meeting the p99 ≤5s latency constraint.

---

## Section 2 — End-to-End System Architecture

### Textual Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  CLIENT LAYER           FastAPI  /query  /execute  /history     │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP
┌────────────────────────────────▼────────────────────────────────┐
│  ORCHESTRATION LAYER    Agent State Machine (Section 2H)        │
│   INTAKE → RETRIEVAL → GENERATION → VALIDATION →               │
│   EXECUTION → RESULT_CHECK → INSIGHT → TERMINAL                │
└──┬───────────┬──────────────┬──────────────┬────────────────────┘
   │           │              │              │
┌──▼──┐  ┌────▼────┐  ┌──────▼──────┐  ┌───▼──────────────────┐
│MEM  │  │RETRIEVAL│  │  LLM LAYER  │  │  TOOLING LAYER       │
│SYS  │  │  LAYER  │  │ GPT-4o via  │  │  SQL Executor        │
│     │  │ FAISS   │  │ func calling│  │  Pandas Executor     │
│STM  │  │ + meta  │  │             │  │  Pre-exec policy     │
│LTM  │  │ filter  │  │             │  │  Sandbox / AST guard │
└──┬──┘  └────┬────┘  └─────────────┘  └──────────────────────┘
   │          │
┌──▼──────────▼──────────────────────────────────────────────────┐
│  KNOWLEDGE LAYER        Schema Store + Embedding Index          │
│  FAISS HNSW · text-embedding-3-small · Hybrid chunks           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  INGESTION LAYER        SQLAlchemy introspection + profiler      │
│  Event-driven trigger · PII flag at ingest · Hash samples       │
└─────────────────────────────────────────────────────────────────┘
```

---

### A. Ingestion Layer

**Core design problem:** Schema metadata must be extracted, profiled, and stored in a form that supports both exact lookup (column existence checks) and semantic retrieval (embedding-based similarity). The extraction must be privacy-aware by default, not by convention.

**Alternative 1 — Manual DDL parsers:** Developers write schema-specific extractors that parse CREATE TABLE statements. Accurate for the tables they cover; fails silently on tables that were never registered.

**Alternative 2 — SQLAlchemy `inspect()` + automated profiler:** `inspect(engine)` returns tables, columns, PKs, FKs, and types at runtime for every accessible table. A follow-on profiling pass computes null rates, cardinality, and sample values. Triggered by schema change events (ALTER TABLE via CDC or scheduled diff against stored snapshot).

**Chosen approach: Alternative 2 with event-driven trigger.** Runtime inspection guarantees 100% table coverage with no manual registration. The event-driven trigger (schema diff on a configurable schedule, defaulting to hourly) keeps metadata fresh without continuous polling overhead. Privacy contract: all sample values pass through a `pii_mask()` step before storage — columns flagged `pii=true` at the column level have their samples SHA-256-hashed; the flag propagates to the pre-execution policy layer.

**Ingestion trigger model:** A schema version hash is stored per `schema_id`. On each scheduled run, the current hash is recomputed. If it differs, ingestion re-runs for the changed schema only (incremental, not full-rebuild). Full rebuild is triggered manually or on first ingest.

---

### B. Knowledge Representation

**Core design problem:** Schema knowledge must be chunked so that individual chunks are semantically coherent (one concept per chunk) and fit within the context budget per retrieved result (~400 tokens each for K=5 at ≤2,000 token schema context).

**Alternative 1 — Table-level chunks:** One chunk per table containing all columns. Simple; a 50-column table produces a chunk that is too large and dilutes retrieval signal.

**Alternative 2 — Column-level chunks:** One chunk per column. Maximum semantic precision; at 50 columns per table, produces 10,000 chunks for 200 tables × 1 schema — retrieval recall degrades because a single query needs columns from multiple tables but retrieves individual column chunks with no structural context.

**Alternative 3 (chosen) — Hybrid table-level chunks, capped:** One chunk per table, containing: table name, business description (if available), all column names with types, nullable flags, cardinality estimates, top-5 sample values (PII-masked), FK relationships. Hard cap: 400 tokens. If the table exceeds 400 tokens, column descriptions are truncated first; column names and types are never truncated. This preserves the structural coherence of table-level chunks while staying within the token budget.

**Metadata enrichment schema per chunk:**
```json
{
  "table": "claims",
  "schema_id": "ins_prod_v3",
  "business_description": "One row per filed insurance claim.",
  "columns": [
    {
      "name": "claim_amount",
      "type": "DECIMAL(12,2)",
      "nullable": true,
      "null_rate": 0.04,
      "cardinality": "high",
      "sample_values": [1200.00, 450.50, 8900.00]
    }
  ],
  "fk_relationships": [
    {"column": "policy_id", "references": "policies.policy_id"}
  ],
  "row_count_estimate": 1200000,
  "pii_flagged": false,
  "chunk_token_count": 312
}
```

---

### C. Embeddings and Indexing

**Core design problem:** Choose an embedding model and vector index that jointly meet recall@5 ≥ 0.85, retrieval latency < 100ms (within the 5s budget), and are operable without dedicated GPU inference infrastructure.

**Embedding model comparison:**

| | BGE-large-en-v1.5 | text-embedding-3-small |
|---|---|---|
| Recall@5 (MTEB) | ~0.87 | ~0.83 |
| Latency | Requires local GPU or inference server | ~50ms via API |
| Cost/1M tokens | Hosting only (~$0.05/hr GPU) | $0.02 |
| Ops complexity | Deploy + maintain embedding server | API call |

**Chosen: text-embedding-3-small.** The 4-point recall delta is acceptable given the grounding check at VALIDATION catches hallucinated columns that slip through retrieval. API-hosted removes the embedding server from the critical path, simplifying Phase 1 delivery. If recall@5 drops below 0.80 in production evaluation, migrate to BGE-large behind a local inference server.

**FAISS index comparison:**

| | Flat (exact) | IVF (approximate) | HNSW (graph) |
|---|---|---|---|
| Recall | 1.0 | ~0.95 | ~0.98 |
| Query latency | O(N) — slow at scale | Fast, needs training data | Fast, no training |
| Suitability at 500 schemas × 200 tables | 100,000 chunks: ~40ms | Best above 1M chunks | Optimal at 10K–500K |

**Chosen: HNSW.** At the scale target (500 schemas × 200 tables × 1 chunk/table = 100,000 chunks), HNSW delivers ~2–5ms query latency with recall ~0.98 — well above the 0.85 target — without requiring an IVF training set or the linear scan of Flat.

**Update strategy:** On schema change event, re-embed and upsert only the affected table chunks. Full-index rebuild is triggered only on initial ingestion or if cumulative upserts exceed 20% of total index size (HNSW accuracy degrades with large incremental updates).

---

### D. Retrieval Layer

**Core design problem:** Embedding similarity alone is insufficient — a query about `claims` should not retrieve chunks from a different schema's `claims` table, and structured metadata constraints (e.g., date columns only) cannot be expressed as a vector distance.

**Alternative 1 — Pure dense retrieval:** Embed the query, return top-K by cosine similarity. Fast; no way to enforce `schema_id` isolation without post-hoc filtering which silently degrades recall.

**Alternative 2 (chosen) — Hybrid: pre-filter by `schema_id` + dense retrieval:** Apply a FAISS `IDSelector` or metadata pre-filter to restrict the index search to chunks belonging to the requested `schema_id` before computing similarity. Then return top-K from that filtered subset.

**Trade-off:** Pre-filtering reduces the effective index size the similarity search operates over, which can reduce HNSW recall slightly (less graph connectivity). At 200 tables per schema, the filtered subset is 200 vectors — at this size, a Flat scan over the filtered subset is ≤1ms and avoids the HNSW recall penalty entirely. Implementation: pre-filter by `schema_id`, then Flat scan the resulting candidate set.

**Query rewriting:** Triggered when the query contains pronouns or implicit references ("that table", "same as before", "those columns"). Detection condition: presence of a coreference token AND a populated session history. Rewriting prepends the last referenced table/column names to the query before embedding. Decomposition (splitting into sub-queries) is triggered only when the query contains an explicit multi-table operator word ("compare", "join", "across both", "versus") — not speculatively.

**Top-K selection:** K=5. At K=3, recall@5 drops to recall@3 (~0.74 at this scale) — insufficient. At K=10, schema context exceeds 4,000 tokens, consuming the entire token budget before session history is appended. K=5 at ~400 tokens/chunk = 2,000 tokens, leaving 3,000 tokens for session history and prompt overhead.

---

### E. LLM Orchestration Layer

**Core design problem:** How should the LLM reason and act — interleaved or in discrete phases — and how does that choice affect latency, failure modes, and retry behavior?

**Alternative 1 — ReAct (reasoning + acting interleaved):** The LLM emits a `Thought:` then an `Action:` in alternating turns. Flexible; well-suited for exploratory multi-step tasks. Failure mode: under ambiguous queries, the model can generate multi-step reasoning chains that drift from the original question. Each reasoning hop adds ~0.5s LLM latency. Retry behavior: the entire chain is re-run from the failed step.

**Alternative 2 — Planner-executor (two-stage):** A planning LLM call decomposes the query into a sub-query DAG; executor calls run each node. Failure mode: planning step adds 1–2s of latency before any execution begins — p99 budget is at risk. Retry: re-plan from scratch, expensive.

**Alternative 3 (chosen) — Function calling (structured tool dispatch):** The LLM is given a schema for `generate_sql` or `generate_pandas_code` as a registered function and returns a structured JSON object. The orchestrator extracts the code, runs it through the validation layer, executes it, and returns the result. No free-form reasoning output to parse. Failure mode: under highly ambiguous queries, function calling collapses ambiguity into a single interpretation without surfacing it — mitigated by the `confidence` field: if confidence < 0.7, the orchestrator surfaces both interpretations before executing. Retry behavior: on validation failure, the error is injected into the next function call as context — one LLM call per retry, not a full chain re-run. Latency per retry: ~1–2s additional.

**Prompting strategy: JSON mode (constrained decoding).** The generation prompt instructs the model to return only a JSON object in a defined schema. This is enforced with `response_format={"type": "json_object"}` in the OpenAI API call. Post-parse validation (Pydantic) catches any deviation. Free-form with post-parse is rejected: it introduces parse failures as a fourth failure mode on top of the three already identified.

---

### F. Tooling Layer

Defined in full in Section 3 (interface contracts) and Section 5 (validation implementation). Design decisions:

**SQL parser choice — sqlglot over sqlparse:** sqlglot provides an AST with named node types (`sqlglot.exp.Column`, `sqlglot.exp.Insert`) enabling programmatic column extraction and mutation detection. sqlparse produces a flat token stream; column extraction requires fragile regex-style token walking. sqlglot also supports multi-dialect transpilation, which is useful for the dialect parameter in the function calling schema.

**Pandas fallback trigger:** SQL is default. Pandas is invoked only on explicit triggers (file input, SQL failure after 2 retries, multi-step transformation) to avoid the security surface area of `exec()` being expanded unnecessarily.

---

### G. Memory System

**Short-term (within session):**
- Conversation buffer: last N=10 turns stored as `TurnRecord` objects in session state (Redis or in-process dict keyed by `session_id`).
- Execution context: last 3 query results stored as compressed JSON (gzip, max 512KB per result). Used to resolve "use the previous result" type queries.
- Entity tracker: a running `set[str]` of table and column names referenced this session. Before retrieval, the entity set is unioned with the top-K results to avoid re-retrieving already-known tables.

**Long-term memory comparison:**

| | Vector memory | Structured (relational) memory |
|---|---|---|
| Storage | Embedding store | PostgreSQL table |
| Retrieval trigger | Similarity to current query | Exact (schema_id, user_id) lookup |
| Staleness handling | TTL on embeddings; hard to enforce | `cached_at` timestamp + staleness policy |
| PII compliance | Embeddings are not reversible, but source queries may contain PII | SQL-level row-level security |

**Chosen: structured memory.** The retrieval use case is "has this user asked a similar question against this schema before?" — this is a keyed lookup (user_id + schema_id + normalized_query_hash), not an open-ended similarity search. Structured storage enables precise staleness control: results older than 15 minutes are evicted. PII-flagged table results are never written to long-term memory (enforced by a flag check before write). Retrieval trigger: only when the normalized query hash matches an existing entry — not speculatively on every query.

---

### H. Orchestration and Agent Control

State machine and transitions are as defined in the specification. Two implementation notes:

**Attempt counter scoping:** The attempt counter is scoped to the `(session_id, turn_id)` pair, not to the session. Each new user query starts at attempt=0. The counter is stored in the agent state object, not in the database.

**Tool selection policy:** SQL is default. The orchestrator checks three conditions before dispatching: (1) is the source a file-type? (2) has SQL failed twice in this turn? (3) does the query contain multi-step transformation keywords? If any condition is true, Pandas executor is selected. Otherwise SQL.

---

## Section 6 — Failure Modes and Guardrails

| Failure Mode | Detection Point | Mitigation | User-Facing Response |
|---|---|---|---|
| Hallucinated column | VALIDATION — `validate_sql` / `validate_python` column existence check | `sqlglot` column extraction + diff against `schema_columns`; grounding check in generation prompt | "I couldn't find a column matching '[term]'. Did you mean one of: [top-3 fuzzy matches via `difflib.get_close_matches`]?" |
| Invalid SQL syntax | VALIDATION — `sqlglot.parse_one()` raises `ParseError` | Inject `error_type`, `error_message`, `error_line` into ERROR_CORRECT prompt | Spinner "Refining query…"; user never sees raw traceback |
| Invalid Python syntax | VALIDATION — `ast.parse()` raises `SyntaxError` | Same as above | Same as above |
| Empty retrieval (K=0) | RETRIEVAL — `len(chunks) == 0` | Transition to INTAKE; ask clarifying question | "I couldn't find a table matching '[query]'. Can you describe the data you're looking for?" |
| Ambiguous query | GENERATION — `confidence < 0.7` in response JSON | Surface both interpretations before execution; require user confirmation | "I found two possible interpretations: [A] or [B]. Which did you mean?" |
| Prompt injection | INTAKE — pre-retrieval sanitizer | Strip `--`, `/**/`, XML tags, instruction-like patterns (`ignore previous`, `you are now`). User input is injected only into delimited user-turn slots — never into the system prompt string. | "Your query contains characters I can't process. Please rephrase." |
| Infinite retry | ERROR_CORRECT — attempt counter ≥ 3 | Hard cap at 3 tracked in agent state. On attempt 4: transition to TERMINAL_ERROR. | Structured `ErrorDetail` with all 3 attempt histories returned; HTTP 200 with `error` populated. |
| Data exfiltration | Pre-execution policy layer — runs before SQL executor receives query | Block LIMIT-less SELECTs on `pii=true` tables. Check is in the policy layer, not in LLM output validation. `pii` flag set at ingestion time per table. | "This query would return an unbounded result from a protected table. Please add a filter or LIMIT clause." |
| Execution timeout | Execution layer — `concurrent.futures.TimeoutError` | Hard kill at `timeout_seconds`; transition to ERROR_CORRECT with timeout as error context | "The query took too long. Try narrowing the date range or adding a filter." |
| Mutation statement | VALIDATION — `sqlglot` mutation type guard | Reject before execution; no retry (not a transient error) | "I can only run read-only queries. This query contains a write operation." |
| OOM in Pandas executor | Execution layer — memory monitor thread | `tracemalloc` snapshot compared to `memory_limit_mb` every 100ms; `MemoryError` raised and caught | "The operation used more memory than allowed. Try reducing the date range or the number of columns." |

**Guardrail implementation notes:**

The pre-execution PII policy layer is a standalone function called by the SQL executor before any query reaches the database connection. It is not LLM-dependent — it parses the SQL with sqlglot to check for missing LIMIT clauses and cross-references the tables in the query against the `pii_flagged` index. This makes the guardrail bypass-proof regardless of what the LLM generates.

Prompt injection sanitization uses a compiled regex at INTAKE, applied before the user string is interpolated into any template. The sanitizer is not responsible for semantic injection (telling the model to behave differently) — that is handled architecturally by the system/user prompt separation. The sanitizer's scope is structural injection only.

---

## Section 11 — Implementation Roadmap

**Team:** 2 ML engineers. **Stack:** FastAPI + PostgreSQL + FAISS + LangChain (or raw OpenAI SDK). **Cadence:** ~3 focused days/week (~15 engineer-days/month).

### Phase 1 — MVP (Weeks 1–4)

**Goal:** One analyst can ask an NL question against one ingested schema and receive a correct SQL result.

| Week | Deliverables | Owner |
|---|---|---|
| 1 | Schema ingestion pipeline: SQLAlchemy `inspect()`, profiler, hybrid chunking, `schema_id` metadata. Embedding with text-embedding-3-small, FAISS HNSW index. | Eng A |
| 1 | PostgreSQL schema store, `pii_flag` column, ingestion trigger (manual). Unit tests: ingest → retrieve round-trip for 5 tables. | Eng B |
| 2 | NL → SQL generation with grounding prompt (no retry). Function calling schema wired to GPT-4o. `validate_sql()` with sqlglot. | Eng A |
| 2 | SQL executor: read-only SQLAlchemy connection, LIMIT cap, mutation guard, timeout. `validate_result()` runtime check. | Eng B |
| 3 | `/query` and `/execute` FastAPI endpoints (no session support). Structured JSON logging per state transition. | Eng A |
| 3 | Integration: retrieval → generation → validation → execution → insight full path. End-to-end smoke test against TPC-H. | Both |
| 4 | Run NL → SQL grounding prompt against 20 hand-crafted test queries (pre-agreed semantic templates). Fix any systematic failures before Phase 2 begins. | Both |

**Highest-risk dependency:** LLM SQL generation quality on the target schema. The Week 4 20-query smoke test is a gate — do not start Phase 2 until executable rate ≥ 70% on this set.

**Phase 1 exit criteria:** `/query` returns a correct SQL result for ≥70% of 20 manually-written test queries against a single ingested PostgreSQL schema. p99 latency < 8s (relaxed from 5s — no async optimizations yet).

---

### Phase 2 — Functional (Weeks 5–9)

**Goal:** Error recovery, multi-turn sessions, Pandas fallback.

| Week | Deliverables |
|---|---|
| 5 | Error correction loop (3-attempt retry, ERROR_CORRECT prompt, attempt counter). |
| 6 | `ast.parse()` + AST visitor validation for Python. Pandas executor with restricted `exec()` namespace. |
| 7 | Session management: `session_id` generation, Redis-backed conversation buffer + entity tracker. `/history` endpoint. |
| 8 | Short-term execution context (last 3 results). Long-term memory: structured table keyed by `(user_id, schema_id, query_hash)`, 15-minute TTL. |
| 9 | Runtime result validation (RESULT_CHECK state). Full state machine wiring. Integration test: 50-query set with injected failures. |

**Highest-risk dependency:** Pandas sandbox security. Before any external user accesses the Pandas executor, the allowed-builtins whitelist and AST visitor must pass a dedicated security review. Do not skip this gate.

**Phase 2 exit criteria:** Error recovery rate ≥ 40% on queries that fail on first attempt. Executable rate ≥ 80% overall. p99 latency < 6s.

---

### Phase 3 — Production (Weeks 10–15)

**Goal:** Observable, scalable, adversarially resilient.

| Week | Deliverables |
|---|---|
| 10 | Pre-execution PII policy layer (LIMIT-less SELECT blocker). Prompt injection sanitizer at INTAKE. |
| 11 | Full structured trace logging per state transition. Elasticsearch / CloudWatch ingest. Monitoring alerts (PagerDuty thresholds). |
| 12 | Evaluation pipeline (Section 8): 200-pair test set, automated correctness runner, CI integration. |
| 13 | Async LLM calls (`AsyncOpenAI`). Concurrent retrieval + session load before LLM dispatch. Query result cache (15-min TTL, non-PII only). |
| 14 | Load test: 50 concurrent users, p99 ≤5s target. Tune K, token budget, and worker count to hit target. |
| 15 | Security review sign-off. Runbook. Phase 3 exit criteria: p99 ≤5s @ 50 concurrent, executable rate ≥ 80%, correctness ≥ 70% on held-out test set. |

**Highest-risk dependency:** p99 latency at 50 concurrent users. The LLM call (GPT-4o, ~2–3s) dominates. Primary mitigations in order of impact: (1) async dispatch — retrieval and session load run concurrently, not serially; (2) query result cache eliminates LLM call for repeated queries; (3) streaming responses reduce perceived latency by showing partial results. If these are insufficient, introduce SQLCoder-7B (Section 9) as a local fallback for simple queries.

---
*Pass 1 complete. Sections 3 and 5 are delivered as standalone Python modules: `interfaces.py` and `validation.py`.*