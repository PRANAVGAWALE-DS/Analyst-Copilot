"""
scripts/load_test.py — Concurrent load test
Data Analyst Copilot · Phase 3 · Python 3.11+

Validates the p99 ≤ 5,000ms target from Section 1 at 50 concurrent users.

Test scenarios (randomised per request):
  1. Simple aggregation query
  2. Filter query
  3. Multi-table join query
  4. Time-series grouping
  5. Ranking query

Usage:
    # Full 50-user test (runs for 60 seconds)
    python scripts/load_test.py --url http://localhost:8000 --users 50 --duration 60

    # Quick 10-user smoke test (10 seconds)
    python scripts/load_test.py --url http://localhost:8000 --users 10 --duration 10 --schema-id ins_dev

    # Ramp test: 10 → 50 users over 2 minutes
    python scripts/load_test.py --url http://localhost:8000 --ramp --users 50 --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

# ---------------------------------------------------------------------------
# Test queries — realistic NL questions against the insurance schema
# ---------------------------------------------------------------------------

_TEST_QUERIES = [
    "What was the average claim amount by policy type last quarter?",
    "Show the total premium collected per state in 2023",
    "How many open claims are there per agent?",
    "Which customers have more than 3 active policies?",
    "What is the monthly claims count for the past 12 months?",
    "List the top 10 agents by total premiums written this year",
    "What percentage of claims were denied by policy type?",
    "Show claims where amount is greater than 50000",
    "What is the average credit score of customers by state?",
    "How many policies are expiring in the next 90 days?",
    "What is the total payout by claim status?",
    "Which policy types have the highest average claim amount?",
    "Show the customer lifetime value distribution by state",
    "How many claims were settled within 30 days of filing?",
    "What is the commission paid per agent by region?",
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    query: str
    status_code: int
    latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class LoadTestReport:
    total_requests: int
    successful: int
    failed: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    mean_ms: float
    rps: float  # requests per second
    p99_target_met: bool
    target_p99_ms: int = 5_000
    results: list[RequestResult] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.p99_target_met else "❌ FAIL"
        lines = [
            "── Load Test Report ─────────────────────────────────────",
            f"  Requests:     {self.total_requests} total, {self.successful} ok, {self.failed} failed",
            f"  Throughput:   {self.rps:.1f} req/s",
            "  Latency:",
            f"    min:        {self.min_ms:.0f} ms",
            f"    mean:       {self.mean_ms:.0f} ms",
            f"    p50:        {self.p50_ms:.0f} ms",
            f"    p95:        {self.p95_ms:.0f} ms",
            f"    p99:        {self.p99_ms:.0f} ms  ← target ≤ {self.target_p99_ms} ms",
            f"    max:        {self.max_ms:.0f} ms",
            f"  p99 target:   {status}",
            "─────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load test runner
# ---------------------------------------------------------------------------


async def run_load_test(
    base_url: str = "http://localhost:8000",
    schema_id: str = "insurance_v1",
    n_users: int = 50,
    duration_seconds: int = 60,
    target_p99_ms: int = 5_000,
    ramp: bool = False,
) -> LoadTestReport:
    """
    Run a concurrent load test for `duration_seconds`.

    Parameters
    ----------
    base_url        : Server base URL.
    schema_id       : Schema ID to use in every request.
    n_users         : Number of concurrent virtual users.
    duration_seconds: How long to run the test.
    target_p99_ms   : SLA target for p99 latency (ms).
    ramp            : If True, ramp from 1 to n_users over the first 30% of duration.
    """
    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(n_users)
    stop_event = asyncio.Event()
    t_start = time.monotonic()

    async def worker(client: httpx.AsyncClient, user_id: int) -> None:
        # M8 FIX: ramp delay moved OUTSIDE the while loop.
        # Previously `await asyncio.sleep(ramp_delay)` executed before EVERY
        # request, so worker 49 of 50 slept 18 s between each query in a 60 s
        # test — effectively eliminating the last 20 % of workers from the
        # load profile and producing misleadingly low p99 results.
        # The intent is staggered startup only; one sleep per worker is correct.
        if ramp:
            ramp_delay = (user_id / n_users) * (duration_seconds * 0.3)
            await asyncio.sleep(ramp_delay)

        while not stop_event.is_set():
            query = random.choice(_TEST_QUERIES)
            result = await _send_query(client, semaphore, schema_id, query)
            results.append(result)

            # Brief pause between requests (simulate real user think time)
            await asyncio.sleep(random.uniform(0.1, 0.5))

    async def stopper() -> None:
        await asyncio.sleep(duration_seconds)
        stop_event.set()

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        user_tasks = [worker(client, i) for i in range(n_users)]
        await asyncio.gather(*user_tasks, stopper(), return_exceptions=True)

    elapsed = time.monotonic() - t_start
    return _compute_report(results, elapsed, target_p99_ms)


async def _send_query(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    schema_id: str,
    query: str,
) -> RequestResult:
    async with semaphore:
        t0 = time.monotonic()
        try:
            resp = await client.post(
                "/query",
                json={
                    "nl_query": query,
                    "schema_id": schema_id,
                    "execution_mode": "auto",
                    "dry_run": False,
                },
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            success = resp.status_code == 200 and not resp.json().get("error")
            return RequestResult(
                query=query[:50],
                status_code=resp.status_code,
                latency_ms=latency_ms,
                success=success,
                error=None if success else str(resp.json().get("error")),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.monotonic() - t0) * 1000)
            return RequestResult(
                query=query[:50],
                status_code=0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )


def _compute_report(
    results: list[RequestResult],
    elapsed: float,
    target_p99_ms: int,
) -> LoadTestReport:
    if not results:
        return LoadTestReport(
            total_requests=0,
            successful=0,
            failed=0,
            p50_ms=0,
            p95_ms=0,
            p99_ms=0,
            max_ms=0,
            min_ms=0,
            mean_ms=0,
            rps=0.0,
            p99_target_met=False,
            target_p99_ms=target_p99_ms,
        )

    latencies = sorted(r.latency_ms for r in results)
    n = len(latencies)
    successful = sum(1 for r in results if r.success)

    p99 = _percentile(latencies, 99)

    return LoadTestReport(
        total_requests=n,
        successful=successful,
        failed=n - successful,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=p99,
        max_ms=latencies[-1],
        min_ms=latencies[0],
        mean_ms=statistics.mean(latencies),
        rps=n / elapsed if elapsed > 0 else 0.0,
        p99_target_met=p99 <= target_p99_ms,
        target_p99_ms=target_p99_ms,
        results=results,
    )


def _percentile(sorted_data: list[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    f, c = int(k), math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Analyst Copilot — load test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--schema-id", default="insurance_v1")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument(
        "--target-p99",
        type=int,
        default=5_000,
        help="p99 SLA target in milliseconds (default: 5000)",
    )
    parser.add_argument(
        "--ramp",
        action="store_true",
        help="Ramp users gradually instead of all at once",
    )
    parser.add_argument("--output", default=None, help="Save full results to JSON")
    args = parser.parse_args()

    print(
        f"Starting load test: {args.users} users × {args.duration}s "
        f"→ {args.url} (schema: {args.schema_id})"
    )

    report = asyncio.run(
        run_load_test(
            base_url=args.url,
            schema_id=args.schema_id,
            n_users=args.users,
            duration_seconds=args.duration,
            target_p99_ms=args.target_p99,
            ramp=args.ramp,
        )
    )

    print(report.summary())

    if args.output:
        import json
        from dataclasses import asdict

        with open(args.output, "w") as f:
            data = asdict(report)
            # Convert results to summary-only to keep file size reasonable
            data["results"] = [
                {
                    "query": r.query,
                    "latency_ms": r.latency_ms,
                    "success": r.success,
                    "status_code": r.status_code,
                }
                for r in report.results
            ]
            json.dump(data, f, indent=2)
        print(f"Full results saved → {args.output}")

    # Exit with non-zero code if SLA not met (useful in CI)
    import sys

    sys.exit(0 if report.p99_target_met else 1)
