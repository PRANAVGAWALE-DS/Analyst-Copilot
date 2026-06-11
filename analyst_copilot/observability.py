"""
observability.py — Structured Observability Stack
Data Analyst Copilot · Python 3.11+ · Section 10

Exports consumed by orchestrator.py (line 49):
  from observability import AgentState, ObservabilityStack, TraceLogger

Interface contracts (derived from every call-site in orchestrator.py):

  AgentState
    Type alias for the set of valid state name strings.

  TraceStore
    .start_trace(session_id, turn_id, *, nl_query_length)
    Passed as first arg to TraceLogger(trace_store, ...)

  TraceLogger(trace_store, session_id, turn_id, *, state, attempt=0, model="")
    Context manager. Inside the `with` block:
      tl.set_input(dict)                 — log state inputs
      tl.set_output(dict)                — log state outputs
      tl.set_tokens(prompt_t, complete_t)— log LLM token usage

  ObservabilityStack
    .trace_store                         — TraceStore instance
    .record_turn_outcome(
        turn_id, terminal_state, latency_ms,
        executable, hit_max_retries
    )

Log schema per event (JSON, one object per line):
  {
    "timestamp": "ISO-8601",
    "session_id": "...",
    "turn_id": "...",
    "state": "GENERATION",
    "attempt": 0,
    "model": "llama-3.3-70b-versatile",
    "duration_ms": 412,
    "input": {...},
    "output": {...},
    "token_usage": {"prompt": 1234, "completion": 87}
  }
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any, Literal

# ---------------------------------------------------------------------------
# AgentState — string literal type mirroring the state machine
# ---------------------------------------------------------------------------

AgentState = Literal[
    "INTAKE",
    "RETRIEVAL",
    "GENERATION",
    "VALIDATION",
    "EXECUTION",
    "RESULT_CHECK",
    "INSIGHT",
    "ERROR_CORRECT",
    "TERMINAL",
    "TERMINAL_ERROR",
]

# ---------------------------------------------------------------------------
# Logging setup — structured JSON to stdout
# ---------------------------------------------------------------------------


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_state_log = _build_logger("copilot.state")
_outcome_log = _build_logger("copilot.outcome")


def _emit(logger: logging.Logger, record: dict[str, Any]) -> None:
    """
    Emit one structured JSON log line.

    Uses print() to sys.stdout / sys.stderr directly rather than going through
    the logging handler, so that pytest's capsys fixture can capture output
    in tests (logging StreamHandlers hold a reference to the original stream
    at import time and bypass capsys patching).
    """
    record["timestamp"] = datetime.now(tz=UTC).isoformat()
    try:
        line = json.dumps(record, default=str)
        target = (
            sys.stderr
            if logger is _outcome_log and record.get("event", "").startswith("ALERT")
            else sys.stdout
        )
        print(line, file=target, flush=True)
    except Exception:  # noqa: BLE001
        pass  # observability must never crash the application


# ---------------------------------------------------------------------------
# TraceStore — per-turn trace lifecycle
# ---------------------------------------------------------------------------


class TraceStore:
    """
    Manages the lifecycle of a single query turn's trace.

    start_trace() is called once per turn (INTAKE entry).
    TraceLogger instances emit individual state-transition events to the
    structured logger, keyed by session_id + turn_id.

    In production: swap the structured-log backend for an Elasticsearch
    or OpenTelemetry exporter without changing the TraceLogger interface.
    """

    def start_trace(
        self,
        session_id: str,
        turn_id: str,
        *,
        nl_query_length: int,
    ) -> None:
        """Emit a TURN_START event."""
        _emit(
            _state_log,
            {
                "event": "TURN_START",
                "session_id": session_id,
                "turn_id": turn_id,
                "nl_query_length": nl_query_length,
            },
        )


# ---------------------------------------------------------------------------
# TraceLogger — context manager for one state transition
# ---------------------------------------------------------------------------


class TraceLogger:
    """
    Context manager that times and logs a single agent state transition.

    Usage (exactly as in orchestrator.py):

        with TraceLogger(
            self._obs.trace_store,
            state.session_id,
            state.turn_id,
            state="GENERATION",
            attempt=attempt,
            model=self._model,
        ) as tl:
            tl.set_input({"code_type": "sql", "prompt_tokens": 1234})
            # ... do work ...
            tl.set_output({"sql_length": 312, "confidence": 0.91})
            tl.set_tokens(gen_resp.prompt_tokens, gen_resp.completion_tokens)

    Emits one JSON log event on __exit__ containing the full trace record.
    Exceptions inside the block are not suppressed — they propagate normally.
    """

    def __init__(
        self,
        trace_store: TraceStore,
        session_id: str,
        turn_id: str,
        *,
        state: str,
        attempt: int = 0,
        model: str = "",
    ) -> None:
        self._trace_store = trace_store  # kept for potential future use
        self._session_id = session_id
        self._turn_id = turn_id
        self._state = state
        self._attempt = attempt
        self._model = model

        self._input: dict[str, Any] = {}
        self._output: dict[str, Any] = {}
        self._token_usage: dict[str, int] = {}
        self._start: float = 0.0

    def __enter__(self) -> TraceLogger:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = int((time.perf_counter() - self._start) * 1000)
        record: dict[str, Any] = {
            "event": "STATE_TRANSITION",
            "session_id": self._session_id,
            "turn_id": self._turn_id,
            "state": self._state,
            "attempt": self._attempt,
            "model": self._model,
            "duration_ms": duration_ms,
            "input": self._input,
            "output": self._output,
        }
        if self._token_usage:
            record["token_usage"] = self._token_usage
        if exc_type is not None:
            record["exception"] = str(exc_val)
        _emit(_state_log, record)
        # Never suppress exceptions

    # -- setters called inside the `with` block --

    def set_input(self, data: dict[str, Any]) -> None:
        """Record state input metadata."""
        self._input.update(data)

    def set_output(self, data: dict[str, Any]) -> None:
        """Record state output metadata."""
        self._output.update(data)

    def set_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM token usage for this state."""
        self._token_usage = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
        }


# ---------------------------------------------------------------------------
# ObservabilityStack — top-level container
# ---------------------------------------------------------------------------


class ObservabilityStack:
    """
    Top-level observability container injected into the Orchestrator.

    Attributes
    ----------
    trace_store : TraceStore
        Passed to TraceLogger constructors throughout the orchestrator.

    Methods
    -------
    record_turn_outcome(...)
        Called once per turn at the very end of Orchestrator.run() to emit
        a TURN_OUTCOME event for roll-up monitoring.

    Monitoring thresholds (Section 10 alerts):
        - executable_rate (last 100 turns) < 70%   → emit ALERT_EXECUTABLE_RATE
        - turn latency > 8 000 ms (per-turn)        → emit ALERT_LATENCY_SPIKE
        - retry_rate (last 100 turns) > 15%         → emit ALERT_RETRY_RATE
        - TERMINAL_ERROR count > 10 in last 100     → emit ALERT_TERMINAL_ERRORS

    NOTE: thresholds are evaluated over a fixed sliding window of the last
    _WINDOW_SIZE (100) turns, NOT a time-based 5-minute window.  On busy
    systems 100 turns may span <1 min; on quiet ones >1 hour.  Replace
    self._recent_outcomes with a time-bucketed structure (e.g. a deque of
    (timestamp, outcome) tuples with TTL eviction) for true rate-per-minute
    semantics in Phase 3.

    In production: pipe ALERT_* events to PagerDuty / Slack webhook.
    This implementation logs them to stderr so they are visible immediately
    without requiring external tooling to be configured first.
    """

    # Rolling window for basic in-process metric accumulation.
    # Replace with a proper metrics backend (Prometheus, Datadog) in Phase 3.
    _WINDOW_SIZE = 100  # last N turns kept in memory

    def __init__(self) -> None:
        self.trace_store = TraceStore()
        # M-17 FIX: use deque(maxlen=_WINDOW_SIZE) instead of a plain list.
        # list.pop(0) shifts all N elements left on every record_turn_outcome
        # call — O(N) per call.  deque with maxlen handles eviction implicitly
        # in O(1) at both ends with no manual length check or pop() needed.
        self._recent_outcomes: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=self._WINDOW_SIZE
        )
        # Tracks which alerts are currently firing so we only emit on
        # condition transitions (False→True), not on every turn.
        self._alert_active: dict[str, bool] = {}

    def record_turn_outcome(
        self,
        *,
        turn_id: str,
        terminal_state: str,
        latency_ms: int,
        executable: bool,
        hit_max_retries: bool,
        schema_id: str = "",
        code_type: str = "",
    ) -> None:
        """
        Emit a TURN_OUTCOME event and evaluate rolling alert thresholds.

        Parameters
        ----------
        turn_id        : Turn identifier (for log correlation).
        terminal_state : One of TERMINAL / TERMINAL_ERROR / INTAKE (clarification).
        latency_ms     : Total wall-clock time for the turn.
        executable     : True if execution succeeded (code ran without error).
        hit_max_retries: True if attempt_count reached MAX_ATTEMPTS - 1.
        schema_id      : Schema used for this turn — enables per-schema roll-up.
        code_type      : "sql" or "pandas" — enables per-executor roll-up.
        """
        outcome: dict[str, Any] = {
            "event": "TURN_OUTCOME",
            "turn_id": turn_id,
            "terminal_state": terminal_state,
            "latency_ms": latency_ms,
            "executable": executable,
            "hit_max_retries": hit_max_retries,
            "schema_id": schema_id,
            "code_type": code_type,
        }
        _emit(_outcome_log, outcome)

        # Rolling window update — deque(maxlen) evicts oldest entry automatically.
        self._recent_outcomes.append(outcome)

        self._evaluate_alerts(latency_ms, executable, hit_max_retries, terminal_state)

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------

    def _evaluate_alerts(
        self,
        latency_ms: int,
        executable: bool,
        hit_max_retries: bool,
        terminal_state: str,
    ) -> None:
        """
        Evaluate rolling alert thresholds over the in-memory window.
        Emits a structured ALERT_* event only when a condition transitions
        from inactive to active (False → True), preventing per-turn log spam.
        """
        n = len(self._recent_outcomes)
        if n < 10:
            return  # not enough data yet

        exec_rate = sum(1 for o in self._recent_outcomes if o["executable"]) / n
        retry_rate = sum(1 for o in self._recent_outcomes if o["hit_max_retries"]) / n
        terminal_errors = sum(
            1 for o in self._recent_outcomes if o["terminal_state"] == "TERMINAL_ERROR"
        )

        self._fire_alert(
            "ALERT_EXECUTABLE_RATE",
            f"Rolling executable rate {exec_rate:.1%} is below 70% threshold " f"(last {n} turns).",
            condition=exec_rate < 0.70,
        )
        self._fire_alert(
            # Renamed from ALERT_LATENCY_P99: this alert fires on any single turn
            # exceeding 8,000 ms (instantaneous spike), not a rolling p99.
            # Use ALERT_LATENCY_SPIKE to accurately reflect the semantics.
            "ALERT_LATENCY_SPIKE",
            f"Turn latency {latency_ms}ms exceeds 8,000ms threshold.",
            condition=latency_ms > 8_000,
        )
        self._fire_alert(
            "ALERT_RETRY_RATE",
            f"Rolling retry rate {retry_rate:.1%} exceeds 15% threshold " f"(last {n} turns).",
            condition=retry_rate > 0.15,
        )
        self._fire_alert(
            "ALERT_TERMINAL_ERRORS",
            f"{terminal_errors} TERMINAL_ERROR events in the last {n} turns.",
            condition=terminal_errors > 10,
        )

    def _fire_alert(self, alert_type: str, message: str, *, condition: bool) -> None:
        """Emit alert on False→True transition; emit resolved on True→False."""
        was_active = self._alert_active.get(alert_type, False)
        self._alert_active[alert_type] = condition
        if condition and not was_active:
            self._alert(alert_type, message)
        elif not condition and was_active:
            # L2 FIX: emit a resolved event so operators know when the
            # condition cleared.  Without this, alerts fired but never had a
            # corresponding "all-clear" signal in the log stream.
            self._alert(
                f"{alert_type}_RESOLVED",
                f"Condition cleared: {alert_type} is no longer firing.",
            )

    @staticmethod
    def _alert(alert_type: str, message: str) -> None:
        record: dict[str, Any] = {
            "event": alert_type,
            "message": message,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        # Emit to stderr so alerts are distinguishable from normal trace logs
        with contextlib.suppress(Exception):  # noqa: BLE001
            print(json.dumps(record, default=str), file=sys.stderr, flush=True)
