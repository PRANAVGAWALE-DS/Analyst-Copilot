"""
tests/test_observability.py
Unit tests for observability.py.

Covers:
  - TraceStore.start_trace() emits without error
  - TraceLogger context manager: set_input, set_output, set_tokens
  - TraceLogger emits on __exit__ (including exception path — no suppression)
  - ObservabilityStack.record_turn_outcome() emits TURN_OUTCOME
  - Alert threshold: executable_rate < 70% triggers ALERT_EXECUTABLE_RATE
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import pytest
from observability import ObservabilityStack, TraceLogger, TraceStore


class TestTraceStore:
    def test_start_trace_does_not_raise(self, capsys: pytest.CaptureFixture) -> None:
        store = TraceStore()
        store.start_trace("sess_001", "turn_001", nl_query_length=42)
        captured = capsys.readouterr()
        record = json.loads(captured.out.strip())
        assert record["event"] == "TURN_START"
        assert record["session_id"] == "sess_001"
        assert record["turn_id"] == "turn_001"
        assert record["nl_query_length"] == 42


class TestTraceLogger:
    def test_context_manager_emits_on_exit(self, capsys: pytest.CaptureFixture) -> None:
        store = TraceStore()
        with TraceLogger(store, "sess_1", "turn_1", state="GENERATION", attempt=0) as tl:
            tl.set_input({"code_type": "sql"})
            tl.set_output({"sql_length": 120})
            tl.set_tokens(800, 200)

        captured = capsys.readouterr()
        # TraceStore.start_trace not called here — only TraceLogger exits
        lines = [ln for ln in captured.out.strip().splitlines() if ln]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "STATE_TRANSITION"
        assert record["state"] == "GENERATION"
        assert record["input"]["code_type"] == "sql"
        assert record["output"]["sql_length"] == 120
        assert record["token_usage"]["prompt"] == 800
        assert record["token_usage"]["completion"] == 200
        assert record["duration_ms"] >= 0

    def test_exception_is_not_suppressed(self, capsys: pytest.CaptureFixture) -> None:
        store = TraceStore()
        with (
            pytest.raises(ValueError, match="intentional"),
            TraceLogger(store, "sess_2", "turn_2", state="EXECUTION") as tl,
        ):
            tl.set_input({"step": "exec"})
            raise ValueError("intentional error")

    def test_exception_is_recorded_in_log(self, capsys: pytest.CaptureFixture) -> None:
        store = TraceStore()
        try:
            with TraceLogger(store, "sess_3", "turn_3", state="VALIDATION") as tl:
                tl.set_output({"valid": False})
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().splitlines() if ln]
        record = json.loads(lines[0])
        assert "exception" in record
        assert "boom" in record["exception"]

    def test_multiple_set_input_calls_merge(self, capsys: pytest.CaptureFixture) -> None:
        store = TraceStore()
        with TraceLogger(store, "sess_4", "turn_4", state="INTAKE") as tl:
            tl.set_input({"a": 1})
            tl.set_input({"b": 2})
        captured = capsys.readouterr()
        record = json.loads(captured.out.strip().splitlines()[0])
        assert record["input"]["a"] == 1
        assert record["input"]["b"] == 2

    def test_no_tokens_omits_token_usage(self, capsys: pytest.CaptureFixture) -> None:
        store = TraceStore()
        with TraceLogger(store, "sess_5", "turn_5", state="INSIGHT"):
            pass
        captured = capsys.readouterr()
        record = json.loads(captured.out.strip().splitlines()[0])
        assert "token_usage" not in record


class TestObservabilityStack:
    def test_record_turn_outcome_emits(self, capsys: pytest.CaptureFixture) -> None:
        obs = ObservabilityStack()
        obs.record_turn_outcome(
            turn_id="t1",
            terminal_state="TERMINAL",
            latency_ms=350,
            executable=True,
            hit_max_retries=False,
        )
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().splitlines() if ln]
        record = json.loads(lines[0])
        assert record["event"] == "TURN_OUTCOME"
        assert record["terminal_state"] == "TERMINAL"
        assert record["executable"] is True

    def test_alert_fires_on_low_executable_rate(self, capsys: pytest.CaptureFixture) -> None:
        obs = ObservabilityStack()
        # Submit 15 failed turns to breach the 70% executable threshold
        for i in range(15):
            obs.record_turn_outcome(
                turn_id=f"t{i}",
                terminal_state="TERMINAL_ERROR",
                latency_ms=100,
                executable=False,
                hit_max_retries=True,
            )
        captured = capsys.readouterr()
        stderr_lines = [ln for ln in captured.err.strip().splitlines() if ln]
        alert_events = [json.loads(ln)["event"] for ln in stderr_lines if ln.startswith("{")]
        assert "ALERT_EXECUTABLE_RATE" in alert_events

    def test_alert_fires_on_high_latency(self, capsys: pytest.CaptureFixture) -> None:
        obs = ObservabilityStack()
        # Need 10 turns in window before alerts are evaluated
        for i in range(9):
            obs.record_turn_outcome(
                turn_id=f"t{i}",
                terminal_state="TERMINAL",
                latency_ms=100,
                executable=True,
                hit_max_retries=False,
            )
        obs.record_turn_outcome(
            turn_id="t_high",
            terminal_state="TERMINAL",
            latency_ms=9_000,  # > 8,000ms threshold
            executable=True,
            hit_max_retries=False,
        )
        captured = capsys.readouterr()
        stderr_lines = [ln for ln in captured.err.strip().splitlines() if ln]
        alert_events = [json.loads(ln)["event"] for ln in stderr_lines if ln.startswith("{")]
        # BUG-7 FIX: renamed from ALERT_LATENCY_P99 to ALERT_LATENCY_SPIKE.
        # The alert fires on a single turn exceeding 8,000ms (instantaneous
        # spike), not on a rolling p99 — the old name was misleading.
        assert (
            "ALERT_LATENCY_SPIKE" in alert_events
        ), f"Expected ALERT_LATENCY_SPIKE in alerts, got: {alert_events}"

    def test_turn_outcome_includes_schema_id_and_code_type(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """OBS-1: TURN_OUTCOME must carry schema_id and code_type for roll-up."""
        obs = ObservabilityStack()
        obs.record_turn_outcome(
            turn_id="t_obs1",
            terminal_state="TERMINAL",
            latency_ms=400,
            executable=True,
            hit_max_retries=False,
            schema_id="ins_prod_v3",
            code_type="sql",
        )
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().splitlines() if ln]
        record = json.loads(lines[0])
        assert record["event"] == "TURN_OUTCOME"
        assert (
            record["schema_id"] == "ins_prod_v3"
        ), "schema_id missing from TURN_OUTCOME — OBS-1 fix not applied"
        assert (
            record["code_type"] == "sql"
        ), "code_type missing from TURN_OUTCOME — OBS-1 fix not applied"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
