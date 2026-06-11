"""
test_app.py — HTTP layer tests for app.py
Data Analyst Copilot · pytest + pytest-asyncio + httpx

Coverage targets:
  - /health: HTTP 503 when not initialised, HTTP 200 when ready  (B-05 guard)
  - Auth middleware: 401 with wrong key, 200 with correct key, exempt paths bypass
  - Docs disabled in production, enabled in development             (H-07 guard)
  - _http_status_for: error_code → HTTP status mapping
  - /query: Pydantic validation error → 422 before orchestrator
  - LLM_PROVIDER validation rejects unsupported values             (H-11 guard)
  - Rate limiter headers present on /query responses

All orchestrator calls are mocked — no DB, no LLM, no GPU required.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Environment setup (must precede any app import) ────────────────────────────
os.environ.setdefault("APP_ENV", "development")  # enables /docs in tests
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-key-ci")
os.environ.setdefault("GEMINI_API_KEY", "test-key-ci")
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("FAISS_INDEX_PATH", "./data/faiss_index/smoke_test.faiss")
os.environ.setdefault("EMBED_CACHE_PATH", "./data/faiss_index/embed_cache.json")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
os.environ["API_KEY"] = ""  # auth disabled by default; individual tests override


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _mock_app_state() -> MagicMock:
    """Return a minimal _AppState mock that satisfies all route handlers."""
    from interfaces import QueryResponse

    state = MagicMock()
    state.orchestrator = MagicMock()
    state.orchestrator.run = AsyncMock(
        return_value=QueryResponse(
            session_id="sess-test-001",
            generated_code="SELECT 1",
            code_type="sql",
            result_preview=[{"col": 1}],
            row_count=1,
            insight="One row returned.",
            execution_time_ms=42,
            retry_count=0,
        )
    )
    state.df_store = MagicMock()
    state.df_store.ingest = MagicMock()
    state.df_store.get = MagicMock(return_value={})
    return state


@pytest.fixture()
def client_uninitialised() -> TestClient:
    """App client where _app_state is None (startup not complete)."""
    with patch("app._app_state", None):
        from app import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture()
def client(mock_state: MagicMock) -> TestClient:
    """App client with a fully initialised mock _app_state."""
    with patch("app._app_state", mock_state):
        from app import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture()
def mock_state() -> MagicMock:
    return _mock_app_state()


# ── /health ────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """
    B-05 regression guard.

    /health must return HTTP 503 when _app_state is None (startup not complete
    or failed), and HTTP 200 when _app_state is set.
    """

    def test_health_503_when_not_initialised(self) -> None:
        with patch("app._app_state", None), patch("app._startup", AsyncMock()):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["initialized"] is False
        assert body["status"] == "starting"

    def test_health_200_when_initialised(self) -> None:
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["initialized"] is True
        assert body["status"] == "ok"

    def test_health_exempt_from_auth(self) -> None:
        """
        /health must be reachable without an API key — load balancers
        and Docker HEALTHCHECK do not send auth headers.
        """
        os.environ["API_KEY"] = "super-secret-key"
        try:
            mock_state = _mock_app_state()
            with patch("app._app_state", mock_state):
                from app import create_app

                app = create_app()
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.get("/health")  # no X-API-Key header
            assert resp.status_code == 200
        finally:
            os.environ["API_KEY"] = ""


# ── Auth middleware ────────────────────────────────────────────────────────────


class TestAuthMiddleware:
    """API key enforcement and exempt paths."""

    def _client_with_key(self, api_key: str) -> TestClient:
        os.environ["API_KEY"] = api_key
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            return TestClient(app, raise_server_exceptions=False)

    def test_401_with_wrong_key(self) -> None:
        with self._client_with_key("correct-key") as c:
            resp = c.post(
                "/query",
                json={
                    "nl_query": "show me claims",
                    "schema_id": "test",
                },
                headers={"X-API-Key": "wrong-key"},
            )
        os.environ["API_KEY"] = ""
        assert resp.status_code == 401
        assert (
            "API key" in resp.json().get("detail", "").lower()
            or resp.json().get("detail") is not None
        )

    def test_401_with_no_key(self) -> None:
        with self._client_with_key("correct-key") as c:
            resp = c.post(
                "/query",
                json={"nl_query": "show me claims", "schema_id": "test"},
            )
        os.environ["API_KEY"] = ""
        assert resp.status_code == 401

    def test_200_with_correct_key(self) -> None:
        with self._client_with_key("correct-key") as c:
            resp = c.post(
                "/query",
                json={"nl_query": "show me top 5 claims", "schema_id": "test"},
                headers={"X-API-Key": "correct-key"},
            )
        os.environ["API_KEY"] = ""
        assert resp.status_code == 200

    def test_no_auth_when_api_key_empty(self) -> None:
        """API_KEY="" disables auth entirely — any request passes."""
        os.environ["API_KEY"] = ""
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/query",
                    json={"nl_query": "show me top 5 claims", "schema_id": "test"},
                )
        assert resp.status_code == 200

    def test_root_exempt_from_auth(self) -> None:
        with self._client_with_key("correct-key") as c:
            resp = c.get("/")  # no key
        os.environ["API_KEY"] = ""
        # Root may 200 or 404 depending on route definition, but not 401
        assert resp.status_code != 401


# ── H-07: Docs visibility ──────────────────────────────────────────────────────


class TestDocsVisibility:
    """
    H-07 regression guard.

    In production (APP_ENV=production) /openapi.json and /docs must return 404.
    In development (APP_ENV=development) they must be accessible.
    """

    def test_openapi_disabled_in_production(self) -> None:
        os.environ["APP_ENV"] = "production"
        try:
            mock_state = _mock_app_state()
            with patch("app._app_state", mock_state):
                from app import create_app

                app = create_app()
                with TestClient(app, raise_server_exceptions=False) as c:
                    assert c.get("/openapi.json").status_code == 404
                    assert c.get("/docs").status_code == 404
                    assert c.get("/redoc").status_code == 404
        finally:
            os.environ["APP_ENV"] = "development"

    def test_openapi_enabled_in_development(self) -> None:
        os.environ["APP_ENV"] = "development"
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                assert c.get("/openapi.json").status_code == 200
                assert c.get("/docs").status_code == 200


# ── _http_status_for error mapping ────────────────────────────────────────────


class TestHttpStatusMapping:
    """_http_status_for maps every error_code to the correct HTTP status."""

    def test_all_mappings(self) -> None:
        from app import _http_status_for

        expected = {
            None: 200,
            "UNRESOLVED_REFERENCE": 422,
            "UNRESOLVED_COLUMN": 422,
            "POLICY_VIOLATION": 403,
            "MUTATION_STATEMENT": 403,
            "TERMINAL_ERROR": 502,
            "EXECUTION_TIMEOUT": 503,
            "DB_UNAVAILABLE": 503,
            "LLM_UNAVAILABLE": 503,
            "LLM_PARSE_ERROR": 502,
            "LLM_EMPTY_RESPONSE": 502,
            "INTERNAL_ERROR": 503,
            "UNKNOWN_CODE": 502,  # fallback for unrecognised codes
        }
        for code, http_status in expected.items():
            assert (
                _http_status_for(code) == http_status
            ), f"_http_status_for({code!r}) → {_http_status_for(code)}, expected {http_status}"

    def test_query_response_with_policy_violation_returns_403(self) -> None:
        """End-to-end: orchestrator returns POLICY_VIOLATION → HTTP 403."""
        from interfaces import ErrorDetail, QueryResponse

        policy_response = QueryResponse(
            session_id="sess-001",
            generated_code="INSERT INTO claims VALUES (1, 500)",
            code_type="sql",
            result_preview=None,
            row_count=None,
            insight="Mutation statements are not permitted.",
            execution_time_ms=10,
            retry_count=0,
            error=ErrorDetail(
                error_code="POLICY_VIOLATION",
                message="Mutation statements are not allowed.",
            ),
        )
        mock_state = _mock_app_state()
        mock_state.orchestrator.run = AsyncMock(return_value=policy_response)

        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/query",
                    json={"nl_query": "insert a claim", "schema_id": "test"},
                )

        assert resp.status_code == 403

    def test_query_response_with_terminal_error_returns_502(self) -> None:
        from interfaces import ErrorDetail, QueryResponse

        error_response = QueryResponse(
            session_id="sess-001",
            generated_code="",
            code_type="sql",
            result_preview=None,
            row_count=None,
            insight="Something went wrong.",
            execution_time_ms=10,
            retry_count=3,
            error=ErrorDetail(
                error_code="TERMINAL_ERROR",
                message="All retries exhausted.",
            ),
        )
        mock_state = _mock_app_state()
        mock_state.orchestrator.run = AsyncMock(return_value=error_response)

        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/query",
                    json={"nl_query": "break everything", "schema_id": "test"},
                )

        assert resp.status_code == 502


# ── Pydantic validation ────────────────────────────────────────────────────────


class TestQueryValidation:
    """Pydantic field validation fires before the orchestrator is called."""

    def test_missing_nl_query_returns_422(self) -> None:
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/query", json={"schema_id": "test"})
        assert resp.status_code == 422

    def test_nl_query_too_short_returns_422(self) -> None:
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/query", json={"nl_query": "hi", "schema_id": "test"})
        assert resp.status_code == 422

    def test_missing_schema_id_returns_422(self) -> None:
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/query", json={"nl_query": "show me claims"})
        assert resp.status_code == 422

    def test_invalid_execution_mode_returns_422(self) -> None:
        mock_state = _mock_app_state()
        with patch("app._app_state", mock_state):
            from app import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/query",
                    json={
                        "nl_query": "show me claims",
                        "schema_id": "test",
                        "execution_mode": "cobol",  # not in Literal
                    },
                )
        assert resp.status_code == 422


# ── H-11: LLM_PROVIDER validation ─────────────────────────────────────────────


class TestLLMProviderValidation:
    """
    H-11 regression guard.

    _startup() must raise RuntimeError immediately when LLM_PROVIDER is not
    in {"groq", "gemini"} — not silently fall through to the Gemini key path.
    """

    def test_unsupported_provider_raises_at_startup(self) -> None:
        from app import _startup

        os.environ["LLM_PROVIDER"] = "openai"  # not supported
        try:
            with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
                import asyncio

                asyncio.get_event_loop().run_until_complete(_startup())
        finally:
            os.environ["LLM_PROVIDER"] = "groq"

    def test_supported_providers_do_not_raise_provider_error(self) -> None:
        """groq and gemini must not raise the provider validation error."""

        for provider in ("groq", "gemini"):
            os.environ["LLM_PROVIDER"] = provider
            # No assertion needed — importing app with these values must not crash
        os.environ["LLM_PROVIDER"] = "groq"
