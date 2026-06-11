# Makefile — analyst-copilot
# Standard targets for local development and CI parity.
#
# Usage:
#   make test           — run full test suite with coverage
#   make lint           — ruff check + format
#   make typecheck      — mypy --strict on analyst_copilot/
#   make build          — docker compose build (no cache)
#   make up             — start full stack (postgres + redis + api)
#   make down           — stop and remove containers
#   make smoke          — post-deploy smoke test against running stack
#   make smoke URL=...  — smoke test against a custom URL
#   make clean          — remove .pytest_cache, coverage artefacts
#   make all            — lint + typecheck + test (pre-push gate)

.PHONY: test lint typecheck build up down logs smoke clean all

PIP           ?= pip
DOCKER        ?= docker
COMPOSE       ?= docker compose
PYTEST_FLAGS  ?= -v --tb=short --asyncio-mode=auto
COVERAGE_MIN  ?= 38
MYPY_FLAGS    ?= --strict --ignore-missing-imports --disable-error-code type-arg --disable-error-code no-any-return --disable-error-code unused-ignore
SMOKE_URL     ?= http://localhost:8000
SMOKE_SCHEMA  ?= ins_prod_v3

# ── Colour helpers (suppress on non-TTY CI runners and Windows shells) ─────────
ifeq ($(OS),Windows_NT)
PYTHON ?= python
RESET  :=
BOLD   :=
GREEN  :=
RED    :=
YELLOW :=
else
PYTHON ?= python3
RESET  := $(shell tput sgr0    2>/dev/null || echo "")
BOLD   := $(shell tput bold    2>/dev/null || echo "")
GREEN  := $(shell tput setaf 2 2>/dev/null || echo "")
RED    := $(shell tput setaf 1 2>/dev/null || echo "")
YELLOW := $(shell tput setaf 3 2>/dev/null || echo "")
endif

_header = @echo $(BOLD)$(GREEN)==^> $(1)$(RESET)

# ── Default target ─────────────────────────────────────────────────────────────
all: lint typecheck test

# ── Dependencies ───────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(call _header,Installing dependencies)
	$(PIP) install -r requirements.txt -r requirements-dev.txt

.PHONY: install-mypy
install-mypy:
	$(PIP) install \
		mypy==1.10.0 \
		types-redis \
		types-tqdm \
		pandas-stubs

# ── Lint ───────────────────────────────────────────────────────────────────────
lint:
	$(call _header,Lint - ruff check)
	ruff check .
	$(call _header,Lint - ruff format check)
	ruff format --check .

.PHONY: lint-fix
lint-fix:
	$(call _header,Lint - ruff fix + format)
	ruff check --fix .
	ruff format .

# ── Type checking ──────────────────────────────────────────────────────────────
typecheck:
	$(call _header,Type check - mypy)
	mypy analyst_copilot/ $(MYPY_FLAGS) --exclude 'analyst_copilot/__init__\.py'

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	$(call _header,Tests - pytest + coverage \(min $(COVERAGE_MIN)%\))
	$(PYTHON) -m pytest tests/ $(PYTEST_FLAGS) --cov=analyst_copilot --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=$(COVERAGE_MIN)

# Run a specific test file quickly (no coverage)
# Usage: make test-file FILE=tests/test_orchestrator.py
.PHONY: test-file
test-file:
	$(PYTHON) -m pytest $(FILE) $(PYTEST_FLAGS)

# Run only the new orchestrator and app tests
.PHONY: test-core
test-core:
	$(call _header,Tests - orchestrator + app layer only)
	$(PYTHON) -m pytest tests/test_orchestrator.py tests/test_app.py $(PYTEST_FLAGS) --no-cov

# ── Docker ─────────────────────────────────────────────────────────────────────
build:
	$(call _header,Docker - build \(no cache\))
	$(COMPOSE) build --no-cache

.PHONY: build-cached
build-cached:
	$(call _header,Docker - build \(with cache\))
	$(COMPOSE) build

up:
	$(call _header,Docker - start stack)
	$(COMPOSE) up -d
	@echo "$(YELLOW)Waiting for services to become healthy...$(RESET)"
	@sleep 5
	$(COMPOSE) ps

.PHONY: up-dev
up-dev:
	$(call _header,Docker - start stack in dev mode \(hot reload\))
	$(COMPOSE) --profile dev up -d
	$(COMPOSE) ps

down:
	$(call _header,Docker - stop stack)
	$(COMPOSE) down

.PHONY: down-volumes
down-volumes:
	$(call _header,Docker - stop stack and remove volumes)
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f api

.PHONY: ps
ps:
	$(COMPOSE) ps

# ── Smoke test ─────────────────────────────────────────────────────────────────
smoke:
	$(call _header,Smoke test - $(SMOKE_URL))
	@chmod +x smoke_test.ps1
	./smoke_test.ps1 \
		--url "$(SMOKE_URL)" \
		--schema-id "$(SMOKE_SCHEMA)" \
		$(if $(API_KEY),--api-key "$(API_KEY)",)

# ── Pre-push gate (mirrors CI job order) ──────────────────────────────────────
.PHONY: pre-push
pre-push: lint typecheck test
	@echo "$(BOLD)$(GREEN)✓ All pre-push checks passed$(RESET)"

# ── Dev workflow shortcut: build → up → smoke ──────────────────────────────────
.PHONY: deploy-local
deploy-local: build up smoke

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	$(call _header,Clean - removing build artefacts)
	rm -rf .pytest_cache htmlcov coverage.xml .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
