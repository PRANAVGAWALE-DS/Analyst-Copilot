#!/usr/bin/env bash
# smoke_test.sh — Post-deploy smoke test for analyst-copilot
# Runs against the Docker stack (or any reachable server).
# Linux/macOS equivalent of smoke_test.ps1 (H-05 fix).
#
# Usage:
#   ./smoke_test.sh                          # defaults: localhost:8000, no auth
#   ./smoke_test.sh --url http://host:8000   # custom host
#   ./smoke_test.sh --api-key <key>          # authenticated server
#   ./smoke_test.sh --schema-id ins_prod_v3  # non-default schema
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed (details printed to stderr)
#
# Dependencies: curl, jq (both available in ubuntu-latest GitHub Actions runner)

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────────
BASE_URL="http://localhost:8000"
API_KEY=""
SCHEMA_ID="ins_prod_v3"
TIMEOUT=30          # seconds per curl call
WAIT_RETRIES=12     # × 5s = 60s max wait for /health 200
WAIT_SLEEP=5

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --url)       BASE_URL="$2";   shift 2 ;;
    --api-key)   API_KEY="$2";    shift 2 ;;
    --schema-id) SCHEMA_ID="$2";  shift 2 ;;
    --timeout)   TIMEOUT="$2";    shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
FAILURES=()

_auth_header() {
  if [[ -n "$API_KEY" ]]; then
    echo "-H" "X-API-Key: ${API_KEY}"
  fi
}

_curl() {
  # Usage: _curl <method> <path> [extra curl args...]
  local method="$1"; shift
  local path="$1";   shift
  curl --silent --fail-with-body \
    --max-time "$TIMEOUT" \
    -X "$method" \
    $(_auth_header) \
    "${BASE_URL}${path}" \
    "$@"
}

_check() {
  local name="$1"; shift
  local result
  if result=$(eval "$@" 2>&1); then
    echo "  [PASS] ${name}"
    (( PASS++ )) || true
    echo "$result"
  else
    echo "  [FAIL] ${name}" >&2
    FAILURES+=("$name")
    (( FAIL++ )) || true
    echo "$result" >&2
  fi
}

_assert_field() {
  # Assert that JSON field $2 equals $3 in stdin JSON
  local json="$1" field="$2" expected="$3"
  local actual
  actual=$(echo "$json" | jq -r ".${field}" 2>/dev/null)
  if [[ "$actual" == "$expected" ]]; then
    return 0
  else
    echo "Field '${field}': expected '${expected}', got '${actual}'" >&2
    return 1
  fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  analyst-copilot smoke test"
echo "  Target : ${BASE_URL}"
echo "  Schema : ${SCHEMA_ID}"
echo "  Auth   : $([ -n "$API_KEY" ] && echo 'yes' || echo 'no')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Wait for server ready ───────────────────────────────────────────────────
echo "▶ Waiting for server to become healthy..."
healthy=false
for i in $(seq 1 "$WAIT_RETRIES"); do
  response=$(curl --silent --max-time 5 "${BASE_URL}/health" 2>/dev/null || true)
  http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --max-time 5 "${BASE_URL}/health" 2>/dev/null || echo "000")

  if [[ "$http_code" == "200" ]]; then
    healthy=true
    echo "  Server healthy after $((i * WAIT_SLEEP))s"
    break
  fi
  echo "  Attempt ${i}/${WAIT_RETRIES}: HTTP ${http_code} — waiting ${WAIT_SLEEP}s..."
  sleep "$WAIT_SLEEP"
done

if [[ "$healthy" != "true" ]]; then
  echo "" >&2
  echo "FATAL: Server did not become healthy within $((WAIT_RETRIES * WAIT_SLEEP))s" >&2
  echo "Check: docker compose ps && docker compose logs api" >&2
  exit 1
fi
echo ""

# ── 2. /health ─────────────────────────────────────────────────────────────────
echo "▶ Check 1: /health returns HTTP 200 with initialized=true"
health_json=$(_curl GET /health)
_check "/health initialized=true" \
  _assert_field "$health_json" "initialized" "true"
_check "/health status=ok" \
  _assert_field "$health_json" "status" "ok"
echo ""

# ── 3. /health is auth-exempt ──────────────────────────────────────────────────
echo "▶ Check 2: /health reachable without API key (load-balancer probe)"
health_no_auth=$(curl --silent --max-time "$TIMEOUT" "${BASE_URL}/health")
_check "/health no-auth-key" \
  _assert_field "$health_no_auth" "status" "ok"
echo ""

# ── 4. Auth middleware (only when API_KEY is set) ──────────────────────────────
if [[ -n "$API_KEY" ]]; then
  echo "▶ Check 3: /query with wrong API key returns 401"
  bad_auth_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --max-time "$TIMEOUT" \
    -X POST \
    -H "X-API-Key: definitely-wrong-key" \
    -H "Content-Type: application/json" \
    -d '{"nl_query":"test","schema_id":"'"$SCHEMA_ID"'"}' \
    "${BASE_URL}/query")
  _check "wrong-key → 401" \
    [[ "$bad_auth_code" == "401" ]]
  echo ""
fi

# ── 5. /query happy path ───────────────────────────────────────────────────────
echo "▶ Check 4: /query returns HTTP 200 with generated_code non-empty"
query_json=$(_curl POST /query \
  -H "Content-Type: application/json" \
  -d '{
    "nl_query": "How many claims are in the dataset?",
    "schema_id": "'"$SCHEMA_ID"'",
    "execution_mode": "sql",
    "dry_run": false
  }')

query_code=$(echo "$query_json" | jq -r '.generated_code // empty' 2>/dev/null)
_check "/query generated_code non-empty" \
  [[ -n "$query_code" ]]

query_session=$(echo "$query_json" | jq -r '.session_id // empty' 2>/dev/null)
_check "/query session_id present" \
  [[ -n "$query_session" ]]

query_insight=$(echo "$query_json" | jq -r '.insight // empty' 2>/dev/null)
_check "/query insight present" \
  [[ -n "$query_insight" ]]
echo ""

# ── 6. Pydantic validation ─────────────────────────────────────────────────────
echo "▶ Check 5: /query with missing nl_query returns 422"
validation_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
  --max-time "$TIMEOUT" \
  $(_auth_header) \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"schema_id":"'"$SCHEMA_ID"'"}' \
  "${BASE_URL}/query")
_check "missing nl_query → 422" \
  [[ "$validation_code" == "422" ]]
echo ""

# ── 7. Docs disabled in production ────────────────────────────────────────────
echo "▶ Check 6: /openapi.json returns 404 (H-07: docs disabled in production)"
openapi_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
  --max-time "$TIMEOUT" \
  "${BASE_URL}/openapi.json")
if [[ "$openapi_code" == "404" ]]; then
  echo "  [PASS] /openapi.json → 404 (production mode confirmed)"
  (( PASS++ )) || true
elif [[ "$openapi_code" == "200" ]]; then
  echo "  [WARN] /openapi.json → 200 (development mode — expected in non-production)"
  # Not a failure — may be running with APP_ENV=development intentionally
else
  echo "  [FAIL] /openapi.json → ${openapi_code} (unexpected status)" >&2
  FAILURES+=("/openapi.json unexpected status")
  (( FAIL++ )) || true
fi
echo ""

# ── 8. Multi-turn session continuity ──────────────────────────────────────────
echo "▶ Check 7: multi-turn session (session_id reuse)"
turn2_json=$(_curl POST /query \
  -H "Content-Type: application/json" \
  -d '{
    "nl_query": "What is the average claim amount?",
    "schema_id": "'"$SCHEMA_ID"'",
    "session_id": "'"$query_session"'",
    "execution_mode": "sql"
  }' 2>/dev/null || echo "{}")

turn2_session=$(echo "$turn2_json" | jq -r '.session_id // empty' 2>/dev/null)
_check "session_id consistent across turns" \
  [[ "$turn2_session" == "$query_session" ]]
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total=$(( PASS + FAIL ))
echo "  Results: ${PASS}/${total} passed"

if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "  Failed checks:" >&2
  for f in "${FAILURES[@]}"; do
    echo "    ✗ ${f}" >&2
  done
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi

echo "  All checks passed."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit 0