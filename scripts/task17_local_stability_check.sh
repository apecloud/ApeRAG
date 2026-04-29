#!/usr/bin/env bash
set -euo pipefail

# task #17 SRE acceptance helper.
#
# This script is intentionally deployment-agnostic: it can validate a local
# docker-compose target, a port-forwarded Kubernetes target, or the Singapore
# demo target. It focuses on the runtime hard-cut signals that are easy to
# verify from outside the app:
#   - API health probes stay light and fast.
#   - API/worker DB pool budgets fit under PostgreSQL max_connections.
#   - Kubernetes topology has a separate indexing-worker deployment.
#   - Optional worker restart does not disturb the API health endpoints.
#
# Heavy product indexing load is driven by the separate load/e2e scripts. Run
# this script before, during, and after that load to catch API health and pool
# regressions while indexing pressure is present.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK17_BASE_URL="${TASK17_BASE_URL:-${E2E_BASE_URL:-http://127.0.0.1:8000}}"
TASK17_HEALTH_SAMPLES="${TASK17_HEALTH_SAMPLES:-20}"
TASK17_HEALTH_P95_MS="${TASK17_HEALTH_P95_MS:-500}"
TASK17_CURL_TIMEOUT_SECONDS="${TASK17_CURL_TIMEOUT_SECONDS:-3}"
TASK17_OBSERVE_SECONDS="${TASK17_OBSERVE_SECONDS:-0}"
TASK17_OBSERVE_INTERVAL_SECONDS="${TASK17_OBSERVE_INTERVAL_SECONDS:-2}"

TASK17_API_REPLICAS="${TASK17_API_REPLICAS:-2}"
TASK17_API_DB_POOL_SIZE="${TASK17_API_DB_POOL_SIZE:-5}"
TASK17_API_DB_MAX_OVERFLOW="${TASK17_API_DB_MAX_OVERFLOW:-5}"
TASK17_WORKER_REPLICAS="${TASK17_WORKER_REPLICAS:-1}"
TASK17_WORKER_DB_POOL_SIZE="${TASK17_WORKER_DB_POOL_SIZE:-10}"
TASK17_WORKER_DB_MAX_OVERFLOW="${TASK17_WORKER_DB_MAX_OVERFLOW:-10}"
TASK17_ROLLOUT_SURGE_BUDGET="${TASK17_ROLLOUT_SURGE_BUDGET:-10}"
TASK17_DIAGNOSTICS_RESERVED="${TASK17_DIAGNOSTICS_RESERVED:-5}"
TASK17_POSTGRES_RESERVED="${TASK17_POSTGRES_RESERVED:-10}"
TASK17_POSTGRES_MAX_CONNECTIONS="${TASK17_POSTGRES_MAX_CONNECTIONS:-100}"
TASK17_SAFETY_RATIO="${TASK17_SAFETY_RATIO:-0.70}"

TASK17_K8S_NAMESPACE="${TASK17_K8S_NAMESPACE:-}"
TASK17_API_DEPLOYMENT="${TASK17_API_DEPLOYMENT:-api}"
TASK17_WORKER_DEPLOYMENT="${TASK17_WORKER_DEPLOYMENT:-indexing-worker}"
TASK17_API_SELECTOR="${TASK17_API_SELECTOR:-app.aperag.io/component=api}"
TASK17_WORKER_SELECTOR="${TASK17_WORKER_SELECTOR:-app.aperag.io/component=indexing-worker}"
TASK17_SCAN_API_LOGS="${TASK17_SCAN_API_LOGS:-1}"
TASK17_RESTART_WORKER="${TASK17_RESTART_WORKER:-0}"

TASK17_DATABASE_URL="${TASK17_DATABASE_URL:-${DATABASE_URL:-}}"
TASK17_DIAGNOSTICS_URL="${TASK17_DIAGNOSTICS_URL:-}"
TASK17_DIAGNOSTICS_BEARER_TOKEN="${TASK17_DIAGNOSTICS_BEARER_TOKEN:-}"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

failures=0
skips=0

log() {
  printf '[task17] %s\n' "$*"
}

pass() {
  printf '[task17] PASS: %s\n' "$*"
}

fail() {
  printf '[task17] FAIL: %s\n' "$*" >&2
  failures=$((failures + 1))
}

skip() {
  printf '[task17] SKIP: %s\n' "$*"
  skips=$((skips + 1))
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

curl_probe() {
  local path="$1"
  local url="${TASK17_BASE_URL}${path}"
  local response status seconds

  response="$(
    curl --silent --show-error \
      --output /dev/null \
      --write-out '%{http_code} %{time_total}' \
      --max-time "${TASK17_CURL_TIMEOUT_SECONDS}" \
      "${url}" 2>/dev/null || true
  )"
  status="${response%% *}"
  seconds="${response##* }"

  if [[ -z "${response}" || "${status}" == "000" || "${status}" -lt 200 || "${status}" -ge 300 ]]; then
    return 1
  fi

  awk -v s="${seconds}" 'BEGIN { printf "%.0f\n", s * 1000 }'
}

percentile_from_file() {
  local file="$1"
  local percentile="$2"
  sort -n "${file}" | awk -v p="${percentile}" '
    { values[++n] = $1 }
    END {
      if (n == 0) {
        exit 1
      }
      idx = int(p * n + 0.999999)
      if (idx < 1) idx = 1
      if (idx > n) idx = n
      print values[idx]
    }
  '
}

assert_le() {
  local actual="$1"
  local expected="$2"
  awk -v a="${actual}" -v e="${expected}" 'BEGIN { exit !(a <= e) }'
}

check_endpoint_latency() {
  local path="$1"
  local samples="${2:-${TASK17_HEALTH_SAMPLES}}"
  local max_p95_ms="${3:-${TASK17_HEALTH_P95_MS}}"
  local tmp
  tmp="$(mktemp)"
  trap 'rm -f "${tmp}"' RETURN

  for ((i = 1; i <= samples; i++)); do
    local ms
    if ! ms="$(curl_probe "${path}")"; then
      rm -f "${tmp}"
      trap - RETURN
      fail "${path} failed before collecting ${samples} samples"
      return
    fi
    printf '%s\n' "${ms}" >>"${tmp}"
    if [[ "${TASK17_OBSERVE_SECONDS}" != "0" && "${i}" -lt "${samples}" ]]; then
      sleep "${TASK17_OBSERVE_INTERVAL_SECONDS}"
    fi
  done

  local p95 p99
  p95="$(percentile_from_file "${tmp}" 0.95)"
  p99="$(percentile_from_file "${tmp}" 0.99)"
  rm -f "${tmp}"
  trap - RETURN

  if assert_le "${p95}" "${max_p95_ms}"; then
    pass "${path} p95=${p95}ms p99=${p99}ms over ${samples} samples"
  else
    fail "${path} p95=${p95}ms exceeds ${max_p95_ms}ms over ${samples} samples"
  fi
}

check_health_contract() {
  if [[ "${TASK17_SKIP_HEALTH:-0}" == "1" ]]; then
    skip "API health check disabled by TASK17_SKIP_HEALTH=1"
    return
  fi

  local samples="${TASK17_HEALTH_SAMPLES}"
  if [[ "${TASK17_OBSERVE_SECONDS}" != "0" ]]; then
    samples="$(
      awk -v duration="${TASK17_OBSERVE_SECONDS}" -v interval="${TASK17_OBSERVE_INTERVAL_SECONDS}" \
        'BEGIN { printf "%d\n", int((duration + interval - 0.000001) / interval) }'
    )"
    if [[ "${samples}" -lt 1 ]]; then
      samples=1
    fi
  fi

  log "Checking API probe endpoints at ${TASK17_BASE_URL}"
  check_endpoint_latency "/health" "${samples}"
  check_endpoint_latency "/health/live" "${samples}"
  check_endpoint_latency "/health/ready" "${samples}"

  if [[ -n "${TASK17_DIAGNOSTICS_URL}" ]]; then
    local auth_args=()
    if [[ -n "${TASK17_DIAGNOSTICS_BEARER_TOKEN}" ]]; then
      auth_args+=(--header "Authorization: Bearer ${TASK17_DIAGNOSTICS_BEARER_TOKEN}")
    fi
    local status
    status="$(
      curl --silent --show-error \
        --output /dev/null \
        --write-out '%{http_code}' \
        --max-time "${TASK17_CURL_TIMEOUT_SECONDS}" \
        "${auth_args[@]}" \
        "${TASK17_DIAGNOSTICS_URL}" 2>/dev/null || true
    )"
    case "${status}" in
      200) pass "diagnostics endpoint reachable with explicit opt-in URL" ;;
      401|403) pass "diagnostics endpoint is protected (status=${status})" ;;
      *) fail "diagnostics endpoint unexpected status=${status}" ;;
    esac
  else
    skip "diagnostics endpoint check not enabled; set TASK17_DIAGNOSTICS_URL to verify it"
  fi
}

check_connection_budget() {
  local api_total worker_total total allowed
  api_total=$((TASK17_API_REPLICAS * (TASK17_API_DB_POOL_SIZE + TASK17_API_DB_MAX_OVERFLOW)))
  worker_total=$((TASK17_WORKER_REPLICAS * (TASK17_WORKER_DB_POOL_SIZE + TASK17_WORKER_DB_MAX_OVERFLOW)))
  total=$((api_total + worker_total + TASK17_ROLLOUT_SURGE_BUDGET + TASK17_DIAGNOSTICS_RESERVED + TASK17_POSTGRES_RESERVED))
  allowed="$(awk -v max="${TASK17_POSTGRES_MAX_CONNECTIONS}" -v ratio="${TASK17_SAFETY_RATIO}" 'BEGIN { printf "%.0f\n", max * ratio }')"

  log "DB budget: api=${api_total}, worker=${worker_total}, surge=${TASK17_ROLLOUT_SURGE_BUDGET}, diagnostics=${TASK17_DIAGNOSTICS_RESERVED}, postgres_reserved=${TASK17_POSTGRES_RESERVED}, total=${total}, allowed=${allowed}"

  if ((total < allowed)); then
    pass "DB pool budget fits under safety ratio"
  else
    fail "DB pool budget total=${total} is not below allowed=${allowed}; lower replicas or pool/overflow values"
  fi
}

check_postgres_live_connections() {
  if [[ -z "${TASK17_DATABASE_URL}" ]]; then
    skip "PostgreSQL live connection check not enabled; set TASK17_DATABASE_URL"
    return
  fi
  if ! have_cmd psql; then
    skip "psql not found; cannot query PostgreSQL live connection counts"
    return
  fi

  local output
  if ! output="$(
    psql "${TASK17_DATABASE_URL}" -v ON_ERROR_STOP=1 -Atc \
      "SELECT current_setting('max_connections') AS max_conn,
              count(*) FILTER (WHERE datname = current_database()) AS current_db_conn,
              count(*) FILTER (WHERE datname = current_database() AND state = 'active') AS active_conn
       FROM pg_stat_activity;"
  )"; then
    fail "PostgreSQL connection query failed"
    return
  fi

  local max_conn current_db_conn active_conn
  IFS='|' read -r max_conn current_db_conn active_conn <<<"${output}"
  pass "PostgreSQL connections current_db=${current_db_conn}, active=${active_conn}, max_connections=${max_conn}"
}

kubectl_ns() {
  kubectl -n "${TASK17_K8S_NAMESPACE}" "$@"
}

check_k8s_topology() {
  if [[ -z "${TASK17_K8S_NAMESPACE}" ]]; then
    skip "Kubernetes topology check not enabled; set TASK17_K8S_NAMESPACE"
    return
  fi
  if ! have_cmd kubectl; then
    skip "kubectl not found; cannot check Kubernetes topology"
    return
  fi

  if kubectl_ns get deploy "${TASK17_API_DEPLOYMENT}" >/dev/null 2>&1; then
    pass "API deployment exists: ${TASK17_API_DEPLOYMENT}"
  else
    fail "API deployment missing: ${TASK17_API_DEPLOYMENT}"
  fi

  if kubectl_ns get deploy "${TASK17_WORKER_DEPLOYMENT}" >/dev/null 2>&1; then
    pass "indexing-worker deployment exists: ${TASK17_WORKER_DEPLOYMENT}"
  else
    fail "indexing-worker deployment missing: ${TASK17_WORKER_DEPLOYMENT}"
  fi

  local api_ready worker_ready
  api_ready="$(kubectl_ns get deploy "${TASK17_API_DEPLOYMENT}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)"
  worker_ready="$(kubectl_ns get deploy "${TASK17_WORKER_DEPLOYMENT}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)"
  log "Kubernetes readyReplicas: api=${api_ready:-0}, indexing-worker=${worker_ready:-0}"

  if [[ "${TASK17_SCAN_API_LOGS}" == "1" ]]; then
    local logs
    logs="$(kubectl_ns logs -l "${TASK17_API_SELECTOR}" --tail=500 --since=30m 2>/dev/null || true)"
    if [[ -z "${logs}" ]]; then
      skip "No API logs found for selector ${TASK17_API_SELECTOR}; cannot scan execution-plane uniqueness"
    elif printf '%s\n' "${logs}" | grep -E "run_.*worker|run_reconcile_loop|run_cleanup_loop|aperag\\.cli\\.indexing_worker" >/dev/null; then
      fail "API logs contain worker/reconciler/cleanup startup markers"
    else
      pass "API logs do not show worker/reconciler/cleanup startup markers"
    fi
  fi
}

check_worker_restart_optional() {
  if [[ "${TASK17_RESTART_WORKER}" != "1" ]]; then
    skip "Worker restart check not enabled; set TASK17_RESTART_WORKER=1"
    return
  fi
  if [[ -z "${TASK17_K8S_NAMESPACE}" ]] || ! have_cmd kubectl; then
    skip "Worker restart check requires TASK17_K8S_NAMESPACE and kubectl"
    return
  fi

  local pod
  pod="$(kubectl_ns get pod -l "${TASK17_WORKER_SELECTOR}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -z "${pod}" ]]; then
    fail "No indexing-worker pod found for selector ${TASK17_WORKER_SELECTOR}"
    return
  fi

  log "Deleting indexing-worker pod ${pod} to verify API remains healthy"
  kubectl_ns delete pod "${pod}" --wait=false >/dev/null
  sleep 2
  if curl_probe "/health/live" >/dev/null && curl_probe "/health/ready" >/dev/null; then
    pass "API health stayed available immediately after indexing-worker pod deletion"
  else
    fail "API health failed after indexing-worker pod deletion"
  fi
  kubectl_ns rollout status deploy/"${TASK17_WORKER_DEPLOYMENT}" --timeout=180s >/dev/null \
    && pass "indexing-worker deployment recovered after pod deletion" \
    || fail "indexing-worker deployment did not recover after pod deletion"
}

main() {
  log "Root: ${ROOT_DIR}"
  log "Base URL: ${TASK17_BASE_URL}"

  check_connection_budget
  check_health_contract
  check_postgres_live_connections
  check_k8s_topology
  check_worker_restart_optional

  if ((failures > 0)); then
    log "Completed with ${failures} failure(s), ${skips} skipped check(s)"
    exit 1
  fi

  log "Completed successfully with ${skips} skipped optional check(s)"
}

main "$@"
