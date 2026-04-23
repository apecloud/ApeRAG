#!/usr/bin/env bash
# Capture provider-aware E2E failure evidence before the compose teardown wipes
# the containers. Called from ``run_compose_full.sh`` cleanup on non-zero exit,
# and from the CI workflow "diagnose" step so we have a second chance to fetch
# evidence even if the cleanup hook itself failed.
#
# Output bundle: ``tests/e2e_http/.diagnostic-artifacts/`` — uploaded by the
# GitHub workflow as ``e2e-http-provider-diagnostic-artifacts`` on failure.
#
# Secrets: ``E2E_ALIBABACLOUD_API_KEY`` must not appear in the bundle. We rewrite
# it to ``<REDACTED_ALIBABACLOUD_API_KEY>`` in every captured stream, and drop
# any ``Authorization: Bearer ...`` / ``"api_key": "..."`` payloads. Additional
# ``E2E_EXTRA_REDACT`` values (``:``-separated) can be passed in to scrub more
# secrets without code changes.
#
# See ``#模块化重构`` task #11 for the causation checklist this script feeds.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DIAG_DIR="${E2E_DIAG_DIR:-${ROOT_DIR}/tests/e2e_http/.diagnostic-artifacts}"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

mkdir -p "${DIAG_DIR}"

_redact() {
  # Accepts stream on stdin, writes redacted stream on stdout.
  local sed_args=()
  if [[ -n "${E2E_ALIBABACLOUD_API_KEY:-}" ]]; then
    sed_args+=(-e "s|${E2E_ALIBABACLOUD_API_KEY}|<REDACTED_ALIBABACLOUD_API_KEY>|g")
  fi
  if [[ -n "${E2E_OPENROUTER_API_KEY:-}" ]]; then
    sed_args+=(-e "s|${E2E_OPENROUTER_API_KEY}|<REDACTED_OPENROUTER_API_KEY>|g")
  fi
  if [[ -n "${E2E_EXTRA_REDACT:-}" ]]; then
    local IFS=":"
    local extra
    for extra in ${E2E_EXTRA_REDACT}; do
      [[ -z "${extra}" ]] && continue
      sed_args+=(-e "s|${extra}|<REDACTED>|g")
    done
  fi
  sed_args+=(-e 's|\(Authorization:\s*Bearer \)[A-Za-z0-9._-]\{8,\}|\1<REDACTED>|g')
  sed_args+=(-e 's|\("api_key"\s*:\s*"\)[^"]*|\1<REDACTED>|g')
  sed_args+=(-e 's|\("apiKey"\s*:\s*"\)[^"]*|\1<REDACTED>|g')
  sed "${sed_args[@]}"
}

echo "[provider_diagnostic] capturing to ${DIAG_DIR}"

{
  echo "=== docker compose ps ==="
  docker compose -f "${COMPOSE_FILE}" ps
  echo
  echo "=== docker compose version ==="
  docker compose version
  echo
  echo "=== uname + runner env ==="
  uname -a
  echo "GITHUB_RUN_ID=${GITHUB_RUN_ID:-<local>}"
  echo "RUNNER_OS=${RUNNER_OS:-<local>}"
} 2>&1 | _redact > "${DIAG_DIR}/compose-ps.txt" || true

for svc in api celeryworker celerybeat postgres redis qdrant es; do
  out="${DIAG_DIR}/compose-logs-${svc}.txt"
  # --tail=4000 keeps the bundle small even for long-running workers while
  # still covering the failing hurl request window (~1-2s).
  docker compose -f "${COMPOSE_FILE}" logs --no-color --timestamps --tail=4000 "${svc}" 2>&1 \
    | _redact > "${out}" || true
done

# Preserve the bootstrap env so reviewers can see base URL / run id / model
# config without dumping live secrets. Bootstrap never writes the raw keys
# into this file, but we redact defensively.
BOOTSTRAP_ENV="${ROOT_DIR}/tests/e2e_http/bootstrap/.generated/e2e.env"
if [[ -f "${BOOTSTRAP_ENV}" ]]; then
  _redact < "${BOOTSTRAP_ENV}" > "${DIAG_DIR}/e2e.env.redacted" || true
fi

# Alibaba direct smoke: same Alibaba embedding model (``text-embedding-v3``),
# same runner network, same secret. Failure → points at key / quota / service /
# network. Success → points at ApeRAG/LiteLLM adaptation layer.
ALIBABA_DIRECT_BASE="${E2E_ALIBABA_DIRECT_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
if [[ -n "${E2E_ALIBABACLOUD_API_KEY:-}" ]]; then
  {
    echo "=== Alibaba direct smoke ${ALIBABA_DIRECT_BASE}/embeddings (text-embedding-v3) ==="
    curl --silent --show-error --include \
      --max-time 20 \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${E2E_ALIBABACLOUD_API_KEY}" \
      -X POST "${ALIBABA_DIRECT_BASE}/embeddings" \
      -d '{"model":"text-embedding-v3","input":["ApeRAG is a retrieval augmented generation platform.","Embeddings convert text into vectors."]}' \
      || echo "[provider_diagnostic] alibaba direct smoke curl exit=$?"
  } 2>&1 | _redact > "${DIAG_DIR}/alibaba-direct-smoke.txt" || true
else
  echo "[provider_diagnostic] E2E_ALIBABACLOUD_API_KEY unset; skipping direct smoke" \
    > "${DIAG_DIR}/alibaba-direct-smoke.txt"
fi

# Secret-leak self-check: fail loudly if the raw Alibaba key somehow survived
# redaction in any captured file. Prevents CI from publishing a bundle that
# contains the key.
if [[ -n "${E2E_ALIBABACLOUD_API_KEY:-}" ]]; then
  if grep --recursive --fixed-strings --silent -- "${E2E_ALIBABACLOUD_API_KEY}" "${DIAG_DIR}" 2>/dev/null; then
    echo "[provider_diagnostic] ERROR: Alibaba API key leaked into diagnostic bundle; scrubbing and aborting" >&2
    # Wipe the bundle rather than upload a leaky one.
    rm -rf "${DIAG_DIR}"
    mkdir -p "${DIAG_DIR}"
    echo "Diagnostic bundle suppressed: raw API key leaked past redaction. Investigate sed rules in provider_diagnostic.sh." \
      > "${DIAG_DIR}/REDACTION_FAILURE.txt"
    exit 1
  fi
fi

echo "[provider_diagnostic] bundle ready: $(ls -1 "${DIAG_DIR}" | tr '\n' ' ')"
