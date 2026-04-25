#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${E2E_REMOVE_VOLUMES:-0}" == "1" ]]; then
  docker compose --profile neo4j --profile nebula -f docker-compose.yml down -v
else
  docker compose --profile neo4j --profile nebula -f docker-compose.yml down
fi
