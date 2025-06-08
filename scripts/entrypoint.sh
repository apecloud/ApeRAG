#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

python3 << END
import sys
import time

import psycopg2

suggest_unrecoverable_after = 30
start = time.time()

while True:
    try:
        psycopg2.connect(
            dbname="${POSTGRES_DB}",
            user="${POSTGRES_USER}",
            password="${POSTGRES_PASSWORD}",
            host="${POSTGRES_HOST}",
            port="${POSTGRES_PORT}",
        )
        break
    except psycopg2.OperationalError as error:
        sys.stderr.write("Waiting for PostgreSQL to become available...\n")

        if time.time() - start > suggest_unrecoverable_after:
            sys.stderr.write("  This is taking longer than expected. The following exception may be indicative of an unrecoverable error: '{}'\n".format(error))

    time.sleep(1)
END

>&2 echo 'PostgreSQL is available'

# FIXME: Temporary solution - create pgvector extension. Should be removed in the future
# and handled by proper database migration or initialization script
python3 << END
import sys
import psycopg2

try:
    conn = psycopg2.connect(
        dbname="${POSTGRES_DB}",
        user="${POSTGRES_USER}",
        password="${POSTGRES_PASSWORD}",
        host="${POSTGRES_HOST}",
        port="${POSTGRES_PORT}",
    )
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cursor.close()
    conn.close()
    sys.stderr.write("pgvector extension created successfully\n")
except Exception as error:
    sys.stderr.write("Failed to create pgvector extension (this is non-critical): {}\n".format(error))
END

# Build DATABASE_URL from components
if [[ -n "${POSTGRES_HOST:-}" && -n "${POSTGRES_USER:-}" && -n "${POSTGRES_PASSWORD:-}" ]]; then
    export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-postgres}"
fi

# Build CELERY_BROKER_URL from Redis components
if [[ -n "${REDIS_HOST:-}" && -n "${REDIS_USER:-}" && -n "${REDIS_PASSWORD:-}" ]]; then
    export CELERY_BROKER_URL="redis://${REDIS_USER}:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT:-6379}/0"
fi

# Build MEMORY_REDIS_URL from Redis components
if [[ -n "${REDIS_HOST:-}" && -n "${REDIS_USER:-}" && -n "${REDIS_PASSWORD:-}" ]]; then
    export MEMORY_REDIS_URL="redis://${REDIS_USER}:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT:-6379}/1"
fi

# Build ES_HOST from Elasticsearch components
if [[ -n "${ES_HOST_NAME:-}" ]]; then
    if [[ -n "${ES_USER:-}" && -n "${ES_PASSWORD:-}" ]]; then
        export ES_HOST="${ES_PROTOCOL:-http}://${ES_USER}:${ES_PASSWORD}@${ES_HOST_NAME}:${ES_PORT:-9200}"
    else
        export ES_HOST="${ES_PROTOCOL:-http}://${ES_HOST_NAME}:${ES_PORT:-9200}"
    fi
fi

exec "$@"
