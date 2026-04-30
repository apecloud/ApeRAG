# Compose Runner

This runner is the first convenience launcher for the HTTP E2E suite.

Responsibilities:
- start the ApeRAG environment
- wait for `GET /health/ready`
- leave execution of bootstrap and Hurl requests to other layers

Non-responsibilities:
- creating test users
- creating collections or documents
- embedding provider setup
- defining what smoke means

This runner is intentionally replaceable. The suite must remain runnable against future K8s-backed environments with the same bootstrap and Hurl files.

## Deployment shapes (`SHAPE`)

ApeRAG supports several real production deployment forms — different choices of vector backend (Qdrant or pgvector) and graph backend (PostgreSQL, Neo4j, or Nebula). Each named combination is a *shape*.

```
SHAPE=lite              ./tests/e2e_http/runners/compose/up.sh
SHAPE=qdrant-postgres   ./tests/e2e_http/runners/compose/up.sh
SHAPE=qdrant-neo4j      ./tests/e2e_http/runners/compose/up.sh
SHAPE=qdrant-nebula     ./tests/e2e_http/runners/compose/up.sh
SHAPE=pgvector-neo4j    ./tests/e2e_http/runners/compose/up.sh
SHAPE=pgvector-nebula   ./tests/e2e_http/runners/compose/up.sh
```

Each shape file under `tests/e2e_http/shapes/<shape>.env` declares:

| Variable | Purpose |
|---|---|
| `SHAPE_VECTOR_DB_TYPE` | `qdrant` or `pgvector` — written into `.env` so the api picks it up |
| `SHAPE_GRAPH_DB_TYPE` | `postgresql` / `neo4j` / `nebula` — same treatment |
| `SHAPE_COMPOSE_SERVICES` | space-separated service list passed to `docker compose up` |
| `SHAPE_COMPOSE_PROFILES` | profile flags (e.g. `--profile neo4j`) |

Adding a new shape = adding one new file. The runner / Makefile / CI all reference the shape by name, so the new combination is reachable from every entry point.

### Available shapes (today)

Shapes are named `<vector>-<graph>`, with **lite** as the special name for the single-PG (pgvector + PG-graph) ApeRAG-Lite deployment:

- **lite** — single-PG ApeRAG-Lite: pgvector + PG-graph, no qdrant/neo4j/nebula containers
- **qdrant-postgres** — distributed vector, PG graph: Qdrant + PostgreSQL graph
- **qdrant-neo4j** — distributed: Qdrant + Neo4j
- **qdrant-nebula** — distributed: Qdrant + Nebula
- **pgvector-neo4j** — mixed: pgvector + Neo4j
- **pgvector-nebula** — mixed: pgvector + Nebula

The six shapes cover the full 2 x 3 backend matrix: vector backend
(`qdrant` / `pgvector`) crossed with graph backend (`postgresql` /
`neo4j` / `nebula`). The extended mixed shapes run from manual or
backend-surface-targeted CI workflows so everyday PRs keep the original
lite + qdrant graph-backend signal cost.

### Backward compatibility

Callers that haven't migrated to `SHAPE` may still pass `VECTOR_DB_TYPE` / `GRAPH_DB_TYPE` directly; the runner derives services and profiles from those values. Defaults (no env vars at all) preserve the historical Qdrant + PG-graph behavior.
