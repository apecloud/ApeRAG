# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pgvector backend for the vector-store abstraction.

Layout
------
Mirrors Qdrant's multi-tenant layout: one physical table per
``(vector_size, distance)`` pair, shared by all ApeRAG collections.
Each row carries ``tenant_id`` (the ApeRAG collection id) and every
read / write filters on it.

Physical table name: ``aperag_vectors_{size}_{distance}``, exactly the
same shape-identifier prefix Qdrant uses. This is deliberate — it means
``purge_all_shards`` logic and operational runbooks translate 1:1 across
backends.

Schema (illustrative, for size=1024, distance=cosine):

.. code-block:: sql

    CREATE TABLE IF NOT EXISTS aperag_vectors_1024_cosine (
        id          UUID PRIMARY KEY,
        tenant_id   TEXT NOT NULL,
        embedding   vector(1024) NOT NULL,
        payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_<t>_tenant ON <t> (tenant_id);
    CREATE INDEX IF NOT EXISTS idx_<t>_embedding
        ON <t> USING hnsw (embedding vector_cosine_ops)
        WITH (m=16, ef_construction=64);
    CREATE INDEX IF NOT EXISTS idx_<t>_payload ON <t> USING GIN (payload);

Extension requirement: ``CREATE EXTENSION IF NOT EXISTS vector;`` is
issued on ``ensure_collection``. The connector requires CREATE/DDL
privilege on the target database; for hosted Postgres services you may
need to run the extension enable once as a superuser.

Deployment
----------
By default the connector reuses ApeRAG's main Postgres (the
``DATABASE_URL`` that already backs the ORM) — this is the
"ApeRAG-Lite / private-delivery" topology where the full product runs on
Postgres + Redis with no separate vector service. Set
``PGVECTOR_DATABASE_URL`` to point at a dedicated Postgres when vector
volume / QPS warrants isolation.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from aperag.vectorstore.base import (
    UnsupportedFilterError,
    VectorStoreConnector,
    denormalize_threshold_to_native,
    normalize_score,
)
from aperag.vectorstore.dto import (
    QueryRequest,
    SearchHit,
    TenantRef,
    VectorPoint,
    VectorShape,
)
from aperag.vectorstore.filters import And, Eq, In, IsEmpty, Not, Or, VectorFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TENANT_COLUMN = "tenant_id"

# Shared prefix for every multi-tenant shard; same string as Qdrant so
# purge / audit tooling treats both backends identically.
_MULTITENANT_PREFIX = "aperag_vectors_"

# Canonical ``distance`` → (pgvector opclass, distance operator, "score =
# 1 - distance?" flag). The boolean matters because for cosine we report
# similarity (higher = better) while for L2 / dot we report the raw
# pgvector distance (smaller = closer, so the connector negates it to
# keep "higher-score-is-better" semantics uniform across backends).
_DISTANCE_SPEC: Dict[str, Tuple[str, str, str]] = {
    # distance -> (opclass, distance_op, score_expr_template)
    # score_expr_template is a SQL fragment that produces the hit score
    # from ``:q`` (query vector) and the stored ``embedding``. We use
    # ``CAST(:q AS vector)`` rather than ``:q::vector`` because
    # SQLAlchemy's :-placeholder parser does not always round-trip the
    # PG ``::`` cast (seen on psycopg2: ``:q::vector`` stays literal).
    "cosine": ("vector_cosine_ops", "<=>", "1 - (embedding <=> CAST(:q AS vector))"),
    "euclid": ("vector_l2_ops", "<->", "-(embedding <-> CAST(:q AS vector))"),
    "dot": ("vector_ip_ops", "<#>", "-(embedding <#> CAST(:q AS vector))"),
}

# HNSW defaults for a first-time table creation; callers can override via
# ``ctx["pgvector_hnsw_m"]`` / ``ctx["pgvector_hnsw_ef_construction"]``.
_DEFAULT_HNSW_M = 16
_DEFAULT_HNSW_EF_CONSTRUCTION = 64

# Table name safety: vector shapes must fit this pattern before we
# interpolate them into DDL. Defence in depth — shape.size/distance are
# already validated by ``VectorShape.__post_init__``, but DDL-by-string
# concatenation is unforgiving and we want a second line of defence.
_SAFE_TABLE_NAME_RE = re.compile(r"^aperag_vectors_\d+_(cosine|euclid|dot)$")


# Process-level cache of (engine, table) pairs we've already ensured.
# Mirrors the Qdrant ``_ENSURED_COLLECTIONS`` pattern and serves the same
# purpose — avoid ``CREATE IF NOT EXISTS`` DDL on every connector init.
_ENSURED_TABLES: set = set()
_ENSURE_LOCK = threading.Lock()

# Process-level engine pool keyed by DATABASE_URL. Two connectors with the
# same URL share a single Engine (and therefore a single connection pool).
_ENGINE_CACHE: Dict[str, Engine] = {}
_ENGINE_LOCK = threading.Lock()


def _get_or_create_engine(database_url: str) -> Engine:
    """Return a process-shared ``sqlalchemy.Engine`` for the given URL."""
    cached = _ENGINE_CACHE.get(database_url)
    if cached is not None:
        return cached
    with _ENGINE_LOCK:
        cached = _ENGINE_CACHE.get(database_url)
        if cached is not None:
            return cached
        # ``future=True`` for SA 2.x Core API; ``pool_pre_ping`` because we
        # share with the main ApeRAG pool semantics and outage-aware
        # behavior is a must in shared-DB mode.
        engine = create_engine(
            _to_sync_pg_url(database_url),
            pool_pre_ping=True,
            future=True,
        )
        _ENGINE_CACHE[database_url] = engine
        return engine


def _reset_engine_cache() -> None:
    """Clear the pgvector engine cache. Tests only."""
    with _ENGINE_LOCK:
        for e in list(_ENGINE_CACHE.values()):
            try:
                e.dispose()
            except Exception:
                pass
        _ENGINE_CACHE.clear()
    with _ENSURE_LOCK:
        _ENSURED_TABLES.clear()


def _to_sync_pg_url(url: str) -> str:
    """Normalize an asyncpg-style URL to a psycopg-compatible sync URL.

    ApeRAG's main config stores ``postgresql+asyncpg://...`` for the
    async engine. pgvector queries run sync; we normalise to
    ``postgresql://`` (driver default) so SQLAlchemy picks psycopg2 / 3
    via whatever is installed.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


def _table_name(shape: VectorShape) -> str:
    """Produce the physical table name for a given ``VectorShape``.

    Raises if the resulting name doesn't match the safe pattern — this
    is a hard stop before DDL string interpolation, not a Pydantic-style
    coerce-and-warn.
    """
    name = f"{_MULTITENANT_PREFIX}{int(shape.size)}_{shape.canonical}"
    if not _SAFE_TABLE_NAME_RE.match(name):
        raise ValueError(f"refusing to build unsafe pgvector table name: {name!r}")
    return name


# ---------------------------------------------------------------------------
# VectorFilter DSL -> SQL fragment + bind params
# ---------------------------------------------------------------------------


class _SqlFilter:
    """Accumulator for translating a ``VectorFilter`` tree into SQL.

    We generate **parameterised** SQL — never substitute values into the
    SQL text. Parameter names are auto-generated (``f0``, ``f1``, …) so
    nested filters never collide.
    """

    __slots__ = ("_params", "_counter")

    def __init__(self) -> None:
        self._params: Dict[str, Any] = {}
        self._counter = 0

    def translate(self, flt: Optional[VectorFilter]) -> Tuple[str, Dict[str, Any]]:
        if flt is None:
            return "", {}
        sql = self._walk(flt)
        return sql, dict(self._params)

    def _bind(self, value: Any) -> str:
        name = f"f{self._counter}"
        self._counter += 1
        # Values are JSONB-compared via ``->>`` which yields text. Stringify
        # so "42" == "42"; callers that need type-strict matching should
        # add a dedicated DSL node, not overload Eq.
        self._params[name] = _scalar_to_text(value)
        return f":{name}"

    def _walk(self, flt: VectorFilter) -> str:
        if isinstance(flt, Eq):
            return f"payload->>'{_escape_json_key(flt.key)}' = {self._bind(flt.value)}"
        if isinstance(flt, In):
            if not flt.values:
                raise ValueError(f"In filter on key {flt.key!r} has empty values list")
            placeholders = ", ".join(self._bind(v) for v in flt.values)
            return f"payload->>'{_escape_json_key(flt.key)}' IN ({placeholders})"
        if isinstance(flt, IsEmpty):
            # JSONB has no real "null" distinct from "missing" for our
            # purposes; both should match IsEmpty.
            key = _escape_json_key(flt.key)
            return f"(NOT (payload ? '{key}') OR payload->'{key}' = 'null'::jsonb)"
        if isinstance(flt, And):
            parts = [self._walk(p) for p in flt.parts]
            return "(" + " AND ".join(parts) + ")"
        if isinstance(flt, Or):
            parts = [self._walk(p) for p in flt.parts]
            return "(" + " OR ".join(parts) + ")"
        if isinstance(flt, Not):
            return f"NOT ({self._walk(flt.inner)})"
        raise UnsupportedFilterError(
            f"Unsupported VectorFilter node: {type(flt).__name__}. "
            "Add a branch in aperag.vectorstore.pgvector_connector._SqlFilter._walk"
        )


def _scalar_to_text(value: Any) -> str:
    """Normalize a scalar value to the text we compare ``payload->>'k'`` to.

    Matches Qdrant's keyword-match semantics: booleans become ``"true"``
    / ``"false"`` (lowercase, same as JSON), numbers become their decimal
    repr (matching how ``to_jsonb`` stringifies them via ``->>'k'``).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _escape_json_key(key: str) -> str:
    """Defence against a key that accidentally contains a single quote.

    We interpolate keys directly (not parameter-bind them) because
    PostgreSQL has no placeholder syntax for the JSON path accessors. The
    DSL is only ever constructed from business code using hard-coded
    keys (``indexer``, ``chat_id``, …), but leaving the escape as
    belt-and-braces makes this layer safe even if a future caller wires
    a user-supplied key in.
    """
    if "'" in key or "\\" in key:
        raise ValueError(f"refusing to interpolate unsafe JSON key: {key!r}")
    return key


def _translate_filter(flt: Optional[VectorFilter]) -> Tuple[str, Dict[str, Any]]:
    """Top-level translator entry. Returns a ``(where_fragment, params)``
    tuple; ``where_fragment`` is empty when ``flt`` is ``None``."""
    return _SqlFilter().translate(flt)


# ---------------------------------------------------------------------------
# vector encoding
# ---------------------------------------------------------------------------


def _vector_literal(vec: Sequence[float]) -> str:
    """Encode a Python vector as pgvector's text literal form.

    pgvector's SQL input is ``'[1.0, 2.0, 3.0]'::vector``; we produce the
    quoted array part and cast at bind site. Using text is a deliberate
    choice over registering ``pgvector.sqlalchemy.Vector``: we avoid
    requiring callers to install the SQLAlchemy adapter pattern and keep
    the connector usable with bare SQLAlchemy Core.
    """
    # Format floats with full precision (repr) so round-trip is lossless.
    inside = ",".join(repr(float(x)) for x in vec)
    return "[" + inside + "]"


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


class PgvectorVectorStoreConnector(VectorStoreConnector):
    """pgvector implementation of ``VectorStoreConnector``."""

    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)

        # --------- tenant (always multi-tenant; legacy mode would
        # require a dedicated table per tenant which defeats the point
        # of sharing PG with the main ApeRAG DB).
        tenant_raw = ctx.get("collection")
        if not tenant_raw:
            raise ValueError(
                "PgvectorVectorStoreConnector requires ctx['collection'] "
                "(the ApeRAG collection id used as tenant key); got empty/missing."
            )
        self._tenant = TenantRef(id=str(tenant_raw))

        # --------- shape
        self._shape = VectorShape(
            size=int(ctx.get("vector_size", 1536)),
            distance=str(ctx.get("distance", "Cosine")),
        )

        # --------- HNSW knobs
        self._hnsw_m = int(ctx.get("pgvector_hnsw_m", _DEFAULT_HNSW_M))
        self._hnsw_ef_construction = int(ctx.get("pgvector_hnsw_ef_construction", _DEFAULT_HNSW_EF_CONSTRUCTION))
        self._hnsw_ef_search_default = int(ctx.get("pgvector_hnsw_ef_search", 40))

        # --------- engine (pool-reused)
        database_url = ctx.get("pgvector_database_url") or ctx.get("database_url")
        if not database_url:
            raise ValueError(
                "PgvectorVectorStoreConnector requires ctx['pgvector_database_url'] "
                "(or a fallback ctx['database_url']); neither was set."
            )
        self._database_url = database_url
        self.engine = _get_or_create_engine(database_url)

        # --------- physical table
        self.table_name = _table_name(self._shape)

        # --------- opclass + ops
        spec = _DISTANCE_SPEC[self._shape.canonical]
        self._opclass, self._distance_op, self._score_expr = spec

        # Eagerly ensure the collection so first write / search don't
        # race with DDL.
        self.ensure_collection()

    # -------------------------------------------------------------- metadata
    @property
    def tenant(self) -> TenantRef:
        return self._tenant

    @property
    def shape(self) -> VectorShape:
        return self._shape

    @property
    def tenant_id(self) -> str:
        return self._tenant.id

    # ============================================================= ensure
    def ensure_collection(self) -> None:
        """Idempotently create table + indexes for this shape.

        DDL runs in its own transaction; repeat callers hit the process
        cache before re-issuing ``CREATE IF NOT EXISTS``.
        """
        cache_key = (self._database_url, self.table_name)
        if cache_key in _ENSURED_TABLES:
            return

        with _ENSURE_LOCK:
            if cache_key in _ENSURED_TABLES:
                return
            logger.info("pgvector: ensuring table %s on %s", self.table_name, self._database_url)
            try:
                with self.engine.begin() as conn:
                    # Extension — harmless if already installed. Requires
                    # CREATE privilege; hosted Postgres may need a
                    # superuser to do this once out-of-band.
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.execute(
                        text(
                            f"""
                            CREATE TABLE IF NOT EXISTS {self.table_name} (
                                id          UUID PRIMARY KEY,
                                tenant_id   TEXT NOT NULL,
                                embedding   vector({self._shape.size}) NOT NULL,
                                payload     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                            )
                            """
                        )
                    )
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tenant ON {self.table_name} (tenant_id)"
                        )
                    )
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding "
                            f"ON {self.table_name} USING hnsw (embedding {self._opclass}) "
                            f"WITH (m={self._hnsw_m}, ef_construction={self._hnsw_ef_construction})"
                        )
                    )
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_payload "
                            f"ON {self.table_name} USING GIN (payload)"
                        )
                    )
            except SQLAlchemyError:
                # Don't poison the cache on DDL failure; next call retries.
                logger.exception("pgvector: DDL failed for %s", self.table_name)
                raise

            _ENSURED_TABLES.add(cache_key)

    def drop_tenant(self, *, purge_all_shards: bool = False) -> None:
        """Remove all rows for this tenant.

        ``purge_all_shards=True`` scans every ``aperag_vectors_*`` table in
        the current database and deletes rows tagged with this tenant.
        Same safety net as Qdrant: used when the embedding provider has
        been removed and the connector's ``vector_size`` no longer matches
        the shard that actually holds the tenant's data.
        """
        if purge_all_shards:
            self._purge_tenant_from_all_tables()
            return

        with self.engine.begin() as conn:
            try:
                conn.execute(
                    text(f"DELETE FROM {self.table_name} WHERE tenant_id = :t"),
                    {"t": self._tenant.id},
                )
            except SQLAlchemyError as e:
                # If the table doesn't exist yet (tenant never wrote
                # anything) treat as already-deleted.
                if "does not exist" in str(e).lower() or "undefinedtable" in str(e).lower():
                    return
                raise

    def _purge_tenant_from_all_tables(self) -> None:
        """Scan every ``aperag_vectors_*`` table and delete this tenant's rows."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename LIKE :pat"),
                {"pat": f"{_MULTITENANT_PREFIX}%"},
            ).all()

            for (tbl,) in rows:
                if not _SAFE_TABLE_NAME_RE.match(tbl):
                    # Defense in depth — do not DELETE FROM a table name
                    # we didn't build ourselves, even if the prefix matches.
                    logger.warning("pgvector: skipping unsafe table name %s", tbl)
                    continue
                try:
                    conn.execute(
                        text(f"DELETE FROM {tbl} WHERE tenant_id = :t"),
                        {"t": self._tenant.id},
                    )
                    logger.info("pgvector: purged tenant %s from %s", self._tenant.id, tbl)
                except SQLAlchemyError:
                    logger.exception("pgvector: failed to purge %s from %s", self._tenant.id, tbl)

    # ============================================================= upsert
    def upsert(self, points: Sequence[VectorPoint]) -> List[str]:
        if not points:
            return []

        # Build bulk INSERT ... ON CONFLICT. We emit a single statement per
        # batch; SQLAlchemy parameter binding handles the common case of
        # "a few hundred points". For very large batches (>10k) callers
        # should chunk upstream.
        rows = []
        params: Dict[str, Any] = {}
        for idx, p in enumerate(points):
            pid = _coerce_uuid(p.id)
            payload = dict(p.payload)
            # Defense-in-depth: stamp our tenant id so a later query or
            # delete can filter by it. Identical semantics to Qdrant.
            payload["collection_id"] = self._tenant.id
            # Use ``CAST(:x AS T)`` rather than ``:x::T``: SQLAlchemy's
            # :name-placeholder parser does not always play nicely with
            # PG's ``::type`` postfix cast (seen on psycopg2, where
            # ``:emb0::vector`` gets left as literal text). CAST is
            # portable SQL and unambiguous.
            rows.append(f"(:id{idx}, :tid{idx}, CAST(:emb{idx} AS vector), CAST(:payload{idx} AS jsonb))")
            params[f"id{idx}"] = pid
            params[f"tid{idx}"] = self._tenant.id
            params[f"emb{idx}"] = _vector_literal(p.vector)
            params[f"payload{idx}"] = json.dumps(payload)

        sql = (
            f"INSERT INTO {self.table_name} (id, tenant_id, embedding, payload) "
            f"VALUES {', '.join(rows)} "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"tenant_id = EXCLUDED.tenant_id, "
            f"embedding = EXCLUDED.embedding, "
            f"payload = EXCLUDED.payload"
        )
        with self.engine.begin() as conn:
            conn.execute(text(sql), params)
        return [p.id for p in points]

    # ============================================================= search
    def search(self, request: QueryRequest) -> List[SearchHit]:
        where_parts: List[str] = ["tenant_id = :tenant"]
        params: Dict[str, Any] = {
            "tenant": self._tenant.id,
            "q": _vector_literal(request.embedding),
            "k": int(request.top_k),
        }

        flt_sql, flt_params = _translate_filter(request.flt)
        if flt_sql:
            where_parts.append(flt_sql)
            params.update(flt_params)

        score_expr = self._score_expr
        # ``request.score_threshold`` is on the normalized [0, 1] scale per
        # base.py P0-B contract. Push it down by inverting back to the
        # raw-score range so pgvector can prune via WHERE before LIMIT,
        # then re-apply the equivalent cutoff after normalization to
        # absorb any inverse-roundoff. Caller-visible behaviour is
        # identical to a Python post-filter.
        if request.score_threshold is not None:
            native_threshold = denormalize_threshold_to_native(self._shape.canonical, float(request.score_threshold))
            # ``-inf`` means "all rows pass", skip the SQL filter entirely.
            if native_threshold != float("-inf"):
                where_parts.append(f"({score_expr}) >= :score_threshold")
                params["score_threshold"] = native_threshold

        select_vector = "embedding" if request.with_vectors else "NULL AS embedding"
        sql = (
            f"SELECT id, payload, {select_vector}, ({score_expr}) AS score "
            f"FROM {self.table_name} "
            f"WHERE {' AND '.join(where_parts)} "
            f"ORDER BY embedding {self._distance_op} CAST(:q AS vector) "
            f"LIMIT :k"
        )

        # ef_search can be set per-transaction for HNSW queries. This is a
        # pgvector convention: higher values trade recall for latency.
        ef_search = (
            request.hints.get("ef_search", self._hnsw_ef_search_default)
            if request.hints
            else self._hnsw_ef_search_default
        )

        hits: List[SearchHit] = []
        with self.engine.connect() as conn:
            if ef_search:
                conn.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": int(ef_search)})
            result = conn.execute(text(sql).bindparams(**_bind_casts(params)), params)
            for row in result:
                vec = None
                if request.with_vectors and row.embedding is not None:
                    vec = _parse_vector(row.embedding)
                normalized = normalize_score(self._shape.canonical, float(row.score))
                # Belt-and-braces: if the inverse-threshold roundoff lets a
                # row through, drop it on the Python side so the contract
                # holds exactly.
                if request.score_threshold is not None and normalized < float(request.score_threshold):
                    continue
                hits.append(
                    SearchHit(
                        id=str(row.id),
                        score=normalized,
                        payload=_ensure_dict(row.payload),
                        vector=vec,
                    )
                )
        return hits

    # =========================================================== retrieve
    def retrieve(
        self,
        ids: Sequence[str],
        *,
        with_vectors: bool = False,
    ) -> List[VectorPoint]:
        if not ids:
            return []

        uuids = [_coerce_uuid(i) for i in ids]
        select_vector = "embedding" if with_vectors else "NULL AS embedding"
        sql = f"SELECT id, payload, {select_vector} FROM {self.table_name} WHERE tenant_id = :tenant AND id = ANY(:ids)"
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(sql),
                {"tenant": self._tenant.id, "ids": uuids},
            ).all()
        out: List[VectorPoint] = []
        for row in rows:
            vector = _parse_vector(row.embedding) if (with_vectors and row.embedding is not None) else []
            out.append(
                VectorPoint(
                    id=str(row.id),
                    vector=vector,
                    payload=_ensure_dict(row.payload),
                )
            )
        return out

    # ============================================================= delete
    def delete(self, ids: Sequence[str]) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        uuids = [_coerce_uuid(i) for i in ids_list]
        with self.engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {self.table_name} WHERE tenant_id = :tenant AND id = ANY(:ids)"),
                {"tenant": self._tenant.id, "ids": uuids},
            )

    def delete_by_filter(self, flt: VectorFilter) -> None:
        flt_sql, flt_params = _translate_filter(flt)
        if not flt_sql:
            raise ValueError("delete_by_filter requires a non-empty filter")

        params = dict(flt_params)
        params["tenant"] = self._tenant.id
        with self.engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {self.table_name} WHERE tenant_id = :tenant AND {flt_sql}"),
                params,
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce_uuid(s: str) -> uuid.UUID:
    """Coerce a point id to UUID; pgvector ``id`` column requires it.

    LlamaIndex generates UUIDv4 strings for node ids, and ApeRAG always
    writes those; a non-UUID input here is a bug at the caller layer and
    should crash loudly rather than silently desync.
    """
    if isinstance(s, uuid.UUID):
        return s
    return uuid.UUID(str(s))


def _ensure_dict(payload: Any) -> Dict[str, Any]:
    """SQLAlchemy may return JSONB as a dict (ideal) or a string; tolerate both."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_vector(raw: Any) -> Optional[List[float]]:
    """Convert pgvector's on-the-wire representation to ``list[float]``.

    Depending on whether ``pgvector.sqlalchemy.Vector`` is registered, the
    column comes back as a list, a numpy array, or a ``'[1.0, 2.0]'`` str
    literal. Normalise all three.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    # numpy array without importing numpy
    if hasattr(raw, "tolist"):
        try:
            return [float(x) for x in raw.tolist()]
        except Exception:
            pass
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1]
            if not inner:
                return []
            try:
                return [float(x) for x in inner.split(",")]
            except ValueError:
                return None
    return None


def _bind_casts(_params: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for per-param bind customisation. Currently unused; kept as a
    seam so future backends that need explicit ``sqlalchemy.bindparam``
    casting can thread it through without refactoring callers."""
    return {}


__all__ = [
    "PgvectorVectorStoreConnector",
    "TENANT_COLUMN",
    "_DISTANCE_SPEC",
    "_translate_filter",
    "_table_name",
    "_vector_literal",
    "_get_or_create_engine",
    "_reset_engine_cache",
    "_ENSURED_TABLES",
]
