#!/usr/bin/env python3
# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Migrate Qdrant data from the per-collection layout to the multitenant layout.

Before: one Qdrant collection per ApeRAG collection (``col<hex>``).
After:  a single global Qdrant collection per ``(vector_size, distance)``
        pair, with a ``collection_id`` payload field + is_tenant keyword index.

Intended to be run once, during a stop-the-world migration window. The script
is **idempotent**: it can be re-run safely if interrupted — we only upsert by
id, and the Qdrant delete of old collections only runs with ``--delete-old``.

Typical workflow:

    # 1. Dry run: just tell me what you would do
    python scripts/migrate_qdrant_multitenancy.py --dry-run

    # 2. Run the actual migration (writes to global collection; keeps old ones)
    python scripts/migrate_qdrant_multitenancy.py

    # 3. After verifying read paths on the new collection, delete old ones
    python scripts/migrate_qdrant_multitenancy.py --delete-old --only-delete

Behavior contract:

* For every non-empty source Qdrant collection whose name matches an ApeRAG
  ``collection.id`` in the database:
    - Resolve the vector_size (from the source collection's config).
    - Ensure the global ``aperag_vectors_{size}_{distance}`` collection exists
      with full INT8 quantization + on-disk HNSW + tenant index.
    - Scroll all points out, batch upsert them into the global collection with
      ``payload.collection_id = <ApeRAG collection id>`` injected.
* Safety rails:
    - By default we do NOT delete source collections (``--delete-old`` opts in).
    - ``--only-delete`` assumes migration already happened and simply drops the
      old per-tenant collections. This is the "commit" phase of a two-stage run.
    - ``--dry-run`` prints the plan and returns without writing.
    - ``--limit N`` restricts the migration to the first N source collections
      (useful for staged rollouts or smoke tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Make sure the repo root is importable regardless of how this script is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as rest  # noqa: E402
from qdrant_client.http.exceptions import UnexpectedResponse  # noqa: E402

from aperag.config import settings  # noqa: E402
from aperag.utils.utils import generate_vector_db_collection_name  # noqa: E402
from aperag.vectorstore.qdrant_connector import (  # noqa: E402
    TENANT_PAYLOAD_KEY,
    _coerce_distance,
    _ensure_tenant_payload_index,
    _hnsw_config,
    _optimizers_config,
    _quantization_config,
    global_collection_name,
)

# Safety assertion: the migration writes ``collection.id`` into
# ``payload.collection_id`` (the tenant key). If somebody ever changes the
# mapping function to add a prefix or hash, our migration would silently
# write the wrong tenant id. Catch that right here.
_probe = "col_migration_probe_0123456789abcdef"
assert generate_vector_db_collection_name(collection_id=_probe) == _probe, (
    "generate_vector_db_collection_name must be an identity-on-str mapping for migration safety; "
    "update this migration script if the mapping has changed."
)
del _probe

logger = logging.getLogger("migrate_qdrant")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _connect_qdrant() -> QdrantClient:
    ctx = json.loads(settings.vector_db_context)
    return QdrantClient(
        url=ctx.get("url", "http://localhost"),
        port=ctx.get("port", 6333),
        grpc_port=ctx.get("grpc_port", 6334),
        prefer_grpc=ctx.get("prefer_grpc", False),
        https=ctx.get("https", False),
        timeout=ctx.get("timeout", 300),
    )


def _load_aperag_collection_ids() -> Set[str]:
    """Return the set of ApeRAG collection ids currently known to the DB.

    We intentionally include both ACTIVE and DELETED rows. An orphan Qdrant
    collection (one that exists in Qdrant but not in the DB at all) is left
    alone by the migration — the operator decides whether to clean it up via
    the separate cleanup script.
    """
    from sqlalchemy import select

    from aperag.config import get_sync_session
    from aperag.db import models as db_models

    ids: Set[str] = set()
    for session in get_sync_session():
        rows = session.execute(select(db_models.Collection.id)).all()
        ids.update(r[0] for r in rows)
    return ids


@dataclass
class SourceInfo:
    name: str
    vector_size: int
    distance: str
    points: int


def _inspect_source_collection(client: QdrantClient, name: str) -> Optional[SourceInfo]:
    try:
        info = client.get_collection(name)
    except UnexpectedResponse:
        return None
    vectors_cfg = info.config.params.vectors
    # In our codebase every collection uses the unnamed-default vector; if a
    # named-vector config turns up, we explicitly bail so a human can decide.
    if isinstance(vectors_cfg, dict):
        logger.warning("collection %s uses named vectors; skipping (requires manual migration)", name)
        return None
    size = int(vectors_cfg.size)
    dist = vectors_cfg.distance.name if hasattr(vectors_cfg.distance, "name") else str(vectors_cfg.distance)
    points = int(getattr(info, "points_count", 0) or 0)
    return SourceInfo(name=name, vector_size=size, distance=dist, points=points)


def _ensure_global_collection(client: QdrantClient, vector_size: int, distance: str, dry_run: bool) -> str:
    dest = global_collection_name(vector_size, distance)
    if dry_run:
        logger.info("[dry-run] would ensure global collection %s", dest)
        return dest
    if not client.collection_exists(dest):
        logger.info("creating global collection %s", dest)
        cfg = {
            "quantization_enabled": True,
            "quantization_type": "int8",
            "quantization_quantile": 0.99,
            "quantization_always_ram": True,
            "hnsw_on_disk": True,
            "default_segment_number": 2,
            "mmap_threshold_kb": 20480,
        }
        client.create_collection(
            collection_name=dest,
            vectors_config=rest.VectorParams(
                size=vector_size,
                # Use the connector's coercion helper rather than Distance[x].
                # Distance member names are UPPERCASE (`COSINE`), values are
                # capitalized (`Cosine`); a naive lookup by the capitalized
                # value would KeyError on the first run (Qdrant ships the
                # capitalized form in GetCollection responses).
                distance=_coerce_distance(distance),
                on_disk=True,
            ),
            hnsw_config=_hnsw_config(cfg),
            optimizers_config=_optimizers_config(cfg),
            quantization_config=_quantization_config(cfg),
            on_disk_payload=True,
        )
    _ensure_tenant_payload_index(client, dest)
    return dest


def _migrate_one(
    client: QdrantClient,
    source: SourceInfo,
    batch_size: int,
    dry_run: bool,
) -> Tuple[int, int]:
    """Move all points from ``source`` into its global collection.

    Returns (migrated, skipped).
    """
    dest = _ensure_global_collection(client, source.vector_size, source.distance, dry_run=dry_run)
    if dry_run:
        logger.info("[dry-run] would migrate %d points: %s -> %s", source.points, source.name, dest)
        return source.points, 0

    offset = None
    total_migrated = 0
    total_skipped = 0
    while True:
        records, next_offset = client.scroll(
            collection_name=source.name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        points: List[rest.PointStruct] = []
        for rec in records:
            # Merge the tenant id into the payload. If somebody already set it
            # (vision / summary indexers do), we overwrite defensively — the
            # source collection's name is the single source of truth.
            payload = dict(rec.payload or {})
            payload[TENANT_PAYLOAD_KEY] = source.name

            vec = rec.vector
            # Scroll may return vectors as a dict for named-vector collections.
            # We ensured earlier that source uses the default vector, so this
            # should always be a plain list[float]; still guard defensively.
            if isinstance(vec, dict):
                # Single entry dict -> unwrap the only value.
                if len(vec) == 1:
                    vec = next(iter(vec.values()))
                else:
                    logger.warning("point %s in %s has multi-vector; skipping", rec.id, source.name)
                    total_skipped += 1
                    continue

            points.append(rest.PointStruct(id=rec.id, vector=vec, payload=payload))

        if points:
            client.upsert(collection_name=dest, points=points, wait=True)
            total_migrated += len(points)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("migrated %s -> %s : %d points (skipped %d)", source.name, dest, total_migrated, total_skipped)
    return total_migrated, total_skipped


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print plan without writing")
    parser.add_argument("--batch-size", type=int, default=512, help="scroll/upsert batch size")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only process the first N source collections (0 = all)",
    )
    parser.add_argument(
        "--only-name",
        type=str,
        default=None,
        help="restrict to a single source collection name (for targeted testing)",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="after successful migration, DELETE source per-collection Qdrant collections",
    )
    parser.add_argument(
        "--only-delete",
        action="store_true",
        help=(
            "skip the migrate phase, only delete per-collection Qdrant collections "
            "whose id is present in the ApeRAG DB (assumes a prior successful migration)"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable DEBUG logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    client = _connect_qdrant()
    aperag_ids = _load_aperag_collection_ids()
    logger.info("ApeRAG DB knows %d collection ids", len(aperag_ids))

    # Enumerate source collections: everything whose name matches an ApeRAG id.
    existing = {c.name for c in client.get_collections().collections}
    logger.info("Qdrant exposes %d collections", len(existing))

    source_names = sorted(existing & aperag_ids)
    if args.only_name:
        if args.only_name not in source_names:
            logger.error("--only-name %s not found among source collections", args.only_name)
            return 2
        source_names = [args.only_name]
    if args.limit > 0:
        source_names = source_names[: args.limit]
    logger.info("will process %d source collections", len(source_names))

    # Inspect each source so we know target global collections up front.
    sources: List[SourceInfo] = []
    per_size: Dict[Tuple[int, str], int] = {}
    for name in source_names:
        info = _inspect_source_collection(client, name)
        if info is None:
            logger.warning("skipping %s (could not inspect)", name)
            continue
        sources.append(info)
        per_size[(info.vector_size, info.distance)] = per_size.get((info.vector_size, info.distance), 0) + info.points

    logger.info("targeted global collections:")
    for (size, dist), pts in sorted(per_size.items()):
        logger.info(
            "  %s : %d points from %d source collections",
            global_collection_name(size, dist),
            pts,
            sum(1 for s in sources if (s.vector_size, s.distance) == (size, dist)),
        )

    if not args.only_delete:
        # Do the migration.
        t0 = time.time()
        total_points = 0
        total_skipped = 0
        for i, src in enumerate(sources, 1):
            if src.points == 0:
                logger.info(
                    "[%d/%d] %s is empty; skipping copy (tenant has nothing to migrate)", i, len(sources), src.name
                )
                continue
            logger.info(
                "[%d/%d] migrating %s (%d points, size=%d)", i, len(sources), src.name, src.points, src.vector_size
            )
            migrated, skipped = _migrate_one(client, src, batch_size=args.batch_size, dry_run=args.dry_run)
            total_points += migrated
            total_skipped += skipped
        logger.info(
            "migration phase done in %.1fs, migrated=%d, skipped=%d", time.time() - t0, total_points, total_skipped
        )

    if args.delete_old or args.only_delete:
        if args.dry_run:
            logger.info("[dry-run] would delete %d source collections", len(sources))
        else:
            for i, src in enumerate(sources, 1):
                try:
                    client.delete_collection(collection_name=src.name)
                    logger.info("[%d/%d] deleted old collection %s", i, len(sources), src.name)
                except UnexpectedResponse as e:
                    logger.warning("delete_collection(%s) failed: %s", src.name, e)

    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
