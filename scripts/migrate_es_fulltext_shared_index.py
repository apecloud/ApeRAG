#!/usr/bin/env python3
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
"""Roll Elasticsearch fulltext storage from per-collection indices to a shared logical index.

This script covers two operational paths:

1. Initial migration:
   - Copy legacy per-collection indices into a shared physical index.
   - Verify counts per collection.
   - Cut the shared alias over once the target is ready.
   - Optionally delete old per-collection indices after verification.

2. Versioned rebuild:
   - Reindex the current shared physical target into a new versioned physical index.
   - Cut the shared alias to the new target.
   - Roll back by switching the alias back to a previous physical index if needed.

The script is deliberately idempotent for the legacy migration path: before each
collection reindex it deletes that collection's docs from the target physical
index. This assumes a controlled rollout window where writers are paused.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Set

from sqlalchemy import select

# Make sure the repo root is importable regardless of how this script is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aperag.config import get_sync_session  # noqa: E402
from aperag.db import models as db_models  # noqa: E402
from aperag.domains.indexing.fulltext_index import (  # noqa: E402
    _get_sync_es,
    count_documents,
    delete_collection_documents,
    delete_index,
    ensure_physical_index_exists,
    migrate_legacy_index,
    resolve_alias_target,
    switch_shared_index_alias,
)
from aperag.utils.utils import (  # noqa: E402
    generate_fulltext_index_alias,
    generate_fulltext_physical_index_name,
    generate_legacy_fulltext_index_name,
)

logger = logging.getLogger("migrate_es_fulltext_shared_index")


@dataclass
class LegacySourceInfo:
    index_name: str
    collection_id: str
    documents: int


def _load_aperag_collection_ids() -> Set[str]:
    ids: Set[str] = set()
    for session in get_sync_session():
        rows = session.execute(select(db_models.Collection.id)).all()
        ids.update(str(row[0]) for row in rows)
    return ids


def _list_physical_indices(es) -> Set[str]:
    indices = es.cat.indices(format="json")
    return {item["index"] for item in indices}


def _discover_legacy_sources(es, limit: int = 0, only_name: Optional[str] = None) -> List[LegacySourceInfo]:
    aperag_ids = _load_aperag_collection_ids()
    existing = _list_physical_indices(es)
    shared_alias = generate_fulltext_index_alias()
    existing.discard(shared_alias)
    current_shared_target = resolve_alias_target(shared_alias, es=es)
    if current_shared_target is not None:
        existing.discard(current_shared_target)

    source_names = sorted(existing & aperag_ids)
    if only_name is not None:
        if only_name not in source_names:
            raise ValueError(f"--only-name {only_name} is not a known legacy fulltext index")
        source_names = [only_name]
    if limit > 0:
        source_names = source_names[:limit]

    sources: List[LegacySourceInfo] = []
    for name in source_names:
        collection_id = generate_legacy_fulltext_index_name(name)
        sources.append(
            LegacySourceInfo(
                index_name=name,
                collection_id=collection_id,
                documents=count_documents(name, es=es),
            )
        )
    return sources


def _build_shared_reindex_body(source_index: str, dest_index: str) -> dict:
    return {"source": {"index": source_index}, "dest": {"index": dest_index}}


def _migrate_legacy_sources(es, sources: List[LegacySourceInfo], target_index: str, dry_run: bool) -> None:
    if not sources:
        logger.info("no legacy per-collection fulltext indices found")
        return

    ensure_physical_index_exists(physical_index=target_index, es=es)
    for idx, source in enumerate(sources, start=1):
        logger.info(
            "[%d/%d] legacy index %s -> %s (%d docs)",
            idx,
            len(sources),
            source.index_name,
            target_index,
            source.documents,
        )
        if dry_run:
            continue

        # Make reruns deterministic inside the rollout window.
        delete_collection_documents(source.collection_id, index=target_index, es=es)
        migrate_legacy_index(source.index_name, source.collection_id, dest_index=target_index, es=es)

        target_docs = count_documents(target_index, collection_id=source.collection_id, es=es)
        if target_docs != source.documents:
            raise RuntimeError(
                f"verification failed for {source.index_name}: source={source.documents}, "
                f"target(collection_id={source.collection_id})={target_docs}"
            )


def _rebuild_from_shared_alias(es, target_index: str, dry_run: bool) -> None:
    source_index = resolve_alias_target(generate_fulltext_index_alias(), es=es)
    if source_index is None:
        raise RuntimeError("shared alias does not exist yet; nothing to rebuild from")

    logger.info("rebuilding shared fulltext target %s -> %s", source_index, target_index)
    if source_index == target_index:
        logger.info("target %s already matches current shared alias target; skipping rebuild", target_index)
        return
    if dry_run:
        return

    ensure_physical_index_exists(physical_index=target_index, es=es)
    es.reindex(
        body=_build_shared_reindex_body(source_index, target_index),
        wait_for_completion=True,
        refresh=True,
        conflicts="proceed",
    )

    source_docs = count_documents(source_index, es=es)
    target_docs = count_documents(target_index, es=es)
    if target_docs != source_docs:
        raise RuntimeError(
            f"verification failed for shared rebuild: source={source_docs}, target={target_docs}, "
            f"source_index={source_index}, target_index={target_index}"
        )


def _delete_legacy_sources(sources: List[LegacySourceInfo], dry_run: bool) -> None:
    if not sources:
        logger.info("no legacy per-collection fulltext indices to delete")
        return

    for idx, source in enumerate(sources, start=1):
        logger.info("[%d/%d] deleting legacy fulltext index %s", idx, len(sources), source.index_name)
        if dry_run:
            continue
        delete_index(source.index_name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("legacy", "shared"),
        default="legacy",
        help="legacy: migrate per-collection indices; shared: rebuild current shared target into a new version",
    )
    parser.add_argument("--target-version", default="v1", help="shared physical index version, e.g. v1 / v2")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N legacy indices (0 = all)")
    parser.add_argument("--only-name", type=str, default=None, help="restrict legacy migration to one collection id")
    parser.add_argument("--cutover", action="store_true", help="switch the shared alias to the target index")
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="delete legacy per-collection indices after successful migration verification",
    )
    parser.add_argument(
        "--only-delete",
        action="store_true",
        help="skip migration and only delete legacy per-collection indices after a prior successful rollout",
    )
    parser.add_argument(
        "--rollback-to",
        type=str,
        default=None,
        help="switch the shared alias back to a specific physical index and exit",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="enable DEBUG logging")
    args = parser.parse_args(argv)

    if args.mode != "legacy" and (args.only_name is not None or args.limit > 0 or args.delete_old or args.only_delete):
        parser.error("--limit/--only-name/--delete-old/--only-delete only apply to --mode legacy")
    if args.rollback_to and args.cutover:
        parser.error("--rollback-to and --cutover are mutually exclusive")
    if args.only_delete and args.mode != "legacy":
        parser.error("--only-delete only applies to --mode legacy")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    es = _get_sync_es()
    shared_alias = generate_fulltext_index_alias()
    current_target = resolve_alias_target(shared_alias, es=es)
    target_index = generate_fulltext_physical_index_name(args.target_version)

    logger.info("shared alias: %s", shared_alias)
    logger.info("current alias target: %s", current_target or "<none>")
    logger.info("requested target index: %s", target_index)

    if args.rollback_to:
        logger.info("rolling back alias %s -> %s", shared_alias, args.rollback_to)
        if not args.dry_run:
            switch_shared_index_alias(args.rollback_to, alias=shared_alias, es=es)
        logger.info("rollback done")
        return 0

    started_at = time.time()
    sources: List[LegacySourceInfo] = []
    if args.mode == "legacy":
        sources = _discover_legacy_sources(es, limit=args.limit, only_name=args.only_name)
        logger.info("legacy migration sources: %d", len(sources))
        for source in sources:
            logger.info("  %s (%d docs)", source.index_name, source.documents)

        if not args.only_delete:
            _migrate_legacy_sources(es, sources, target_index=target_index, dry_run=args.dry_run)
    else:
        _rebuild_from_shared_alias(es, target_index=target_index, dry_run=args.dry_run)

    if args.cutover:
        logger.info("cutting alias %s -> %s", shared_alias, target_index)
        if not args.dry_run:
            switch_shared_index_alias(target_index, alias=shared_alias, es=es)

    if args.delete_old or args.only_delete:
        _delete_legacy_sources(sources, dry_run=args.dry_run)

    logger.info("done in %.1fs", time.time() - started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
