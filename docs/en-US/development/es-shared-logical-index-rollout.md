# ES Shared Logical Index Rollout

This document captures the P1 implementation slice of the Elasticsearch redesign:

- P1-A: shared logical fulltext index
- P1-B: migration / reindex / rollout

It is the follow-up to `es-p0-contract-runtime-hardening.md`. P0 fixed runtime
correctness and explicit contracts; P1 changes the physical fulltext layout.

## Goals

1. Replace the per-collection physical fulltext index model with a shared
   logical index.
2. Make alias / versioned rebuild / cutover / rollback first-class rollout
   primitives instead of one-off manual steps.
3. Preserve per-collection correctness by keeping `collection_id` and `chat_id`
   as explicit fulltext filter fields.
4. Provide a real migration path for existing data instead of leaving rollout as
   a docs-only follow-up.

## Target Runtime Model

### Logical view

- Runtime fulltext reads and writes use the shared alias: `aperag-fulltext`
- The alias points to a concrete versioned physical index:
  - `aperag-fulltext-v1`
  - `aperag-fulltext-v2`
  - ...

### Document contract inside the shared index

Each chunk document stores explicit top-level fields:

- `collection_id`
- `document_id`
- `chunk_id`
- `chat_id`
- `name`
- `content`
- `title`

`metadata` remains a stored payload blob, not the authoritative filter path.

### Query model

- Fulltext recall is now always collection-scoped in ES itself.
- Collection filters run on top-level `collection_id`.
- Chat-scoped recall keeps the P0 dual-read guard:
  - `chat_id`
  - `metadata.chat_id`

This means P1 does not regress existing-data compatibility while the rollout is
still in flight.

## Shared Index Settings

The shared physical index no longer relies on cluster-default topology.

The creation contract is now explicit:

- `ES_FULLTEXT_NUMBER_OF_SHARDS`
- `ES_FULLTEXT_NUMBER_OF_REPLICAS`

Default values:

- `number_of_shards = 1`
- `number_of_replicas = 0`

These defaults are intentionally single-node friendly. Production deployments
can override them explicitly instead of inheriting incompatible cluster
defaults across many tiny indices.

## Routing Strategy

The shared index uses `collection_id` as the routing key for:

- chunk writes
- collection-scoped deletes
- collection-scoped search
- collection-scoped count verification
- legacy reindex migration

This keeps shared-index writes and reads collection-local inside ES without
bringing back per-collection physical index fragmentation.

## Source Of Truth And Rebuild Authority

The source-of-truth model remains unchanged:

- Object store is the source of original documents.
- Parser / chunking output is derived data.
- PostgreSQL remains the authority for ApeRAG collection/document identity.
- Elasticsearch remains a projection for BM25/fulltext retrieval.

Therefore:

- ES loss must be recoverable from the authoritative source path.
- ES migration does not redefine ownership of document data.
- Rollback means alias rollback and, if necessary, rebuild from the true source
  path, not treating ES as authoritative state.

## Rollout Script

The rollout entrypoint is:

```bash
python scripts/migrate_es_fulltext_shared_index.py
```

### Legacy -> shared migration

Dry-run the plan first:

```bash
python scripts/migrate_es_fulltext_shared_index.py --dry-run
```

Copy legacy per-collection indices into the shared physical target:

```bash
python scripts/migrate_es_fulltext_shared_index.py --target-version v1
```

Cut the shared alias after verification:

```bash
python scripts/migrate_es_fulltext_shared_index.py --target-version v1 --cutover
```

Delete old legacy indices after the new path is verified:

```bash
python scripts/migrate_es_fulltext_shared_index.py --only-delete --delete-old
```

### Versioned rebuild

Rebuild the current shared target into a new physical version:

```bash
python scripts/migrate_es_fulltext_shared_index.py --mode shared --target-version v2
```

Cut the alias to the rebuilt target:

```bash
python scripts/migrate_es_fulltext_shared_index.py --mode shared --target-version v2 --cutover
```

Rollback the alias if the cutover needs to be reverted:

```bash
python scripts/migrate_es_fulltext_shared_index.py --rollback-to aperag-fulltext-v1
```

## Verification Contract

The rollout script verifies:

- legacy source document counts
- migrated document counts inside the shared target, scoped by `collection_id`
- shared rebuild total counts for versioned alias rebuilds

Legacy migration is intentionally rerunnable:

- before reindexing a collection, the script deletes that collection's docs from
  the target physical index
- then reindexes the source collection again

This assumes a controlled rollout window where writers are paused.

## Explicit Boundaries

### Included in P1

- shared logical index alias
- versioned physical fulltext indices
- explicit shard / replica settings
- collection-based routing
- legacy per-collection -> shared migration
- alias cutover / rollback
- legacy index cleanup

### Explicitly excluded from P1

- replacing Elasticsearch with another engine
- bucketed physical index as the default layout
- automatic config-flip-driven rebuild orchestration for `enable_fulltext`

If future scale or isolation requirements prove shared is no longer economical,
bucketed physical indices remain a later conditional follow-up, not the default
P1 outcome.
