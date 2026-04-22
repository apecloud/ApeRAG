# ES P0 Contract And Runtime Hardening

This document captures the first implementation slice of the Elasticsearch redesign:

- P0-A: contract hard-fix
- P0-B: runtime hardening

It intentionally does **not** implement the P1 shared logical index redesign or the
P1 migration / reindex rollout. Those steps require separate design and rollout PRs.

## Goals

The P0 slice fixes correctness and runtime issues without introducing physical
index migration risk:

1. Make `enable_fulltext` effective in runtime behavior.
2. Stop routing fulltext search through vector naming helpers.
3. Add explicit filterable fulltext fields for `collection_id`, `document_id`,
   `chunk_id`, and `chat_id`.
4. Stop silently degrading fulltext search failures into unexplained empty recall.
5. Turn IK from an implicit startup assumption into an explicit runtime dependency.

## Scope

### Included in P0

- Fulltext index creation is skipped when `enable_fulltext=false`.
- Fulltext document indexing tasks skip cleanly when the collection disables fulltext.
- Fulltext search uses `generate_fulltext_index_name(...)`, not the vector helper.
- Fulltext chunks store explicit top-level filter fields.
- Chat-scoped fulltext search writes explicit top-level `chat_id`, but keeps a
  temporary dual-read filter on both `chat_id` and legacy `metadata.chat_id`
  until the later reindex / rollout line removes the historical path.
- Fulltext keyword extraction falls back to the raw query token when all extractors
  return nothing.
- Fulltext backend failures are logged as explicit degrade events before returning
  empty recall.
- IK installation behavior is gated by explicit environment flags.

### Explicitly excluded from P0

- Shared logical index.
- Alias / versioned rebuild / cutover.
- Physical fulltext index renaming.
- Per-collection to shared migration.
- Reindex / backfill / rollback orchestration across existing ES data.

## Compatibility Boundary

P0 keeps the current physical per-collection fulltext index model in place.
This keeps the first PR small and avoids mixing correctness fixes with data-plane
migration.

That means:

- No existing ES indices are renamed in this slice.
- No automatic reindex runs in this slice.
- Rollback remains a code rollback, not an ES data rollback.

The physical model changes only in the later P1 implementation.

P0 also does **not** make collection-config flips self-healing at the data plane:

- Turning `enable_fulltext` off stops new runtime reads and writes.
- Turning it back on does not automatically purge or rebuild existing ES
  projections.
- Fulltext projection convergence after config flips still requires an explicit
  rebuild / rollout action.

## Source Of Truth

P0 does not change the source-of-truth model:

- Object store remains the source of the original document.
- Parser / chunking remains a derived layer.
- Elasticsearch remains a projection for fulltext retrieval.

This PR must not turn ES into an authoritative data source.

## Runtime Contract For IK

IK remains the Chinese analyzer dependency in this slice, but it is now treated
as an explicit runtime dependency instead of an implicit startup side effect.

The startup contract is:

- `ES_REQUIRE_IK_PLUGIN=true|false`
- `ES_AUTO_INSTALL_IK=true|false`
- `ES_IK_PLUGIN_URL=<pinned plugin artifact>`

This allows environments to:

- fail fast when IK is required but unavailable, or
- explicitly opt into controlled bootstrap behavior.

Longer-term image baking / artifact pinning still belongs to the broader runtime
hardening line, but P0 removes the hidden dependency behavior.
