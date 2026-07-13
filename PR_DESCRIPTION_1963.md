# Summary
This PR addresses issue #1963 where `PUT /api/v2/collections/{id}` returns stable `500 DATABASE_ERROR` when updating an existing collection config in apemind POC (SG) / SG evaluation environments.

## Problem
- Updating an existing `type=document` collection via `PUT /api/v2/collections/{collection_id}` consistently fails with:
  - `{"success":false,"error_code":"DATABASE_ERROR","code":1050,"message":"数据库出现错误，请稍后重试。"}`
- Retry does not recover; failure is stable.

## Reproduction
1. Create a `type=document` collection (POST path succeeds).
2. Update config with `PUT /api/v2/collections/{collection_id}`:
   - Reproduces when changing only `enable_vector` / `fulltext` / `embedding`.
   - Also reproduces when enabling knowledge graph (`enable_knowledge_graph=true`).
3. Observe stable `500 DATABASE_ERROR`.

## Impact
- Existing collections cannot be reconfigured.
- Typical operation "enable knowledge graph after collection creation" is blocked.
- Current workaround is delete + recreate collection, which is high cost and may lose built index state.

## Isolation Findings
- `POST /api/v2/collections` (CREATE): normal.
- `GET` paths: normal.
- Only `PUT` update path is failing.
- Failure is not knowledge-graph specific.

## Scope in this PR
- Triage and root-cause analysis for collection update failure in update path.
- Confirm ownership boundary between KB domain collection update service and DB layer.
- Implement and validate fix for `DATABASE_ERROR` in update flow.

## Validation Plan
- Reproduce on apemind POC (SG) with an existing collection.
- Verify PUT update succeeds for:
  - Non-graph config-only changes (`enable_vector` / `fulltext` / `embedding`).
  - Graph enablement path (`enable_knowledge_graph=true`).
- Regression check:
  - CREATE remains normal.
  - GET remains normal.
  - No regression in collection update behavior across existing test fixtures.

## Context
- Environment where issue was found: apemind POC (SG).
- Discovery date: 2026-07-01.
- Reporter: @cuiwenbo (崔文博), during `task feat: chat #17`.
- Tracking issue: #1963.
