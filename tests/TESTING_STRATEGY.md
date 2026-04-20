# ApeRAG Testing Strategy

The test tree is now organized by intent instead of by historical implementation details.

## Main Buckets

- `tests/unit_test/`
  Fast, isolated tests for pure logic and service behavior.
- `tests/e2e_http/`
  Black-box HTTP validation for a freshly started ApeRAG deployment.
- `tests/e2e_pytest/`
  Only the residue that Hurl should not own yet, primarily streaming or websocket behavior.
- `tests/integration/`
  Backend contract and service-integration coverage that is not product-level HTTP E2E.

## What Changed

- Removed duplicated pytest product E2E that now overlaps with Hurl coverage:
  - `test_user.py`
  - `test_collection.py`
  - `test_document.py`
  - `test_api_key.py`
- Moved remaining pytest product E2E from `tests/e2e_test/` to `tests/e2e_pytest/`.
- Moved cache and graph storage checks into `tests/integration/`.
- Removed the old `tests/model_test/` audit scripts instead of keeping a parallel manual test bucket.

## Migration Rule

- New product-level API regression coverage should prefer `tests/e2e_http/`.
- Tests that import backend storage classes directly belong in `tests/integration/`.
- Manual provider/model audits belong under `scripts/`, not `tests/`.

## Next Cleanup Targets

- Keep converging duplicated product HTTP cases into `tests/e2e_http/hurl/full/`.
- Trim `tests/e2e_pytest/` down to the smallest useful supplement for:
  - websocket chat
  - streaming chat
- Use `tests/COVERAGE_MATRIX.md` as the source of truth before deleting additional pytest modules.
