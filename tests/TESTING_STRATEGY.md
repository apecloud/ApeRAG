# ApeRAG Testing Strategy

The test tree is now organized by intent instead of by historical implementation details.

## Main Buckets

- `tests/unit_test/`
  Fast, isolated tests for pure logic and service behavior.
- `tests/e2e_http/`
  Black-box HTTP validation for a freshly started ApeRAG deployment.
- `tests/e2e_pytest/`
  Remaining pytest-based product E2E coverage that has not yet been migrated to `tests/e2e_http/`.
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
- Moved manual model audit scripts from `tests/model_test/` to `scripts/model_test/` so they are no longer collected by pytest.

## Migration Rule

- New product-level API regression coverage should prefer `tests/e2e_http/`.
- Tests that import backend storage classes directly belong in `tests/integration/`.
- Manual provider/model audits belong under `scripts/`, not `tests/`.

## Next Cleanup Targets

- Continue migrating remaining pytest product E2E from `tests/e2e_pytest/` into `tests/e2e_http/` when the Hurl suite has equivalent coverage.
- Keep `tests/e2e_pytest/` as a compatibility bucket until each module has a clear replacement.
