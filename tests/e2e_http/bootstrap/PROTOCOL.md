# Bootstrap Protocol

`bootstrap/` is responsible for making a freshly started ApeRAG service testable.

It must not:
- launch the environment
- choose the runner
- create business resources such as collections or documents for smoke tests

It may:
- prepare a test user
- verify login works
- emit a run identifier and naming prefix
- publish file paths and basic execution metadata for the suite

## Inputs

The current bootstrap script reads these environment variables:

- `E2E_BASE_URL`
  HTTP base URL for the target ApeRAG service.
  Default: `http://127.0.0.1:8000`

- `E2E_BOOTSTRAP_MODE`
  Supported values:
  - `public-register` (default and only mode): register the first user through
    `/api/v2/auth/register`. On a fresh database the first registered user is
    automatically promoted to admin by the canonical register flow (see
    `aperag/domains/identity/api/auth_routes.py::register_view` and
    `aperag/domains/identity/service/user_manager.py::on_after_register`),
    so no dev-only API surface is needed.

  Phase 8 task #65 (G5c) removed the historical `dev-api` shortcut: it
  bypassed `on_after_register` and skipped per-user resource initialisation,
  so the canonical flow is strictly better even for tests.

- `E2E_RUN_ID`
  Optional explicit run identifier. If omitted, bootstrap generates one.

- `E2E_USERNAME`
  Optional explicit username. If omitted, bootstrap derives one from `E2E_RUN_ID`.

- `E2E_PASSWORD`
  Optional explicit password. If omitted, bootstrap derives one from `E2E_RUN_ID`.

- `E2E_EMAIL`
  Optional explicit email. If omitted, bootstrap derives one from `E2E_USERNAME`.

- `E2E_ENV_FILE`
  Output file for generated exports.
  Default: `tests/e2e_http/bootstrap/.generated/e2e.env`

## Outputs

Bootstrap writes an env file that can be sourced by runners or suite wrappers.

Current outputs:
- `E2E_BASE_URL`
- `E2E_BOOTSTRAP_MODE`
- `E2E_RUN_ID`
- `E2E_USERNAME`
- `E2E_PASSWORD`
- `E2E_EMAIL`
- `E2E_ARTIFACTS_DIR`
- `E2E_TESTDATA_DIR`

## Contract notes

- The suite consumes bootstrap outputs only through env variables.
- Smoke tests create and clean up their own business resources.
- Provider/model setup is intentionally excluded from v1 bootstrap to keep smoke mostly provider-independent.
- Bootstrap verifies only that a test user can be created and logged in; it does not create collections, documents, bots, or provider resources.
