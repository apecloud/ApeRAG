# ApeRAG Test Coverage Matrix

This matrix is the working contract for converging the test tree toward:

- `tests/unit_test/` for isolated logic
- `tests/e2e_http/` for black-box product HTTP coverage
- `tests/integration/` for backend contract and service-integration coverage
- a very small `tests/e2e_pytest/` residue for cases Hurl should not own yet

## Resource Coverage

| Capability | Unit | HTTP E2E Smoke | HTTP E2E Full | Remaining Pytest E2E | Integration / Contract | Current Decision |
| --- | --- | --- | --- | --- | --- | --- |
| health | no | yes | no | no | no | keep in smoke |
| auth | no | yes | no | no | no | fully owned by Hurl |
| collection CRUD | partial | yes | no | no | no | fully owned by Hurl |
| document upload/detail/delete | partial | yes | no | no | no | fully owned by Hurl |
| document staged/confirm/download/rebuild | no | no | yes | `test_document_download.py` | no | move the primary path to Hurl; keep pytest edge cases until stable |
| document status visibility and list search by name | no | no | yes | no | no | Hurl owns the stable HTTP contract for status fields and document-name search |
| collection search API and search history contract | no | no | yes | no | no | Hurl owns stable search/history HTTP behavior; result quality stays provider-sensitive |
| api key | no | yes | no | no | no | fully owned by Hurl |
| available models | no | no | yes | no | no | fully owned by Hurl |
| llm provider config and model CRUD | no | no | yes | no | no | fully owned by Hurl |
| bot CRUD | no | no | yes | no | no | fully owned by Hurl |
| bot agent config get/update | no | no | yes | no | no | fully owned by Hurl |
| chat create/list/get/update/delete | no | no | yes | no | no | fully owned by Hurl |
| chat title generation contract | no | no | yes | no | no | Hurl owns the current provider-aware chat HTTP contract that still exists after legacy frontend completion removal |
| unsupported `/v1/chat/completions` error contract | no | no | yes | no | no | Hurl asserts the stable not-implemented response; no pytest happy-path remains |
| chat streaming / websocket | no | no | no | `test_chat.py` | no | keep thin pytest supplement |
| graph labels / graph overview / parameter validation | no | no | yes | no | `tests/integration/graphstorage/` | Hurl owns stable HTTP surface; integration keeps backend oracle |
| cache behavior | some | no | no | no | `tests/integration/cache/` | keep integration |
| graph storage backend correctness | no | no | no | no | `tests/integration/graphstorage/` | keep integration |

## File-Level Decisions

| File / Bucket | Decision | Reason |
| --- | --- | --- |
| provider / bot legacy pytest modules | already removed | their HTTP coverage now lives in `tests/e2e_http/hurl/full/10_provider_llm.hurl` and `12_bot.hurl` |
| `tests/e2e_pytest/test_chat.py` | trim now | keep only streaming / websocket coverage that Hurl should not own |
| `tests/e2e_pytest/test_document_download.py` | keep for now | Hurl now covers the main download path, while pytest still carries extra negative cases |
| document status/search residual pytest | no extra pytest to delete in this phase | current delta is Hurl contract expansion rather than duplicate pytest removal |
| `tests/integration/cache/*` | keep | not product-level E2E |
| `tests/integration/graphstorage/*` | keep | backend oracle and contract verification, not product HTTP behavior |
| `tests/e2e_http/hurl/smoke/*` | keep | PR gate for provider-independent deployment contract |
| `tests/e2e_http/hurl/full/*` | expand | broader product HTTP coverage that can depend on providers |

## Near-Term Cleanup Rule

Delete a pytest product E2E module only when:

1. the user-visible HTTP surface is covered in `tests/e2e_http/`
2. the new Hurl path is exercised locally and in CI
3. the remaining pytest value is only streaming / websocket / non-Hurl behavior
