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
| api key | no | yes | no | no | no | fully owned by Hurl |
| available models | no | no | yes | `test_available_models.py` | no | migrate to Hurl and delete pytest duplicate |
| llm provider config and model CRUD | no | no | yes | `test_llm_provider.py` | no | migrate to Hurl and delete pytest duplicate |
| bot CRUD | no | no | yes | `test_bot.py` | no | migrate to Hurl and delete pytest duplicate |
| bot flow get/update | no | no | yes | `test_bot.py` | no | migrate to Hurl and delete pytest duplicate |
| chat create/list/get/update | no | no | yes | no | no | fully owned by Hurl |
| chat frontend non-streaming completion envelope | no | no | yes | no | no | move the frontend HTTP contract to Hurl; keep stronger success-path guarantees separate until the path is stable |
| chat openai-compatible completion | no | no | no | `test_chat.py` | no | keep pytest supplement until a stable black-box replacement is worth the churn |
| chat streaming / websocket | no | no | no | `test_chat.py` | no | keep thin pytest supplement |
| graph labels / graph overview | no | no | yes | no | `tests/integration/graphstorage/` | Hurl owns HTTP surface; integration keeps backend oracle |
| cache behavior | some | no | no | no | `tests/integration/cache/` | keep integration |
| graph storage backend correctness | no | no | no | no | `tests/integration/graphstorage/` | keep integration |

## File-Level Decisions

| File / Bucket | Decision | Reason |
| --- | --- | --- |
| `tests/e2e_pytest/test_available_models.py` | delete after this phase | replaced by `tests/e2e_http/hurl/full/10_provider_llm.hurl` |
| `tests/e2e_pytest/test_llm_provider.py` | delete after this phase | replaced by `tests/e2e_http/hurl/full/10_provider_llm.hurl` |
| `tests/e2e_pytest/test_bot.py` | delete after this phase | replaced by `tests/e2e_http/hurl/full/12_bot.hurl` |
| `tests/e2e_pytest/test_chat.py` | trim later | keep only streaming / websocket / openai-compatible supplements once Hurl parity is validated |
| `tests/e2e_pytest/test_document_download.py` | keep for now | Hurl now covers the main download path, while pytest still carries extra negative cases |
| `tests/integration/cache/*` | keep | not product-level E2E |
| `tests/integration/graphstorage/*` | keep | backend oracle and contract verification, not product HTTP behavior |
| `tests/e2e_http/hurl/smoke/*` | keep | PR gate for provider-independent deployment contract |
| `tests/e2e_http/hurl/full/*` | expand | broader product HTTP coverage that can depend on providers |

## Near-Term Cleanup Rule

Delete a pytest product E2E module only when:

1. the user-visible HTTP surface is covered in `tests/e2e_http/`
2. the new Hurl path is exercised locally and in CI
3. the remaining pytest value is only streaming / websocket / non-Hurl behavior
