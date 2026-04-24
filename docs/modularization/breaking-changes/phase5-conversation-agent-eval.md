# Phase 5 breaking changes — conversation + agent_runtime + evaluation

This document tracks every breaking change that landed with PR #1634
(Phase 5). The HTTP API surface stays byte-stable —
`scripts/export_openapi.py --check` is green on the final sweep — so
there are no API consumer changes. Everything below is at the Python
import surface for code that pulled in internals directly.

Rollback plan: `git revert` the squashed PR commit. The Phase 5 DB
schema is identical to the Phase 4 snapshot (only Python class homes
moved); `alembic check` stays empty-diff, so no migration revert is
required.

## Summary

- Three new domains at `aperag/domains/{conversation,agent_runtime,evaluation}/`
- Legacy package paths (`aperag/service/*_service.py` for six
  conversation services, `aperag/agent_runtime/*`, `aperag/evaluation_v2/*`,
  `aperag/views/{chat,bots_v2,agent_runtime,evaluation_v2}.py`, and
  `aperag/db/repositories/evaluation_v2.py`) collapse to thin
  re-export shims that preserve class / singleton identity. Phase 6
  cleanup removes the shim layer once every caller has migrated.
- Three new consumer-owned Protocols in the conversation domain
  (`QuotaOps`, `ChatCollectionServiceOps`, `AuthenticatedUser`) and
  one in the agent_runtime domain (`PromptTemplateOps`). The
  `ChatDocumentOps` Protocol seeded in 5-S1 was retired in 5-S5b
  (direct cross-domain import after the conversation domain landed).
  Phase 6 subsequently retired `ChatCollectionServiceOps` as well
  — `chat_document_service` now sibling-imports
  `chat_collection_service` directly (both live in the conversation
  domain after 5-S4f).
- One new runtime smoke test: `test_phase5_di_critical_wirings_at_app_startup`
  (G18 alt) asserts the consumer-owned Protocol DI slots are
  non-`None` after `import aperag.app`. Phase 6 shrunk the
  registry from three slots to two once `ChatCollectionServiceOps`
  was retired; the two remaining entries are
  `conversation._quota_ops` and `agent_runtime._prompt_template_ops`.
- Lesson 9a-quatuordec codified as a gate:
  `test_phase5_domain_routes_never_use_pep_563_future_annotations`
  enforces that no `aperag/domains/**/api/routes.py` file declares
  `from __future__ import annotations`.

## Import migration table

### DB symbols (Phase 5 5-S2 — all re-exported from `aperag.db.models`)

| Symbol | New canonical home |
|---|---|
| `Bot`, `BotStatus`, `BotType` | `aperag.domains.conversation.db.models` |
| `Chat`, `ChatStatus`, `ChatPeerType` | `aperag.domains.conversation.db.models` |
| `TurnFeedback`, `TurnFeedbackType`, `TurnFeedbackTag` | `aperag.domains.conversation.db.models` |
| `AgentTurn`, `AgentTurnStatus` | `aperag.domains.agent_runtime.db.models` |
| `AgentTimelineEvent`, `AgentEventActor` | `aperag.domains.agent_runtime.db.models` |
| `AgentArtifact`, `AgentArtifactType` | `aperag.domains.agent_runtime.db.models` |
| `EvaluationDataset`, `EvaluationDatasetItem`, `EvaluationDatasetSourceType` | `aperag.domains.evaluation.db.models` |
| `EvaluationRun`, `EvaluationRunStatus` | `aperag.domains.evaluation.db.models` |
| `EvaluationRunItem`, `EvaluationRunItemStatus` | `aperag.domains.evaluation.db.models` |
| `EvaluationRunItemAttempt`, `EvaluationRunItemAttemptStatus` | `aperag.domains.evaluation.db.models` |
| `EvaluationJudgeMode` | `aperag.domains.evaluation.db.models` |

### Pydantic schemas (Phase 5 5-S3 + 5-S5a — dual-hook re-exported from `aperag.schema.view_models`)

- Conversation (21): `Bot`, `BotConfig`, `BotList`, `BotCreate`, `BotUpdate`,
  `BotUpdateRequest`, `Chat`, `ChatList`, `ChatCreate`, `ChatDetails`,
  `ChatUpdate`, `ChatMessage`, `Reference`, `File`,
  `TitleGenerateRequest`, `TitleGenerateResponse`, `TurnFeedback`,
  `TurnFeedbackList`, `Feedback`, `TurnFeedbackWrite`, `Agent` —
  canonical home `aperag.domains.conversation.schemas`.
- Agent runtime (13): envelope + request + response shapes plus
  `AgentMessage` — canonical home `aperag.domains.agent_runtime.schemas`.

### Services (Phase 5 5-S4 / 5-S5 / 5-S6)

| Legacy import (re-exported) | Canonical home |
|---|---|
| `aperag.service.bot_service` | `aperag.domains.conversation.service.bot_service` |
| `aperag.service.chat_service` | `aperag.domains.conversation.service.chat_service` |
| `aperag.service.chat_document_service` | `aperag.domains.conversation.service.chat_document_service` |
| `aperag.service.chat_collection_service` | `aperag.domains.conversation.service.chat_collection_service` |
| `aperag.service.chat_title_service` | `aperag.domains.conversation.service.chat_title_service` |
| `aperag.service.turn_feedback_service` | `aperag.domains.conversation.service.turn_feedback_service` |
| `aperag.agent_runtime.runtime` | `aperag.domains.agent_runtime.runtime` |
| `aperag.agent_runtime.services` | `aperag.domains.agent_runtime.services` |
| `aperag.agent_runtime.storage` | `aperag.domains.agent_runtime.storage` |
| `aperag.agent_runtime.schemas` | `aperag.domains.agent_runtime.schemas` |
| `aperag.evaluation_v2.services` | `aperag.domains.evaluation.services` |
| `aperag.evaluation_v2.worker` | `aperag.domains.evaluation.worker` |
| `aperag.evaluation_v2.schemas` | `aperag.domains.evaluation.schemas` |
| `aperag.evaluation_v2.constants` | `aperag.domains.evaluation.constants` |
| `aperag.evaluation_v2.judges` | `aperag.domains.evaluation.judges` |
| `aperag.evaluation_v2.tasks` | `aperag.domains.evaluation.tasks` |
| `aperag.db.repositories.evaluation_v2.AsyncEvaluationV2RepositoryMixin` | `aperag.domains.evaluation.db.repositories.evaluation_v2` |

### Views / routes

| Legacy import (re-exported) | Canonical home |
|---|---|
| `aperag.views.chat.router` | `aperag.domains.conversation.api.routes.chat_router` |
| `aperag.views.bots_v2.router` | `aperag.domains.conversation.api.routes.bots_router` |
| `aperag.views.agent_runtime.router` | `aperag.domains.agent_runtime.api.routes.router` |
| `aperag.views.evaluation_v2.router` | `aperag.domains.evaluation.api.routes.router` |

## Protocols

| Protocol | Domain | Wire-up | Provider today |
|---|---|---|---|
| `AuthenticatedUser` | conversation / agent_runtime / evaluation (three separate per-domain copies) | implicit (FastAPI `Depends(required_user)`) — identity `User` row satisfies each Protocol structurally | — |
| `KnowledgeBaseCollectionView` | conversation | structural — KB `Collection` ORM satisfies it | Phase 3 knowledge_base.db.models.Collection |
| `QuotaOps` | conversation | `aperag/app.py` `_conv_set_quota_ops(_legacy_quota_service)` | `aperag.service.quota_service` (standalone-infra, permanent seam) |
| `ChatCollectionServiceOps` (**retired** in Phase 6) | conversation | — | replaced by sibling direct import (same domain) |
| `PromptTemplateOps` | agent_runtime | `aperag/app.py` `_ar_set_prompt_template_ops(_PromptTemplateOpsAdapter())` | `aperag.service.prompt_template_service` (standalone-infra, permanent seam) |
| `ChatDocumentOps` (**retired** in 5-S5b) | agent_runtime | — | replaced by direct cross-domain import |

## Lesson 9a-quatuordec (codified as a gate)

PEP 563 and FastAPI's 204 handler check are incompatible: adding
`from __future__ import annotations` to a module that defines a
`-> Response` handler at status code 204 turns the annotation into a
string, which `fastapi.routing.is_body_allowed_for_status_code(204)`
then dereferences by identity-check and rejects. Phase 3 step 5a
triggered the symptom; Phase 5 step 5-S8 adds
`test_phase5_domain_routes_never_use_pep_563_future_annotations` so
no future move can reintroduce the combination.

## `_TERMINAL_RUN_STATUSES` and #23 race fix

The Phase 5 evaluation move (5-S6) preserved the module-level
`_TERMINAL_RUN_STATUSES` frozenset at
`aperag/domains/evaluation/db/repositories/evaluation_v2.py` exactly
as PR #1631 (`54cd86b`) landed it for task #23. The race-guard logic
in `update_run_status` is byte-identical to the legacy implementation;
the move is pure file relocation. Promotion to
`EvaluationRunStatus.is_terminal()` classmethod was deferred — Phase 6
cleanup will evaluate whether to retire the module-level alias.

## Phase 6 cleanup outcome

1. **Done** — `identity_user_ops.set_chat_collection` facade replaced
   the 5-S4f raw `UPDATE users …` in
   `aperag.domains.conversation.service.chat_collection_service`
   (9a-sexdec hierarchy-1 terminal state).
2. **Done** — `ChatCollectionServiceOps` Protocol seam retired;
   `chat_document_service` now sibling-imports
   `chat_collection_service` directly.
3. **Resolved as no-op** — `PromptTemplateOps` is the permanent seam
   for `aperag.service.prompt_template_service`. The service is a
   standalone-infrastructure module (cross-cutting across runtime
   execution, conversation bot-config resolution, indexing, and a
   user-facing `/prompts` REST surface) with no natural domain home;
   the Protocol + DI seam is its canonical long-term integration
   point.
4. **Done** — `_TERMINAL_RUN_STATUSES` promoted to
   `EvaluationRunStatus.is_terminal()` classmethod; the #23 race-fix
   guard is byte-identical with PR #1631 `54cd86b`.
5. **Out of scope for this PR** — blanket legacy shim package
   deletion is a separate initiative and was deliberately not
   combined with the 5-entry cleanup sweep.
