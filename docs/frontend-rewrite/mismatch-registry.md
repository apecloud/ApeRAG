# Frontend Rewrite — Mismatch Registry

**Status**: living document, continuously updated during the frontend rewrite project.
**Owner**: 符炫炜 (lead architect) — sole maintainer, all lanes contribute via PR.
**Scope**: records mismatches between the design prototype bundle at `/Users/earayu/Downloads/aperag/` and ApeRAG's current backend API surface + existing `web/` implementation.

This registry is the SSoT for "what we will NOT build" and "what we chose to interpret differently from the design prototype". Every frontend coding lane PR must self-check against Section 1 (ghost features) before merge.

---

## 0. Design DNA Lock (SSoT)

Canonical source: `/Users/earayu/Downloads/aperag/project/src/tokens.jsx`. Overrides any earlier chat-transcript DNA (e.g. the `#0165ca` blue referenced in `chats/chat1.md` is obsolete — user iterated to the warm amber palette below).

| Axis | Token | Value | Notes |
|------|-------|-------|-------|
| Surface bg | `bg` | `#FCFBF8` | warm off-white page |
| Surface card | `card` | `#FFFFFF` | |
| Surface subtle | `subtle` | `#F5F3EE` | hover / chip |
| Ink primary | `fg` | `#0A0A0A` | |
| Ink secondary | `fgSecondary` | `#3A3A38` | |
| Muted | `muted` | `#6B6A65` | |
| Border | `border` | `#EDEAE2` | bone hairline |
| Border strong | `borderStrong` | `#DDD9CF` | |
| Accent (sole) | `accent` | `#C96442` | warm amber, Claude-adjacent |
| Accent soft | `accentSoft` | `#FBEDE4` | |
| Accent ink | `accentInk` | `#7A3320` | legible on accentSoft |
| Radius base | `radius.lg` | `14px` | card default |
| Font display | `fontSerif` | Fraunces | hero / wordmark / display |
| Font UI | `fontSans` | Manrope | body / UI |
| Font mono | `fontMono` | JetBrains Mono | numbers only; no serif italics for numbers |
| Default theme | — | light | `layout.tsx` force-dark removed |

**Entity palette** (Graph page, 6-tone muted):

| Kind | Hex |
|------|-----|
| person | `#8C5A3D` |
| org | `#3F5E4F` |
| concept | `#5A4F7A` |
| doc | `#6B6A65` |
| product | `#8B5A1E` |
| event | `#7A3320` |

---

## 1. Ghost Features — Design has, Backend doesn't (wontfix this round)

Each row: design element → backend truth → decision.

| # | Design element | Backend status | Decision | Rationale |
|---|----------------|----------------|----------|-----------|
| G1 | Ingest health timeseries dashboard | Per-document `status` field only; no time-series aggregate endpoint | **Wontfix this round** | Backend only exposes point-in-time status; exposing a placeholder chart would mislead |
| G2 | Spend / budget / billing panel | `quota` endpoint only (numeric limits + current usage, no billing) | **Wontfix** | No billing integration exists; quota page keeps numeric display without $ semantics |
| G3 | Team / collaboration / sharing with permissions | Single-user model; no team CRUD | **Wontfix** | Marketplace publish/subscribe is the only sharing primitive today |
| G4 | Folder / nested collection navigation | Collections are a flat list | **Wontfix** | Keep flat list; any "folder" visual in design is cut |
| G5 | Marketplace as independent product experience | Marketplace is a filter view of "published collections" | **Re-interpret** | Design treats Marketplace as a storefront; we ship it as a filtered list sharing the Collections IA |
| G6 | Runtime tweaks panel (primary/accent/density live tweaker) | No per-user theme config endpoint | **Wontfix** | Design prototype's tweaks panel was a design-canvas affordance; not a product feature |
| G7 | Industry-toggle runtime switch (manufacturing / general) | No backend industry config | **Wontfix** | Keep as design-sample only; no UI exposure |
| G8 | Raw tool name / JSON args / JSON result surfaced in Agent trace | Backend emits envelope events with structured fields | **Soften** | UI must render natural-language action phrases; debug payloads move to a collapsible secondary view |
| G9 | "12 communities" / cluster stats on Graph header | No community-detection aggregation endpoint | **Conditional** | Render only fields the API actually returns; omit counters we cannot source |
| G10 | "TRUSTED BY TEAMS" logo row on Landing | N/A (design-only) | **Cut** | User explicitly removed (see chat1.md) |

---

## 2. Existing web/ → Design Deltas (rewrite scope)

Existing `web/` is largely complete; rewrite revises tokens + composition + copy tone, not features or routes.

| # | Area | Current state | Target state |
|---|------|---------------|--------------|
| D1 | Primary color | `#0165ca` blue in `globals.css` | Remove as default; `#C96442` accent is the sole spot color |
| D2 | Font stack | Geist Sans + Geist Mono (via `next/font/google`) | Manrope + Fraunces + JetBrains Mono (+ Noto Serif SC fallback for zh-CN) |
| D3 | Default theme | `'system'` with workspace forced-dark precedent | Default light; dark kept as opt-in |
| D4 | Card radius | `0.5rem` (8px) shadcn default | `0.875rem` (14px) matching `tokens.radius.lg` |
| D5 | Border token | oklch generic border | `#EDEAE2` bone hairline |
| D6 | shadcn Button variants | `default / destructive / outline / secondary / ghost / link` | Keep existing + add `accent` variant (accent bg, white text) |
| D7 | Badge tones | shadcn 4-variant | 6-tone alignment: ghost / accent / ok / warn / danger / ink / outline |
| D8 | Agent trace default render | Mix of tool-name + structured fields | Natural-language action chain as default; debug panel secondary |
| D9 | Graph showcase look | Dark-mode showcase exists | Warm-white palette, match main Graph; do not ship dark showcase as ref |
| D10 | Landing / topbar wordmark | Plain sans | Fraunces italic wordmark |

---

## 3. Backend has, Design didn't depict — render in new style

These features exist in the API and stay in product; the rewrite covers them in the `tokens.jsx` visual language per earayu2 principle #1.

- Prompt template CRUD (`GET/PUT /api/v2/prompts/user`)
- API keys (`GET/POST/PUT/DELETE /api/v1/apikeys`)
- Quotas page (`GET /api/v1/quotas`) — numeric only, no budget framing
- Audit logs (`GET /api/v1/audit-logs`)
- Admin users / invitations (`GET /api/v2/auth/users`, `POST /api/v2/auth/invite`)
- Provider & model configuration (`/api/v2/providers*`)
- Evaluation dataset / run CRUD (`/api/v2/evaluation-datasets*`, `/api/v2/evaluation-runs*`)
- OpenAI-compat chat (`POST /v1/chat/completions`) — dev reference; no product-facing UI change planned
- Web search / web read (`POST /api/v2/web/search`, `/api/v2/web/read`) — already surfaced via chat tool toggle
- MCP tool surface (`/mcp/*`) — documented but no UI beyond the existing chat integration

---

## 4. Fix-forward Log

Appended after each lane PR merges to track drift / late gaps discovered post-merge. Entries use:
`- YYYY-MM-DD PR #NNNN — <lane> — <what was caught / deferred> — <resolution>`

<!-- entries appended by lane owners + architect during rewrite project -->

---

## 5. Canonical Gates (echoed from `#前端重写` channel)

### L1 task #3 Design System gate
- `tokens.jsx` → `globals.css` + `web/src/lib/design-tokens.ts` consistency (no dropped fields)
- Global grep `#0165ca` / `#026ad4` → zero hits after this lane merges
- shadcn `Button` / `Card` / `AppShell` consume the same token set (no component-level hard-code overrides)
- Entity palette / accent / font exported as `const` from `design-tokens.ts` for L2 / L3 consumption

### L2 task #4 Agent Chat gate
1. Preserve `AgentTurnEnvelope` / `AgentTimelineEventEnvelope` / `AgentArtifactEnvelope` + SSE backbone — **no runtime contract change**
2. Default view: natural-language action timeline only (理解 / 搜索 / 阅读 / 比对 / 整理) — no `technical_type`, no JSON args/result, no raw tool name
3. Debug info moves to a secondary entry point (collapsible or dev toggle); does not re-pollute the main view
4. references / feedback / completed-answer hiding current behaviors not regressed
5. Styles only consume L1 tokens; no hard-coded primary color in `agent-turn-card.tsx` or peers

### L3 task #5 Graph gate
1. Preserve `react-force-graph-2d` + `/api/v2/collections/{id}/graphs` + merge-suggestions flow — no data contract change
2. Entity 6-tone palette reused from L1 `design-tokens.ts` (no hard-code in `collection-graph.tsx`)
3. Do not copy the dark `graph-showcase` palette; target warm-white from `tokens.jsx`
4. Keep existing Inspector / Legend / neighbor highlight / fullscreen behaviors
5. Human-readable mapping for `entity_type` / edge properties; no raw JSON exposure in main view

### L4 task #6 Other screens gate
- No new features; visual consistency with L1 token set; all existing API bindings preserved
- Self-check against Section 1 ghost-features list before every commit

---

## 6. Commit Message Convention

Every lane PR includes a `Ghost-check:` line confirming none of Section 1 items were implemented. Example:

```
docs(fe): L2 task #4 — Agent Chat timeline visual rewrite

Ghost-check: G1-G10 all enforced; no JSON args, no billing, no folders, no tweaks panel.
```

This keeps ghost-feature discipline auditable without a separate CI gate.
