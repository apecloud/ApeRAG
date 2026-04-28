# Wave 10：Collection 摘要自动生成

## 1. 当前结论

每个 collection 自动生成一个「摘要」（summary）和一个「简短描述」（description），由 agent runtime 探索 collection 内容后生成。摘要是核心载荷，描述从摘要派生。

核心结论：

1. 两阶段生成：agent → 摘要（长，完整描述） → 派生 → 描述（短，UI 展示）
2. 每个用户配一个隐藏的「摘要机器人」，所有该用户的 collection 共用
3. 机器人的 13 个只读工具用 `bot.type=summary` 在 agent runtime 内 hardcoded 限制
4. 集群级 lease 防止并发 regen 同一 collection
5. 三层 fallback：Tier 1 agent runtime → Tier 2 chunks.jsonl 单 LLM call → Tier 3 transient skip
6. Reconciler 30s 周期巡检，按文档变更触发自动 regen

## 2. 要解决的问题

ApeRAG 现在 collection 描述需要用户手填，且没有自动「我这个 collection 里有什么内容」的智能摘要。希望系统自动：

1. 根据 collection 实际文档内容生成摘要
2. 用户加文档/改文档/删文档后自动重新生成
3. 用户也能手动触发重新生成
4. 私有化部署免维护，不需要用户配机器人或 prompt

## 3. Schema 改动

### 3.1 Collection 表新加 5 字段（已 ship 在 PR #1783）

| 字段 | 类型 | 说明 |
|---|---|---|
| `summary` | TEXT | 摘要（长，canonical） |
| `description` | TEXT | 描述（短，UI 展示，从 summary 派生） |
| `summary_updated_at` | TIMESTAMP | summary 上次更新时间 |
| `description_updated_at` | TIMESTAMP | description 上次更新时间 |
| `regen_lease_owner` | TEXT | 当前持有 regen lease 的 worker ID |
| `regen_lease_expires_at` | TIMESTAMP | lease 过期时间 |

### 3.2 Bot 表新加 `is_system` 列（PR #1786 amend）

```sql
ALTER TABLE bot ADD COLUMN is_system BOOLEAN NOT NULL DEFAULT FALSE;
CREATE UNIQUE INDEX uq_bot_user_type_is_system ON bot ("user", type, is_system) WHERE is_system = TRUE;
```

`is_system=TRUE` 标记系统内部 bot（用户 UI 默认过滤掉）。模仿现有 `ApiKey.is_system` 同款 pattern。

### 3.3 BotType 加 SUMMARY 成员（PR #1786 amend）

`Bot.type` 当前是 VARCHAR(50) backed by Python `BotType(str, Enum)`：

```python
class BotType(str, Enum):
    KNOWLEDGE = "knowledge"
    COMMON = "common"
    AGENT = "agent"
    SUMMARY = "summary"  # 新加
```

加新成员 = **0 DB schema 改**（VARCHAR 自然兼容）。

## 4. Bot 基础设施

### 4.1 设计决策：(c1-extend-hide) + 防御 lazy fallback

每个用户配一个隐藏的「摘要机器人」（`type=summary, is_system=TRUE`），所有该用户的 collection 共用。

- **主路径（c1）**：用户注册时同事务自动建（复用现有 register-time hook）
- **防御 fallback**：Tier 1 调用时 `get_or_create_summary_bot_for_user` 优先 get；missing 则 lazy create 自愈

### 4.2 复用现有 register hook

ApeRAG 现有 `_BotInitOpsAdapter.create_default_bot_for_user`（`aperag/app.py:171-186`）由 `UserManager.on_after_register` 触发，自动建一个 `type=AGENT` 的 default bot。

我们在这个 method 同事务多建一个 summary bot：

```python
class _BotInitOpsAdapter:
    async def create_default_bot_for_user(self, user_id: str) -> None:
        # 现有：建 default agent bot
        await bot_service.create_bot(
            user=user_id,
            bot_in=BotCreate(title="Default Agent Bot", type=BotType.AGENT, ...),
            skip_quota_check=True,
        )
        # 新加：建 summary bot
        await self._create_summary_bot_for_user(user_id)

    async def _create_summary_bot_for_user(self, user_id: str) -> None:
        await bot_service.create_bot(
            user=user_id,
            bot_in=BotCreate(
                title="__summary_gen__",
                type=BotType.SUMMARY,
                description="Internal bot for collection summary regeneration.",
                collection_ids=[],
            ),
            skip_quota_check=True,
            is_system=True,
        )
```

### 4.3 老用户 backfill

Alembic data migration 一次性给所有现有用户补建 summary bot：

```sql
INSERT INTO bot (id, "user", title, type, is_system, description, ...)
SELECT
    gen_random_uuid()::text,
    u.id,
    '__summary_gen__',
    'summary',
    TRUE,
    'Internal bot for collection summary regeneration.',
    ...
FROM "user" u
WHERE NOT EXISTS (
    SELECT 1 FROM bot b
    WHERE b."user" = u.id
      AND b.type = 'summary'
      AND b.is_system = TRUE
);
```

### 4.4 防御 lazy fallback 的必要性

`user_manager.py:137` 当前在 bot 创建失败时只 log error 不 rollback user。这意味着 register hook 偶发失败 = 用户永远没 summary bot = 该用户所有 regen 永远失败。

`get_or_create_summary_bot_for_user(user_id)` 在 Tier 1 调用时检查 + 自愈：

```python
async def get_or_create_summary_bot_for_user(user_id: str, session) -> Bot:
    bot = await session.scalar(
        select(Bot).where(
            Bot.user == user_id,
            Bot.type == BotType.SUMMARY,
            Bot.is_system == True,
        )
    )
    if bot is not None:
        return bot
    # 防御 lazy fallback（register hook 失败 edge case）
    return await _create_summary_bot_for_user(user_id, session)
```

`(user, type, is_system)` unique 索引防止并发 lazy create 写双行。

### 4.5 Tool subset 限制

不在 Bot 表加 `tool_subset` 列。改在 agent runtime 内做 hardcoded mapping：

```python
SUMMARY_BOT_ALLOWED_TOOLS = frozenset([
    "vector_search", "fulltext_search", "graph_search",
    "list_documents", "get_document", "get_collection",
    "list_chunks", "get_chunk", ...  # 共 13 个只读工具
])

def get_bot_tool_subset(bot: Bot) -> frozenset[str]:
    if bot.type == BotType.SUMMARY:
        return SUMMARY_BOT_ALLOWED_TOOLS
    return DEFAULT_TOOL_SUBSET  # 现有逻辑
```

## 5. 两阶段生成模型

### Stage 1: agent runtime → 摘要（canonical）

调用 agent runtime，让机器人自己用 13 个只读工具探索 collection，输出长摘要写入 `Collection.summary`。

调用 pattern 复用 `aperag/domains/evaluation/worker.py:114-180`：

```python
async def regen_summary(collection_id: str) -> str:
    collection = await collection_service.get(collection_id)
    user_id = collection.user
    bot = await get_or_create_summary_bot_for_user(user_id, session)

    chat = await chat_service_global.create_chat(user_id, bot.id)
    turn_request = TurnCreateRequest(
        message=SUMMARY_GEN_PROMPT.format(collection_id=collection_id),
        agent_max_turns=5,
        agent_token_limit=20000,
    )
    chat, bot, turn, _ = await turn_service.create_or_get_turn(user_id, chat.id, turn_request)
    await claim_turn(turn.id)
    await launch_turn(turn=turn, chat=chat, bot=bot, user=user_id, request=turn_request)
    # poll terminal status
    final_turn = await poll_until_terminal(turn.id)
    if final_turn.status != TurnStatus.SUCCESS:
        raise SummaryRegenFailed(...)
    answer = await uimessage_store.read(turn.id)
    return answer.text
```

### Stage 2: 派生描述

从 `summary` 用一次便宜 LLM call 派生短描述写入 `Collection.description`：

```python
async def derive_description_from_summary(collection_id: str, summary: str) -> str:
    language = await detect_language(summary)
    prompt = DESCRIPTION_DERIVE_PROMPT_ZH if language == "zh" else DESCRIPTION_DERIVE_PROMPT_EN
    response = await llm_simple_completion(prompt.format(summary=summary), max_tokens=200)
    return response.text
```

## 6. 三层 fallback chain

Stage 1 内部三层：

```python
async def regen_summary(collection_id: str) -> str | None:
    # Tier 1: agent runtime
    try:
        return await tier1_agent_regen(collection_id)
    except Exception as e:
        log.warning("Tier 1 failed: %s", e)

    # Tier 2: chunks.jsonl + 单 LLM call
    try:
        return await tier2_chunks_regen(collection_id)
    except Exception as e:
        log.warning("Tier 2 failed: %s", e)

    # Tier 3: transient skip（不写 DB，不抛异常）
    log.warning("Tier 3 transient skip for collection %s", collection_id)
    return None
```

### Tier 1 invariant

Bot 永存（register-time + lazy fallback 双保护）。如 `get_or_create_summary_bot_for_user` 仍返 None → raise `SummaryBotMissingError` 表示数据完整性 invariant violation，不被 fallback chain 吞掉。

### 质量门槛 (`is_valid_summary` / `is_valid_description`)

每个 tier 输出后做轻量 quality gate：长度 > 50 char + 无明显 LLM 拒答模板（"Sorry I cannot..."）。失败则 fallback 到下一 tier。

## 7. 集群级 lease

防止两个 reconciler worker 同时 regen 同一个 collection。

```python
async def try_acquire_regen_lease(collection_id: str, worker_id: str, ttl_sec: int = 600) -> bool:
    """原子 UPDATE: 仅当 lease 过期或为本 worker 时获取。"""
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=ttl_sec)
    result = await session.execute(
        update(Collection)
        .where(
            Collection.id == collection_id,
            or_(
                Collection.regen_lease_owner.is_(None),
                Collection.regen_lease_expires_at < now,
                Collection.regen_lease_owner == worker_id,
            )
        )
        .values(regen_lease_owner=worker_id, regen_lease_expires_at=expires_at)
    )
    return result.rowcount > 0
```

释放：

```python
async def release_regen_lease(collection_id: str, worker_id: str):
    await session.execute(
        update(Collection)
        .where(Collection.id == collection_id, Collection.regen_lease_owner == worker_id)
        .values(regen_lease_owner=None, regen_lease_expires_at=None)
    )
```

## 8. Reconciler 触发逻辑

30 秒周期巡检（Pattern B），找需要 regen 的 collection：

```python
async def find_collections_needing_regen() -> list[Collection]:
    """触发条件：summary stale 且 doc 变更累计达到阈值。"""
    return await session.scalars(
        select(Collection).where(
            # summary 比 doc 旧（add/edit/delete 任一场景都触发）
            (
                select(func.count(Document.id))
                .where(
                    Document.collection_id == Collection.id,
                    Document.gmt_updated > Collection.summary_updated_at,
                )
                .scalar_subquery() >= BULK_THRESHOLD  # 默认 10
            )
            # debounce：上次 regen 至少 60 分钟前
            & (Collection.summary_updated_at < datetime.utcnow() - timedelta(minutes=DEBOUNCE_MIN))
            # min-stale：collection 至少 10 分钟没动（避免活跃 indexing 期间反复打扰）
            & (Collection.gmt_updated < datetime.utcnow() - timedelta(minutes=MIN_STALE_MIN))
        )
    )
```

`gmt_updated > summary_updated_at` 自然 cover add/edit/delete 三场景（per mini-pattern #10）。

## 9. OpenAPI 端点

### `POST /api/v1/collections/{id}/summary/regen`

手动触发 regen summary。

- 202 Accepted + `task_id` + `estimated_completion_seconds=60`
- 404 Collection not found
- 423 Locked（lease busy）

成功 = 任务真完成 + DB 写入。**不允许返回 202 但 DB 不写**（supplementary #2 silent failure 修复）。

### `POST /api/v1/collections/{id}/description/regen`

手动触发派生 description（Stage 2）。

- 202 Accepted
- 400 Bad Request when `summary IS NULL`（必须先有 summary）
- 404 Collection not found

## 10. 测试要求

PR #1786 amend 必须覆盖：

1. **Lease atomic semantics**：并发 fixture 两个 worker 同抢，只一个成功
2. **3-tier fallback chain**：mock 各 tier 返 None / value，验证转移正确
3. **State machine**：lease busy / collection deleted / all-tiers-invalid 各路径
4. **API 400 reject**：`POST /description/regen` 当 summary 为空时返 400
5. **Quality gate**：`is_valid_summary` / `is_valid_description` 拒掉短文本和 LLM 拒答模板
6. **Trigger 三场景 coverage**：add/edit/delete doc 都触发 reconciler
7. **Bot lazy fallback**：register-time 失败模拟，Tier 1 调用自动 create
8. **Backfill migration**：在多用户 fixture 上跑，验证只补建 missing 的
9. **Silent failure 修复**：endpoint 不返回 202 但 DB 没写

## 11. 实施顺序（建议）

1. Schema 改动（Bot.is_system + unique index + Python BotType.SUMMARY）— ~30min
2. Alembic data migration backfill — ~15min
3. extend `_BotInitOpsAdapter` 加 `_create_summary_bot_for_user` — ~30min
4. `get_or_create_summary_bot_for_user` service — ~30min
5. Tier 2 chunks.jsonl + LLM call（先 ship 简单 path 验证管道）— ~1h
6. Tier 1 launch_turn 整套（移植 evaluation/worker.py:114-180）— ~3-4h
7. supplementary #1 测试 + #2 silent fix — ~2h
8. Chunk E reconciler — ~2h

**整体 ~1.2-1.3 天**

## 12. 不做（non-goals）

- 不在 Bot 表加 `tool_subset` 列（hardcoded mapping by `bot.type` 即可）
- 不让用户编辑 summary bot（隐藏，hardcoded prompt + tool subset）
- 不在第一阶段做 description 多语言切换（用户切换 collection language 是另一 wave）
- 不做 summary 历史版本（每次 regen 直接 overwrite）
- 不做用户级 quota for summary regen（debounce + min-stale 自然限流）
- 不做 reconciler dashboard / 监控 UI（Wave 10 不做）

## 13. PR hard gate

PR #1786 amend 描述里必须写清楚：

1. ✅ Schema：`Bot.is_system` + unique index + `BotType.SUMMARY` Python enum
2. ✅ Alembic data migration backfill
3. ✅ Register hook extend（`_BotInitOpsAdapter._create_summary_bot_for_user`）
4. ✅ `get_or_create_summary_bot_for_user` 含防御 lazy fallback
5. ✅ Tier 1 `launch_turn` 整套 + Tier 2 chunks.jsonl
6. ✅ 3-tier fallback chain transition tests
7. ✅ Lease atomic concurrent test
8. ✅ Silent failure 修复（不返 202 + DB 不写）
9. ✅ Reconciler 30s 周期 + 三场景 trigger coverage
10. ✅ grep 结果：`type_id` / `_DEFAULT_ENTITY_TYPES` / `tool_subset` 在 Bot 表

继续沿用 Wave 7-11 的 hard-gate 三段：12-invariant + 4-pattern + 11 mini-pattern + simple-stable 4-guardrail。

## 14. 一句话总结

每个用户在注册时被自动配一个隐藏的「摘要机器人」（type=summary, is_system=true）；用户每个 collection 通过这个机器人调 agent runtime 探索文档生成摘要写到 `Collection.summary`，再派生短描述到 `Collection.description`。30 秒周期 reconciler 自动检测文档变更触发 regen，集群级 lease 防并发，三层 fallback 抗失败，私有化部署免维护。
