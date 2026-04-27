# ApeRAG 自动化评估功能设计文档

> 本文是 `#20` **Evaluation v3 simplification** 的技术设计纲要,对齐当前主线。
>
> 如果你要了解产品视角的使用说明,请优先阅读:
> [Evaluation 当前产品状态与使用说明](../reference/evaluation-current-guide.md)。
>
> 早期的 Benchmark / Dataset Version / Question Set 设计稿已经**作废**,保留的只是
> 为了理解历史迁移;所有实现细节以本文件 + `#20` merge 后的 main 为准。

## 1. 目标

让 Evaluation 变成一条"创建数据集 → 录入问题 → 点一下发起评测"的三步流程,删除
用户视角中的 `Benchmark / Dataset Version / Publish / Question Set / 选 Bot` 等
全部非必要概念。后端保留最小的 Dataset + Run + RunItem + Attempt 四个对象。

## 2. 非目标

- 不做 judge / scoring 最终实现(PR-C polish)。
- 不做自动 QA 生成(留 generated `source_type` 占位)。
- 不做跨 Collection 的 dataset 引用(`collection_id` 仅用作 scope/filter)。

## 3. 数据模型(与 `#1588` / `#1590` migration 一致)

### 3.1 `evaluation_datasets`

| 字段 | 说明 |
| ---- | ---- |
| `id` | PK |
| `user_id` | 所有者(硬过滤权限边界) |
| `collection_id?` | 关联 Collection 仅做 scope,不继承 sharing |
| `name` / `description` | 文本字段 |
| `source_type` | `manual` / `import` / `generated` |
| `schema_hint?` | JSON 提示 |
| `item_count` | 冗余计数,便于前端显示和 Start Run 的 gating |
| 审计字段 | `created_at / updated_at / deleted_at` |

### 3.2 `evaluation_dataset_items`

| 字段 | 说明 |
| ---- | ---- |
| `id` | PK |
| `dataset_id` | FK -> `evaluation_datasets.id`, `ON DELETE RESTRICT` |
| `case_key` | 稳定键,留空由后端生成 |
| `input_message` | 必填 |
| `expected_answer?` / `reference_context?` | 可选 |
| `tags?` / `case_metadata?` / `sort_key` | 辅助字段 |

### 3.3 `evaluation_runs`

| 字段 | 说明 |
| ---- | ---- |
| `id` / `user_id` | 基本 |
| `bot_id` | resolved bot id;显式传入或按 default-bot 解析 |
| `dataset_id` | 快照时的 dataset,不跟随 dataset rename/delete |
| `collection_id?` | 冗余过滤用 |
| `dataset_name?` | snapshot,dataset 被删仍可读 |
| `name?` | 运行名称 |
| `status` | `queued / running / completed / failed / cancelled` |
| `summary?` | JSON `{total,pending,running,completed,failed,cancelled,avg_score?}` |
| `judge_config?` / `bot_config_snapshot?` / `model_config_snapshot?` | 快照 |
| `error?` / `created_at / updated_at / started_at? / finished_at?` | 基本 |

### 3.4 `evaluation_run_items`

Run item 是 dataset item 在**创建 run 时的 value-copy 快照**。运行期间不回读
mutable `evaluation_dataset_items`。字段(节选):

- `source_dataset_item_id?`:字符串指针,**不是** FK;用来做可追溯。
- `case_key / sort_key / input_message / expected_answer? / reference_context? / tags? / case_metadata?`:全部快照。
- `status`:`pending / running / completed / failed / cancelled`。
- `best_score? / latest_attempt_id? / latest_attempt? / attempt_count / error?`:执行态。

### 3.5 `evaluation_run_item_attempts`

单次调用记录:`attempt_no`、`agent_chat_id? / agent_turn_id?` 字符串指针(不升级为 FK)、
`answer_text?`、`judge_result?`、`score?`、`latency_ms?`、`token_usage?`、`error?`、
`retry_reason?`、时间戳。

## 4. 公开 API(`/api/v2/*`,`openapi.public.json` 唯一真源)

```
GET    /api/v2/evaluation-datasets                                  ?collection_id&page&page_size
POST   /api/v2/evaluation-datasets
GET    /api/v2/evaluation-datasets/{dataset_id}
PUT    /api/v2/evaluation-datasets/{dataset_id}
DELETE /api/v2/evaluation-datasets/{dataset_id}
GET    /api/v2/evaluation-datasets/{dataset_id}/items               ?page&page_size
POST   /api/v2/evaluation-datasets/{dataset_id}/items
PUT    /api/v2/evaluation-datasets/{dataset_id}/items/{item_id}
DELETE /api/v2/evaluation-datasets/{dataset_id}/items/{item_id}

GET    /api/v2/evaluation-runs                                      ?collection_id&bot_id&dataset_id&page&page_size
POST   /api/v2/evaluation-runs
GET    /api/v2/evaluation-runs/{run_id}
GET    /api/v2/evaluation-runs/{run_id}/items                       ?page&page_size
POST   /api/v2/evaluation-runs/{run_id}/cancel
POST   /api/v2/evaluation-runs/{run_id}/items/{item_id}/retry
GET    /api/v2/evaluation-runs/{run_id}/items/{item_id}/attempts
```

`/api/v2/benchmark-datasets*` + `/versions*` 以及 `dataset_version_id` 字段已在
`#1590` destructive migration 中一次性拆掉,不再提供。

## 5. Default Bot 解析

`EvaluationRunCreate.bot_id` 可选,缺省时:

1. 选择当前用户下 `active=true` 且标题为 `Default Agent Bot` 的 bot。
2. 若上一步无结果,退到 `gmt_created ASC` 的最早 active bot。
3. 若仍无,返回可被 FE 识别的错误文案,FE 替换为"当前没有可用于评测的 Bot,请先创建 Bot 或联系管理员"(见 msg=38d7e74d UX 补丁 G)。

`DEFAULT_AGENT_BOT_TITLE` 放在服务层常量,用单测固定顺序和 active/soft-delete 条件。

## 6. Runtime(PR-1b 边界,落在 `#20 PR-1b`)

- `launch_run()` 触发 Celery task,不同步 no-op 不直接 mark completed。
- `run_evaluation_run(run_id)` 从 `evaluation_run_items` snapshot 读取,**不回读** `evaluation_dataset_items`。
- 每 item 通过 `agent_runtime.runtime.agent_runtime_manager` 派发 turn,不走 HTTP bot route。
- 写 `evaluation_run_item_attempts`,状态机 `PENDING → RUNNING → COMPLETED / FAILED`。
- 增量更新 `evaluation_runs.summary`。
- 不做 judge scoring / best_score / complex retry(PR-C polish)。
- focused test:成功 + 失败 + snapshot-only read 断言 + 不引 `dataset_version_id / benchmark_*`;mock seam 放在 `agent_runtime_manager.dispatch_turn`,不 re-assert `#13` chat persistence 层。

## 7. 前端(PR-2 本 PR 落地)

- 单入口:`/workspace/collections/{collectionId}/evaluations`,Datasets section + Runs section。
- 子入口:`/workspace/collections/{collectionId}/evaluations/datasets/{datasetId}` 管理 dataset items。
- Run 详情:
    - `/workspace/collections/{collectionId}/evaluations/{runId}`(默认入口)
    - `/workspace/bots/{botId}/evaluation/runs/{runId}`(deep link,由 trace 链接跳入)
- Bot 页 `/workspace/bots/{botId}/evaluation` 退化为**只读历史列表**,不再提供 `Create run` 入口,不再有 `dataset_version_id` / Bot 选择输入。
- FE 只消费 `/api/v2/evaluation-*` 和已经完成迁移的 collection/document typed adapter,不再触 `@/api` 老 SDK。
- typed feature adapter:`web/src/features/evaluation/{types,client-api,server-api}.ts`。
- i18n:`page_collection_evaluations` (new namespace) + 清理后的 `page_bot_evaluation`;`page_benchmarks` 整 namespace 删除,`global.ts` typed `Messages` 同步。
- `Start Evaluation` 按钮在 dataset item 数 = 0 时置灰(msg=38d7e74d 补丁 F)。

## 8. 测试

- `tests/unit_test/test_web_typed_api_contract.py::test_evaluation_feature_uses_v2_typed_api_boundary`:正向钉新路径 + 负向钉 0 条 benchmark/dataset_version_id/老 SDK。
- `tests/unit_test/test_evaluation_v2_openapi_contract.py`(PR-1 落地):OpenAPI spec 层负向钉 benchmark 路径/字段。
- `tests/e2e_http/hurl/full/16_evaluation_v2.hurl`:端到端覆盖 dataset CRUD → items append → run create(含 bot_id 显式 + default bot 两条)→ run detail → run items → cancel。
- `#20 PR-1b` 补 runtime focused pytest(PR-1b 范围)。

## 9. 迁移策略

- `#1588`(PR-0)additive foundation:新建 `evaluation_datasets / items`,不动旧 benchmark 表。
- `#1590`(PR-1)destructive switch:drop 旧 `benchmark_*` + 拆 `evaluation_runs.dataset_version_id`,切公开 API 到新路径。
- `#20 PR-1b` runtime minimal。
- `#20 PR-2` FE + docs + hurl(本 PR)。

## 10. 非 scope / 历史约束

- 不在本设计中重新规划 Question Set / Benchmark。这两个概念作为用户可见对象已在 `#1590` 移除。
- 不改 `agent_turn / turn_feedback`(是 `#13` 的 schema 域)。
- `collection_id` 只是 scope metadata,不耦合 document upload/indexing 状态机;也不回引 `/api/v1/collections*` 或旧 generated SDK。
