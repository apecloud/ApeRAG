# Evaluation 当前产品状态与使用说明(v3 简化版)

本文档面向产品、测试、交付和一线支持同学,说明 ApeRAG 当前主线里的 Evaluation 是什么、现在有哪些能力、推荐怎么使用。

注意:

- 本文以当前主线实现为准(`#20` evaluation v3 simplification 合并后)。
- 如果你同时看到 [evaluation-design](../design/evaluation-design.md) 的旧草稿,请把它理解为一份较早期的设计稿,**而不是当前产品说明**。

## 1. 现状与当前能力

当前主线里的 Evaluation 采用**两个核心对象 + 单入口**的工作流:

1. **Evaluation Dataset**:一组用于评测的问答,挂在某个 Collection 下。
2. **Evaluation Run**:对某个 Dataset 发起的一次运行;运行时按用户级默认 Bot 执行,或显式覆盖 `bot_id`。

**操作步数只有 3 步**:创建 Dataset → 录入问题 → 发起评测。

已经**消失的旧概念**(`#20` 简化移除):

- ~~Benchmark~~
- ~~Dataset Version / Publish Version~~
- ~~Question Set(独立于 Dataset 的问题集)~~
- ~~UI 上要求用户手工输入 / 复制粘贴 `dataset_version_id`~~
- ~~发起运行时在 FE 页面选择 Bot~~(改成"默认 Bot 解析 + 可覆盖",一般情况下不需要选)

### 1.1 入口在哪里

**唯一入口:Collection → Evaluations 页面**。

路径:

- `/workspace/collections/{collectionId}/evaluations`

作用:

- 整个 Evaluation 流程的起点与终点。
- 在这里创建/删除 Evaluation Dataset。
- 在这里手动录入问题(每条是一个 dataset item)。
- 在这里点"发起评测"启动一次 Evaluation Run。
- 在这里查看该 Collection 的历史 runs 和进度。

**Dataset 问题管理页面**(进入某个 dataset 后):

- `/workspace/collections/{collectionId}/evaluations/datasets/{datasetId}`

作用:

- 查看/新增/删除这个 dataset 下的问题。
- 注意:dataset item 可以继续修改,但**历史 run 通过 snapshot 保留当时的问答内容**,不会被后续编辑影响。

**Run 详情页**:

- Collection 视角:`/workspace/collections/{collectionId}/evaluations/{runId}`
- Bot 视角(只读):`/workspace/bots/{botId}/evaluation/runs/{runId}`

作用:

- 查看单次 run 的总体进度、summary 汇总。
- 查看每个 run item 的状态、分数、最近一次 attempt、trace/chat 入口。
- 对失败项执行重试;run 处于 queued/running 时可以取消整条 run。

**Bot → Evaluation 页面**(只读历史列表):

路径:

- `/workspace/bots/{botId}/evaluation`

作用:

- 只读展示这个 bot 作为评测对象的历史 runs。
- 不再承担"发起运行"的入口,也没有 `dataset_version_id` / Bot 选择输入。
- 要发起新 run,点页面上的"打开知识库评测"跳回 Collection Evaluations。

### 1.2 当前核心对象

#### Evaluation Dataset

挂在 Collection 下的一组问答。字段:

- `name` / `description`
- `collection_id`:scope 过滤用(不继承 Collection 的 sharing 权限,按 `user_id` 硬过滤)
- `source_type`:`manual` / `import` / `generated`(MVP 主走 `manual`)
- `item_count`:当前问题数

#### Evaluation Dataset Item

一条"问答"。字段:

- `case_key`:稳定标识;留空时后端自动生成
- `input_message`(必填):用户提示
- `expected_answer`:期望答案(可选)
- `reference_context`:参考上下文(可选)
- `tags` / `case_metadata` / `sort_key`:辅助字段

#### Evaluation Run

对某个 Dataset 的一次评测。字段:

- `dataset_id`(必填)
- `bot_id`:省略时后端按"默认 Bot"解析 → 标题为 `Default Agent Bot` 的 active bot 优先,否则选最早创建的 active bot
- `name`:可选运行名称
- `judge`:判分配置(可选,MVP 下判分留 TODO)
- `bot_config_snapshot` / `model_config_snapshot`:调用时的配置快照
- `status`:`queued` → `running` → `completed` / `failed` / `cancelled`
- `summary`:`total / pending / running / completed / failed / cancelled / avg_score?`
- `dataset_name`:dataset 删除或改名后,run 详情仍能显示当时的名称

#### Evaluation Run Item

一条 run item = 一个 dataset item 的快照 + 执行态。**不回读 mutable dataset items**:`input_message / expected_answer / reference_context` 在创建 run 时 value-copy 到 run item。

#### Evaluation Run Item Attempt

一次实际调用机器人的 attempt。挂在 run item 下,通过 `attempt_no` 编号,失败后重试追加。

## 2. 使用流程

1. 进入 `Collections → {你的 collection} → Evaluations`。
2. 点"创建数据集"(Create dataset),填名称。
3. 在数据集卡片上点"管理问题"(Manage questions),把要测的问题逐条加进来。
4. 回到 Evaluations 页面,点"发起评测"(Start evaluation):
    - 选择数据集(必须有至少 1 个问题)。
    - 可选:覆盖 `bot_id`。多数情况下留空,让后端选默认 Bot。
    - 可选:命名本次 run。
5. 进入 run 详情页,看进度、查看每条 item 的结果,必要时重试失败项或取消 run。

## 3. 错误和边界

- **Dataset 没有问题就点"发起评测"**:按钮会置灰,hover 上去提示"请先给数据集添加至少一个问题"。
- **当前用户没有任何可用的 Bot** 就尝试发起评测(没有 `Default Agent Bot` 也没有其它 active bot):FE 会把后端返回的 `ValidationException` 替换成用户可读的提示:"当前没有可用于评测的 Bot,请先创建 Bot 或联系管理员。"
- **Dataset 删除**:历史 run 通过 snapshot 保留;dataset 删除后仍可查看 run 的每条 item(显示 `dataset_name` 而不是挂 FK)。
- **非终态 run** (`queued / running`):run detail 页每 5 秒自动刷新,直到 run 进入终态。

## 4. API 对照

| 动作 | Method | Path |
| ---- | ------ | ---- |
| 列举 dataset | GET | `/api/v2/evaluation-datasets?collection_id=` |
| 创建 dataset | POST | `/api/v2/evaluation-datasets` |
| 更新 dataset | PUT | `/api/v2/evaluation-datasets/{dataset_id}` |
| 删除 dataset | DELETE | `/api/v2/evaluation-datasets/{dataset_id}` |
| 列举 dataset items | GET | `/api/v2/evaluation-datasets/{dataset_id}/items` |
| 追加 items | POST | `/api/v2/evaluation-datasets/{dataset_id}/items` |
| 更新单条 item | PUT | `/api/v2/evaluation-datasets/{dataset_id}/items/{item_id}` |
| 删除单条 item | DELETE | `/api/v2/evaluation-datasets/{dataset_id}/items/{item_id}` |
| 发起 run | POST | `/api/v2/evaluation-runs` |
| 列举 run | GET | `/api/v2/evaluation-runs?collection_id&bot_id&dataset_id` |
| run 详情 | GET | `/api/v2/evaluation-runs/{run_id}` |
| 列举 run items | GET | `/api/v2/evaluation-runs/{run_id}/items` |
| 取消 run | POST | `/api/v2/evaluation-runs/{run_id}/cancel` |
| 重试单条 item | POST | `/api/v2/evaluation-runs/{run_id}/items/{item_id}/retry` |
| 查询 attempts | GET | `/api/v2/evaluation-runs/{run_id}/items/{item_id}/attempts` |

旧路径 `/api/v2/benchmark-datasets*`、`/api/v2/benchmark-datasets/{id}/versions*` 以及请求/响应字段 `dataset_version_id` 都已移除。
