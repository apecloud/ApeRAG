---
title: task #30 — Graph Chunk Window spec v1
description: ApeRAG graph extraction 多 chunk 合并窗口设计 + 4 const co-scale + prompt v2 改造 + A/B benchmark 默认值确定
---

# task #30 — Graph Chunk Window spec v1

> earayu2 directive (`#indexing优化` msg=e7bac0ec / msg=6c4caced / msg=622ca94d / msg=f060f1c6 / msg=9882a93d)：当前 graph index 默认按单 chunk 抽取过小；建议合并 N 个 chunk 后让 LLM extract，跟 LightRAG 原始 entity extraction context size 对齐，目标降本提速 + 提升实体 / 关系上下文完整性。默认值由 A/B benchmark 数据决定，**不允许 hardcode magic number**。

## 1. 现状 inventory（grep 实证）

### 1.1 当前 graph extraction chunk 处理逻辑

`aperag/indexing/graph_extractor.py`:
- `_extract_one_chunk()` 按**单 chunk 调 LLM**：每 chunk 独立 prompt，串行 bootstrap 前 20 + 后续 batch 并发
- 输入给 LLM 的 context 是 chunk 全文 (`text=str(chunk.get("text"))`)
- 没有任何 multi-chunk batching / sliding window / coalesce 半成品

### 1.2 当前 chunk size 配置

`aperag/config.py`:
- `chunk_size: 400` token（默认）
- `chunk_overlap_size: 20`
- vector / fulltext / graph 三个索引共享同一份 chunk 切分

### 1.3 当前 4 个 per-chunk 维度的 caps / timeout / bootstrap

`aperag/indexing/graph_extractor.py:76-99`:
- `_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32`
- `_DEFAULT_MAX_RELATIONS_PER_CHUNK = 32`
- `_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS = 60.0`
- `_BOOTSTRAP_CHUNK_COUNT = 20`（serial bootstrap for entity-type discovery）
- `_MAIN_PASS_BATCH_SIZE = 50`（main batch concurrent）

### 1.4 LightRAG 痕迹

`aperag/indexing/llm.py:145-227` `ENTITY_RELATION_EXTRACTION` prompt 是 LightRAG-style，但**没保留** LightRAG 的 `chunk_token_size` (~1200) / `entity_extract_max_gleaning` 等参数 — fork 时只继承 prompt 模板，没继承 context size 设计。

## 2. 缺口识别

### 2.1 BLOCKER：单 chunk 400 token 远小于现代 LLM context window

`aperag/indexing/graph_extractor.py:_extract_one_chunk` 单 chunk 400 token + ~500 token prompt 模板 = **每次 LLM 调用 ~1500 token / 单 chunk**。

**影响**：
- 现代 LLM（Qwen3 30B / GPT-4 / Claude）context 大（128k+），单 chunk 400 token 显著浪费
- 跨 chunk 关系（A 在 chunk 1 定义、B 在 chunk 3 出现）会被切碎，graph extraction 漏覆盖
- LLM 调用次数 = chunks 总数，cost / latency 完全线性

**根因**：fork LightRAG 时只继承 prompt 模板没继承 context size 设计；ApeRAG 早期 graph extraction 跟 vector / fulltext 共享同一份 chunk 切分（design simplicity）。

### 2.2 P1：4 const co-scale 缺失

如果直接 merge 3 chunk 不调整 caps，`_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32` / `MAX_RELATIONS_PER_CHUNK = 32` / `PER_CHUNK_TIMEOUT = 60s` / `BOOTSTRAP_CHUNK_COUNT = 20` 都会**silently 降低质量**：
- 一个 window 信息量 ≈ 3 chunk → 实体数大概率 > 32 cap → silently drop entities
- 60s timeout 对 3x 输入可能不够 → window 跳过 → 净质量降而不是升
- bootstrap 20 chunk 在 window 后 = 20 windows × 3 chunk = 60 chunk 数据，bootstrap 收敛过头浪费 LLM

（参考 huangheng msg=29f83d1f surface 此 risk + ziang msg=ad7dd311 / huangzhangshu msg=107a16d5 给具体公式）

### 2.3 P1：prompt 不支持多 chunk 边界标记

当前 `ENTITY_RELATION_EXTRACTION` prompt 单 chunk 设计：
- 输入只接受单 text 块
- 输出 entity / relation 关联到单个 chunk_id
- 没有跨 chunk 边界标记 / evidence chunk_ids 输出 / 跨 chunk 关系鼓励

如果 window 后还沿用单 chunk prompt，LLM 不知道 chunk 边界，evidence 归属糊化（例如 entity A 在 chunk 1 / B 在 chunk 3，但 LLM 输出无法区分）。

### 2.4 P2：默认值无 benchmark 依据

earayu2 msg=622ca94d 明确「3 只是拍脑袋数字，benchmark 跑数据决定」+ msg=f060f1c6「这应该是一个可配置参数」。spec 必须 lock：
- 初始代码默认值 = 1（旧行为兼容回退）
- benchmark 矩阵 1/2/3/5 × 多 model
- 数据出来后由 PM + architect + earayu2 confirm 默认值

## 3. 接口改造方向（task #30 主线）

### 3.1 必须做（Hard scope）

#### 3.1.1 5 业务边界（per #indexing优化:e7bac0ec msg=8ed5caf2）

1. **配置化**（earayu2 msg=f060f1c6 hard requirement）：
   - `kg.graph_extraction_window_size`（**初始 default `1` = 旧行为兼容回退**，benchmark 后再 confirm 默认值）
   - `kg.graph_extraction_window_overlap`（默认 `0`，non-overlap group-of-N 第一版不滑窗）
   - `kg.graph_extraction_max_window_tokens`（兜底 cap，超长窗口截断或拆分）
   - **collection-level config**（每个知识库可独立配置，跟 model / 文档类型 align）
2. **窗口边界**：
   - 仅同 doc + 同 parse_version + 连续 chunk 合并
   - 不跨 doc / 不跨 chapter / 不跨 section
   - max_window_tokens 兜底
3. **provenance 保留**：
   - entity / relation `source_chunk_ids` 扩 list（不再单 chunk_id）
   - prompt 让 LLM 在每条 entity / relation 输出关联到 source chunk_id
   - 测试：windows 内 N 个 chunk_id 都挂上 entity / relation provenance
4. **不动 vector / fulltext / chunks.jsonl schema**：
   - 仅 graph extractor 入口做 sliding / non-overlap window
   - 不改 parser / vector / fulltext / citation chunk 粒度
5. **A/B benchmark**：
   - 复用 `tests/benchmarks/graph_extraction/` (PR #1863) framework
   - 加 `--chunk-window-size N` 参数
   - 跑 1 / 2 / 3 / 5 × 多 model 对比

#### 3.1.2 4 const co-scale（per huangheng msg=29f83d1f + ziang msg=ad7dd311）

避免「window 变 N 但 caps 仍 per-chunk」silently 降质量：

| 常量 | 改造 |
| --- | --- |
| `_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32` | `_extract_one_window()` 调用处 `base * len(window_chunks)` 动态计算 |
| `_DEFAULT_MAX_RELATIONS_PER_CHUNK = 32` | 同上 |
| `_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS = 60.0` | 第一版线性 `base * window_size`；benchmark 后看是否回退到 `base * sqrt(window_size)` 防单次过长 |
| `_BOOTSTRAP_CHUNK_COUNT = 20` | `_BOOTSTRAP_WINDOW_COUNT = max(ceil(20 / window_size), MIN)` 保留 type discovery 收敛但减 serial cost |

`window_size=1` 时所有 cap 计算结果跟旧常量字节等价（boundary test 钉死等价 + co-scale 关系不漂）。

#### 3.1.3 prompt v2 — 6 hard requirement（per Weston msg=8e155097 + Planetegg msg=b81e25cf + dongdong msg=1b148f3e）

1. **输入每 chunk 用 `[[chunk_id=X index=Y]]` 边界标记**（不只用空行拼接）
2. **输出 schema 增加 `source_chunk_ids`**（每 entity / relation 必带 evidence chunk ids list）
3. **鼓励跨 chunk 关系**：A 在 chunk 1 定义、B 在 chunk 3 出现 → 抽 + 列 evidence chunks
4. **去重 / 规范化**：同实体跨 chunk 出现合并规范名，不产生别名实体
5. **fail-safe 不编造**：无文本证据不输出 / 低置信度可不输出
6. **max output 指令跟 cap × window 同步 co-scale**（避免 silently drop）

**附加 fold**（per 我 msg=cf5040b3）：
- few-shot 多样性：加 1-2 个中文 example（系统主用户中文）+ 1 个含跨段落关系的 example
- 可选受控 relation schema：collection-level `kg.allowed_relation_types` 默认 free-text 兼容；配置后 prompt 加约束减 relation_type 碎片化

**单立 backlog 不实施**（cost ×2 跟降本 directive 反向）：
- 反向 verify pass（第二轮 prompt 让 LLM 列「漏的低显著度实体」）

### 3.2 不做（YAGNI）

- 不引入跨 doc / 跨 collection 合并 window
- 不改 vector / fulltext chunk 粒度
- 不动 citation 粒度
- 不实施反向 verify pass（cost ×2）
- 不暴露 fine-grained pipeline 能力

## 4. benchmark 矩阵 + 默认值选择规则

### 4.1 benchmark 矩阵（per Planetegg msg=5de92693 + Weston msg=ed215bd7 + ziang msg=1cf6eec7）

复用 `tests/benchmarks/graph_extraction/` (PR #1863) framework，加 `--chunk-window-size` 参数:

| window_size | 用途 |
| --- | --- |
| `1` | 旧行为 baseline（必须可回退） |
| `2` | 中等偏保守候选（小模型可能更稳） |
| `3` | earayu2 直觉默认候选 |
| `5` | 上界候选（仅长 context + 结构化输出强模型考虑） |

| 维度 | 指标 |
| --- | --- |
| 性能 | 每文档 LLM 调用数 / 总 token / 总耗时 / 失败率 / timeout 率 |
| 质量 | JSON parse 率 / 实体命中率 / 关系命中率 / 重复率 / `source_chunk_ids` 有效率 |
| 模型 | 至少 2 个：默认 Qwen3 30B + Gemini/Claude 类强模型对照 |
| 样本 | 复用 PR #1863 3 真实 sample（ASF zh / ESD zh / TI en）跨语言 + 跨领域 |

### 4.2 默认值选择规则（earayu2 directive「中等偏保守」）

- **不锁 3，benchmark 数据决定**
- 优先选「中等偏保守的最小有效窗口」：
  - 如 `2` 已显著降调用 + 提质量 → 不贸然默认 `3` / `5`
  - 只有 `3` 在 JSON 稳定性 + 证据归属 + 质量 + 成本都明显优于 `2` 才默认 `3`
- `5` 仅作实验档不建议第一版默认
- **配置仍保留 collection-level override**，不同模型 / 知识库自己调
- benchmark 数据出来后 PM + architect + earayu2 confirm 最终默认值，再写进代码（**不在 spec 时点 lock**）

## 5. 实施 sub-task 拆分（parallel-friendly）

### Phase A（必须做，并行）

- **#30-A1**：config knob + window assembler
  - collection-level `kg.graph_extraction_window_size` / `window_overlap` / `max_window_tokens` config
  - graph extractor 入口 sliding / non-overlap window builder
  - 同 doc + 同 parse_version + 连续 chunk + token cap 边界
  - 推荐 owner：@ziang 或 @Bryce（熟 indexing pipeline）
- **#30-A2**：4 const co-scale + boundary test
  - 4 const 改 `_extract_one_window()` 调用处动态计算
  - boundary test 钉「`window_size=1` 字节等价旧行为」+ 钉 cap × window co-scale 关系
  - 推荐 owner：@huangheng（boundary test lane）— 跟 task #32 A3 + cr-checklist follow-up 子 PR 同 lane
- **#30-A3**：provenance + prompt v2
  - entity / relation `source_chunk_ids` 扩 list
  - prompt 模板加 `[[chunk_id=X]]` 边界标记 + 6 hard requirement + few-shot 多样性 + 可选受控 relation schema
  - 推荐 owner：@Bryce（task #14 issue #1861 graph extractor 改造熟悉）

### Phase B（A 后启动）

- **#30-B1**：A/B benchmark harness 扩展
  - PR #1863 framework 加 `--chunk-window-size` 参数
  - 多 model + 多 sample × window_size 矩阵
  - 推荐 owner：@冬柏（PR #1863 framework 熟悉）
- **#30-B2**：benchmark 跑数据 + cost / quality 对比
  - 真实 provider 实测（OR token / Bailian）
  - 7 维度指标收集
  - 推荐 owner：@Planetegg（SRE / 真实 provider 验证）
- **#30-B3**：默认值 lock + spec amend
  - benchmark 数据呈现给 PM + architect + earayu2
  - confirm 后 amend spec + 改代码默认值

## 6. 验收口径

### 6.1 Phase A 完成标准

- `window_size=1` 时所有行为字节等价旧实现（boundary test 钉死）
- `window_size=N` 时 LLM 调用数减 ~`1/N`（per Planetegg msg=a6225720 验收口径）
- entity / relation `source_chunk_ids` 扩 list + window 内全部 chunk_id 都挂上
- prompt v2 6 hard requirement 全实施（chunk 边界 + provenance + 跨 chunk 关系 + 去重 + fail-safe + max output co-scale）

### 6.2 boundary test gate（CI must pass）

- `tests/boundaries/test_graph_window_caps_co_scale.py` 钉:
  - `window_size=1` cap = `_DEFAULT_MAX_ENTITIES_PER_CHUNK` (32)
  - `window_size=N` cap = base * N（线性放大公式）
  - bootstrap 公式 `_BOOTSTRAP_WINDOW_COUNT = max(ceil(20 / window_size), MIN)` 不漂移
- 现有 G1-G19 + `test_modularization_boundaries.py` + `test_worker_di_parity.py` + `test_no_rerank_in_mcp.py` 不破坏

### 6.3 Phase B 数据驱动验收（per huangzhangshu msg=107a16d5 + Planetegg msg=a6225720）

- A/B 必须比 **per-document 总实体 / 关系数量**（不是 per-chunk 平均）
- 每文档 LLM call count 降幅接近 `1/window_size`
- per-window timeout / failure rate 不显著上升
- token usage + 总耗时真下降（单次调用变长不能抵消收益）
- `source_chunk_ids` 有效率（agent 用 `read_document_chunk` 真实消费验证）
- 多 model 分维度看：小模型最佳 window 可能 ≠ 强模型最佳

## 7. 关联文档

- task #32 MCP 审计 spec v1：[`task-32-mcp-audit-spec-v1.md`](./task-32-mcp-audit-spec-v1.md)（评估 graph window 改动后是否需要 update MCP graph tool 输出 schema）
- task #17 任务系统不变式：[`task-system-invariants.md`](./task-system-invariants.md)
- 模块化重构 canonical SSoT：[`docs/modularization/architecture.md`](../../modularization/architecture.md)
- earayu2 directive：`#indexing优化` msg=e7bac0ec (graph extract 多 chunk) + msg=6c4caced (开始讨论) + msg=622ca94d (中等偏保守 benchmark 决定) + msg=f060f1c6 (可配置参数 hard requirement) + msg=9882a93d (顺手优化 prompt)
- thread sediment：`#indexing优化:e7bac0ec` 完整 7+ 团队成员 input + msg=8ed5caf2 我 lock spec frame

## 8. CR mandatory checklist（spec 落地时必走）

按 `docs/zh-CN/architecture/task-17-cr-review-checklist.md` 既有 framework + huangheng follow-up 子 PR 待 fold 的 sediment：

- **Lesson #11 v5**（entry-point migration sub-check）— 不适用，无 process split
- **Lesson #12 v4**（PR `lint-and-unit` CI 全绿是 mandatory ratify gate）
- **Lesson #12 v5**（CI status 解读 trust framing 反模式）
- **Lesson #12 v6 / v6.1 / v6.2 / v6.3**（grep line number ≠ 执行顺序，必 walk function / API endpoint / data type scope）
- **Lesson #12 v7 + v7.1**（caller signature → backend schema → runtime fallback / composite key invariant + backend 投影 + acceptance chained chain 双层 verify）
- **Lesson #13 v2.1 + v2.2 + v3**（删 source 必删 obsolete test + 同步 update test data assertion + boundary 不重复事实保证）
- **Migration chain 时序 invariant**（如本 task 涉及 DB schema 改动 — 看 source_chunk_ids 是否需要 migration）
- **Lesson #14**（架构 invariant 删除多轮迭代收尾 — 不直接适用，但「window 单位变化必 grep 全部 per-chunk 命名常量 co-evolve」是同类工程常态）
- **简单稳定 + 私有化部署免维护 4 guardrail**（不无限扩 scope / 尽快上线 / 简单稳定优于复杂 / 免维护）

---

**起草**：@符炫炜（总架构师）
**日期**：2026-04-30
**版本**：v1（task #30 spec lock 候选；team review + earayu2 ratify 后由 PM @不穷 派单实施 Phase A）
