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
   - **配置 schema path lock**（per ziang msg=c0ea4ecc 实施点 1）：`collection.config.knowledge_graph_config.graph_extraction_window_size` / `.graph_extraction_max_window_tokens`（跟 `aperag/schema/common.py:220` `knowledge_graph_config` + `graph_extractor.py:_resolve_kg_config_value()` 现有 schema 对齐，避免配置面漂移；spec 内文用 `kg.*` 简写表示同一路径）
   - `graph_extraction_window_size`（**初始 default `1` = 旧行为兼容回退**，benchmark 后再 confirm 默认值）
   - `graph_extraction_max_window_tokens`（兜底 cap，超长窗口截断或拆分）
   - **collection-level config**（每个知识库可独立配置，跟 model / 文档类型 align）
   - **`window_overlap` 移到 backlog**（per Weston msg=a29f94ab NIT 2）：第一版仅 non-overlap (overlap=0 hardcoded)，避免重复 provenance / 重复实体 / benchmark 解读复杂度；earayu2 hard requirement 是 `window_size` configurable，不是 overlap configurable，sliding overlap 是 future feature
2. **窗口边界**（per Weston msg=a29f94ab NIT 1 修正）：
   - **第一版 hard gate**：同 doc + 同 parse_version + chunk 顺序 + `max_window_tokens` 兜底
   - chapter / section 边界：仅在 chunk metadata `section_path` / `heading_anchor` **存在且连续** 时才作为 hard boundary，否则不强制（当前 parser chunk record 的 section_path 大多为 None，强制 chapter/section 边界无法稳定 enforce）
   - 不跨 doc / 不跨 parse_version 是 hard 不可破
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

#### 3.1.2 5 const co-scale（per huangheng msg=29f83d1f + ziang msg=ad7dd311 + Bryce msg=1ce25f3a concern 3）

避免「window 变 N 但 caps 仍 per-chunk」silently 降质量 + prompt size 撑爆 model context：

| 常量 | 改造 |
| --- | --- |
| `_DEFAULT_MAX_ENTITIES_PER_CHUNK = 32` | `_extract_one_window()` 调用处 `base * len(window_chunks)` 动态计算 |
| `_DEFAULT_MAX_RELATIONS_PER_CHUNK = 32` | 同上 |
| `_DEFAULT_PER_CHUNK_TIMEOUT_SECONDS = 60.0` | 第一版线性 `base * window_size`；benchmark 后看是否回退到 `base * sqrt(window_size)` 防单次过长 |
| `_BOOTSTRAP_CHUNK_COUNT = 20` | `_BOOTSTRAP_WINDOW_COUNT = max(ceil(20 / window_size), _MIN_BOOTSTRAP_WINDOW_COUNT)`，其中 `_MIN_BOOTSTRAP_WINDOW_COUNT = 1`（per Weston msg=a29f94ab NIT 3 显式命名给值，否则 boundary test 没法钉公式） — 保留 type discovery 收敛但减 serial cost |
| **`MAX_PROMPT_TOKENS` 兜底**（新增，per Bryce concern 3） | window_size × chunk_size + prompt 模板 (~500) + few-shot opt-in (+200~400) 之和不能超过 `model.max_input_tokens` 的 80%（保留 LLM 输出空间）；超过时降级 window_size 或 disable few-shot；boundary test 钉死公式 |

`window_size=1` 时所有 cap 计算结果跟旧常量字节等价（boundary test 钉死等价 + co-scale 关系不漂）。

#### 3.1.3 prompt v2 — 7 hard requirement（per Weston msg=8e155097 + Planetegg msg=b81e25cf + dongdong msg=1b148f3e + Bryce msg=1ce25f3a concern 2）

1. **输入每 chunk 用 `[[chunk_id=X index=Y]]` 边界标记**（不只用空行拼接）
2. **输出 schema 增加 `source_chunk_ids`**（每 entity / relation 必带 evidence chunk ids list）
3. **鼓励跨 chunk 关系**：A 在 chunk 1 定义、B 在 chunk 3 出现 → 抽 + 列 evidence chunks
4. **去重 / 规范化**：同实体跨 chunk 出现合并规范名，不产生别名实体
5. **fail-safe 不编造**：无文本证据不输出 / 低置信度可不输出
6. **max output 指令跟 cap × window 同步 co-scale**（避免 silently drop）
7. **`response_format=json_object` 必保留**（per Bryce msg=1ce25f3a concern 2）：task #14 issue #1861 PR #1877 已 wired 进 graph extractor 入口，prompt v2 改造（特别是加 few-shot + chunk 边界 token）必须验证 caller `aperag/indexing/llm.py:build_graph_llm_callable` 不动 `response_format` kwarg；否则 LLM 可能学习自由格式输出（few-shot 隐式诱导）或回写 markdown 围栏破坏 JSON parse

**附加 fold**（per 我 msg=cf5040b3）：
- few-shot 多样性：**default 不带 few-shot**（控 prompt size + 减 token cost），通过 collection-level `kg.graph_extraction_few_shot_locale` 可选 opt-in（值如 `zh` / `en` / `mixed`，配置后 prompt 加 1-2 个对应语言 example + 1 个跨段落示例）
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

### 4.2 默认值 lock（task #30 B3，2026-04-30）

**`graph_extraction_window_size = 2`** — 总架构师拍板甜蜜点（per earayu2 directive `msg=adb0c366`「效果稍微降低一点是可以接受的，总架构师拍板一个甜蜜点，默认至少是 2，根据性价比」+ Planetegg B2 `msg=096e0089` + Planetegg `msg=a33607aa` + Weston `msg=9ae48560` + 架构师 `msg=08ebb696` / `msg=f1feb2f1` 三方收敛）。

#### 4.2.1 B2 全矩阵数据（3 sample × 8 cell）

| model | window | calls | wall_s | cost | json_ok | source_valid | entity_hit | relation_hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 30B | 1 | 12 | 147.2 | $0.0042 | 1.000 | 1.000 | **0.930** | 0.686 |
| **Qwen3 30B** | **2** | **6** | **82.9** | **$0.0031** | **1.000** | **0.992** | **0.860** | **0.714** |
| Qwen3 30B | 3 | 6 | 102.6 | $0.0033 | 1.000 | 1.000 | 0.912 | 0.657 |
| Qwen3 30B | 5 | 3 | 75.2 | $0.0025 | 1.000 | 1.000 | 0.754 | 0.543 |
| Gemini 2.5 Flash | 1 | 12 | 57.3 | $0.0312 | 1.000 | 1.000 | **0.965** | 0.686 |
| **Gemini 2.5 Flash** | **2** | **6** | **49.6** | **$0.0260** | **1.000** | **1.000** | **0.930** | **0.714** |
| Gemini 2.5 Flash | 3 | 6 | 49.7 | $0.0226 | **0.833** ⚠️ | 1.000 | 0.667 | 0.514 |
| Gemini 2.5 Flash | 5 | 3 | 37.2 | $0.0255 | 1.000 | 1.000 | 0.947 | 0.714 |

#### 4.2.2 default=2 sweet spot rationale

1. **跨模型稳定**：Qwen + Gemini 都 `json_ok=1.0` / `source_valid≥0.992`（window=3 Gemini 1/6 json drift `0.833` ⚠️ — 不能默认）
2. **效果降低 acceptable**（per earayu2 directive）：Qwen entity -0.07 + relation +0.028 ≈ 净 -0.04 / Gemini entity -0.035 + relation +0.028 ≈ 净 -0.01；earayu2 明确「效果稍微降低可以接受」
3. **性价比显著**：calls -50% / Qwen cost -26% wall -44% / Gemini cost -17% wall -13%
4. **风险低**：跨模型一致；window=3 Gemini json drift / window=5 Qwen entity 跪 (0.754) — 都不适合默认

#### 4.2.3 collection-level override 推荐

- **保守 / 质量优先**：`window_size=1`（旧行为兼容回退，Qwen entity 0.930 baseline）
- **强模型实验**：Gemini 2.5 Flash `window_size=5`（entity 0.947 / cost $0.0255 优于 default=2，model-specific）
- **Qwen 不推荐 opt-in 大窗口**：`window_size=3` Qwen entity 略好但 relation 跌且 wall 反而更长，不性价比；`window_size=5` 质量明显跪
- **Qwen window=2 = 默认**：cost / 调用数 / 关系召回最佳平衡

#### 4.2.4 sample 限制免责

3 个 benchmark 文档不足以支撑「按模型自动改默认」(per Weston `msg=4b7f2357` + Planetegg `msg=181518f2`)。default=2 lock 是「按当前数据最稳健甜蜜点」，未来更大样本 + 多模型同时不退步证据可调。后续改默认必须满足：(a) ≥10 个样本跨语言/跨领域；(b) 至少 3 个 model 同时不退步；(c) PM + architect + earayu2 三方 confirm。

#### 4.2.5 实施改动 (B3 spec amend PR)

1. `aperag/indexing/graph_extractor.py:81` `_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE = 1` → **`2`** + docstring fold sweet spot rationale
2. `aperag/schema/common.py:167` `KnowledgeGraphConfig.graph_extraction_window_size` description 「default 1 if unset」→「default 2 if unset」+ override 推荐文案
3. `docs/zh-CN/architecture/indexing-retrieval-kg.md` 加 model × window 实验参考表
4. 本 spec § 4.2 改 lock 章节（本次 amend）+ § 5 B3 sub-task 收口

## 5. 实施 sub-task 拆分（parallel-friendly）

### Phase A（必须做，并行）

- **#30-A1**：config knob + window assembler
  - collection-level config（per ziang msg=e0812e7e NIT 同步）：
    - `collection.config.knowledge_graph_config.graph_extraction_window_size`
    - `collection.config.knowledge_graph_config.graph_extraction_max_window_tokens`
    - `window_overlap` 第一版 hardcoded `0`（移到 backlog，per Weston msg=a29f94ab NIT 2）
  - graph extractor 入口 non-overlap window builder（第一版不滑窗）
  - 同 doc + 同 parse_version + 连续 chunk + token cap 边界
  - 推荐 owner：@ziang 或 @Bryce（熟 indexing pipeline）
- **#30-A2**：5 const co-scale + boundary test（per Bryce msg=1ce25f3a concern 3 加 5th const）
  - 5 const 改 `_extract_one_window()` 调用处动态计算（max_entities / max_relations / timeout / bootstrap / max_prompt_tokens）
  - boundary test 钉「`window_size=1` window assembler / caps / timeout / bootstrap **结构等价**旧行为」+ 钉 cap × window co-scale 关系（**结构等价非字节等价**，prompt v2 schema 改造跟 task #32 evidence_refs 路径一致非回退，per huangzhangshu msg=0d497539 + Weston msg=a29f94ab + Bryce msg=1ce25f3a 三方 BLOCKER 1 修订）
  - 推荐 owner：@huangheng（boundary test lane）— 跟 task #32 A3 + cr-checklist follow-up 子 PR 同 lane
- **#30-A3**：provenance + prompt v2（**7 hard requirement**，per Bryce msg=1ce25f3a concern 2 加第 7 项 `response_format=json_object` 必保留）
  - entity / relation `source_chunk_ids` 扩 list（schema 跟 task #32 PR #1909 `GraphEvidenceRef` 同源 — composite key (`document_id, chunk_id, parse_version?`)，`chunk_id` 非全局唯一 invariant）
  - prompt v2 改造：7 hard requirement 全实施（chunk 边界 `[[chunk_id=X index=Y]]` 标记 + `source_chunk_ids` 输出 + 跨 chunk 关系鼓励 + 去重规范化 + fail-safe 不编造 + max output co-scale + `response_format=json_object` 必保留）
  - **parser invariant**（per Weston msg=a29f94ab BLOCKER 2 + ziang msg=c0ea4ecc 实施点 2）：`_parse_extraction_response` / `_entity_from_dict` / `_relation_from_dict` 接收 `allowed_chunk_ids` 参数 + `source_chunk_ids` normalize 必须是 `allowed_chunk_ids` 非空子集 + window_size=1 字段缺失 fallback 到唯一 chunk_id（兼容旧 schema）+ window_size>1 缺失或过滤后为空 → skip record + warning log
  - **few-shot default off**（per Bryce msg=1ce25f3a concern 3 防 prompt size 撑爆）：通过 collection-level `kg.graph_extraction_few_shot_locale` opt-in（值 `zh` / `en` / `mixed`，配置后加 1-2 个对应语言 example + 1 个跨段落示例）
  - 可选受控 relation schema：collection-level `kg.allowed_relation_types` 默认 free-text 兼容
  - 推荐 owner：@Bryce（task #14 issue #1861 graph extractor 改造熟悉）

### Phase B（A 全 close 后 sequential 启动，per 冬柏 msg=39e7034a）

**Phase B 不并行 A** — B1 harness 调真实 `_extract_one_window()` runtime（不 mock），必须等 A1（config + window assembler）+ A2（co-scale const）+ A3（prompt v2 + provenance）全 merge 后才启动。

- **#30-B1**：A/B benchmark harness 扩展
  - PR #1863 framework 加 `--window-sizes 1,2,3,5` batch runner（现 framework 是「1 sample × 1 model = 1 run」，B1 矩阵 `4 × ≥2 × 3 = 24+ runs` 需 batch 一层）
  - 实现 7 维度指标聚合：每 doc LLM call count / input+output token / wall time / timeout-failure rate / 实体+关系总量 / 重复率 / `source_chunk_ids` 有效率
  - **新指标 `source_chunk_ids` 有效率**：window 内每 chunk_id 都至少被 1 entity 或 relation 引用（per A3 prompt v2 输出 schema）
  - 推荐 owner：@冬柏（PR #1863 framework 熟悉，msg=39e7034a 已 claim sub-task ownership）
- **#30-B2**：benchmark 跑数据 + cost / quality 对比
  - 真实 provider 实测（OR token / Bailian），先小矩阵 (Qwen3 30B + Gemini/Claude 类) × 3 sample 跑 JSON parse 稳定后再扩
  - per-document 聚合（不看单次调用变少），按 model 分维度收集
  - 推荐 owner：@Planetegg（SRE / 真实 provider 验证，msg=ea7efa7b 已 ack 验收 3 执行细节）
- **#30-B3**：默认值 lock + spec amend
  - benchmark 数据呈现给 PM + architect + earayu2
  - 「中等偏保守的最小有效窗口」选择规则 confirm 默认值
  - amend spec + 改代码默认值（同 PR）

## 6. 验收口径

### 6.1 Phase A 完成标准（per huangzhangshu msg=0d497539 + Weston msg=a29f94ab + Bryce msg=1ce25f3a 三方 BLOCKER 1 修订）

**关键澄清**：`window_size=1` 等价**仅限于 window assembler / caps / timeout / bootstrap 公式**层面，**不要求** prompt 文本 / LLM 输出 schema 字节等价（prompt v2 是有意 schema 改造，跟 task #32 evidence_refs 链路一致非回退）：

- `window_size=1` 时**结构等价**：
  - window assembler 输出 1 个 window 仅含 1 chunk
  - `MAX_ENTITIES = base = 32` / `MAX_RELATIONS = base = 32` / `TIMEOUT = base = 60s`
  - `BOOTSTRAP_WINDOW_COUNT = max(ceil(20/1), 1) = 20` 跟 `_BOOTSTRAP_CHUNK_COUNT = 20` 等价
  - 字段缺失时 parser fallback 到唯一 chunk_id（兼容旧 schema）
- `window_size=1` 不要求 prompt 文本字节等价（prompt v2 改造 schema/provenance 是 hard scope）
- `window_size=N` 时 LLM 调用数减 ~`1/N`（per Planetegg msg=a6225720 验收口径）
- entity / relation `source_chunk_ids` 扩 list + window 内全部 chunk_id 都挂上
- prompt v2 7 hard requirement 全实施（chunk 边界 + provenance + 跨 chunk 关系 + 去重 + fail-safe + max output co-scale + response_format=json_object 保留）

### 6.2 boundary test gate（CI must pass）

- `tests/boundaries/test_graph_window_caps_co_scale.py` 钉:
  - `window_size=1` cap = base = 32 (window assembler / caps / timeout / bootstrap **结构等价**层级，不锁 prompt 文本)
  - `window_size=N` cap = base * N（线性放大公式）
  - bootstrap 公式 `_BOOTSTRAP_WINDOW_COUNT = max(ceil(20 / window_size), _MIN_BOOTSTRAP_WINDOW_COUNT)` 不漂移
- **A3 parser source_chunk_ids 验证 invariant**（per Weston msg=a29f94ab BLOCKER 2）：
  - `_parse_extraction_response` / `_entity_from_dict` / `_relation_from_dict` 接收 `allowed_chunk_ids` 参数
  - LLM 输出 `source_chunk_ids` normalize 后必须是 `allowed_chunk_ids` 的非空子集（防 LLM hallucinate window 外 chunk_id）
  - `window_size=1` 字段缺失时 fallback 到唯一 chunk_id（兼容旧 schema）
  - `window_size>1` 字段缺失或过滤后为空 → skip record（fail-safe 不编造）+ warning log
  - boundary test 覆盖：invalid id 被过滤 / missing ids 单 chunk fallback / multi chunk skip
- 现有 G1-G19 + `test_modularization_boundaries.py` + `test_worker_di_parity.py` + `test_no_rerank_in_mcp.py` 不破坏

### 6.3 Phase B 数据驱动验收（per huangzhangshu msg=107a16d5 + Planetegg msg=a6225720 + huangzhangshu msg=0d497539 BLOCKER 2 修订）

- A/B 必须比 **per-document 总实体 / 关系数量**（不是 per-chunk 平均）
- 每文档 LLM call count 降幅接近 `1/window_size`
- per-window timeout / failure rate 不显著上升
- token usage + 总耗时真下降（单次调用变长不能抵消收益）
- **provenance 验证用 composite key 路径**（不裸接 chunk_id）：
  - graph extraction 内部 provenance：entity / relation lineage 必须保留 `document_id + parse_version + source_chunk_ids`（同 task #32 PR #1909 `GraphEvidenceRef` schema 路径）
  - agent / MCP 消费验收：通过 task #32 已落地的 `evidence_refs[{document_id, chunk_id, parse_version?}]` 投影，或用 `(document_id, source_chunk_ids[i])` composite key 调 `read_document_chunk`
  - **`chunk_id` 非全局唯一**，spec 任何位置都不允许只用裸 chunk_id 喂 `read_document_chunk`（task #32 Lesson #12 v7.1 composite key invariant）
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
