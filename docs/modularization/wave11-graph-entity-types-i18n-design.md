# Wave 11：Graph Entity Type 动态生成与 Collection 语言

## 1. 当前结论

这次 Wave 11 采用 **hard cut + 简单枚举** 方案。

核心结论：

1. 不做老数据迁移。
2. 不建 `collection_entity_type` 新表。
3. 不做 `type_id` / `label` 分离。
4. 不保留 `_DEFAULT_ENTITY_TYPES` 作为默认兜底类型。
5. 每个 collection 的 entity type 就是一个字符串列表：`entity_types: list[str]`。
6. collection 语言固定后，graph 里新生成的 type、entity 描述、relation 描述都遵守这个语言。
7. LLM 可以根据文档内容提出新的 entity type，系统把新 type 原子合并回 collection 配置。

例子：

- 中文 collection：`["人物", "组织", "疾病", "药物"]`
- 英文 collection：`["Person", "Organization", "Disease", "Drug"]`

同一个 collection 不切换语言，所以不需要再拆成内部 ID 和展示名。

## 2. 要解决的问题

现在 graph indexing 的 entity type 基本还是预定义列表：

```python
organization, person, geo, event, product, technology, date, category
```

这有几个问题：

1. 不同 collection 的领域完全不同。医疗、法律、个人知识库不应该共用一套固定 type。
2. Prompt 现在会让 LLM 只使用给定 type，不合适就跳过，这会丢掉有效 entity。
3. Collection 已经有 `language` 配置，但 graph prompt 是否稳定遵守 collection language 还不够清楚。
4. 用户希望系统少维护，最好让 type 随真实文档自然出现，而不是人工预先设计 ontology。

Wave 11 的目标是把这些规则一次钉死。

## 3. 当前代码入口

主要涉及这些文件：

- `aperag/schema/common.py`
  - 已有 `CollectionConfig.language`
  - 已有 `KnowledgeGraphConfig.entity_types`
- `aperag/indexing/graph_extractor.py`
  - 负责解析 collection config
  - 负责调用 LLM 和解析 graph extraction 结果
- `aperag/indexing/llm.py`
  - graph extraction prompt 在这里
- `aperag/indexing/graph.py`
  - `EntityRecord.entity_type` 是最终写入 graph 的字符串字段
- `aperag/indexing/graph_storage/*`
  - Postgres / Neo4j / Nebula 后端都继续存 `entity_type` 字符串
- `aperag/domains/knowledge_graph/service.py`
  - `get_graph_labels()` 继续返回 graph 里已有的 type 字符串

好消息是：schema 里已经有 `knowledge_graph_config.entity_types`，所以不需要新增 DB 表。

## 4. 最终数据模型

### 4.1 Collection 配置

继续使用现有配置字段：

```python
Collection.config.knowledge_graph_config.entity_types: list[str]
```

规则：

- 新 collection 初始值是空列表：`[]`
- 如果用户手动配置了初始 type，就用用户配置
- 如果没有配置，就让 LLM 从第一批文档里生成
- LLM 后续发现新 type，就追加到这个列表

不再使用硬编码默认值：

```python
_DEFAULT_ENTITY_TYPES = (...)
```

这个默认值要删掉，不允许再作为空 collection 的 fallback。

### 4.2 Entity 存储

Graph entity 仍然只存字符串：

```python
EntityRecord.entity_type = "疾病"
```

不改成：

```python
EntityRecord.type_id = "disease"
```

原因：

- collection 语言不切换
- 用户看到的 type 和系统存的 type 可以是同一个字符串
- 这样 graph storage、graph label filter、graph curation 都不用大改

### 4.3 不做 registry 表

这次不建这种表：

```text
collection_entity_type
- type_id
- label
- description
- examples
- status
- source
- created_from_document_id
```

这些字段以后如果真的需要管理、合并、禁用 entity type，可以再做。Wave 11 先只解决当前问题。

## 5. 语言规则

Collection 的 `language` 是 graph 输出语言的唯一来源。

需要把代码里的语言 code 映射成 prompt 里更清楚的语言名：

| collection language | prompt 语言 |
| --- | --- |
| `zh-CN` | `Chinese (Simplified)` |
| `en-US` | `English` |
| `ja-JP` | `Japanese` |
| `ko-KR` | `Korean` |

规则：

1. Entity name 保留原文。
   - `OpenAI` 不要翻译成 `开放人工智能`
   - 人名、公司名、产品名、代码名尽量按原文保留
2. Entity description 用 collection language。
3. Relation description 用 collection language。
4. 新生成的 entity type 用 collection language。
5. 不引入 `auto` 语言检测。

例子：

英文文档里有：

```text
OpenAI released GPT-4o.
```

如果 collection language 是 `zh-CN`，graph extraction 可以输出：

```json
{
  "entities": [
    {
      "name": "OpenAI",
      "entity_type": "组织",
      "description": "一家人工智能研究与产品公司。"
    },
    {
      "name": "GPT-4o",
      "entity_type": "产品",
      "description": "OpenAI 发布的多模态模型。"
    }
  ]
}
```

## 6. Prompt 规则

旧规则：

```text
只能使用这些 entity types。如果没有合适 type，就跳过这个 entity。
```

新规则：

```text
优先复用已有 entity types。
如果没有合适 type，可以创建一个新的、简短的 entity type。
新 type 必须使用 collection language。
不要因为 type 不在列表里就丢掉有效 entity。
```

Prompt 里传给 LLM 的内容包括：

- collection language
- 当前 `entity_types` 列表
- 文档 chunk 内容

如果 `entity_types=[]`，LLM 就从当前文档开始生成第一批 type。

## 7. LLM 输出格式

继续使用简单格式，不引入复杂 registry 对象。

推荐输出：

```json
{
  "entities": [
    {
      "name": "糖尿病",
      "entity_type": "疾病",
      "description": "一种慢性代谢性疾病。"
    },
    {
      "name": "胰岛素",
      "entity_type": "药物",
      "description": "用于控制血糖的治疗药物。"
    }
  ],
  "relations": [
    {
      "source": "胰岛素",
      "target": "糖尿病",
      "relation_type": "治疗",
      "description": "胰岛素可用于治疗糖尿病。"
    }
  ]
}
```

Parser 兼容已有字段：

- `entity_type`
- legacy `type`

但最终内部统一成 `entity_type`。

## 8. 新 type 如何追加

每次 LLM 输出 entity 后，系统检查：

```python
entity.entity_type not in collection.config.knowledge_graph_config.entity_types
```

如果发现新 type，就合并回 collection 配置。

例如当前列表是：

```json
["人物", "组织"]
```

LLM 新输出：

```json
{"name": "糖尿病", "entity_type": "疾病"}
```

那么合并后变成：

```json
["人物", "组织", "疾病"]
```

## 9. 字符串清理规则

只做轻量清理，不做复杂语义合并。

清理规则：

1. 去掉前后空格。
2. 多个连续空白合并成一个空格。
3. 空字符串丢掉。
4. 太长的 type 字段丢掉，初始上限 64 个字符；entity 本身保留，`entity_type=""`，这个空字符串不合并进 collection 配置。
5. 英文 type 做大小写去重：
   - `Person`
   - `person`
   - 只保留一个，规则是 first-write-wins：列表里已有的写法胜出，新来的大小写变体被忽略
6. 中文、日文、韩文按原字符串去重：
   - `疾病`
   - `病症`
   - 这两个不自动合并

不做这些事情：

- 不把 `疾病` 转成 `disease`
- 不把 `Medical Condition` 转成 `medical_condition`
- 不自动判断 `公司` 和 `组织` 是不是同义词

## 10. 并发写入规则

多个 worker 可能同时处理同一个 collection 的不同文档。

例如两个 worker 都发现新 type：

```json
"疾病"
```

不能写出重复数据，也不能让其中一个 worker 因为重复 type 报错。

所以需要一个异步原子合并函数：

```python
async def merge_entity_types(
    session: AsyncSession,
    collection_id: str,
    new_types: Sequence[str],
) -> list[str]
```

语义：

1. 开启数据库事务。
2. 用 `SELECT ... FOR UPDATE` 锁住 collection 这一行。
3. 读取当前 `collection.config`。
4. 解析出当前 `entity_types`。
5. 合并新 type。
6. 去重。
7. 如果有变化，写回 collection config。
8. 返回最终列表。

这样两个 worker 同时写 `疾病`，最终列表里也只有一个 `疾病`。

实现必须走 `SELECT ... FOR UPDATE` 这一条路径，不允许在 Wave 11 里改成 sync wrapper，也不允许改成乐观重试 fallback。

如果 `merge_entity_types()` 失败：

1. 记录 warning，包含 `collection_id`、`document_id`、失败原因、新 type 列表。
2. 不阻塞已经完成的 graph entity 写入。
3. 不在 indexing task 内无限重试。
4. 后续文档再次出现同一个 type 时会重新尝试合并，形成轻量 self-healing。

## 11. Runtime 流程

一次 graph indexing 的流程：

1. 读取 collection。
2. 读取 collection language。
3. 读取 `knowledge_graph_config.entity_types`。
4. 把 language 和 entity_types 放进 prompt。
5. LLM 提取 entities / relations。
6. Parser 接受新旧字段格式。
7. entity 写入 graph，`entity_type` 就是字符串。
8. 以 document 为粒度收集本文档输出里出现的新 type。
9. 一个文档处理完成、entity 已 flush 到 graph 后，调用一次 `merge_entity_types()` 合并回 collection config。

调用粒度锁定为 **per-document**：

- 不做 per-chunk 合并，避免一个文档内多次 DB 写。
- 不做 per-indexing-run 合并，避免 worker 中途崩溃后 graph 里已有 entity type 没进 collection 配置。
- per-document 合并失败不回滚已经写入 graph 的 entity。

Graph backend 不需要知道这个 type 是用户手动配置的，还是 LLM 新生成的。

## 12. API / UI 影响

Wave 11 不要求新增 API。

现有 graph labels API 可以继续返回字符串：

```json
["人物", "组织", "疾病"]
```

前端 filter 也继续按字符串过滤。

如果以后要让用户编辑 type list，可以直接编辑：

```json
knowledge_graph_config.entity_types
```

但这不是第一版必须做的 UI。

## 13. Graph Curation 兼容性

Wave 7 的 graph curation 逻辑里也有 `entity_type`。

这次不改字段形态，所以它继续是字符串：

```python
CurationEntity.entity_type: str
```

需要在实现 PR 里 grep 验证：

- 没有引入 `type_id`
- 没有引入 `collection_entity_type`
- `_DEFAULT_ENTITY_TYPES` 没有剩余 caller
- graph curation 仍然按字符串 entity_type 工作

## 14. 实施计划

### PR 1：文档 + language + dynamic type

内容：

1. 加这份设计文档。
2. 加 centralized language mapping helper。
3. 改 graph prompt：
   - 不再说 unknown type 就 skip
   - 允许 LLM 创建新 type
   - 明确新 type 和 description 要遵守 collection language
4. 删除 `_DEFAULT_ENTITY_TYPES` fallback。
5. 允许 `entity_types=[]`。
6. 改 parser：
   - 接受 unknown `entity_type`
   - 不再因为 type 不在列表里丢 entity
   - 收集本次出现的新 type
7. 加 `merge_entity_types()`。
8. 每个 document graph indexing 完成后，把本文档新 type 合并回 collection config。
9. 更新测试。

### PR 2：可选 UI / API polish

只有产品明确需要时再做：

1. UI 上展示 / 编辑 entity_types。
2. API 校验手动编辑的 type list。
3. graph label 使用统计。

不在 Wave 11 第一版做：

- type 生命周期
- approve / deprecated 状态
- synonym merge
- registry table

## 15. 测试要求

PR 1 必须覆盖：

1. Language mapping：
   - `zh-CN` 能映射成 `Chinese (Simplified)`
   - prompt 里能看到这个语言要求
2. Empty entity types：
   - `entity_types=[]` 合法
   - prompt 不崩
3. Prompt contract：
   - prompt 不再说 unknown type 必须跳过
   - prompt 明确允许创建新 type
4. Parser：
   - unknown `entity_type` 不被丢掉
   - legacy `type` 仍可解析
5. Dynamic append：
   - LLM 输出新 type 后，collection config 里能看到新 type
6. Atomic merge：
   - 重复 merge 同一个 type，最终只出现一次
   - 并发 merge 不丢失 type
7. Config preservation：
   - merge entity_types 不破坏 collection config 里的其他字段
8. i18n：
   - 英文文档 + 中文 collection，可以输出中文 description 和中文 type
   - entity name 保留原文
9. Graph curation：
   - `CurationEntity.entity_type` 仍然是字符串
   - candidate pair 逻辑不因为 Wave 11 破坏
10. Failure behavior：
   - `merge_entity_types()` 失败时不阻塞已写入 graph 的 entity
   - 有 warning 日志

## 16. PR hard gate

每个 Wave 11 PR 描述里必须写清楚：

1. hard cut：没有老数据迁移。
2. 没有新增 `collection_entity_type` 表。
3. 没有 `type_id` / `label` 双层模型。
4. `_DEFAULT_ENTITY_TYPES` 是否已经删除。
5. `entity_types=[]` 是否合法。
6. `merge_entity_types()` 是否是 async + `SELECT ... FOR UPDATE` + per-document 调用。
7. grep 结果：
   - `type_id`
   - `collection_entity_type`
   - `CurationEntity.entity_type`
   - `_DEFAULT_ENTITY_TYPES`
8. 测试命令和结果。

同时继续沿用 Wave 7-10 的 hard-gate 三段：

- 12-invariant cross-check
- 4-pattern + 11 mini-pattern pre-check
- simple-stable 4-guardrail

## 17. 明确不做

Wave 11 不做：

- 老数据迁移
- lazy seed
- backfill
- 新 DB 表
- `type_id`
- localized `label`
- `description/examples/status/source` 等复杂字段
- 默认 `_DEFAULT_ENTITY_TYPES`
- 自动语言检测
- 语义同义词合并
- entity type 审批 / 禁用流程
- 因为新增 type 自动触发 graph reindex
- type 列表 size cap
- type 列表 LRU 淘汰

## 18. 最终一句话

每个 collection 自己维护一个字符串列表 `entity_types`。LLM 先复用这个列表，不合适就按 collection
语言生成新字符串。系统把新字符串原子合并回 collection 配置。Graph 里仍然只存 `entity_type` 字符串。
