---
title: Prompt 自定义
position: 50
---

# Prompt 自定义管理指南

本文面向"想定制 Agent 回答风格 / 知识图谱抽取规则 / 索引摘要提示"的管理员与高级用户，说明 ApeRAG 的 **Prompt 分层覆盖机制** 与 **用户层 / Bot 层 / Collection 层** 的配置路径。面向前端集成的 API curl 示例见 [`reference/prompt-api.md`](../reference/prompt-api.md)；后端架构面（`PromptTemplateOps` Protocol + DI + `prompt_template_service` 作为 standalone-infra permanent seam）见 [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md#protocol-promptTemplateOps)。

## 1. Prompt 的 5 种类型与用途

| 类型 | 作用 | 主要覆盖位置 |
| --- | --- | --- |
| `agent_system` | Agent 的"人格/系统指令" — 决定身份、风格、约束 | Bot 配置 > 用户默认 > 系统默认 > 代码硬编码 |
| `agent_query` | 每次对话的 query 模板 — 把用户输入 + 场景注入 agent | Bot 配置 > 用户默认 > 系统默认 > 代码硬编码 |
| `index_graph` | 知识图谱实体关系抽取 prompt | Collection 配置 > 用户默认 > 系统默认 > 代码硬编码 |
| `index_summary` | 文档摘要生成 prompt | Collection 配置 > 用户默认 > 系统默认 > 代码硬编码 |
| `index_vision` | 图片内容识别 prompt | Collection 配置 > 用户默认 > 系统默认 > 代码硬编码 |

所有 prompt 都支持 Jinja2 变量渲染；运行时后端会按 prompt 类型注入一组标准变量（例如 `{{ query }}`、`{{ collections }}`、`{{ language }}`、`{{ has_chat_files }}` 等）。

## 2. 三层优先级（高 → 低）

Prompt 的生效顺序是典型的"越靠近这次调用的配置，越先赢"：

### 2.1 Agent prompt（`agent_system` / `agent_query`）

```
1. Bot.config.agent.system_prompt_template / query_prompt_template
       ↓ 若为空
2. prompt_template 表：scope='user', user_id=<me>, prompt_type=<agent_system|agent_query>
       ↓ 若为空
3. prompt_template 表：scope='system', prompt_type=<agent_system|agent_query>
       ↓ 若仍为空
4. 代码硬编码（APERAG_AGENT_INSTRUCTION 中英文分版）
```

典型场景：

- 某个 Bot 专门服务"医学文献问答"：把专用系统提示写进 Bot.config.agent.system_prompt_template，其他所有 Bot 不受影响。
- 当前用户希望所有 Bot 默认用中文强约束的系统提示：通过 `/prompts/user` 把 `agent_system` 改成自己的模板，会被所有未显式设置 Bot prompt 的 Bot 继承。
- 整个部署希望调整一套默认提示：管理员修改 `prompt_template` 表中 `scope='system'` 的行（目前通过数据库迁移或人工运维操作管理，不开放用户 API）。

### 2.2 Index prompt（`index_graph` / `index_summary` / `index_vision`）

```
1. Collection.config.index_prompts.{graph|summary|vision}
       ↓ 若为空
2. prompt_template 表：scope='user', user_id=<collection owner>, prompt_type=<index_graph|...>
       ↓ 若为空
3. prompt_template 表：scope='system', prompt_type=<index_graph|...>
       ↓ 若仍为空
4. 代码硬编码（LightRAG PROMPTS['entity_extraction'] 等）
```

典型场景：

- 某 Collection 是"法律知识库"，需要更精细的实体类型：在 Collection 的 `config.index_prompts.graph` 写定制 prompt，不影响其他 Collection。
- 用户全部 Collection 都想使用同一套定制化 vision prompt：通过 `/prompts/user` 把 `index_vision` 设置为自定义版本，所有未显式配置 `index_prompts.vision` 的 Collection 都会继承。

## 3. 用户层自定义（推荐的常用入口）

用户层是介于"全局默认"和"单个 Bot/Collection 覆盖"之间的中间层，`/api/v2/prompts/*` 完整支持：

| 动作 | 接口 | 说明 |
| --- | --- | --- |
| 查看我的 5 种 prompt 当前生效内容 + 来源 | GET `/prompts/user` | 见 [reference/prompt-api.md §1](../reference/prompt-api.md) |
| 更新一种或多种 prompt | PUT `/prompts/user` | 见 §2 |
| 查看系统默认（参考） | GET `/prompts/system` | 见 §3 |
| 重置单条用户自定义 | DELETE `/prompts/user/{type}` | 见 §4 |
| 批量重置（全部 / 指定列表） | POST `/prompts/user/reset` | 见 §5 |
| 预览渲染效果 | POST `/prompts/preview` | 见 §6 |
| 校验 Jinja2 语法 | POST `/prompts/validate` | 见 §7 |

更新/重置会直接影响**所有未显式在 Bot/Collection 层配置**该 prompt 的场景。

### 3.1 注意事项

- **变更即生效**：新的 prompt 会作用于随后的所有 Agent 调用或 Collection 索引构建请求；已经在跑的任务不会回溯修改。
- **索引类 prompt 改动**要配合重新构建索引：`index_graph` / `index_summary` / `index_vision` 只对新触发的索引任务生效；已经入库的内容不会自动回溯重抽。
- **preview / validate** 用于"所见即所得"的前端编辑体验，不会修改任何持久化数据。

## 4. Bot 层配置（只覆盖单个 Bot）

在 Bot 的 `config` 里写入 `agent.system_prompt_template` 或 `agent.query_prompt_template`：

```json
{
  "agent": {
    "system_prompt_template": "你是法律助手，只引用...",
    "query_prompt_template": "{{ query }}\n\n请引用收录在 {{ collections }} 中的法条编号"
  }
}
```

写入后该 Bot 的 agent prompt 将直接使用这里的模板，完全绕过用户默认 / 系统默认。若想回退到用户默认，把这两个字段清空即可。

## 5. Collection 层配置（只覆盖单个 Collection）

在 Collection 的 `config.index_prompts` 里写入 `graph` / `summary` / `vision`：

```json
{
  "index_prompts": {
    "graph": "从以下文本中抽取医疗领域的实体和关系...",
    "summary": "请用 3 句话概括以下药品说明书...",
    "vision": "识别图中医疗器械并给出结构化描述..."
  }
}
```

写入后这一 Collection 的索引任务会使用这里的 prompt，完全绕过用户默认 / 系统默认。清空字段即可回退。

## 6. 排查常用问题

**Q：改了 `/prompts/user` 但 Agent 回答没变**

- 先排查该用户使用的 Bot 是否自带了 `agent.system_prompt_template` / `agent.query_prompt_template`；Bot 层会覆盖用户层。
- GET `/prompts/user`：确认 `source = user`、`customized = true`。若 `source` 仍是 `system` / `hardcoded`，说明保存失败（可能语法校验未通过）。
- 注意：修改用户层后已在流式中的 chat turn 使用的是旧 prompt；新发起的 turn 才会用新 prompt。

**Q：改了 `index_prompts.graph` 但历史实体没更新**

- Index prompt 只影响"新触发的索引任务"。已入库的 entity/relation 不会自动重抽；在 Collection 页面上手动重建索引或删除后重新导入。

**Q：PUT `/prompts/user` 报 400 或 422**

- 用 POST `/prompts/validate` 先跑一次 Jinja2 语法校验；常见原因是 `{% for ... %}` 写错、变量引用中多余空格、或缺少必须的变量（`warnings` 列出建议变量）。

**Q：如何知道"系统默认"的版本**

- GET `/prompts/system?type=agent_system` 只读查看；该接口不受用户自定义影响，可用来对照"你改过什么"。

## 7. 服务归属说明

Prompt 相关的所有逻辑集中在 standalone-infra 模块 `aperag.service.prompt_template_service`：该模块跨 agent_runtime（运行时调用）/ conversation（bot-config 解析）/ indexing（reconciler）/ `/api/v2/prompts` REST 四个场景共享，没有自然的 domain 归属。Phase 8 #49 G3 起，REST 路由本身已碎入 `aperag/domains/model_platform/api/prompts_routes.py`，通过 `model_platform.ports.PromptCRUDOps` Protocol + `aperag/app.py::set_prompt_crud_ops()` wire 同一个单例；agent_runtime 仍通过 `aperag.domains.agent_runtime.ports.PromptTemplateOps` Protocol + `aperag/app.py::_PromptTemplateOpsAdapter` wire-up 访问它。具体架构背景见 [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md#protocol-promptTemplateOps)。
