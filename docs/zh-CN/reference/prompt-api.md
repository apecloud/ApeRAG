# Prompt API 参考

本文提供 `/api/v1/prompts/*` 系列接口的 curl 示例与期望行为说明，便于前端集成、自动化测试以及回归验证。关于 prompt 的三层优先级、Bot 配置交互与管理 UX，见 [`admin-guide/prompt-customization.md`](../admin-guide/prompt-customization.md)。关于 prompt_template_service 在后端架构中的 standalone-infra + Protocol+DI 定位，见 [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md#protocol-promptTemplateOps)。

- **Base URL**：`http://localhost:8000/api/v1`
- **Auth**：`Authorization: Bearer sk-<your-api-key>`
- **Route host**：当前仍落在 `aperag/views/prompts.py`（standalone-infra legacy view）；调用的服务是 `aperag.service.prompt_template_service` 单例。

所有示例把 Bearer token 写成占位符 `sk-<your-api-key>`，请在本地自行替换。接口的返回值以 JSON 表示，为了简洁本文只列出关键字段。

---

## 1. GET `/prompts/user` — 获取用户的 prompt 配置

返回当前用户 5 种 prompt 的**有效内容**（合并过三层优先级后的结果），并在每一项上标注来源。

```bash
curl -X GET 'http://localhost:8000/api/v1/prompts/user' \
  -H 'Authorization: Bearer sk-<your-api-key>'
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `agent_system` / `agent_query` / `index_graph` / `index_summary` / `index_vision` | 每种 prompt 的有效内容 |
| `source` | `user` / `system` / `hardcoded` — 说明内容来自哪一层 |
| `customized` | `true` 表示用户层已自定义，否则为 `false` |

未做任何自定义时，`source` 通常为 `system` 或 `hardcoded`、`customized` 均为 `false`。

## 2. PUT `/prompts/user` — 更新用户层的 prompt

只更新请求体里给出的字段，未出现的字段保持不变。

```bash
curl -X PUT 'http://localhost:8000/api/v1/prompts/user' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<your-api-key>' \
  -d '{
    "prompts": {
      "agent_system": "你是一名专注于技术支持的中文智能助理。",
      "index_graph": "请从文本中抽取医疗领域的实体与关系。"
    }
  }'
```

返回体中的 `updated: [...]` 列出了本次实际写入的字段。再次 GET `/prompts/user`，对应字段的 `source` 会从 `system` / `hardcoded` 变为 `user`，`customized` 会变为 `true`。

## 3. GET `/prompts/system` — 查看系统默认 prompt

只读接口。可以整表返回，也可以按 `type` 查询单项；不受用户自定义影响。

```bash
# 获取所有系统默认
curl -X GET 'http://localhost:8000/api/v1/prompts/system' \
  -H 'Authorization: Bearer sk-<your-api-key>'

# 仅获取 agent_system 一项
curl -X GET 'http://localhost:8000/api/v1/prompts/system?type=agent_system' \
  -H 'Authorization: Bearer sk-<your-api-key>'
```

## 4. DELETE `/prompts/user/{type}` — 重置单条用户自定义

把用户层对某种 prompt 的覆盖删掉，恢复到系统默认或硬编码。

```bash
# 正常重置
curl -X DELETE 'http://localhost:8000/api/v1/prompts/user/agent_system' \
  -H 'Authorization: Bearer sk-<your-api-key>'
```

返回体包含重置后的有效内容（`source` 为 `system` / `hardcoded`）。若该 type 未曾自定义，接口返回 `404`：

```json
{ "detail": "User has not customized agent_query prompt" }
```

传入不支持的 type 会返回 `400`，并提示合法 type 列表。

## 5. POST `/prompts/user/reset` — 批量重置

```bash
# 重置指定类型
curl -X POST 'http://localhost:8000/api/v1/prompts/user/reset' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<your-api-key>' \
  -d '{"types": ["agent_system", "index_graph"]}'

# 省略 types：重置所有已自定义项
curl -X POST 'http://localhost:8000/api/v1/prompts/user/reset' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<your-api-key>' \
  -d '{}'
```

返回体中的 `reset: [...]` 列出了实际被重置的类型；未自定义的类型不会出现在该数组中。

## 6. POST `/prompts/preview` — 预览 Jinja2 渲染结果

前端在"编辑 prompt"页可以用该接口预览变量替换后的效果。

```bash
curl -X POST 'http://localhost:8000/api/v1/prompts/preview' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<your-api-key>' \
  -d '{
    "template": "Hello {{ name }}, you have {{ count }} messages.",
    "variables": {"name": "Alice", "count": 5}
  }'
```

返回：`rendered: "Hello Alice, you have 5 messages."`

## 7. POST `/prompts/validate` — 校验模板合法性

```bash
# 合法模板：可能会返回 warnings（提示缺少建议变量）
curl -X POST 'http://localhost:8000/api/v1/prompts/validate' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<your-api-key>' \
  -d '{"type": "agent_query", "template": "{{ query }} {{ collections }}"}'

# 非法 Jinja2 语法：valid=false + errors
curl -X POST 'http://localhost:8000/api/v1/prompts/validate' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<your-api-key>' \
  -d '{"type": "agent_query", "template": "{% for x in %}broken{% endfor %}"}'
```

返回体里 `valid: true/false` 指明语法是否合法；合法时 `warnings` 会列出缺少的建议变量，非法时 `errors` 会列出 Jinja2 的错误位置。

---

## Prompt 类型速查

| 类型 | 作用 | 三层配置来源（高 → 低） |
| --- | --- | --- |
| `agent_system` | Agent 人格与行为约束 | Bot 配置 → 用户默认 → 系统默认 |
| `agent_query` | 每次对话的 query 模板 | Bot 配置 → 用户默认 → 系统默认 |
| `index_graph` | 知识图谱实体关系抽取 | Collection 配置 → 用户默认 → 系统默认 |
| `index_summary` | 文档摘要生成 | Collection 配置 → 用户默认 → 系统默认 |
| `index_vision` | 图片内容识别 | Collection 配置 → 用户默认 → 系统默认 |

Prompt 解析策略、覆盖顺序与 Bot / Collection 的绑定关系详见 [`admin-guide/prompt-customization.md`](../admin-guide/prompt-customization.md)。
