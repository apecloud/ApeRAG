# Prompt API 测试手册

**Base URL**: `http://localhost:8000/api/v1`  
**Auth**: `Bearer sk-85fc1342e0df44378ad73184ca8005b5`
（请替换SK为真实SK）

---

## 1. GET /prompts/user — 获取用户 Prompt 配置

返回所有 5 种 prompt 的当前生效内容，以及来源（`source: user/system/hardcoded`）和是否自定义（`customized: true/false`）。

```bash
curl -X GET 'http://localhost:8000/api/v1/prompts/user' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5'
```

**期待结果**：返回 `agent_system`、`agent_query`、`index_graph`、`index_summary`、`index_vision` 五个字段，初始状态下 `source` 均为 `system` 或 `hardcoded`，`customized` 均为 `false`。

---

## 2. PUT /prompts/user — 更新用户 Prompt 配置

只更新提供的字段，未提供的字段保持不变。

```bash
curl -X PUT 'http://localhost:8000/api/v1/prompts/user' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5' \
  -d '{
    "prompts": {
      "agent_system": "You are a helpful assistant specialized in technical support.",
      "index_graph": "Extract medical entities and relationships from the text."
    }
  }'
```

**期待结果**：`updated: ["agent_system", "index_graph"]`。再次调用接口 1，可见这两个字段的 `source` 变为 `user`，`customized` 变为 `true`。

---

## 3. GET /prompts/system — 查看系统默认 Prompt

只读接口，供用户参考系统默认内容。

```bash
# 查看所有系统默认
curl -X GET 'http://localhost:8000/api/v1/prompts/system' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5'
```

```bash
# 查看指定类型
curl -X GET 'http://localhost:8000/api/v1/prompts/system?type=agent_system' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5'
```

**期待结果**：返回系统内置的 prompt 内容，不受用户自定义影响。

---

## 4. DELETE /prompts/user/{type} — 重置单个 Prompt

删除用户对某个 prompt 的自定义，回退到系统默认。

```bash
# 正常重置
curl -X DELETE 'http://localhost:8000/api/v1/prompts/user/agent_system' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5'
```

**期待结果**：返回重置后生效的内容，`source` 为 `system` 或 `hardcoded`。

```bash
# 重置一个未自定义的 prompt
curl -X DELETE 'http://localhost:8000/api/v1/prompts/user/agent_query' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5'
```

**期待结果**：`404`，`detail: "User has not customized agent_query prompt"`。

```bash
# 传入非法类型
curl -X DELETE 'http://localhost:8000/api/v1/prompts/user/invalid_type' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5'
```

**期待结果**：`400`，提示合法的 type 列表。

---

## 5. POST /prompts/user/reset — 批量重置 Prompt

```bash
# 重置指定类型
curl -X POST 'http://localhost:8000/api/v1/prompts/user/reset' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5' \
  -d '{"types": ["agent_system", "index_graph"]}'
```

```bash
# 重置所有（不传 types）
curl -X POST 'http://localhost:8000/api/v1/prompts/user/reset' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5' \
  -d '{}'
```

**期待结果**：`reset` 数组列出实际被重置的类型（未自定义的不会出现在列表中）。

---

## 6. POST /prompts/preview — 预览 Prompt 渲染效果

用于前端展示"变量填入后的效果"。

```bash
curl -X POST 'http://localhost:8000/api/v1/prompts/preview' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5' \
  -d '{
    "template": "Hello {{ name }}, you have {{ count }} messages.",
    "variables": {"name": "Alice", "count": 5}
  }'
```

**期待结果**：`rendered: "Hello Alice, you have 5 messages."`

---

## 7. POST /prompts/validate — 校验 Prompt 语法

```bash
# 合法模板（但缺少建议变量，会有 warnings）
curl -X POST 'http://localhost:8000/api/v1/prompts/validate' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5' \
  -d '{"type": "agent_query", "template": "{{ query }} {{ collections }}"}'
```

**期待结果**：`valid: true`，`warnings` 中提示缺少 `language`、`chat_id` 等建议变量。

```bash
# 非法 Jinja2 语法
curl -X POST 'http://localhost:8000/api/v1/prompts/validate' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-85fc1342e0df44378ad73184ca8005b5' \
  -d '{"type": "agent_query", "template": "{% for x in %}broken{% endfor %}"}'
```

**期待结果**：`valid: false`，`errors` 中包含 Jinja2 语法错误信息。

---

## Prompt 类型说明

| 类型 | 用途 | 配置位置 |
|---|---|---|
| `agent_system` | Agent 人格/行为定义 | Bot 配置 > 用户默认 > 系统默认 |
| `agent_query` | 每次对话的查询 prompt 模板 | Bot 配置 > 用户默认 > 系统默认 |
| `index_graph` | 知识图谱实体关系抽取 | Collection 配置 > 用户默认 > 系统默认 |
| `index_summary` | 文档摘要生成 | Collection 配置 > 用户默认 > 系统默认 |
| `index_vision` | 图片内容提取 | Collection 配置 > 用户默认 > 系统默认 |