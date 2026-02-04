# Prompt自定义功能设计文档

## 架构概述

Prompt自定义功能采用**配置优先 + 简化表存储**的方案，提供三层优先级的配置继承机制。

---

## 数据模型

### prompt_template表

用于存储用户默认和系统默认prompt。

```sql
CREATE TABLE prompt_template (
    id VARCHAR(24) PRIMARY KEY,
    prompt_type VARCHAR(50) NOT NULL,     -- agent_system, agent_query, index_graph, etc.
    scope VARCHAR(20) NOT NULL,           -- 'user' or 'system'
    user_id VARCHAR(256),                 -- NULL for system, user_id for user
    language VARCHAR(10) NOT NULL,        -- zh-CN, en-US
    content TEXT NOT NULL,
    description TEXT,
    gmt_created TIMESTAMP,
    gmt_updated TIMESTAMP,
    gmt_deleted TIMESTAMP
);
```

### Bot配置（已存在）

```json
{
  "agent": {
    "system_prompt_template": "Bot专属system prompt",
    "query_prompt_template": "Bot专属query prompt"
  }
}
```

### Collection配置（新增index_prompts）

```json
{
  "index_prompts": {
    "graph": "Collection专属graph prompt",
    "summary": "Collection专属summary prompt",
    "vision": "Collection专属vision prompt"
  }
}
```

---

## 三层优先级系统

### Agent Prompt解析

```
优先级1: Bot.config.agent.system_prompt_template
    ↓
优先级2: prompt_template (scope='user', prompt_type='agent_system')
    ↓
优先级3: prompt_template (scope='system', prompt_type='agent_system')
    ↓
优先级4: 代码硬编码 (APERAG_AGENT_INSTRUCTION_ZH/EN)
```

### 索引Prompt解析

```
优先级1: Collection.config.index_prompts.graph
    ↓
优先级2: prompt_template (scope='user', prompt_type='index_graph')
    ↓
优先级3: prompt_template (scope='system', prompt_type='index_graph')
    ↓
优先级4: 代码硬编码 (LightRAG PROMPTS["entity_extraction"])
```

---

## 架构分层

```
┌─────────────────────────────────────┐
│ View层 (prompts.py)                 │
│ - HTTP请求处理                       │
│ - 参数验证                           │
│ - 错误处理                           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ Service层 (PromptTemplateService)   │
│ ┌─────────────────────────────────┐ │
│ │ 用户配置管理（给View用）         │ │
│ │ - get_user_prompts()            │ │
│ │ - update_user_prompts()         │ │
│ │ - delete_user_prompt()          │ │
│ │ - reset_user_prompts()          │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ Prompt解析（给Agent/LightRAG用） │ │
│ │ - resolve_agent_system_prompt() │ │
│ │ - resolve_agent_query_prompt()  │ │
│ │ - resolve_index_prompt()        │ │
│ └─────────────────────────────────┘ │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ Repository层                         │
│ (AsyncPromptTemplateRepositoryMixin)│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ Database (prompt_template表)        │
└─────────────────────────────────────┘
```

---

## 支持的Prompt类型

| Prompt类型 | 用途 | 存储位置 |
|-----------|------|---------|
| agent_system | Agent人格定义 | Bot.config.agent.system_prompt_template |
| agent_query | 查询模板 | Bot.config.agent.query_prompt_template |
| index_graph | 实体关系抽取 | Collection.config.index_prompts.graph |
| index_summary | 文档摘要 | Collection.config.index_prompts.summary |
| index_vision | 图像提取 | Collection.config.index_prompts.vision |

---

## API设计

### RESTful资源模型

```
/prompts/user              # 用户的prompt配置（资源）
/prompts/system            # 系统的prompt配置（资源）
/prompts/preview           # 工具
/prompts/validate          # 工具
```

### 核心API

- `GET /prompts/user` - 获取用户配置（含优先级解析）
- `PUT /prompts/user` - 批量更新用户配置
- `DELETE /prompts/user/{type}` - 重置单个配置
- `POST /prompts/user/reset` - 批量重置
- `GET /prompts/system` - 获取系统默认

---

## 核心实现

### PromptTemplateService

**位置**：`aperag/service/prompt_template_service.py`

**核心方法**：

```python
class PromptTemplateService:
    # 用户配置管理（给View层用）
    async def get_user_prompts(user_id, language) -> Dict
    async def update_user_prompts(user_id, language, prompts) -> List[str]
    async def delete_user_prompt(user_id, prompt_type, language) -> Dict
    async def reset_user_prompts(user_id, language, types) -> List[str]
    
    # Prompt解析（给Agent/LightRAG用）
    async def resolve_agent_system_prompt(bot, user_id, language) -> str
    async def resolve_agent_query_prompt(bot, user_id, language) -> str
    async def resolve_index_prompt(collection, prompt_type, user_id) -> str
```

---

## 技术特点

1. **配置内聚**：对象级配置跟随对象存储（Bot/Collection的config字段）
2. **简化表结构**：prompt_template表只存储默认配置
3. **三层优先级**：对象 > 用户默认 > 系统默认 > 硬编码
4. **RESTful API**：资源导向，语义清晰
5. **分层架构**：View → Service → Repository

---

## 数据流示例

### 用户获取配置

```
1. 用户请求：GET /prompts/user?language=zh-CN
   ↓
2. View层：prompts.py 接收请求
   ↓
3. Service层：prompt_template_service.get_user_prompts()
   - 遍历所有prompt_type
   - 对每个type执行优先级查找：
     a. 查询 prompt_template (scope='user')
     b. 如无，查询 prompt_template (scope='system')
     c. 如无，使用代码硬编码
   - 组装响应：content + source + customized + language
   ↓
4. Repository层：query_prompt_template()
   ↓
5. Database：prompt_template表
   ↓
6. 返回响应给前端
```

### Agent对话使用prompt

```
1. 用户发起对话
   ↓
2. agent_chat_service.py
   ↓
3. 调用：prompt_template_service.resolve_agent_system_prompt(bot, user_id, language)
   - 优先级1：检查 bot.config.agent.system_prompt_template
   - 优先级2：查询 prompt_template (scope='user')
   - 优先级3：查询 prompt_template (scope='system')
   - 优先级4：使用 APERAG_AGENT_INSTRUCTION_ZH
   ↓
4. 返回解析后的prompt给Agent使用
```

---

## 后续集成

详见：[prompt_customization_integration_todo.md](./prompt_customization_integration_todo.md)

核心任务：
- Agent对话集成（agent_chat_service.py）
- LightRAG集成（lightrag_manager.py）
- Summary/Vision索引集成
