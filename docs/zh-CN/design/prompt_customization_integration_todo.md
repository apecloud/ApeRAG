# Prompt自定义功能后续集成指南

本文档记录Prompt自定义功能的后续集成工作。

## 已完成的基础设施

- ✅ `prompt_template` 数据库表
- ✅ Repository层CRUD方法
- ✅ `PromptTemplateService`类（完整业务逻辑）
- ✅ RESTful API (`/prompts/user/*`)
- ✅ Collection Schema扩展（`index_prompts`字段）
- ✅ 辅助API（preview、validate、system）

## 后续集成任务

需要将prompt解析服务集成到现有的Agent对话和索引构建流程中。

---

## 一、Prompt解析服务实现

### 1.1 文件位置
`aperag/service/prompt_template_service.py`

### 1.2 需要添加的方法

#### resolve_agent_system_prompt
```python
async def resolve_agent_system_prompt(bot, user_id: str, language: str) -> str:
    """
    解析Agent系统prompt
    
    优先级：
    1. Bot.config.agent.system_prompt_template
    2. prompt_template表（scope='user', prompt_type='agent_system'）
    3. prompt_template表（scope='system', prompt_type='agent_system'）
    4. 代码硬编码（APERAG_AGENT_INSTRUCTION_EN/ZH）
    
    Args:
        bot: Bot对象
        user_id: 用户ID
        language: 语言代码 (en-US, zh-CN)
        
    Returns:
        解析后的system prompt内容
    """
    from aperag.db.ops import async_db_ops
    
    # 层级1：Bot配置
    if bot.config:
        try:
            import json
            config_dict = json.loads(bot.config) if isinstance(bot.config, str) else bot.config
            if config_dict.get("agent", {}).get("system_prompt_template"):
                return config_dict["agent"]["system_prompt_template"]
        except:
            pass
    
    # 层级2：用户默认
    user_default = await async_db_ops.query_prompt_template(
        prompt_type="agent_system",
        scope="user",
        user_id=user_id,
        language=language
    )
    if user_default:
        return user_default.content
    
    # 层级3：系统默认
    system_default = await async_db_ops.query_prompt_template(
        prompt_type="agent_system",
        scope="system",
        user_id=None,
        language=language
    )
    if system_default:
        return system_default.content
    
    # 层级4：代码硬编码
    if language == "zh-CN":
        return APERAG_AGENT_INSTRUCTION_ZH
    else:
        return APERAG_AGENT_INSTRUCTION_EN
```

#### resolve_agent_query_prompt
```python
async def resolve_agent_query_prompt(bot, user_id: str, language: str) -> str:
    """
    解析Agent查询prompt模板
    
    优先级：
    1. Bot.config.agent.query_prompt_template
    2. prompt_template表（scope='user', prompt_type='agent_query'）
    3. prompt_template表（scope='system', prompt_type='agent_query'）
    4. 代码硬编码（DEFAULT_AGENT_QUERY_PROMPT_EN/ZH）
    
    Args:
        bot: Bot对象
        user_id: 用户ID
        language: 语言代码
        
    Returns:
        解析后的query prompt模板内容
    """
    # 实现逻辑与 resolve_agent_system_prompt 类似
    # ...
```

#### resolve_index_prompt
```python
async def resolve_index_prompt(
    collection, 
    prompt_type: str,  # "graph", "summary", "vision"
    user_id: str
) -> str:
    """
    解析索引prompt
    
    优先级：
    1. Collection.config.index_prompts.{type}
    2. prompt_template表（scope='user', prompt_type='index_{type}'）
    3. prompt_template表（scope='system', prompt_type='index_{type}'）
    4. 代码硬编码
    
    Args:
        collection: Collection对象
        prompt_type: Prompt类型 (graph, summary, vision)
        user_id: 用户ID
        
    Returns:
        解析后的index prompt内容
    """
    from aperag.db.ops import async_db_ops
    
    # 层级1：Collection配置
    if collection.config:
        try:
            import json
            config_dict = json.loads(collection.config) if isinstance(collection.config, str) else collection.config
            index_prompts = config_dict.get("index_prompts", {})
            if index_prompts.get(prompt_type):
                return index_prompts[prompt_type]
        except:
            pass
    
    # 层级2：用户默认
    db_prompt_type = f"index_{prompt_type}"
    collection_language = "zh-CN"  # 从collection.config.language获取
    try:
        config_dict = json.loads(collection.config) if isinstance(collection.config, str) else collection.config
        collection_language = config_dict.get("language", "zh-CN")
    except:
        pass
    
    user_default = await async_db_ops.query_prompt_template(
        prompt_type=db_prompt_type,
        scope="user",
        user_id=user_id,
        language=collection_language
    )
    if user_default:
        return user_default.content
    
    # 层级3：系统默认
    system_default = await async_db_ops.query_prompt_template(
        prompt_type=db_prompt_type,
        scope="system",
        user_id=None,
        language=collection_language
    )
    if system_default:
        return system_default.content
    
    # 层级4：代码硬编码
    return get_hardcoded_index_prompt(prompt_type)


def get_hardcoded_index_prompt(prompt_type: str) -> str:
    """获取代码硬编码的索引prompt（最终fallback）"""
    if prompt_type == "graph":
        from aperag.graph.lightrag.prompt import PROMPTS
        return PROMPTS["entity_extraction"]
    elif prompt_type == "summary":
        return """Provide a comprehensive summary of the following document..."""
    elif prompt_type == "vision":
        return """Analyze the provided image and extract its content with high fidelity..."""
    else:
        return None
```

---

## 二、Agent对话集成

### 2.1 文件位置
`aperag/service/agent_chat_service.py`

### 2.2 改造点1：_get_agent_session方法

**位置**：约第402-461行

**当前代码**：
```python
# 约437-439行
system_prompt = (
    custom_system_prompt if custom_system_prompt 
    else get_agent_system_prompt(language=agent_message.language)
)
```

**改造后**：
```python
from aperag.service.prompt_template_service import resolve_agent_system_prompt

system_prompt = await resolve_agent_system_prompt(
    bot=bot,
    user_id=user,
    language=agent_message.language
)
```

**影响**：
- 需要将bot对象传递到这个方法中
- 当前方法签名可能需要调整

### 2.3 改造点2：process_agent_message方法

**位置**：约第463-551行

**当前代码**：
```python
# 约518-521行
comprehensive_prompt = build_agent_query_prompt(
    chat_id, agent_message=merged_agent_message, user=user, custom_template=custom_query_prompt
)
```

**改造后**：
```python
from aperag.service.prompt_template_service import resolve_agent_query_prompt

query_prompt_template = await resolve_agent_query_prompt(
    bot=bot,
    user_id=user,
    language=merged_agent_message.language
)

comprehensive_prompt = build_agent_query_prompt(
    chat_id, 
    agent_message=merged_agent_message, 
    user=user, 
    custom_template=query_prompt_template
)
```

**注意事项**：
- 需要确保bot对象在此方法中可用
- 当前代码使用custom_query_prompt参数，需要替换为解析服务

---

## 三、Graph索引集成（LightRAG）

### 3.1 文件位置
`aperag/graph/lightrag_manager.py`

### 3.2 改造点：create_lightrag_instance函数

**位置**：约第59-123行

**当前代码**：
```python
async def create_lightrag_instance(collection: Collection) -> LightRAG:
    # ... 获取配置 ...
    
    rag = LightRAG(
        working_dir=working_dir,
        entity_types=entity_types,
        # ... 其他配置 ...
    )
    
    return rag
```

**改造后**：
```python
from aperag.service.prompt_template_service import resolve_index_prompt

async def create_lightrag_instance(collection: Collection) -> LightRAG:
    # ... 获取配置 ...
    
    # 解析自定义graph prompt
    custom_graph_prompt = await resolve_index_prompt(
        collection=collection,
        prompt_type="graph",
        user_id=collection.user
    )
    
    # 创建LightRAG实例
    rag = LightRAG(
        working_dir=working_dir,
        entity_types=entity_types,
        # ... 其他配置 ...
    )
    
    # 如果有自定义prompt，需要覆盖LightRAG的默认prompt
    # 方案1：扩展LightRAG支持custom_prompts参数（需要修改lightrag.py）
    # 方案2：运行时替换（临时方案，但要注意线程安全）
    
    return rag
```

### 3.3 LightRAG扩展（可选）

**文件位置**：`aperag/graph/lightrag/lightrag.py`

**建议扩展**：
```python
@dataclass
class LightRAG:
    # ... 现有字段 ...
    
    # 新增：自定义prompts字典
    custom_prompts: Optional[Dict[str, str]] = None
    
    def _get_prompt(self, key: str) -> str:
        """获取prompt，支持自定义覆盖"""
        if self.custom_prompts and key in self.custom_prompts:
            return self.custom_prompts[key]
        return PROMPTS[key]
```

**使用**：
```python
rag = LightRAG(
    working_dir=working_dir,
    entity_types=entity_types,
    custom_prompts={
        "entity_extraction": custom_graph_prompt
    } if custom_graph_prompt else None,
    # ... 其他配置 ...
)
```

---

## 四、Summary索引集成

### 4.1 文件位置
`aperag/index/summary_index.py`

### 4.2 改造点：create_index方法

**位置**：约第46-87行

**当前代码**：
```python
def create_index(self, document_id: str, content: str, doc_parts: List[Any], collection, **kwargs):
    # ... 现有逻辑 ...
    
    # 使用默认的map-reduce prompt生成摘要
    summary = self._generate_document_summary(content, doc_parts, collection)
```

**改造后**：
```python
from aperag.service.prompt_template_service import resolve_index_prompt

def create_index(self, document_id: str, content: str, doc_parts: List[Any], collection, **kwargs):
    # ... 现有逻辑 ...
    
    # 解析自定义summary prompt
    custom_summary_prompt = await resolve_index_prompt(
        collection=collection,
        prompt_type="summary",
        user_id=collection.user
    )
    
    if custom_summary_prompt:
        # 使用自定义prompt生成摘要
        summary = self._generate_summary_with_custom_prompt(
            content, doc_parts, collection, custom_summary_prompt
        )
    else:
        # 使用默认的map-reduce逻辑
        summary = self._generate_document_summary(content, doc_parts, collection)
```

**注意事项**：
- `create_index` 可能是同步方法，需要确认是否可以改为异步
- 可能需要新增 `_generate_summary_with_custom_prompt` 方法

---

## 五、Vision索引集成

### 5.1 文件位置
`aperag/index/vision_index.py`

### 5.2 改造点：create_index方法

**位置**：约第146行附近

**当前代码**：
```python
prompt = """Analyze the provided image and extract its content with high fidelity. Follow these instructions precisely..."""
```

**改造后**：
```python
from aperag.service.prompt_template_service import resolve_index_prompt

# 解析自定义vision prompt
custom_vision_prompt = await resolve_index_prompt(
    collection=collection,
    prompt_type="vision",
    user_id=collection.user
)

prompt = custom_vision_prompt if custom_vision_prompt else """
Analyze the provided image and extract its content with high fidelity...
"""
```

---

## 六、Collection API扩展

### 6.1 说明
Collection的API已经通过Schema扩展支持`index_prompts`字段，无需额外改造。

### 6.2 使用示例

**更新Collection配置**：
```bash
PUT /api/v1/collections/{collection_id}
Content-Type: application/json

{
  "title": "医疗知识库",
  "config": {
    "enable_knowledge_graph": true,
    "enable_summary": true,
    "knowledge_graph_config": {
      "entity_types": ["疾病", "药物", "症状", "治疗方案"]
    },
    "index_prompts": {
      "graph": "从医疗文本中提取实体和关系。实体类型：{entity_types}。要求：1. 识别中文医疗术语... 2. 提取疾病-药物、症状-疾病等关系...",
      "summary": "生成医疗文档的结构化摘要，包括：1. 主要诊断 2. 治疗方案 3. 用药建议 4. 注意事项"
    }
  }
}
```

### 6.3 可选增强：变更提示

如果希望在用户修改`index_prompts`后给出"需重建索引"的提示，可以在Collection更新API中添加逻辑：

**文件**：`aperag/service/collection_service.py`

**位置**：`update_collection`方法

**增强逻辑**：
```python
async def update_collection(self, user: str, collection_id: str, update_data: dict):
    # ... 现有更新逻辑 ...
    
    # 检测index_prompts是否变更
    warnings = []
    if "config" in update_data and "index_prompts" in update_data["config"]:
        warnings.append("索引Prompt配置已变更，建议重建相关索引以使新配置生效")
    
    # 在返回结果中包含warnings
    return {
        "collection": updated_collection,
        "warnings": warnings
    }
```

---

## 七、实施优先级建议

### 高优先级（核心功能）
1. ✅ **Prompt解析服务**：实现三个resolve方法
2. **Agent集成**：改造agent_chat_service.py
3. **Graph索引集成**：改造lightrag_manager.py和LightRAG

### 中优先级（常用功能）
4. **Summary索引集成**：改造summary_index.py
5. **Collection API增强**：添加变更提示

### 低优先级（可选功能）
6. **Vision索引集成**：改造vision_index.py
7. **性能优化**：添加缓存机制
8. **使用统计**：记录prompt使用情况

---

## 八、验证和测试

### 8.1 API测试

**用户默认Prompt**：
```bash
# 1. 设置用户默认的Agent System Prompt
curl -X PUT http://localhost:8000/api/v1/prompts/defaults/agent \
  -H "Content-Type: application/json" \
  -d '{
    "language": "zh-CN",
    "system": "你是一个专业的技术支持助手，擅长解决软件问题",
    "query": "{% set collection_list = [] %}..."
  }'

# 2. 获取用户默认配置
curl http://localhost:8000/api/v1/prompts/defaults?language=zh-CN

# 3. 获取系统默认配置（参考）
curl http://localhost:8000/api/v1/prompts/system-defaults?type=agent_system&language=zh-CN

# 4. 预览prompt渲染
curl -X POST http://localhost:8000/api/v1/prompts/preview \
  -H "Content-Type: application/json" \
  -d '{
    "type": "agent_query",
    "template": "用户查询：{{ query }}",
    "variables": {"query": "测试查询"}
  }'

# 5. 验证prompt语法
curl -X POST http://localhost:8000/api/v1/prompts/validate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "agent_query",
    "template": "{% set x = 1 %}{{ query }}"
  }'
```

**Collection索引Prompt**：
```bash
# 更新Collection配置
curl -X PUT http://localhost:8000/api/v1/collections/{collection_id} \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "index_prompts": {
        "graph": "自定义的图索引prompt...",
        "summary": "自定义的摘要prompt..."
      }
    }
  }'
```

### 8.2 集成测试流程

**Agent对话测试**：
1. 创建Bot（不配置prompt） → 应使用用户默认
2. 用户设置默认Agent prompt
3. 发起对话 → 验证使用了用户默认prompt
4. 更新Bot配置（设置prompt） → 应优先使用Bot配置

**索引构建测试**：
1. 创建Collection（不配置index_prompts） → 应使用系统默认
2. 用户设置默认索引prompt
3. 上传文档构建索引 → 验证使用了用户默认prompt
4. 更新Collection配置（设置index_prompts）
5. 重建索引 → 验证使用了Collection配置

---

## 九、常见问题

### Q1: Bot.config是字符串还是对象？
**A**: 数据库中是Text类型（JSON字符串），读取后需要json.loads()解析。

### Q2: 异步方法在同步context中如何调用？
**A**: 如果indexer的create_index是同步方法，可能需要：
- 改为异步方法（推荐）
- 或使用asyncio.run()包装（不推荐，可能有事件循环冲突）

### Q3: LightRAG的PROMPTS是全局变量，如何实现实例级覆盖？
**A**: 
- 方案1：扩展LightRAG支持custom_prompts参数（推荐）
- 方案2：创建实例时深拷贝PROMPTS字典
- 方案3：运行时临时替换（需注意线程安全）

### Q4: 用户修改索引prompt后，旧索引怎么办？
**A**: 
- 旧索引仍然有效，但是用旧prompt生成的
- 建议在API响应中提示用户"需重建索引"
- 可以添加Collection.index_config_changed字段标记（可选）

---

## 十、下一步行动清单

- [ ] 实现 `resolve_agent_system_prompt()`
- [ ] 实现 `resolve_agent_query_prompt()`
- [ ] 实现 `resolve_index_prompt()`
- [ ] 改造 `agent_chat_service.py`（两处）
- [ ] 改造 `lightrag_manager.py`
- [ ] 扩展 `LightRAG` 支持 `custom_prompts`
- [ ] 改造 `summary_index.py`
- [ ] 改造 `vision_index.py`
- [ ] 添加Collection更新提示
- [ ] 编写集成测试用例
- [ ] 更新用户文档

---

## 十一、参考文档

- [设计方案](../../.cursor/plans/自定义prompt模板系统_8a863299.plan.md)
- [OpenAPI定义](../../aperag/api/components/schemas/prompt.yaml)
- [Repository实现](../../aperag/db/repositories/prompt_template.py)
- [API实现](../../aperag/views/prompts.py)
