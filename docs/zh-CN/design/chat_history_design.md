# ApeRAG 聊天历史消息数据流程

## 概述

本文档详细说明ApeRAG项目中聊天历史消息的完整数据流程，从前端API调用到后端存储的全链路实现。

**核心接口**: `GET /api/v1/bots/{bot_id}/chats/{chat_id}`

## 数据流图

```
┌─────────────────┐
│    Frontend     │
│   (Next.js)     │
└────────┬────────┘
         │ GET /api/v1/bots/{bot_id}/chats/{chat_id}
         ▼
┌─────────────────────────────────────────────┐
│  View Layer                                 │
│  aperag/views/chat.py                       │
│  - get_chat_view()                          │
│  - JWT身份验证                               │
│  - 参数验证                                  │
└────────┬────────────────────────────────────┘
         │ chat_service_global.get_chat()
         ▼
┌─────────────────────────────────────────────┐
│  Service Layer                              │
│  aperag/service/chat_service.py             │
│  - get_chat()                               │
│  - 业务逻辑编排                              │
└────────┬────────────────────────────────────┘
         │
         ├──────────────┬─────────────┐
         │              │             │
         ▼              ▼             ▼
┌────────────┐   ┌───────────┐   ┌──────────────┐
│ PostgreSQL │   │   Redis   │   │ PostgreSQL   │
│  chat表    │   │ 消息历史   │   │ feedback表   │
│(基本信息)   │   │(会话内容)  │   │(用户反馈)     │
└────────────┘   └───────────┘   └──────────────┘
       │              │                  │
       └──────────────┴──────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │  ChatDetails │
               │  (组装响应)   │
               └──────────────┘
```

## 完整流程详解

### 1. View层 - HTTP请求处理

**文件**: `aperag/views/chat.py`

```python
@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat_view(
    request: Request, 
    bot_id: str, 
    chat_id: str, 
    user: User = Depends(required_user)
) -> view_models.ChatDetails:
    return await chat_service_global.get_chat(str(user.id), bot_id, chat_id)
```

**职责**:
- 接收HTTP GET请求
- JWT Token身份验证
- 提取路径参数 (bot_id, chat_id)
- 调用Service层
- 返回`ChatDetails`响应

### 2. Service层 - 业务逻辑编排

**文件**: `aperag/service/chat_service.py`

```python
async def get_chat(self, user: str, bot_id: str, chat_id: str) -> view_models.ChatDetails:
    from aperag.utils.history import query_chat_messages
    
    # Step 1: 从PostgreSQL查询Chat基本信息
    chat = await self.db_ops.query_chat(user, bot_id, chat_id)
    if chat is None:
        raise ChatNotFoundException(chat_id)
    
    # Step 2: 从Redis查询聊天消息历史
    messages = await query_chat_messages(user, chat_id)
    
    # Step 3: 构建响应对象（消息中已包含feedback信息）
    chat_obj = self.build_chat_response(chat)
    return ChatDetails(**chat_obj.model_dump(), history=messages)
```

**核心逻辑**:

1. **查询Chat元数据** (PostgreSQL)
2. **查询消息历史** (Redis + PostgreSQL反馈信息)
3. **组装完整响应**

### 3. 数据存储层

#### 3.1 PostgreSQL - Chat基本信息

**表**: `chat`

**文件**: `aperag/db/models.py`

```python
class Chat(Base):
    __tablename__ = "chat"
    
    id = Column(String(24), primary_key=True)           # chat_xxxx
    user = Column(String(256), nullable=False)          # 用户ID
    bot_id = Column(String(24), nullable=False)         # Bot ID
    title = Column(String(256))                         # 会话标题
    peer_type = Column(EnumColumn(ChatPeerType))        # 对话类型
    peer_id = Column(String(256))                       # 对话ID
    status = Column(EnumColumn(ChatStatus))             # 状态
    gmt_created = Column(DateTime(timezone=True))       # 创建时间
    gmt_updated = Column(DateTime(timezone=True))       # 更新时间
    gmt_deleted = Column(DateTime(timezone=True))       # 删除时间（软删除）
```

**用途**: 存储Chat会话的元数据，不包含具体消息内容

#### 3.2 Redis - 聊天消息历史

**文件**: `aperag/utils/history.py`

**Key格式**: `message_store:{chat_id}`

**数据结构**: Redis List (使用LPUSH，最新消息在前)

**核心类**:

```python
class RedisChatMessageHistory:
    def __init__(self, session_id: str, key_prefix: str = "message_store:"):
        self.session_id = session_id
        self.key_prefix = key_prefix
    
    @property
    def key(self) -> str:
        return self.key_prefix + self.session_id  # message_store:chat_abc123
    
    @property
    async def messages(self) -> List[StoredChatMessage]:
        # 从Redis读取所有消息
        _items = await self.redis_client.lrange(self.key, 0, -1)
        # 反转为时间顺序（因为LPUSH导致最新在前）
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        return [storage_dict_to_message(item) for item in items]
```

**消息查询函数**:

```python
async def query_chat_messages(user: str, chat_id: str):
    """查询聊天消息并转换为前端格式"""
    
    # 1. 从Redis获取消息历史
    chat_history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    stored_messages = await chat_history.messages
    
    if not stored_messages:
        return []
    
    # 2. 从PostgreSQL获取反馈信息
    feedbacks = await async_db_ops.query_chat_feedbacks(user, chat_id)
    feedback_map = {feedback.message_id: feedback for feedback in feedbacks}
    
    # 3. 转换为前端格式并附加反馈信息
    result = []
    for stored_message in stored_messages:
        # 转换为前端格式
        chat_message_list = stored_message.to_frontend_format()
        
        # 为AI消息添加反馈数据
        for chat_msg in chat_message_list:
            feedback = feedback_map.get(chat_msg.id)
            if feedback and chat_msg.role == "ai":
                chat_msg.feedback = Feedback(
                    type=feedback.type,
                    tag=feedback.tag,
                    message=feedback.message
                )
        
        result.append(chat_message_list)
    
    return result  # [[message1_parts], [message2_parts], [message3_parts], ...]
```

#### 3.3 PostgreSQL - 用户反馈信息

**表**: `message_feedback`

```python
class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    
    user = Column(String(256), nullable=False)          # 用户ID
    chat_id = Column(String(24), primary_key=True)      # 会话ID
    message_id = Column(String(256), primary_key=True)  # 消息ID
    type = Column(EnumColumn(MessageFeedbackType))      # like/dislike
    tag = Column(EnumColumn(MessageFeedbackTag))        # 反馈标签
    message = Column(Text)                              # 反馈内容
    question = Column(Text)                             # 原始问题
    original_answer = Column(Text)                      # 原始回答
    status = Column(EnumColumn(MessageFeedbackStatus))  # 状态
    gmt_created = Column(DateTime(timezone=True))
    gmt_updated = Column(DateTime(timezone=True))
```

**用途**: 存储用户对AI回复的反馈（点赞/点踩），用于质量监控和模型优化

## 数据格式详解

### 存储格式 (Redis)

消息在Redis中以JSON格式存储，采用**Part-Based设计**：

#### StoredChatMessage - 一条完整消息

```python
class StoredChatMessage(BaseModel):
    """一条完整消息（用户的一条消息 或 AI的一条消息）"""
    parts: List[StoredChatMessagePart]  # 消息的多个部分
    files: List[Dict[str, Any]]         # 关联的上传文件
```

#### StoredChatMessagePart - 消息的一个部分

```python
class StoredChatMessagePart(BaseModel):
    """消息的单个部分（原子单元）"""
    
    # 标识信息
    chat_id: str              # 所属会话
    message_id: str           # 所属消息（同一条消息的多个part共享）
    part_id: str              # 部分的唯一ID
    timestamp: float          # 生成时间戳
    
    # 内容分类
    type: Literal["message", "tool_call_result", "thinking", "references"]
    role: Literal["human", "ai", "system"]
    content: str
    
    # 扩展字段
    references: List[Dict]    # 文档引用
    urls: List[str]           # URL引用
    metadata: Optional[Dict]  # 额外元数据
```

#### Part类型说明

| Type | 说明 | 包含在LLM上下文 |
|------|------|---------------|
| `message` | 主要对话内容 | ✅ 是 |
| `tool_call_result` | 工具调用过程 | ❌ 否（仅展示） |
| `thinking` | AI思考过程 | ❌ 否（仅展示） |
| `references` | 文档引用和链接 | ❌ 否（仅展示） |

**设计原因**: AI的一条回复包含多个阶段（工具调用、思考、回答、引用），这些内容按时序产生且互相穿插，单一字段无法表达。用户的消息通常只有1个part（type="message"），但也支持多个part以保持结构一致性。

#### Redis存储示例

**用户消息**:
```json
{
  "parts": [
    {
      "chat_id": "chat_abc123",
      "message_id": "uuid-1",
      "part_id": "uuid-part-1",
      "timestamp": 1699999999.0,
      "type": "message",
      "role": "human",
      "content": "什么是LightRAG？",
      "references": [],
      "urls": [],
      "metadata": null
    }
  ],
  "files": []
}
```

**AI回复（包含多个part）**:
```json
{
  "parts": [
    {
      "message_id": "uuid-2",
      "part_id": "uuid-part-2",
      "type": "tool_call_result",
      "role": "ai",
      "content": "正在检索知识库...",
      "timestamp": 1699999999.1
    },
    {
      "message_id": "uuid-2",
      "part_id": "uuid-part-3",
      "type": "message",
      "role": "ai",
      "content": "LightRAG是一个轻量级的RAG框架，由ApeCloud团队深度改造...",
      "timestamp": 1699999999.5
    },
    {
      "message_id": "uuid-2",
      "part_id": "uuid-part-4",
      "type": "references",
      "role": "ai",
      "content": "",
      "references": [
        {
          "score": 0.95,
          "text": "LightRAG架构说明...",
          "metadata": {"source": "lightrag_doc.pdf", "page": 3}
        }
      ],
      "urls": ["https://github.com/HKUDS/LightRAG"],
      "timestamp": 1699999999.6
    }
  ],
  "files": []
}
```

### API响应格式

**ChatDetails Schema** (`aperag/api/components/schemas/chat.yaml`):

```yaml
chatDetails:
  type: object
  properties:
    id: string                    # chat_abc123
    title: string                 # 会话标题
    bot_id: string                # bot_xyz
    peer_id: string
    peer_type: string             # system/feishu/weixin/web
    status: string                # active/archived
    created: string               # ISO 8601
    updated: string               # ISO 8601
    history:                      # 二维数组
      type: array
      description: 对话历史，每个元素是一条消息
      items:
        type: array
        description: 一条消息包含多个parts（工具调用、思考、回答、引用等）
        items:
          $ref: '#/chatMessage'
```

**ChatMessage Schema**:

```yaml
chatMessage:
  type: object
  properties:
    id: string                    # message_id（同一轮次相同）
    part_id: string               # part_id（每个part唯一）
    type: string                  # message/tool_call_result/thinking/references
    timestamp: number             # Unix时间戳
    role: string                  # human/ai
    data: string                  # 消息内容
    references:                   # 文档引用（可选）
      type: array
      items:
        type: object
        properties:
          score: number
          text: string
          metadata: object
    urls:                         # URL引用（可选）
      type: array
      items:
        type: string
    feedback:                     # 用户反馈（可选）
      type: object
      properties:
        type: string              # like/dislike
        tag: string
        message: string
    files:                        # 关联文件（可选）
      type: array
```

### 前端接收示例

```json
{
  "id": "chat_abc123",
  "title": "关于LightRAG的讨论",
  "bot_id": "bot_xyz",
  "status": "active",
  "created": "2025-01-01T00:00:00Z",
  "updated": "2025-01-01T01:00:00Z",
  "history": [
    [
      {
        "id": "uuid-1",
        "part_id": "uuid-part-1",
        "type": "message",
        "timestamp": 1699999999.0,
        "role": "human",
        "data": "什么是LightRAG？",
        "files": []
      }
    ],
    [
      {
        "id": "uuid-2",
        "part_id": "uuid-part-2",
        "type": "tool_call_result",
        "timestamp": 1699999999.1,
        "role": "ai",
        "data": "正在检索知识库...",
        "files": []
      },
      {
        "id": "uuid-2",
        "part_id": "uuid-part-3",
        "type": "message",
        "timestamp": 1699999999.5,
        "role": "ai",
        "data": "LightRAG是一个轻量级的RAG框架...",
        "files": []
      },
      {
        "id": "uuid-2",
        "part_id": "uuid-part-4",
        "type": "references",
        "timestamp": 1699999999.6,
        "role": "ai",
        "data": "",
        "references": [
          {
            "score": 0.95,
            "text": "LightRAG架构说明...",
            "metadata": {"source": "lightrag_doc.pdf"}
          }
        ],
        "urls": ["https://github.com/HKUDS/LightRAG"],
        "files": []
      }
    ]
  ]
}
```

**注意**: `history`是二维数组，第一维是消息序列（按时间顺序），第二维是该条消息的多个part。例如：
- `history[0]` = 用户的第1条消息的parts（通常只有1个part）
- `history[1]` = AI的第1条回复的parts（可能有多个part：工具调用、思考、回答、引用）
- `history[2]` = 用户的第2条消息的parts
- `history[3]` = AI的第2条回复的parts
- ...

## 消息写入流程

### WebSocket实时聊天

**接口**: `WS /api/v1/bots/{bot_id}/chats/{chat_id}/connect`

```python
async def handle_websocket_chat(websocket: WebSocket, user: str, bot_id: str, chat_id: str):
    await websocket.accept()
    
    while True:
        # 1. 接收用户消息
        data = json.loads(await websocket.receive_text())
        message_content = data.get("data")
        message_id = str(uuid.uuid4())
        
        # 2. 写入用户消息到Redis
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
        await history.add_user_message(message_content, message_id, files=data.get("files", []))
        
        # 3. 执行Flow获取AI响应
        flow = FlowParser.parse(flow_config)
        engine = FlowEngine()
        _, system_outputs = await engine.execute_flow(flow, initial_data)
        
        # 4. 流式传输AI响应
        async for chunk in async_generator():
            await websocket.send_text(success_response(message_id, chunk))
            full_message += chunk
        
        # 5. AI消息写入Redis（由Flow内部的Runner自动调用history.add_ai_message()）
```

**写入方法**:

```python
# 用户消息
await history.add_user_message(
    message="用户的问题",
    message_id="uuid-1",
    files=[{"id": "file1", "name": "doc.pdf"}]
)

# AI消息
await history.add_ai_message(
    content="AI的回答",
    chat_id="chat_abc123",
    message_id="uuid-2",
    tool_use_list=[{"data": "工具调用信息"}],  # 可选
    references=[{"score": 0.95, "text": "..."}],  # 可选
    urls=["https://example.com"]  # 可选
)
```

## 设计特点

### 1. 混合存储架构

| 存储 | 内容 | 原因 |
|------|------|------|
| PostgreSQL | Chat元数据 | 持久化、支持复杂查询 |
| Redis | 消息历史 | 高性能读写、支持TTL |
| PostgreSQL | 用户反馈 | 持久化、用于分析 |

**优势**:
- 性能优化：消息历史使用Redis快速读写
- 数据持久化：重要元数据存储在PostgreSQL
- 灵活性：可独立配置TTL、备份策略

### 2. Part-Based消息设计

**核心价值**:
- ✅ 支持复杂的AI回复流程（工具调用→思考→回答→引用）
- ✅ 前端可差异化渲染不同类型的内容
- ✅ 完整记录时序关系（通过timestamp）
- ✅ 灵活扩展（新增type无需改表结构）

**为什么一条消息需要多个part**:

AI的一条回复过程是时序产生、互相穿插的，例如：
1. 🔍 Part1 (tool_call_result): "正在查询数据库..."
2. 💭 Part2 (thinking): "找到了327条记录..."
3. 🔍 Part3 (tool_call_result): "正在计算增长率..."
4. 💭 Part4 (thinking): "环比增长15%..."
5. 💬 Part5 (message): "根据数据分析，Q4表现优秀..."
6. 📚 Part6 (references): [文档1, 文档2]

这6个part属于AI的**一条消息**（共享同一个message_id），单一字段无法表达这种复杂的时序关系。

### 3. 格式转换解耦

提供三种格式转换：

```python
class StoredChatMessage:
    def to_frontend_format(self) -> List[ChatMessage]:
        """转换为前端展示格式"""
        # 包含所有types的parts
        
    def to_openai_format(self) -> List[Dict]:
        """转换为LLM调用格式"""
        # 只包含type="message"的parts
        
    def get_main_content(self) -> str:
        """获取主要回答内容"""
        # 第一个type="message"的content
```

**优势**:
- 内部存储格式与外部接口解耦
- 支持不同的消费场景
- LLM上下文只包含实际对话内容，不包含工具调用和思考过程

### 4. 三级ID设计

```python
chat_id = "chat_abc123"           # 会话级别
message_id = "uuid-msg-1"         # 消息级别（同一条消息的多个part共享）
part_id = "uuid-part-1"           # 部分级别（每个part独立）
```

**作用**:
- `chat_id`: 标识一个聊天会话
- `message_id`: 将同一条消息的多个part分组（用于前端展示和反馈关联）
- `part_id`: 每个part独立标识（用于单独操作，如复制、引用）

## 性能考虑

### Redis优化
- **List数据结构**: LPUSH O(1), LRANGE O(N)
- **可选TTL**: 自动过期历史消息
- **连接池复用**: 全局Redis客户端

### PostgreSQL优化
- **索引**: user, bot_id, chat_id, status字段
- **软删除**: 使用gmt_deleted
- **分页查询**: list_chats支持分页

### 传输优化
- **WebSocket流式**: 边生成边发送
- **增量更新**: 只传输新的part
- **按需加载**: 懒加载历史消息

## 相关文件

### 核心实现
- `aperag/views/chat.py` - View层接口
- `aperag/service/chat_service.py` - Service层业务逻辑
- `aperag/utils/history.py` - Redis消息历史管理
- `aperag/chat/history/message.py` - 消息数据结构
- `aperag/db/models.py` - 数据库模型
- `aperag/db/repositories/chat.py` - Chat数据库操作
- `aperag/api/components/schemas/chat.yaml` - OpenAPI Schema

### 前端实现
- `web/src/app/workspace/bots/[botId]/chats/[chatId]/page.tsx` - 聊天详情页面
- `web/src/components/chat/chat-messages.tsx` - 消息展示组件

## 总结

ApeRAG的聊天历史消息系统采用**混合存储 + Part-Based消息设计**：

1. **PostgreSQL**存储Chat元数据和反馈（持久化、可查询）
2. **Redis**存储消息历史（高性能、支持过期）
3. **Part-Based设计**支持复杂的AI回复流程（工具调用、思考、回答、引用）
4. **三级ID设计**支持消息分组和独立操作
5. **清晰的分层架构**（View → Service → Repository → Storage）

这种设计既保证了性能，又支持复杂的AI交互场景，同时具有良好的可扩展性。
