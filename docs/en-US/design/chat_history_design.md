# ApeRAG Chat History Message Data Flow

## Overview

This document details the complete data flow of chat history messages in the ApeRAG project, covering the full-stack implementation from frontend API calls to backend storage.

**Core API**: `GET /api/v1/bots/{bot_id}/chats/{chat_id}`

## Data Flow Diagram

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
│  - JWT Authentication                       │
│  - Parameter Validation                     │
└────────┬────────────────────────────────────┘
         │ chat_service_global.get_chat()
         ▼
┌─────────────────────────────────────────────┐
│  Service Layer                              │
│  aperag/service/chat_service.py             │
│  - get_chat()                               │
│  - Business Logic Orchestration             │
└────────┬────────────────────────────────────┘
         │
         ├──────────────┬─────────────┐
         │              │             │
         ▼              ▼             ▼
┌────────────┐   ┌───────────┐   ┌──────────────┐
│ PostgreSQL │   │   Redis   │   │ PostgreSQL   │
│ chat table │   │  Message  │   │feedback table│
│ (Metadata) │   │  History  │   │(User Feedback)│
└────────────┘   └───────────┘   └──────────────┘
       │              │                  │
       └──────────────┴──────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │ ChatDetails  │
               │  (Assemble)  │
               └──────────────┘
```

## Detailed Flow

### 1. View Layer - HTTP Request Handling

**File**: `aperag/views/chat.py`

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

**Responsibilities**:
- Receive HTTP GET requests
- JWT Token authentication
- Extract path parameters (bot_id, chat_id)
- Call Service layer
- Return `ChatDetails` response

### 2. Service Layer - Business Logic Orchestration

**File**: `aperag/service/chat_service.py`

```python
async def get_chat(self, user: str, bot_id: str, chat_id: str) -> view_models.ChatDetails:
    from aperag.utils.history import query_chat_messages
    
    # Step 1: Query Chat metadata from PostgreSQL
    chat = await self.db_ops.query_chat(user, bot_id, chat_id)
    if chat is None:
        raise ChatNotFoundException(chat_id)
    
    # Step 2: Query message history from Redis
    messages = await query_chat_messages(user, chat_id)
    
    # Step 3: Build response object (messages already include feedback info)
    chat_obj = self.build_chat_response(chat)
    return ChatDetails(**chat_obj.model_dump(), history=messages)
```

**Core Logic**:

1. **Query Chat Metadata** (PostgreSQL)
2. **Query Message History** (Redis + PostgreSQL feedback)
3. **Assemble Complete Response**

### 3. Data Storage Layer

#### 3.1 PostgreSQL - Chat Metadata

**Table**: `chat`

**File**: `aperag/db/models.py`

```python
class Chat(Base):
    __tablename__ = "chat"
    
    id = Column(String(24), primary_key=True)           # chat_xxxx
    user = Column(String(256), nullable=False)          # User ID
    bot_id = Column(String(24), nullable=False)         # Bot ID
    title = Column(String(256))                         # Chat title
    peer_type = Column(EnumColumn(ChatPeerType))        # Peer type
    peer_id = Column(String(256))                       # Peer ID
    status = Column(EnumColumn(ChatStatus))             # Status
    gmt_created = Column(DateTime(timezone=True))       # Created time
    gmt_updated = Column(DateTime(timezone=True))       # Updated time
    gmt_deleted = Column(DateTime(timezone=True))       # Deleted time (soft delete)
```

**Purpose**: Store Chat session metadata without actual message content

#### 3.2 Redis - Message History

**File**: `aperag/utils/history.py`

**Key Format**: `message_store:{chat_id}`

**Data Structure**: Redis List (using LPUSH, newest messages first)

**Core Class**:

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
        # Read all messages from Redis
        _items = await self.redis_client.lrange(self.key, 0, -1)
        # Reverse to chronological order (LPUSH puts newest first)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        return [storage_dict_to_message(item) for item in items]
```

**Message Query Function**:

```python
async def query_chat_messages(user: str, chat_id: str):
    """Query chat messages and convert to frontend format"""
    
    # 1. Get message history from Redis
    chat_history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    stored_messages = await chat_history.messages
    
    if not stored_messages:
        return []
    
    # 2. Get feedback info from PostgreSQL
    feedbacks = await async_db_ops.query_chat_feedbacks(user, chat_id)
    feedback_map = {feedback.message_id: feedback for feedback in feedbacks}
    
    # 3. Convert to frontend format and attach feedback info
    result = []
    for stored_message in stored_messages:
        # Convert to frontend format
        chat_message_list = stored_message.to_frontend_format()
        
        # Add feedback data for AI messages
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

#### 3.3 PostgreSQL - User Feedback

**Table**: `message_feedback`

```python
class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    
    user = Column(String(256), nullable=False)          # User ID
    chat_id = Column(String(24), primary_key=True)      # Chat ID
    message_id = Column(String(256), primary_key=True)  # Message ID
    type = Column(EnumColumn(MessageFeedbackType))      # like/dislike
    tag = Column(EnumColumn(MessageFeedbackTag))        # Feedback tag
    message = Column(Text)                              # Feedback content
    question = Column(Text)                             # Original question
    original_answer = Column(Text)                      # Original answer
    status = Column(EnumColumn(MessageFeedbackStatus))  # Status
    gmt_created = Column(DateTime(timezone=True))
    gmt_updated = Column(DateTime(timezone=True))
```

**Purpose**: Store user feedback on AI responses (like/dislike) for quality monitoring and model optimization

## Data Format Specification

### Storage Format (Redis)

Messages in Redis are stored in JSON format using **Part-Based Design**:

#### StoredChatMessage - A Complete Message

```python
class StoredChatMessage(BaseModel):
    """A complete message (either a user message or an AI message)"""
    parts: List[StoredChatMessagePart]  # Multiple parts of the message
    files: List[Dict[str, Any]]         # Associated uploaded files
```

#### StoredChatMessagePart - A Message Part

```python
class StoredChatMessagePart(BaseModel):
    """A single part of a message (atomic unit)"""
    
    # Identification
    chat_id: str              # Chat session ID
    message_id: str           # Message ID (shared by multiple parts of the same message)
    part_id: str              # Unique part ID
    timestamp: float          # Generation timestamp
    
    # Content Classification
    type: Literal["message", "tool_call_result", "thinking", "references"]
    role: Literal["human", "ai", "system"]
    content: str
    
    # Extended Fields
    references: List[Dict]    # Document references
    urls: List[str]           # URL references
    metadata: Optional[Dict]  # Additional metadata
```

#### Part Type Descriptions

| Type | Description | Included in LLM Context |
|------|-------------|------------------------|
| `message` | Main conversation content | ✅ Yes |
| `tool_call_result` | Tool calling process | ❌ No (display only) |
| `thinking` | AI thinking process | ❌ No (display only) |
| `references` | Document references and links | ❌ No (display only) |

**Design Rationale**: A single AI response contains multiple stages (tool calling, thinking, answering, references), and these contents are generated sequentially and interleaved. A single field cannot express this. User messages typically have only 1 part (type="message"), but also support multiple parts to maintain structural consistency.

#### Redis Storage Example

**User Message**:
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
      "content": "What is LightRAG?",
      "references": [],
      "urls": [],
      "metadata": null
    }
  ],
  "files": []
}
```

**AI Response (with multiple parts)**:
```json
{
  "parts": [
    {
      "message_id": "uuid-2",
      "part_id": "uuid-part-2",
      "type": "tool_call_result",
      "role": "ai",
      "content": "Searching knowledge base...",
      "timestamp": 1699999999.1
    },
    {
      "message_id": "uuid-2",
      "part_id": "uuid-part-3",
      "type": "message",
      "role": "ai",
      "content": "LightRAG is a lightweight RAG framework, deeply modified by the ApeCloud team...",
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
          "text": "LightRAG architecture description...",
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

### API Response Format

**ChatDetails Schema** (`aperag/api/components/schemas/chat.yaml`):

```yaml
chatDetails:
  type: object
  properties:
    id: string                    # chat_abc123
    title: string                 # Chat title
    bot_id: string                # bot_xyz
    peer_id: string
    peer_type: string             # system/feishu/weixin/web
    status: string                # active/archived
    created: string               # ISO 8601
    updated: string               # ISO 8601
    history:                      # 2D array
      type: array
      description: Conversation history, each element is a message
      items:
        type: array
        description: A message contains multiple parts (tool calls, thinking, answer, references, etc.)
        items:
          $ref: '#/chatMessage'
```

**ChatMessage Schema**:

```yaml
chatMessage:
  type: object
  properties:
    id: string                    # message_id (same for one turn)
    part_id: string               # part_id (unique for each part)
    type: string                  # message/tool_call_result/thinking/references
    timestamp: number             # Unix timestamp
    role: string                  # human/ai
    data: string                  # Message content
    references:                   # Document references (optional)
      type: array
      items:
        type: object
        properties:
          score: number
          text: string
          metadata: object
    urls:                         # URL references (optional)
      type: array
      items:
        type: string
    feedback:                     # User feedback (optional)
      type: object
      properties:
        type: string              # like/dislike
        tag: string
        message: string
    files:                        # Associated files (optional)
      type: array
```

### Frontend Response Example

```json
{
  "id": "chat_abc123",
  "title": "Discussion about LightRAG",
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
        "data": "What is LightRAG?",
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
        "data": "Searching knowledge base...",
        "files": []
      },
      {
        "id": "uuid-2",
        "part_id": "uuid-part-3",
        "type": "message",
        "timestamp": 1699999999.5,
        "role": "ai",
        "data": "LightRAG is a lightweight RAG framework...",
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
            "text": "LightRAG architecture description...",
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

**Note**: `history` is a 2D array. The first dimension is the message sequence (in chronological order), and the second dimension is the multiple parts of that message. For example:
- `history[0]` = Parts of user's 1st message (usually only 1 part)
- `history[1]` = Parts of AI's 1st response (may have multiple parts: tool calls, thinking, answer, references)
- `history[2]` = Parts of user's 2nd message
- `history[3]` = Parts of AI's 2nd response
- ...

## Message Write Flow

### WebSocket Real-time Chat

**Endpoint**: `WS /api/v1/bots/{bot_id}/chats/{chat_id}/connect`

```python
async def handle_websocket_chat(websocket: WebSocket, user: str, bot_id: str, chat_id: str):
    await websocket.accept()
    
    while True:
        # 1. Receive user message
        data = json.loads(await websocket.receive_text())
        message_content = data.get("data")
        message_id = str(uuid.uuid4())
        
        # 2. Write user message to Redis
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
        await history.add_user_message(message_content, message_id, files=data.get("files", []))
        
        # 3. Execute Flow to get AI response
        flow = FlowParser.parse(flow_config)
        engine = FlowEngine()
        _, system_outputs = await engine.execute_flow(flow, initial_data)
        
        # 4. Stream AI response
        async for chunk in async_generator():
            await websocket.send_text(success_response(message_id, chunk))
            full_message += chunk
        
        # 5. Write AI message to Redis (automatically called by Flow Runner's history.add_ai_message())
```

**Write Methods**:

```python
# User message
await history.add_user_message(
    message="User's question",
    message_id="uuid-1",
    files=[{"id": "file1", "name": "doc.pdf"}]
)

# AI message
await history.add_ai_message(
    content="AI's answer",
    chat_id="chat_abc123",
    message_id="uuid-2",
    tool_use_list=[{"data": "Tool call info"}],  # Optional
    references=[{"score": 0.95, "text": "..."}],  # Optional
    urls=["https://example.com"]  # Optional
)
```

## Design Features

### 1. Hybrid Storage Architecture

| Storage | Content | Reason |
|---------|---------|--------|
| PostgreSQL | Chat metadata | Persistence, complex queries |
| Redis | Message history | High-performance read/write, TTL support |
| PostgreSQL | User feedback | Persistence, for analysis |

**Advantages**:
- Performance optimization: Message history uses Redis for fast read/write
- Data persistence: Important metadata stored in PostgreSQL
- Flexibility: Independent TTL and backup strategy configuration

### 2. Part-Based Message Design

**Core Value**:
- ✅ Support complex AI response flow (tool calling → thinking → answer → references)
- ✅ Frontend can render different types of content differently
- ✅ Complete temporal relationship recording (via timestamp)
- ✅ Flexible extension (adding new types doesn't require schema changes)

**Why does a single message need multiple parts?**

A single AI response is generated sequentially and interleaved, for example:
1. 🔍 Part1 (tool_call_result): "Querying database..."
2. 💭 Part2 (thinking): "Found 327 records..."
3. 🔍 Part3 (tool_call_result): "Calculating growth rate..."
4. 💭 Part4 (thinking): "15% QoQ growth..."
5. 💬 Part5 (message): "Based on data analysis, Q4 performance is excellent..."
6. 📚 Part6 (references): [Document 1, Document 2]

These 6 parts belong to **one AI message** (sharing the same message_id). A single field cannot express such complex temporal relationships.

### 3. Format Conversion Decoupling

Three format conversions are provided:

```python
class StoredChatMessage:
    def to_frontend_format(self) -> List[ChatMessage]:
        """Convert to frontend display format"""
        # Include all types of parts
        
    def to_openai_format(self) -> List[Dict]:
        """Convert to LLM call format"""
        # Only include type="message" parts
        
    def get_main_content(self) -> str:
        """Get main answer content"""
        # Content of the first type="message" part
```

**Advantages**:
- Internal storage format decoupled from external interfaces
- Support different consumption scenarios
- LLM context only includes actual conversation content, not tool calls and thinking processes

### 4. Three-Level ID Design

```python
chat_id = "chat_abc123"           # Session level
message_id = "uuid-msg-1"         # Message level (shared by parts of the same message)
part_id = "uuid-part-1"           # Part level (each part is independent)
```

**Purpose**:
- `chat_id`: Identifies a chat session
- `message_id`: Groups parts of the same message (for frontend display and feedback association)
- `part_id`: Uniquely identifies each part (for individual operations like copy, reference)

## Performance Considerations

### Redis Optimization
- **List Data Structure**: LPUSH O(1), LRANGE O(N)
- **Optional TTL**: Automatic expiration of historical messages
- **Connection Pool Reuse**: Global Redis client

### PostgreSQL Optimization
- **Indexes**: user, bot_id, chat_id, status fields
- **Soft Delete**: Using gmt_deleted
- **Paginated Queries**: list_chats supports pagination

### Transmission Optimization
- **WebSocket Streaming**: Generate and send simultaneously
- **Incremental Updates**: Only transmit new parts
- **Lazy Loading**: Load historical messages on demand

## Related Files

### Core Implementation
- `aperag/views/chat.py` - View layer interface
- `aperag/service/chat_service.py` - Service layer business logic
- `aperag/utils/history.py` - Redis message history management
- `aperag/chat/history/message.py` - Message data structures
- `aperag/db/models.py` - Database models
- `aperag/db/repositories/chat.py` - Chat database operations
- `aperag/api/components/schemas/chat.yaml` - OpenAPI Schema

### Frontend Implementation
- `web/src/app/workspace/bots/[botId]/chats/[chatId]/page.tsx` - Chat detail page
- `web/src/components/chat/chat-messages.tsx` - Message display component

## Summary

ApeRAG's chat history message system adopts **Hybrid Storage + Part-Based Message Design**:

1. **PostgreSQL** stores Chat metadata and feedback (persistence, queryable)
2. **Redis** stores message history (high performance, expiration support)
3. **Part-Based Design** supports complex AI response flows (tool calling, thinking, answering, references)
4. **Three-Level ID Design** supports message grouping and independent operations
5. **Clear Layered Architecture** (View → Service → Repository → Storage)

This design ensures both performance and support for complex AI interaction scenarios, while maintaining good scalability.

