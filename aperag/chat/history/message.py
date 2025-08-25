import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from aperag.schema.view_models import ChatMessage


class StoredChatMessagePart(BaseModel):
    """Single part of a chat message with complete identification"""

    # Core identifiers
    chat_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID to group related parts in the same turn")
    part_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique part ID")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
    timestamp: float = Field(default_factory=time.time, description="Part timestamp")

    # Message content
    type: Literal["message", "tool_call_result", "thinking", "references"] = Field(
        default="message", description="Part type"
    )
    role: Literal["human", "ai", "system"] = Field(default="ai", description="Message role")
    content: str = Field(default="", description="Part content")

    # Extended fields
    references: List[Dict[str, Any]] = Field(default_factory=list, description="Document references")
    urls: List[str] = Field(default_factory=list, description="URL references")
    feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class StoredChatMessage(BaseModel):
    """Complete chat message as a collection of parts"""

    # Message parts - this is now the core content
    parts: List[StoredChatMessagePart] = Field(default_factory=list, description="Message parts")

    @property
    def chat_id(self) -> Optional[str]:
        """Get chat_id from first part (convenience property)"""
        return self.parts[0].chat_id if self.parts else None

    @property
    def message_id(self) -> Optional[str]:
        """Get message_id from first part (convenience property)"""
        return self.parts[0].message_id if self.parts else None

    @property
    def timestamp(self) -> Optional[float]:
        """Get timestamp from first part (convenience property)"""
        return self.parts[0].timestamp if self.parts else None
    
    @property
    def role(self) -> Optional[float]:
        """Get timestamp from first part (convenience property)"""
        return self.parts[0].role if self.parts else None

    def to_frontend_format(self) -> List[ChatMessage]:
        """Convert parts to frontend ChatMessage format"""
        frontend_messages = []
        for part in self.parts:
            chatMessage = ChatMessage(
                id=part.message_id,
                part_id=part.part_id,
                type=part.type,
                timestamp=part.timestamp,
                role="human" if part.role == "user" else part.role,  # Convert "user" to "human" for frontend
                data=part.content,  # Frontend expects "data" field
                references=part.references if part.references else None,
                urls=part.urls if part.urls else None,
                feedback=part.feedback,
            )
            frontend_messages.append(chatMessage)
        return frontend_messages

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI ChatML format for LLM consumption (only conversation parts)"""
        openai_messages = []
        for part in self.parts:
            if part.type == "message":  # Only include actual conversation content
                # Map roles for OpenAI compatibility
                openai_role = part.role
                if part.role == "ai":
                    openai_role = "assistant"
                elif part.role == "human":
                    openai_role = "user"

                openai_messages.append({"role": openai_role, "content": part.content})
        return openai_messages

    def add_part(self, part: StoredChatMessagePart) -> None:
        """Add a part to this message"""
        self.parts.append(part)

    def get_main_content(self) -> str:
        """Get the main message content (first 'message' type part)"""
        for part in self.parts:
            if part.type == "message":
                return part.content
        return ""

    def get_thinking_steps(self) -> List[str]:
        """Get all thinking steps"""
        return [part.content for part in self.parts if part.type in ["thinking", "tool_call_result"]]

    def get_references_and_urls(self) -> tuple[List[Dict[str, Any]], List[str]]:
        """Get all references and URLs from all parts"""
        all_references = []
        all_urls = []
        for part in self.parts:
            all_references.extend(part.references)
            all_urls.extend(part.urls)
        return all_references, all_urls


# Helper functions for creating messages
def create_user_message(
    content: str,
    chat_id: str,
    message_id: str = None,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> StoredChatMessage:
    """Create a user message"""
    if not message_id:
        message_id = str(uuid.uuid4())

    part = StoredChatMessagePart(
        chat_id=chat_id,
        message_id=message_id,
        trace_id=trace_id,
        type="message",
        role="human",
        content=content,
        metadata=metadata,
    )

    return StoredChatMessage(parts=[part])


def create_assistant_message(
    content: str,
    chat_id: str,
    message_id: str = None,
    tool_use_list: List[str] = None,
    references: List[Dict[str, Any]] = None,
    urls: List[str] = None,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> StoredChatMessage:
    """Create an assistant message with optional thinking steps and references"""
    if not message_id:
        message_id = str(uuid.uuid4())

    parts = []

    # Add thinking steps
    if tool_use_list:
        for step in tool_use_list:
            parts.append(
                StoredChatMessagePart(
                    chat_id=chat_id,
                    message_id=message_id,
                    trace_id=trace_id,
                    type="tool_call_result",
                    role="ai",
                    content=step.get("data"),
                    metadata=metadata,
                )
            )

    # Add main content
    parts.append(
        StoredChatMessagePart(
            chat_id=chat_id,
            message_id=message_id,
            trace_id=trace_id,
            type="message",
            role="ai",
            content=content,
            metadata=metadata,
        )
    )

    # Add references and URLs if any
    if references or urls:
        parts.append(
            StoredChatMessagePart(
                chat_id=chat_id,
                message_id=message_id,
                trace_id=trace_id,
                type="references",
                role="ai",
                content="",  # Empty content for references-only part
                references=references or [],
                urls=urls or [],
                metadata=metadata,
            )
        )

    return StoredChatMessage(parts=parts)


# Conversion functions
def message_to_storage_dict(message: StoredChatMessage) -> Dict[str, Any]:
    """Convert StoredChatMessage to storage dictionary"""
    return message.model_dump()


def storage_dict_to_message(data: Dict[str, Any]) -> StoredChatMessage:
    """Convert storage dictionary to StoredChatMessage"""
    return StoredChatMessage.model_validate(data)


def group_messages_by_message_id(messages: List[StoredChatMessage]) -> Dict[str, List[StoredChatMessage]]:
    """Group messages by message_id"""
    groups = {}
    for message in messages:
        for part in message.parts:
            if part.message_id not in groups:
                groups[part.message_id] = []
            groups[part.message_id].append(message)
    return groups


def group_parts_by_message_id(parts: List[StoredChatMessagePart]) -> Dict[str, List[StoredChatMessagePart]]:
    """Group parts by message_id"""
    groups = {}
    for part in parts:
        if part.message_id not in groups:
            groups[part.message_id] = []
        groups[part.message_id].append(part)
    return groups


def messages_to_frontend_format(messages: List[StoredChatMessage]) -> List[ChatMessage]:
    """Convert multiple StoredChatMessage objects to frontend format"""
    frontend_messages = []
    for message in messages:
        frontend_messages.extend(message.to_frontend_format())
    return frontend_messages


def messages_to_openai_format(messages: List[StoredChatMessage]) -> List[Dict[str, Any]]:
    """Convert multiple StoredChatMessage objects to OpenAI format"""
    openai_messages = []
    for message in messages:
        openai_messages.extend(message.to_openai_format())
    return openai_messages
