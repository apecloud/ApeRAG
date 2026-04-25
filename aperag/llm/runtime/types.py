from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelCapability(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ResolvedModelInvocation(BaseModel):
    model_id: str
    account_id: str
    provider_type: str
    provider_model_id: str
    capability: ModelCapability
    runner_type: str
    base_url: str
    api_key: str
    runner_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    supports_vision: bool = False
    supports_tool_calling: bool = False


class ModelRuntimeError(RuntimeError):
    pass


class ModelUnavailableError(ModelRuntimeError):
    pass
