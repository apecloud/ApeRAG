# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent-runtime-domain Pydantic request / response schemas.

Phase 5 step 5-S5a carves ``AgentMessage`` out of
``aperag.schema.view_models`` so the follow-up step 5-S5b can move
``aperag/agent_runtime/runtime.py`` into this domain without tripping
G1. The dual-hook shim in ``aperag.schema.view_models`` mirrors the
Phase 3 step 4b / Phase 5 step 5-S3 pattern so pre-migration callers
(``aperag/agent_runtime/schemas.py`` local to this module at
5-S5a time, plus any ``from aperag.schema.view_models import
AgentMessage`` caller) keep resolving the same class object.

Step 5-S5b will fold the pre-existing legacy ``aperag/agent_runtime/schemas.py``
envelope classes into this module; 5-S5a intentionally keeps this
file single-schema so the carve change stays reviewable in isolation.
Cross-domain ``Collection`` + ``File`` references are direct imports
(G1 allows domain→domain).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from aperag.domains.conversation.schemas import File
from aperag.domains.knowledge_base.schemas import Collection
from aperag.schema.common import ModelSpec


class AgentMessage(BaseModel):
    """
    Message format for agent-type bots with additional capabilities
    """

    query: str = Field(..., description="User query", examples=["Tell me about ApeRAG features"])
    collections: list[Collection] = Field(
        ...,
        description="List of collection objects to search in",
        examples=[
            [
                {"id": "col_123", "title": "Example Collection"},
                {"id": "col_456", "title": "Another Collection"},
            ]
        ],
    )
    completion: Optional[ModelSpec] = Field(
        None,
        description="Model specification for completion including provider and model details",
    )
    web_search_enabled: Optional[bool] = Field(False, description="Whether to enable web search", examples=[True])
    language: Optional[
        Literal[
            "en-US",
            "zh-CN",
            "zh-TW",
            "ja-JP",
            "ko-KR",
            "fr-FR",
            "de-DE",
            "es-ES",
            "it-IT",
            "pt-BR",
            "ru-RU",
        ]
    ] = Field("en-US", description="Language preference for the response", examples=["en-US"])
    files: Optional[list[File]] = None


__all__ = ["AgentMessage"]


# Phase 5 step 5-S5a dual-hook back-compat: write AgentMessage onto
# the legacy ``aperag.schema.view_models`` namespace if this module
# loads before view_models. Symmetric with the ``try:`` block in
# view_models that imports AgentMessage from here; together they
# guarantee pre-migration callers resolve the same class object
# regardless of module load order. Write-only per lesson 9a-quad.


def _bind_view_models_reexports() -> None:
    import sys

    _vm = sys.modules.get("aperag.schema.view_models")
    if _vm is None:  # pragma: no cover - ordering-dependent
        return
    for name in __all__:
        setattr(_vm, name, globals()[name])


_bind_view_models_reexports()
