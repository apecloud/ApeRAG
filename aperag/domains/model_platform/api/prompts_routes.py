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

"""Prompt template HTTP router (Phase 8 #49 G3 carve, D7 v2 hard-cut).

Carved from legacy ``aperag/views/prompts.py``. URL prefix migrated
from ``/api/v1/prompts*`` to ``/api/v2/prompts*`` per D7 canonical
(msg=94f663f2 §3.2.2). The user-CRUD ops are still backed by the
legacy ``aperag/service/prompt_template_service.py`` (Layer A
permanent seam, shared with agent_runtime); this module accesses
the singleton through the ``PromptCRUDOps`` Protocol seam wired at
app startup via ``set_prompt_crud_ops()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from jinja2 import Template, TemplateSyntaxError
from pydantic import BaseModel, Field

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.model_platform.ports import AuthenticatedUser, PromptCRUDOps

logger = logging.getLogger(__name__)

# Module-level DI seam: ``aperag/app.py`` calls ``set_prompt_crud_ops()``
# at startup to wire the legacy ``prompt_template_service`` singleton
# (the same one already wired into agent_runtime). The singleton
# structurally satisfies the ``PromptCRUDOps`` Protocol — its public
# user-CRUD method names map 1:1.
_prompt_crud_ops: Optional[PromptCRUDOps] = None


def set_prompt_crud_ops(ops: PromptCRUDOps) -> None:
    """Inject the ``PromptCRUDOps`` Protocol implementation at app startup.

    Idempotent so repeated wiring (e.g. test re-initialization) does
    not raise.
    """
    global _prompt_crud_ops
    _prompt_crud_ops = ops


def _get_prompt_crud_ops() -> PromptCRUDOps:
    if _prompt_crud_ops is None:
        raise RuntimeError(
            "PromptCRUDOps not wired. aperag/app.py must call "
            "set_prompt_crud_ops() at startup."
        )
    return _prompt_crud_ops


router = APIRouter(tags=["prompts"])


# Request models
class PromptsPayload(BaseModel):
    agent_system: Optional[str] = Field(None, description="Agent system prompt (persona definition)")
    agent_query: Optional[str] = Field(None, description="Agent query prompt template")
    index_graph: Optional[str] = Field(None, description="Graph index prompt for entity/relation extraction")
    index_summary: Optional[str] = Field(None, description="Summary index prompt for document summarization")
    index_vision: Optional[str] = Field(None, description="Vision index prompt for image content extraction")


class UpdateUserPromptsRequest(BaseModel):
    prompts: PromptsPayload = Field(..., description="Prompts to update (only provided fields will be updated)")


class ResetPromptsRequest(BaseModel):
    types: Optional[List[str]] = Field(None, description="Prompt types to reset, omit to reset all")


class PreviewRequest(BaseModel):
    template: str
    variables: Optional[Dict[str, Any]] = None


class ValidateRequest(BaseModel):
    type: str = Field(..., pattern="^(agent_system|agent_query|index_graph|index_summary|index_vision)$")
    template: str


# === User prompt configuration management ===


@router.get("/prompts/user", tags=["prompts"])
async def get_user_prompts(
    request: Request,
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, Any]:
    """
    Get user's prompt configuration with priority resolution.

    Returns current effective prompts for the user, including:
    - content: Actual prompt content (resolved with priority)
    - source: Where the prompt comes from (user/system/hardcoded)
    - customized: Whether user has customized this prompt
    - description: Optional description
    """
    ops = _get_prompt_crud_ops()
    return await ops.get_user_prompts(user_id=str(user.id))


@router.put("/prompts/user", tags=["prompts"])
async def update_user_prompts(
    request: Request,
    body: UpdateUserPromptsRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, Any]:
    """
    Batch update user's prompt configurations.

    Only updates the prompts provided in the request body.
    Prompts not included will remain unchanged.
    """
    prompts_dict = body.prompts.model_dump(exclude_none=True)
    if not prompts_dict:
        raise HTTPException(status_code=400, detail="No prompts provided to update")

    # Validate Jinja2 template syntax
    try:
        for content in prompts_dict.values():
            Template(content)
    except TemplateSyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Template syntax error: {str(e)}")

    ops = _get_prompt_crud_ops()
    updated = await ops.update_user_prompts(user_id=str(user.id), prompts=prompts_dict)

    return {"message": "Prompts updated successfully", "updated": updated}


@router.delete("/prompts/user/{prompt_type}", tags=["prompts"])
async def delete_user_prompt(
    request: Request,
    prompt_type: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, Any]:
    """
    Delete user's specific prompt configuration (reset to system default).

    Returns the new effective content after deletion.
    """
    ops = _get_prompt_crud_ops()
    if prompt_type not in ops.PROMPT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt type: {prompt_type}. Valid types: {ops.PROMPT_TYPES}",
        )

    result = await ops.delete_user_prompt(user_id=str(user.id), prompt_type=prompt_type)

    if not result["deleted"]:
        raise HTTPException(status_code=404, detail=f"User has not customized {prompt_type} prompt")

    return {
        "message": "Prompt reset to default",
        "type": prompt_type,
        "new_content": result["new_content"],
        "source": result["source"],
    }


@router.post("/prompts/user/reset", tags=["prompts"])
async def reset_user_prompts(
    request: Request,
    body: ResetPromptsRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, Any]:
    """
    Batch reset user's prompt configurations.

    If 'types' is not provided, resets all prompts.
    """
    ops = _get_prompt_crud_ops()
    if body.types:
        invalid_types = [t for t in body.types if t not in ops.PROMPT_TYPES]
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prompt types: {invalid_types}. Valid types: {ops.PROMPT_TYPES}",
            )

    reset = await ops.reset_user_prompts(user_id=str(user.id), types=body.types)

    return {"message": "Prompts reset successfully", "reset": reset}


# === System defaults (read-only, for reference) ===


@router.get("/prompts/system", tags=["prompts"])
async def get_system_prompts(
    request: Request,
    type: Optional[str] = None,
    user: AuthenticatedUser = Depends(required_user),
):
    """
    Get system default prompts (for reference).

    Can query a specific type or all types.
    """
    ops = _get_prompt_crud_ops()
    if type and type not in ops.PROMPT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt type: {type}. Valid types: {ops.PROMPT_TYPES}",
        )
    return await ops.get_system_prompts(prompt_type=type)


# === Helper utilities ===


@router.post("/prompts/preview", tags=["prompts"])
async def preview_prompt(
    request: Request,
    body: PreviewRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, str]:
    """
    Preview how a prompt template will be rendered with given variables.
    """
    ops = _get_prompt_crud_ops()
    try:
        rendered = ops.preview_prompt(body.template, body.variables or {})
        return {"rendered": rendered}
    except TemplateSyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Template syntax error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Template rendering error: {str(e)}")


@router.post("/prompts/validate", tags=["prompts"])
async def validate_prompt(
    request: Request,
    body: ValidateRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> Dict[str, Any]:
    """
    Validate prompt template syntax (Jinja2) and check for required variables.
    """
    ops = _get_prompt_crud_ops()
    result = ops.validate_prompt(body.type, body.template)
    return result
