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

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from jinja2 import TemplateSyntaxError
from pydantic import BaseModel, Field

from aperag.db.models import User
from aperag.service.prompt_template_service import PromptTemplateService, prompt_template_service
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prompts"])


# Request models
class UpdateUserPromptsRequest(BaseModel):
    language: str = Field(default="zh-CN", pattern="^(en-US|zh-CN)$")
    prompts: Dict[str, str] = Field(..., description="Prompts to update, e.g., {'agent_system': 'content'}")


class ResetPromptsRequest(BaseModel):
    language: str = Field(default="zh-CN", pattern="^(en-US|zh-CN)$")
    types: Optional[List[str]] = Field(None, description="Prompt types to reset, None means all")


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
    language: str = Query(default="zh-CN", pattern="^(en-US|zh-CN)$"),
    user: User = Depends(required_user),
) -> Dict[str, Any]:
    """
    Get user's prompt configuration with priority resolution.

    Returns current effective prompts for the user, including:
    - content: Actual prompt content (resolved with priority)
    - source: Where the prompt comes from (user/system/hardcoded)
    - customized: Whether user has customized this prompt
    - description: Optional description
    """
    return await prompt_template_service.get_user_prompts(user_id=str(user.id), language=language)


@router.put("/prompts/user", tags=["prompts"])
async def update_user_prompts(
    request: Request, body: UpdateUserPromptsRequest, user: User = Depends(required_user)
) -> Dict[str, Any]:
    """
    Batch update user's prompt configurations.

    Only updates the prompts provided in the request body.
    Prompts not included will remain unchanged.
    """
    # Validate prompt types
    invalid_types = [t for t in body.prompts.keys() if t not in PromptTemplateService.PROMPT_TYPES]
    if invalid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt types: {invalid_types}. Valid types: {PromptTemplateService.PROMPT_TYPES}",
        )

    # Validate templates (optional, can be skipped for better performance)
    try:
        for content in body.prompts.values():
            from jinja2 import Template

            Template(content)  # Quick syntax check
    except TemplateSyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Template syntax error: {str(e)}")

    # Update via service
    updated = await prompt_template_service.update_user_prompts(
        user_id=str(user.id), language=body.language, prompts=body.prompts
    )

    return {"message": "Prompts updated successfully", "updated": updated}


@router.delete("/prompts/user/{prompt_type}", tags=["prompts"])
async def delete_user_prompt(
    request: Request,
    prompt_type: str,
    language: str = Query(default="zh-CN", pattern="^(en-US|zh-CN)$"),
    user: User = Depends(required_user),
) -> Dict[str, Any]:
    """
    Delete user's specific prompt configuration (reset to system default).

    Returns the new effective content after deletion.
    """
    # Validate prompt type
    if prompt_type not in PromptTemplateService.PROMPT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt type: {prompt_type}. Valid types: {PromptTemplateService.PROMPT_TYPES}",
        )

    # Delete via service
    result = await prompt_template_service.delete_user_prompt(
        user_id=str(user.id), prompt_type=prompt_type, language=language
    )

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
    request: Request, body: ResetPromptsRequest, user: User = Depends(required_user)
) -> Dict[str, Any]:
    """
    Batch reset user's prompt configurations.

    If 'types' is not provided, resets all prompts.
    """
    # Validate types if provided
    if body.types:
        invalid_types = [t for t in body.types if t not in PromptTemplateService.PROMPT_TYPES]
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prompt types: {invalid_types}. Valid types: {PromptTemplateService.PROMPT_TYPES}",
            )

    # Reset via service
    reset = await prompt_template_service.reset_user_prompts(
        user_id=str(user.id), language=body.language, types=body.types
    )

    return {"message": "Prompts reset successfully", "reset": reset}


# === System defaults (read-only, for reference) ===


@router.get("/prompts/system", tags=["prompts"])
async def get_system_prompts(
    request: Request,
    language: str = Query(default="zh-CN", pattern="^(en-US|zh-CN)$"),
    type: Optional[str] = Query(
        default=None, pattern="^(agent_system|agent_query|index_graph|index_summary|index_vision)$"
    ),
    user: User = Depends(required_user),
):
    """
    Get system default prompts (for reference).

    Can query a specific type or all types.
    """
    return await prompt_template_service.get_system_prompts(language=language, prompt_type=type)


# === Helper utilities ===


@router.post("/prompts/preview", tags=["prompts"])
async def preview_prompt(request: Request, body: PreviewRequest, user: User = Depends(required_user)) -> Dict[str, str]:
    """
    Preview how a prompt template will be rendered with given variables.
    """
    try:
        rendered = prompt_template_service.preview_prompt(body.template, body.variables or {})
        return {"rendered": rendered}
    except TemplateSyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Template syntax error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Template rendering error: {str(e)}")


@router.post("/prompts/validate", tags=["prompts"])
async def validate_prompt(
    request: Request, body: ValidateRequest, user: User = Depends(required_user)
) -> Dict[str, Any]:
    """
    Validate prompt template syntax (Jinja2) and check for required variables.
    """
    result = prompt_template_service.validate_prompt(body.type, body.template)
    return result
