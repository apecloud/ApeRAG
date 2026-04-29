import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.model_platform.ports import AuthenticatedUser
from aperag.domains.model_platform.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ModelCapability,
)
from aperag.domains.model_platform.service.model_service import model_platform_service
from aperag.llm.runtime.invocation_service import model_invocation_service
from aperag.llm.runtime.types import ModelUnavailableError
from aperag.utils.audit_decorator import audit

logger = logging.getLogger(__name__)
router = APIRouter()


async def _resolve_model_id(
    *,
    user_id: str,
    body_model_id: str | None,
    legacy_model: str | None,
    legacy_provider: str | None,
    legacy_custom_llm_provider: str | None,
    capability: ModelCapability,
) -> str:
    """Resolve a request's effective ``model_id``.

    The route accepts either the new ``{model_id}`` shape or the legacy
    ``{model, model_service_provider, custom_llm_provider}`` triple
    (Blocker A — Weston msg=80e873c1). Direct ``model_id`` wins; the
    triple is resolved via the model-platform service. Raises
    ``HTTPException(422)`` when neither path resolves.
    """
    if body_model_id:
        return body_model_id
    resolved = await model_platform_service.resolve_legacy_model_id(
        user_id,
        provider_name=legacy_provider,
        provider_model_name=legacy_model,
        custom_llm_provider=legacy_custom_llm_provider,
        capability=capability,
    )
    if not resolved:
        raise HTTPException(
            status_code=422,
            detail=(
                "Either ``model_id`` or the legacy "
                "``{model, model_service_provider}`` pair must resolve to a known model."
            ),
        )
    return resolved


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["llm"])
@audit(resource_type="llm", api_name="CreateEmbeddings")
async def create_embeddings(
    http_request: Request, request: EmbeddingRequest, user: AuthenticatedUser = Depends(required_user)
):
    try:
        model_id = await _resolve_model_id(
            user_id=str(user.id),
            body_model_id=request.model_id,
            legacy_model=request.model,
            legacy_provider=request.model_service_provider,
            legacy_custom_llm_provider=request.custom_llm_provider,
            capability=ModelCapability.EMBEDDING,
        )
        input_texts = [request.input] if isinstance(request.input, str) else request.input
        response = await model_invocation_service.embed(model_id, str(user.id), input_texts)
        data = getattr(response, "data", None) or response.get("data", [])
        embeddings = [getattr(item, "embedding", None) or item["embedding"] for item in data]
        total_tokens = sum(len(text.split()) for text in input_texts)
        return EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(object="embedding", embedding=embedding, index=i)
                for i, embedding in enumerate(embeddings)
            ],
            model=model_id,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )
    except HTTPException:
        raise
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
