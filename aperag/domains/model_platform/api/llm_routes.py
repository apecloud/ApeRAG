import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.model_platform.ports import AuthenticatedUser
from aperag.domains.model_platform.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankUsage,
)
from aperag.llm.runtime.invocation_service import model_invocation_service
from aperag.llm.runtime.types import ModelUnavailableError
from aperag.utils.audit_decorator import audit

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["llm"])
@audit(resource_type="llm", api_name="CreateEmbeddings")
async def create_embeddings(
    http_request: Request, request: EmbeddingRequest, user: AuthenticatedUser = Depends(required_user)
):
    try:
        input_texts = [request.input] if isinstance(request.input, str) else request.input
        response = await model_invocation_service.embed(request.model_id, str(user.id), input_texts)
        data = getattr(response, "data", None) or response.get("data", [])
        embeddings = [getattr(item, "embedding", None) or item["embedding"] for item in data]
        total_tokens = sum(len(text.split()) for text in input_texts)
        return EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(object="embedding", embedding=embedding, index=i)
                for i, embedding in enumerate(embeddings)
            ],
            model=request.model_id,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rerank", response_model=RerankResponse, tags=["llm"])
@audit(resource_type="llm", api_name="CreateRerank")
async def create_rerank(
    http_request: Request, request: RerankRequest, user: AuthenticatedUser = Depends(required_user)
):
    try:
        documents = [doc if isinstance(doc, str) else doc.text for doc in request.documents]
        ranked = await model_invocation_service.rerank(
            request.model_id,
            str(user.id),
            request.query,
            documents,
            top_n=request.top_k or len(documents),
        )
        ranked = ranked[: request.top_k or len(ranked)]
        data = []
        for item in ranked:
            idx = item["index"]
            document = None
            if request.return_documents:
                document = {"text": documents[idx], "metadata": {}}
            data.append(
                RerankDocument(
                    index=idx,
                    relevance_score=float(item.get("relevance_score", item.get("score", 0.0))),
                    document=document,
                )
            )
        total_tokens = len(request.query.split()) + sum(len(doc.split()) for doc in documents)
        return RerankResponse(
            object="list",
            data=data,
            model=request.model_id,
            usage=RerankUsage(total_tokens=total_tokens),
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Rerank failed")
        raise HTTPException(status_code=500, detail=str(exc))
