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

from fastapi import APIRouter, Depends, HTTPException, Request

from aperag.db.models import User
from aperag.db.ops import async_db_ops
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import (
    EmbeddingError,
    InvalidConfigurationError,
    InvalidDocumentError,
    ModelNotFoundError,
    ProviderNotFoundError,
    RerankError,
    TooManyDocumentsError,
)
from aperag.llm.rerank.rerank_service import RerankService
from aperag.query.query import DocumentWithScore
from aperag.schema.view_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankUsage,
)
from aperag.utils.audit_decorator import audit
from aperag.views.auth import current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
@audit(resource_type="llm", api_name="CreateEmbeddings")
async def create_embeddings(http_request: Request, request: EmbeddingRequest, user: User = Depends(current_user)):
    """
    Create embeddings for the given input text(s).
    Compatible with OpenAI embeddings API format.
    """
    try:
        # Validate and normalize input
        input_texts = [request.input] if isinstance(request.input, str) else request.input
        if not input_texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        # Query database for provider and model information
        provider_info = await _get_provider_info(request.provider, request.model, str(user.id), "embedding")

        # Create embedding service
        embedding_service = EmbeddingService(
            embedding_provider=provider_info["custom_llm_provider"],
            embedding_model=request.model,
            embedding_service_url=provider_info["base_url"],
            embedding_service_api_key=provider_info["api_key"],
            embedding_max_chunks_in_batch=10,  # Default batch size
        )

        # Generate embeddings
        embeddings = embedding_service.embed_documents(input_texts)

        # Calculate token usage (approximation)
        total_tokens = sum(len(text.split()) for text in input_texts)

        # Format response in OpenAI format
        embedding_data = [
            EmbeddingData(object="embedding", embedding=embedding, index=i) for i, embedding in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=request.model,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except (ProviderNotFoundError, ModelNotFoundError, InvalidConfigurationError) as e:
        logger.warning(f"Configuration error for user {user.id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except EmbeddingError as e:
        logger.error(f"Embedding generation failed for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
    except Exception:
        logger.exception(f"Unexpected error in embedding endpoint for user {user.id}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _get_provider_info(provider: str, model: str, user_id: str, api_type: str) -> dict:
    """
    Get provider configuration including API key, base URL, and custom LLM provider.

    Args:
        provider: Provider name
        model: Model name
        user_id: User ID
        api_type: API type (embedding or rerank)

    Returns:
        Dict with api_key, base_url, custom_llm_provider

    Raises:
        ProviderNotFoundError: If provider not found
        ModelNotFoundError: If model not found for provider
        InvalidConfigurationError: If configuration is invalid
    """
    try:
        # 1. Get LLM provider configuration
        llm_provider = await async_db_ops.query_llm_provider_by_name(provider)
        if not llm_provider:
            raise ProviderNotFoundError(provider, api_type)

        # 2. Get model configuration for embedding API
        llm_model = await async_db_ops.query_llm_provider_model(provider_name=provider, api=api_type, model=model)
        if not llm_model:
            raise ModelNotFoundError(model, provider, api_type)

        # 3. Get user's API key from MSP
        api_key = await async_db_ops.query_provider_api_key(provider, user_id)
        if not api_key:
            raise InvalidConfigurationError("api_key", None, f"API KEY not found for LLM Provider: {provider}")

        # 4. Validate base URL
        if not llm_provider.base_url:
            raise InvalidConfigurationError(
                "base_url", llm_provider.base_url, f"Base URL not configured for provider '{provider}'"
            )

        return {
            "api_key": api_key,
            "base_url": llm_provider.base_url,
            "custom_llm_provider": llm_model.custom_llm_provider,
        }

    except (ProviderNotFoundError, ModelNotFoundError, InvalidConfigurationError):
        # Re-raise our custom errors
        raise
    except Exception as e:
        logger.error(f"Failed to get provider info for {provider}/{model}: {e}")
        raise InvalidConfigurationError(
            "database", str(e), f"Failed to retrieve configuration for provider '{provider}'"
        ) from e


@router.post("/rerank", response_model=RerankResponse)
@audit(resource_type="llm", api_name="CreateRerank")
async def create_rerank(http_request: Request, request: RerankRequest, user: User = Depends(current_user)):
    """
    Rerank documents based on relevance to a query.
    Compatible with industry-standard rerank API format used by Cohere, Jina AI, etc.
    """
    try:
        # Validate and normalize input
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")

        # Validate top_k parameter
        top_k = request.top_k or len(request.documents)
        if top_k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be positive")
        if top_k > len(request.documents):
            raise HTTPException(
                status_code=400, detail=f"top_k value {top_k} exceeds document count {len(request.documents)}"
            )

        # Query database for provider and model information
        provider_info = await _get_provider_info(request.provider, request.model, str(user.id), "rerank")

        # Convert input documents to DocumentWithScore format
        input_documents = []
        for i, doc in enumerate(request.documents):
            if isinstance(doc, str):
                # Simple text document
                doc_obj = DocumentWithScore(text=doc, score=0.0)
            else:
                # Structured document with metadata
                doc_obj = DocumentWithScore(
                    text=doc.text,
                    score=0.0,
                    metadata=doc.metadata if hasattr(doc, "metadata") else {},
                )
            input_documents.append(doc_obj)

        # Create rerank service
        # For alibabacloud, pass the actual provider name instead of custom_llm_provider
        effective_provider = (
            request.provider if request.provider == "alibabacloud" else provider_info["custom_llm_provider"]
        )

        rerank_service = RerankService(
            rerank_provider=effective_provider,
            rerank_model=request.model,
            rerank_service_url=provider_info["base_url"],
            rerank_service_api_key=provider_info["api_key"],
        )

        # Perform reranking
        reranked_documents = await rerank_service.rank(request.query, input_documents)

        # Apply top_k limit
        reranked_documents = reranked_documents[:top_k]

        # Calculate token usage (approximation)
        query_tokens = len(request.query.split())
        doc_tokens = sum(len(doc.text.split()) for doc in input_documents)
        total_tokens = query_tokens + doc_tokens

        # Format response
        rerank_data = []
        for ranked_doc in reranked_documents:
            # Find original index
            original_index = -1
            for i, orig_doc in enumerate(input_documents):
                if orig_doc.text == ranked_doc.text:
                    original_index = i
                    break

            # Create response item
            response_item = RerankDocument(
                index=original_index, relevance_score=ranked_doc.score if hasattr(ranked_doc, "score") else 0.0
            )

            # Add document content if requested
            if request.return_documents:
                doc_content = {"text": ranked_doc.text}
                if hasattr(ranked_doc, "metadata") and ranked_doc.metadata:
                    doc_content["metadata"] = ranked_doc.metadata
                response_item.document = doc_content

            rerank_data.append(response_item)

        return RerankResponse(
            object="list",
            data=rerank_data,
            model=request.model,
            usage=RerankUsage(total_tokens=total_tokens),
        )

    except (ProviderNotFoundError, ModelNotFoundError, InvalidConfigurationError) as e:
        logger.warning(f"Configuration error for user {user.id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except (InvalidDocumentError, TooManyDocumentsError) as e:
        logger.warning(f"Document validation error for user {user.id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RerankError as e:
        logger.error(f"Rerank operation failed for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Rerank operation failed: {e}")
    except Exception:
        logger.exception(f"Unexpected error in rerank endpoint for user {user.id}")
        raise HTTPException(status_code=500, detail="Internal server error")
