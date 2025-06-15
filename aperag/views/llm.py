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

from fastapi import APIRouter, Depends, HTTPException

from aperag.db.models import User
from aperag.db.ops import async_db_ops
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import (
    EmbeddingError,
    InvalidConfigurationError,
    ModelNotFoundError,
    ProviderNotFoundError,
)
from aperag.schema.view_models import EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage
from aperag.views.auth import current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, user: User = Depends(current_user)):
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
        provider_info = await _get_provider_info(request.provider, request.model, str(user.id))

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


async def _get_provider_info(provider: str, model: str, user_id: str) -> dict:
    """
    Get provider configuration including API key, base URL, and custom LLM provider.

    Args:
        provider: Provider name
        model: Model name
        user_id: User ID

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
            raise ProviderNotFoundError(provider, "Embedding")

        # 2. Get model configuration for embedding API
        llm_model = await async_db_ops.query_llm_provider_model(provider_name=provider, api="embedding", model=model)
        if not llm_model:
            raise ModelNotFoundError(model, provider, "Embedding")

        # 3. Get user's API key from MSP
        msp_dict = await async_db_ops.query_msp_dict(user_id)
        if provider not in msp_dict:
            available_providers = list(msp_dict.keys())
            logger.warning(f"Provider {provider} not configured for user {user_id}. Available: {available_providers}")
            raise ProviderNotFoundError(provider, "Embedding")

        msp = msp_dict[provider]
        if not msp.api_key:
            raise InvalidConfigurationError("api_key", None, f"API key not configured for provider '{provider}'")

        # 4. Validate base URL
        if not llm_provider.base_url:
            raise InvalidConfigurationError(
                "base_url", llm_provider.base_url, f"Base URL not configured for provider '{provider}'"
            )

        return {
            "api_key": msp.api_key,
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
