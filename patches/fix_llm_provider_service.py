#!/usr/bin/env python3
"""
Patch for aperag/domains/model_platform/service/llm_provider_service.py

Fixes:
1. custom_llm_provider is auto-derived from provider dialect instead of
   using the UI label (which LiteLLM does not understand)
2. First local completion model gets default_for_background_task and
   default_for_agent_completion tags automatically
3. First local embedding model gets default_for_embedding tag automatically
"""

import re
import shutil
from pathlib import Path

TARGET = Path("aperag/domains/model_platform/service/llm_provider_service.py")
BACKUP = TARGET.with_suffix(".py.bak")

# Read current content
content = TARGET.read_text()

# Make backup
shutil.copy(TARGET, BACKUP)
print(f"Backed up to {BACKUP}")

# ----------------------------------------------------------------
# Fix 1: Add dialect-to-LiteLLM mapping near top of file
# ----------------------------------------------------------------
MAPPING_CODE = '''
# ---------------------------------------------------------------------------
# LiteLLM provider name mapping
# Maps ApeRAG dialect names to LiteLLM-recognized provider identifiers.
# LiteLLM uses these names to route requests to the correct backend.
# "openai" covers any OpenAI-compatible endpoint (including local Ollama).
# ---------------------------------------------------------------------------
_DIALECT_TO_LITELLM: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "ollama": "ollama_chat",
    "ollama_chat": "ollama_chat",
    "jina_ai": "openai",  # jina uses openai-compatible for completion
}


def _resolve_custom_llm_provider(dialect: str, custom_llm_provider: str | None) -> str:
    """Resolve the LiteLLM provider name from dialect.

    If the caller already supplied a valid LiteLLM provider name, keep it.
    Otherwise, derive it from the provider's dialect setting.
    """
    # Known LiteLLM provider names — if the supplied value is already one
    # of these, trust it and return as-is.
    known_litellm_providers = set(_DIALECT_TO_LITELLM.values()) | {
        "openai", "anthropic", "gemini", "ollama", "ollama_chat",
        "azure", "bedrock", "cohere", "huggingface", "together_ai",
        "deepseek", "mistral", "groq", "perplexity", "sagemaker",
    }
    if custom_llm_provider and custom_llm_provider.lower() in known_litellm_providers:
        return custom_llm_provider

    # Derive from dialect
    return _DIALECT_TO_LITELLM.get(dialect, "openai")

'''

# Insert after the imports block (after the last import line near the top)
insert_after = 'PUBLIC_USER_ID = "public"'
if '_DIALECT_TO_LITELLM' not in content:
    content = content.replace(
        insert_after,
        insert_after + "\n" + MAPPING_CODE,
        1
    )
    print("✓ Added _DIALECT_TO_LITELLM mapping and _resolve_custom_llm_provider()")
else:
    print("⚠ _DIALECT_TO_LITELLM already present, skipping")

# ----------------------------------------------------------------
# Fix 2: In create_llm_provider_model, auto-resolve custom_llm_provider
# and auto-assign default scenario tags for first local model
# ----------------------------------------------------------------

OLD_CREATE_MODEL = '''async def create_llm_provider_model(provider_name: str, model_data: dict, user_id: str, is_admin: bool = False):
    """Create a new model for a specific provider or restore a soft-deleted one with the same combination

    Args:
        provider_name: Name of the provider
        model_data: Model configuration data
        user_id: User ID creating the model
        is_admin: Whether the user is an admin
    """
    # Check if provider exists
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)

    if not provider:
        raise ResourceNotFoundException("Provider", provider_name)

    # Check edit permission for the provider
    _check_edit_permission(user_id, is_admin, provider.user_id, provider_name)

    # First check if there's an active model with the same combination
    active_existing = await async_db_ops.query_llm_provider_model(provider_name, model_data["api"], model_data["model"])

    if active_existing:
        raise invalid_param(
            "model",
            f"Model \'{model_data[\'model\']}\' for API \'{model_data[\'api\']}\' already exists for provider \'{provider_name}\'",
        )

    # Try to restore a soft-deleted model if it exists
    model = await async_db_ops.restore_llm_provider_model(provider_name, model_data["api"], model_data["model"])

    if model:
        # Update the restored model with new data
        model = await async_db_ops.update_llm_provider_model(
            provider_name=provider_name,
            api=model_data["api"],
            model=model_data["model"],
            custom_llm_provider=model_data["custom_llm_provider"],
            context_window=model_data.get("context_window"),
            max_input_tokens=model_data.get("max_input_tokens"),
            max_output_tokens=model_data.get("max_output_tokens"),
            tags=model_data.get("tags", []),
        )
    else:
        # Create new model
        model = await async_db_ops.create_llm_provider_model(
            provider_name=provider_name,
            api=model_data["api"],
            model=model_data["model"],
            custom_llm_provider=model_data["custom_llm_provider"],
            context_window=model_data.get("context_window"),
            max_input_tokens=model_data.get("max_input_tokens"),
            max_output_tokens=model_data.get("max_output_tokens"),
            tags=model_data.get("tags", []),
        )'''

NEW_CREATE_MODEL = '''async def create_llm_provider_model(provider_name: str, model_data: dict, user_id: str, is_admin: bool = False):
    """Create a new model for a specific provider or restore a soft-deleted one with the same combination

    Args:
        provider_name: Name of the provider
        model_data: Model configuration data
        user_id: User ID creating the model
        is_admin: Whether the user is an admin
    """
    # Check if provider exists
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)

    if not provider:
        raise ResourceNotFoundException("Provider", provider_name)

    # Check edit permission for the provider
    _check_edit_permission(user_id, is_admin, provider.user_id, provider_name)

    # --- Fix 1: Resolve custom_llm_provider from dialect, not UI label ---
    api_type = model_data["api"]
    dialect = (
        provider.completion_dialect if api_type == "completion"
        else provider.embedding_dialect if api_type == "embedding"
        else provider.rerank_dialect
    )
    resolved_custom_llm_provider = _resolve_custom_llm_provider(
        dialect,
        model_data.get("custom_llm_provider"),
    )

    # --- Fix 2: Auto-assign default scenario tags for first model of each type ---
    existing_tags = list(model_data.get("tags") or [])

    # Check if this provider already has models of this api type with default tags
    # If not, auto-assign default tags so the system works out of the box
    _DEFAULT_TAG_MAP = {
        "completion": ["default_for_background_task", "default_for_agent_completion"],
        "embedding": ["default_for_embedding"],
        "rerank": ["default_for_rerank"],
    }
    candidate_default_tags = _DEFAULT_TAG_MAP.get(api_type, [])

    if candidate_default_tags:
        # Check if any model already holds these default tags for this user
        for default_tag in candidate_default_tags:
            existing_default_models = await async_db_ops.find_models_by_tag(user_id, default_tag)
            if not existing_default_models:
                # No model has this default tag yet — assign it to this new model
                if default_tag not in existing_tags:
                    existing_tags.append(default_tag)

    # First check if there's an active model with the same combination
    active_existing = await async_db_ops.query_llm_provider_model(provider_name, api_type, model_data["model"])

    if active_existing:
        raise invalid_param(
            "model",
            f"Model \'{model_data[\'model\']}\' for API \'{api_type}\' already exists for provider \'{provider_name}\'",
        )

    # Try to restore a soft-deleted model if it exists
    model = await async_db_ops.restore_llm_provider_model(provider_name, api_type, model_data["model"])

    if model:
        # Update the restored model with new data
        model = await async_db_ops.update_llm_provider_model(
            provider_name=provider_name,
            api=api_type,
            model=model_data["model"],
            custom_llm_provider=resolved_custom_llm_provider,
            context_window=model_data.get("context_window"),
            max_input_tokens=model_data.get("max_input_tokens"),
            max_output_tokens=model_data.get("max_output_tokens"),
            tags=existing_tags,
        )
    else:
        # Create new model
        model = await async_db_ops.create_llm_provider_model(
            provider_name=provider_name,
            api=api_type,
            model=model_data["model"],
            custom_llm_provider=resolved_custom_llm_provider,
            context_window=model_data.get("context_window"),
            max_input_tokens=model_data.get("max_input_tokens"),
            max_output_tokens=model_data.get("max_output_tokens"),
            tags=existing_tags,
        )'''

if OLD_CREATE_MODEL in content:
    content = content.replace(OLD_CREATE_MODEL, NEW_CREATE_MODEL, 1)
    print("✓ Patched create_llm_provider_model()")
else:
    print("⚠ Could not find create_llm_provider_model() block — check manually")

# Write patched content
TARGET.write_text(content)
print(f"✓ Written to {TARGET}")
