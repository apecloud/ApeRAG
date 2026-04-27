#!/usr/bin/env python3
"""
Patch the CONTAINER runtime file directly.
Targets: /app/aperag/service/llm_provider_service.py (extracted from container)
"""
import shutil
from pathlib import Path

TARGET = Path("patches/llm_provider_service_CONTAINER.py")
BACKUP = Path("patches/llm_provider_service_CONTAINER.py.bak")

content = TARGET.read_text()
shutil.copy(TARGET, BACKUP)
print(f"Backed up to {BACKUP}")

# ----------------------------------------------------------------
# Fix 1: Add dialect mapping after PUBLIC_USER_ID constant
# ----------------------------------------------------------------
MAPPING_CODE = '''
# ---------------------------------------------------------------------------
# LiteLLM provider name mapping
# Maps ApeRAG dialect names to LiteLLM-recognized provider identifiers.
# ---------------------------------------------------------------------------
_DIALECT_TO_LITELLM: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "ollama": "ollama_chat",
    "ollama_chat": "ollama_chat",
    "jina_ai": "openai",
}


def _resolve_custom_llm_provider(dialect: str, custom_llm_provider: str | None) -> str:
    """Resolve the LiteLLM provider name from dialect.

    If the caller already supplied a valid LiteLLM provider name, keep it.
    Otherwise, derive it from the provider dialect setting.
    """
    known_litellm_providers = set(_DIALECT_TO_LITELLM.values()) | {
        "openai", "anthropic", "gemini", "ollama", "ollama_chat",
        "azure", "bedrock", "cohere", "huggingface", "together_ai",
        "deepseek", "mistral", "groq", "perplexity", "sagemaker",
    }
    if custom_llm_provider and custom_llm_provider.lower() in known_litellm_providers:
        return custom_llm_provider
    return _DIALECT_TO_LITELLM.get(dialect, "openai")

'''

INSERT_AFTER = 'PUBLIC_USER_ID = "public"'

if '_DIALECT_TO_LITELLM' not in content:
    if INSERT_AFTER in content:
        content = content.replace(INSERT_AFTER, INSERT_AFTER + "\n" + MAPPING_CODE, 1)
        print("✓ Added _DIALECT_TO_LITELLM and _resolve_custom_llm_provider()")
    else:
        print(f"⚠ Could not find '{INSERT_AFTER}' — printing first 30 lines:")
        for i, line in enumerate(content.splitlines()[:30], 1):
            print(f"  {i:3d}: {line}")
        raise SystemExit(1)
else:
    print("⚠ _DIALECT_TO_LITELLM already present")

# ----------------------------------------------------------------
# Fix 2: Patch create_llm_provider_model to use resolved provider
# ----------------------------------------------------------------
OLD_CREATE = '''    # First check if there's an active model with the same combination
    active_existing = await async_db_ops.query_llm_provider_model(provider_name, model_data["api"], model_data["model"])

    if active_existing:
        raise invalid_param(
            "model",
            f"Model '{model_data[\'model\']}' for API '{model_data[\'api\']}' already exists for provider '{provider_name}'",
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

NEW_CREATE = '''    # --- Resolve custom_llm_provider from dialect, not the UI label ---
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

    # --- Auto-assign default scenario tags for first model of each type ---
    existing_tags = list(model_data.get("tags") or [])
    _DEFAULT_TAG_MAP = {
        "completion": ["default_for_background_task", "default_for_agent_completion"],
        "embedding":  ["default_for_embedding"],
        "rerank":     ["default_for_rerank"],
    }
    for default_tag in _DEFAULT_TAG_MAP.get(api_type, []):
        existing_default_models = await async_db_ops.find_models_by_tag(user_id, default_tag)
        if not existing_default_models:
            if default_tag not in existing_tags:
                existing_tags.append(default_tag)

    # First check if there's an active model with the same combination
    active_existing = await async_db_ops.query_llm_provider_model(provider_name, api_type, model_data["model"])

    if active_existing:
        raise invalid_param(
            "model",
            f"Model '{model_data[\'model\']}' for API '{api_type}' already exists for provider '{provider_name}'",
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

if OLD_CREATE in content:
    content = content.replace(OLD_CREATE, NEW_CREATE, 1)
    print("✓ Patched create_llm_provider_model()")
else:
    print("⚠ Could not find create block — dumping search context:")
    for i, line in enumerate(content.splitlines()):
        if 'restore_llm_provider_model' in line:
            start = max(0, i-5)
            end = min(len(content.splitlines()), i+5)
            for j, l in enumerate(content.splitlines()[start:end], start=start):
                print(f"  {j:4d}: {l}")
    raise SystemExit(1)

# ----------------------------------------------------------------
# Fix 3: Patch update_llm_provider_model to also use resolved provider
# ----------------------------------------------------------------
OLD_UPDATE = '''    # Update model using the DatabaseOps method
    model_obj = await async_db_ops.update_llm_provider_model(
        provider_name=provider_name,
        api=api,
        model=model,
        custom_llm_provider=update_data.get("custom_llm_provider"),'''

NEW_UPDATE = '''    # Resolve custom_llm_provider from dialect on update too
    update_dialect = (
        provider.completion_dialect if api == "completion"
        else provider.embedding_dialect if api == "embedding"
        else provider.rerank_dialect
    )
    resolved_update_provider = _resolve_custom_llm_provider(
        update_dialect,
        update_data.get("custom_llm_provider"),
    )

    # Update model using the DatabaseOps method
    model_obj = await async_db_ops.update_llm_provider_model(
        provider_name=provider_name,
        api=api,
        model=model,
        custom_llm_provider=resolved_update_provider,'''

if OLD_UPDATE in content:
    content = content.replace(OLD_UPDATE, NEW_UPDATE, 1)
    print("✓ Patched update_llm_provider_model()")
else:
    print("⚠ Could not find update block — skipping (non-critical)")

TARGET.write_text(content)
print(f"✓ Written to {TARGET}")
