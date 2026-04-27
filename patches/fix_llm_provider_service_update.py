#!/usr/bin/env python3
import shutil
from pathlib import Path

TARGET = Path("aperag/domains/model_platform/service/llm_provider_service.py")
BACKUP = TARGET.with_suffix(".py.bak2")

content = TARGET.read_text()
shutil.copy(TARGET, BACKUP)
print(f"Backed up to {BACKUP}")

OLD = '''    # Update model using the DatabaseOps method
    model_obj = await async_db_ops.update_llm_provider_model(
        provider_name=provider_name,
        api=api,
        model=model,
        custom_llm_provider=update_data.get("custom_llm_provider"),
        context_window=update_data.get("context_window"),
        max_input_tokens=update_data.get("max_input_tokens"),
        max_output_tokens=update_data.get("max_output_tokens"),
        tags=update_data.get("tags"),
    )'''

NEW = '''    # Resolve custom_llm_provider from provider dialect, not the UI label
    dialect = (
        provider.completion_dialect if api == "completion"
        else provider.embedding_dialect if api == "embedding"
        else provider.rerank_dialect
    )
    resolved_custom_llm_provider = _resolve_custom_llm_provider(
        dialect,
        update_data.get("custom_llm_provider"),
    )

    # Update model using the DatabaseOps method
    model_obj = await async_db_ops.update_llm_provider_model(
        provider_name=provider_name,
        api=api,
        model=model,
        custom_llm_provider=resolved_custom_llm_provider,
        context_window=update_data.get("context_window"),
        max_input_tokens=update_data.get("max_input_tokens"),
        max_output_tokens=update_data.get("max_output_tokens"),
        tags=update_data.get("tags"),
    )'''

if OLD not in content:
    print("Could not find update block to patch")
    raise SystemExit(1)

content = content.replace(OLD, NEW, 1)
TARGET.write_text(content)
print(f"Patched {TARGET}")
