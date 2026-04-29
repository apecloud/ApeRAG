from __future__ import annotations

from typing import Any

from aperag.llm.runtime.types import ModelCapability, ResolvedModelInvocation


def _get_value(record: Any, key: str, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def infer_runner_type(*, provider_type: str, capability: str) -> str:
    if provider_type == "pydantic_ai":
        return "pydantic_ai"
    return "openai_compatible"


def resolve_model_invocation_from_records(*, model: Any, account: Any) -> ResolvedModelInvocation:
    capability = _get_value(model, "capability")
    provider_type = _get_value(account, "provider_type")
    runner_type = _get_value(model, "runner_type") or infer_runner_type(
        provider_type=provider_type,
        capability=capability,
    )
    api_key = _get_value(account, "api_key") or _get_value(account, "encrypted_api_key")

    return ResolvedModelInvocation(
        model_id=_get_value(model, "id"),
        account_id=_get_value(account, "id"),
        provider_type=provider_type,
        provider_model_id=_get_value(model, "provider_model_id"),
        capability=ModelCapability(capability),
        runner_type=runner_type,
        base_url=_get_value(account, "base_url"),
        api_key=api_key,
        runner_config=_get_value(model, "runner_config", {}) or {},
        context_window=_get_value(model, "context_window"),
        max_input_tokens=_get_value(model, "max_input_tokens"),
        max_output_tokens=_get_value(model, "max_output_tokens"),
        embedding_dimensions=_get_value(model, "embedding_dimensions"),
        supports_vision=bool(_get_value(model, "supports_vision", False)),
        supports_tool_calling=bool(_get_value(model, "supports_tool_calling", False)),
    )
