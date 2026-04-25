import logging

from aperag.db.ops import db_ops
from aperag.llm.completion.completion_service import CompletionService
from aperag.llm.llm_error_types import InvalidConfigurationError, ModelNotFoundError
from aperag.llm.runtime.resolver import resolve_model_invocation_from_records
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


def get_completion_service(model_id: str, user_id: str, temperature: float = 0.1) -> CompletionService:
    if not model_id:
        raise InvalidConfigurationError("model_id", model_id, "Model id cannot be empty")
    row = db_ops.query_model_runtime(model_id, user_id)
    if not row:
        raise ModelNotFoundError(model_id, "model", "Completion")
    model, account = row
    invocation = resolve_model_invocation_from_records(model=model, account=account)
    provider = invocation.runner_config.get("provider")
    if not provider:
        provider = "openai" if invocation.runner_type == "openai_compatible" else invocation.provider_type
    return CompletionService(
        provider=provider,
        model=invocation.provider_model_id,
        base_url=invocation.base_url,
        api_key=invocation.api_key,
        temperature=temperature,
        vision=invocation.supports_vision,
    )


def get_collection_completion_service_sync(collection) -> CompletionService:
    config = parseCollectionConfig(collection.config)
    if not config.completion:
        raise InvalidConfigurationError("collection.config.completion.model_id", None, "Completion model is required")
    spec = config.completion
    model_id = spec.model_id
    if not model_id and spec.has_legacy_triple():
        # Pre-#1697 back-compat: resolve the legacy
        # ``{model, model_service_provider}`` triple via the sync
        # repository helper (Weston msg=80e873c1 / Blocker C).
        from aperag.domains.model_platform.service.model_service import _normalise_provider_type

        provider_type = _normalise_provider_type(spec.legacy_provider) or _normalise_provider_type(
            spec.legacy_custom_llm_provider
        )
        if provider_type and spec.legacy_model:
            model_id = db_ops.resolve_legacy_model_id(
                user_id=collection.user,
                provider_type=provider_type,
                provider_model_id=spec.legacy_model,
                capability="chat",
            )
            if model_id:
                spec.model_id = model_id
    if not model_id:
        raise InvalidConfigurationError("collection.config.completion.model_id", None, "Completion model is required")
    temperature = spec.temperature or 0.1
    return get_completion_service(model_id, collection.user, temperature=temperature)
