import hashlib
import logging

from aperag.cache import NAMESPACE_EMBEDDING_DIMENSION, application_cache_policy, get_sync_application_cache
from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import EmbeddingError, InvalidConfigurationError, ModelNotFoundError
from aperag.llm.runtime.resolver import resolve_model_invocation_from_records
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


def _get_embedding_dimension(embedding_svc: EmbeddingService, model_id: str, configured_dimension: int | None) -> int:
    if configured_dimension:
        return configured_dimension

    def compute_dimension() -> int:
        vec = embedding_svc.embed_query("dimension_probe")
        if not vec:
            raise EmbeddingError("Failed to obtain embedding vector while probing dimension", {"model_id": model_id})
        return len(vec[0]) if isinstance(vec[0], (list, tuple)) else len(vec)

    cache = get_sync_application_cache()
    return int(
        cache.get_or_compute(
            namespace=NAMESPACE_EMBEDDING_DIMENSION,
            key_data={
                "model_id": model_id,
                "provider": embedding_svc.embedding_provider,
                "model": embedding_svc.model,
                "api_base": embedding_svc.api_base,
                "api_key_hash": hashlib.sha256((embedding_svc.api_key or "").encode("utf-8")).hexdigest(),
            },
            compute=compute_dimension,
            policy=application_cache_policy(NAMESPACE_EMBEDDING_DIMENSION),
        )
    )


def get_embedding_service(model_id: str, user_id: str) -> tuple[EmbeddingService, int]:
    if not model_id:
        raise InvalidConfigurationError("model_id", model_id, "Model id cannot be empty")
    row = db_ops.query_model_runtime(model_id, user_id)
    if not row:
        raise ModelNotFoundError(model_id, "model", "Embedding")
    model, account = row
    invocation = resolve_model_invocation_from_records(model=model, account=account)
    provider = invocation.runner_config.get("provider")
    if not provider:
        provider = "openai" if invocation.runner_type == "openai_compatible" else invocation.provider_type
    # Wave 5 P2 chunk 3 (per §G.2.5.1 spec amend item 3): prefer the
    # typed ``Model.supports_multimodal_embedding`` column when present,
    # fall back to the legacy ``runner_config["multimodal"]`` JSON dict
    # entry so collections / models created before the typed column
    # landed keep working (operators who edited the JSON keep the same
    # behaviour without re-saving the model row).
    multimodal = bool(getattr(model, "supports_multimodal_embedding", False)) or bool(
        invocation.runner_config.get("multimodal", False)
    )
    embedding_svc = EmbeddingService(
        embedding_provider=provider,
        embedding_model=invocation.provider_model_id,
        embedding_service_url=invocation.base_url,
        embedding_service_api_key=invocation.api_key,
        embedding_max_chunks_in_batch=settings.embedding_max_chunks_in_batch,
        embedding_max_workers=settings.embedding_max_workers,
        multimodal=multimodal,
    )
    return embedding_svc, _get_embedding_dimension(embedding_svc, model_id, invocation.embedding_dimensions)


def get_collection_embedding_service_sync(collection) -> tuple[EmbeddingService, int]:
    config = parseCollectionConfig(collection.config)
    if not config.embedding:
        raise InvalidConfigurationError("collection.config.embedding.model_id", None, "Embedding model is required")
    spec = config.embedding
    model_id = spec.model_id
    if not model_id and spec.has_legacy_triple():
        # Pre-#1697 collections wrote ``{model, model_service_provider,
        # custom_llm_provider}`` instead of ``{model_id}``. Resolve via
        # the sync repository helper so existing rows continue to work
        # after the model-platform refactor (Weston msg=80e873c1).
        from aperag.domains.model_platform.service.model_service import _normalise_provider_type

        provider_type = _normalise_provider_type(spec.legacy_provider) or _normalise_provider_type(
            spec.legacy_custom_llm_provider
        )
        if provider_type and spec.legacy_model:
            model_id = db_ops.resolve_legacy_model_id(
                user_id=collection.user,
                provider_type=provider_type,
                provider_model_id=spec.legacy_model,
                capability="embedding",
            )
            if model_id:
                spec.model_id = model_id
    if not model_id:
        raise InvalidConfigurationError("collection.config.embedding.model_id", None, "Embedding model is required")
    return get_embedding_service(model_id, collection.user)
