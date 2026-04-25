import logging

from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import EmbeddingError, InvalidConfigurationError, ModelNotFoundError
from aperag.llm.runtime.resolver import resolve_model_invocation_from_records
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)
_dimension_cache: dict[str, int] = {}


def _get_embedding_dimension(embedding_svc: EmbeddingService, model_id: str, configured_dimension: int | None) -> int:
    if configured_dimension:
        return configured_dimension
    if model_id in _dimension_cache:
        return _dimension_cache[model_id]
    vec = embedding_svc.embed_query("dimension_probe")
    if not vec:
        raise EmbeddingError("Failed to obtain embedding vector while probing dimension", {"model_id": model_id})
    dim = len(vec[0]) if isinstance(vec[0], (list, tuple)) else len(vec)
    _dimension_cache[model_id] = dim
    return dim


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
    embedding_svc = EmbeddingService(
        embedding_provider=provider,
        embedding_model=invocation.provider_model_id,
        embedding_service_url=invocation.base_url,
        embedding_service_api_key=invocation.api_key,
        embedding_max_chunks_in_batch=settings.embedding_max_chunks_in_batch,
        multimodal=invocation.runner_config.get("multimodal", False),
    )
    return embedding_svc, _get_embedding_dimension(embedding_svc, model_id, invocation.embedding_dimensions)


def get_collection_embedding_service_sync(collection) -> tuple[EmbeddingService, int]:
    config = parseCollectionConfig(collection.config)
    if not config.embedding or not config.embedding.model_id:
        raise InvalidConfigurationError("collection.config.embedding.model_id", None, "Embedding model is required")
    return get_embedding_service(config.embedding.model_id, collection.user)
