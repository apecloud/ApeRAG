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

import json
import logging

from aperag.domains.marketplace.schemas import SharedCollectionConfig
from aperag.schema.common import CollectionConfig, ModelSpec

logger = logging.getLogger(__name__)


def parseCollectionConfig(config: str) -> CollectionConfig:
    try:
        config_dict = json.loads(config)
        collection_config = CollectionConfig.model_validate(config_dict)
        return collection_config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse collection config: {str(e)}")


def dumpCollectionConfig(collection_config: CollectionConfig) -> str:
    return collection_config.model_dump_json()


async def resolve_model_spec_legacy(spec: ModelSpec | None, *, user_id: str, capability) -> ModelSpec | None:
    """If ``spec`` was parsed from the pre-#1697 ``{model,
    model_service_provider, custom_llm_provider}`` triple, look up the
    matching new-schema ``Model`` row and populate ``spec.model_id``.

    Idempotent — returns ``spec`` unchanged when ``model_id`` is already
    set or when the legacy triple is empty. Logs (not raises) when the
    legacy data references a model that no longer exists in the new
    schema; the caller's ``model_id`` will stay ``None`` in that case
    and the existing ``InvalidConfigurationError`` path fires.
    """
    if spec is None:
        return None
    if spec.model_id:
        return spec
    if not spec.has_legacy_triple():
        return spec
    # Lazy import to avoid a circular dependency between
    # aperag.schema.common <- aperag.domains.model_platform.* <- aperag.schema.*
    from aperag.domains.model_platform.service.model_service import model_platform_service

    resolved = await model_platform_service.resolve_legacy_model_id(
        user_id,
        provider_name=spec.legacy_provider,
        provider_model_name=spec.legacy_model,
        custom_llm_provider=spec.legacy_custom_llm_provider,
        capability=capability,
    )
    if resolved:
        spec.model_id = resolved
    else:
        logger.warning(
            "Legacy ModelSpec triple did not resolve to a model_id",
            extra={
                "legacy_model": spec.legacy_model,
                "legacy_provider": spec.legacy_provider,
                "user_id": user_id,
            },
        )
    return spec


def convertToSharedCollectionConfig(config: CollectionConfig) -> SharedCollectionConfig:
    """Convert CollectionConfig to SharedCollectionConfig for marketplace display"""
    return SharedCollectionConfig(
        enable_vector=config.enable_vector if config.enable_vector is not None else True,
        enable_fulltext=config.enable_fulltext if config.enable_fulltext is not None else True,
        enable_knowledge_graph=config.enable_knowledge_graph if config.enable_knowledge_graph is not None else True,
        enable_summary=config.enable_summary if config.enable_summary is not None else False,
        enable_vision=config.enable_vision if config.enable_vision is not None else False,
    )
