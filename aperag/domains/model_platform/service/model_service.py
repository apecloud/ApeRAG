from __future__ import annotations

from aperag.db.ops import async_db_ops
from aperag.domains.model_platform.schemas import (
    Model,
    ModelAccount,
    ModelAccountCreate,
    ModelAccountList,
    ModelAccountUpdate,
    ModelCapability,
    ModelCreate,
    ModelList,
    ModelProvider,
    ModelProviderList,
    ModelUpdate,
    ModelUse,
    ModelUseList,
    ModelUseScenario,
    ModelUseUpdate,
    ModelValidationResponse,
)
from aperag.exceptions import PermissionDeniedError, ResourceNotFoundException, ValidationException
from aperag.llm.runtime.resolver import infer_runner_type

ALLOWED_SCENARIOS_EXTRA_KEY = "allowed_scenarios"

CAPABILITY_SCENARIOS: dict[ModelCapability, tuple[ModelUseScenario, ...]] = {
    ModelCapability.CHAT: (
        ModelUseScenario.AGENT_CHAT,
        ModelUseScenario.COLLECTION_COMPLETION,
        ModelUseScenario.BACKGROUND_TASK,
    ),
    ModelCapability.COMPLETION: (
        ModelUseScenario.AGENT_CHAT,
        ModelUseScenario.COLLECTION_COMPLETION,
        ModelUseScenario.BACKGROUND_TASK,
    ),
    ModelCapability.EMBEDDING: (ModelUseScenario.COLLECTION_EMBEDDING,),
}

SCENARIO_CAPABILITY: dict[ModelUseScenario, ModelCapability] = {
    ModelUseScenario.AGENT_CHAT: ModelCapability.CHAT,
    ModelUseScenario.COLLECTION_COMPLETION: ModelCapability.CHAT,
    ModelUseScenario.COLLECTION_EMBEDDING: ModelCapability.EMBEDDING,
    ModelUseScenario.BACKGROUND_TASK: ModelCapability.CHAT,
}


def _capability_value(capability) -> str:
    return capability.value if hasattr(capability, "value") else str(capability)


def _scenario_value(scenario) -> str:
    return scenario.value if hasattr(scenario, "value") else str(scenario)


def _normalise_capability(capability) -> ModelCapability:
    if isinstance(capability, ModelCapability):
        return capability
    return ModelCapability(str(capability))


def _normalise_scenario(scenario) -> ModelUseScenario:
    if isinstance(scenario, ModelUseScenario):
        return scenario
    return ModelUseScenario(str(scenario))


def default_allowed_scenarios(capability) -> list[ModelUseScenario]:
    """Return the backward-compatible scenario set for models without explicit configuration."""
    return list(CAPABILITY_SCENARIOS[_normalise_capability(capability)])


def allowed_scenarios_for_model(model) -> list[ModelUseScenario]:
    extra = model.extra or {}
    raw = extra.get(ALLOWED_SCENARIOS_EXTRA_KEY)
    if raw is None:
        return default_allowed_scenarios(model.capability)
    scenarios: list[ModelUseScenario] = []
    for item in raw:
        try:
            scenario = _normalise_scenario(item)
        except ValueError as exc:
            raise ValidationException(f"Unknown model scenario '{item}'") from exc
        if scenario not in scenarios:
            scenarios.append(scenario)
    return scenarios


def _validate_allowed_scenarios(capability, scenarios: list[ModelUseScenario] | None) -> list[str] | None:
    if scenarios is None:
        return None
    model_capability = _normalise_capability(capability)
    valid_scenarios = set(CAPABILITY_SCENARIOS[model_capability])
    normalised: list[ModelUseScenario] = []
    for item in scenarios:
        scenario = _normalise_scenario(item)
        if scenario not in valid_scenarios:
            raise ValidationException(
                f"Scenario '{scenario.value}' is not valid for model capability '{model_capability.value}'"
            )
        if scenario not in normalised:
            normalised.append(scenario)
    return [scenario.value for scenario in normalised]


def _model_extra_with_allowed_scenarios(
    extra: dict | None,
    capability,
    scenarios: list[ModelUseScenario] | None,
) -> dict:
    next_extra = dict(extra or {})
    allowed = _validate_allowed_scenarios(capability, scenarios)
    if allowed is not None:
        next_extra[ALLOWED_SCENARIOS_EXTRA_KEY] = allowed
    return next_extra


BUILTIN_PROVIDERS = [
    ModelProvider(
        provider_type="openai",
        display_name="OpenAI",
        supported_capabilities=[ModelCapability.CHAT, ModelCapability.EMBEDDING],
        default_base_url="https://api.openai.com/v1",
        sort_order=10,
    ),
    ModelProvider(
        provider_type="dashscope",
        display_name="阿里百炼",
        supported_capabilities=[ModelCapability.CHAT, ModelCapability.EMBEDDING],
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        sort_order=20,
    ),
    ModelProvider(
        provider_type="jina",
        display_name="Jina AI",
        supported_capabilities=[ModelCapability.EMBEDDING],
        default_base_url="https://api.jina.ai/v1",
        sort_order=30,
    ),
    ModelProvider(
        provider_type="openai_compatible",
        display_name="OpenAI Compatible",
        supported_capabilities=[ModelCapability.CHAT, ModelCapability.EMBEDDING],
        sort_order=100,
    ),
]


def _account_to_schema(account) -> ModelAccount:
    return ModelAccount(
        id=account.id,
        user_id=account.user_id,
        provider_type=account.provider_type,
        name=account.name,
        display_name=account.display_name,
        base_url=account.base_url,
        status=account.status,
        auth_config=account.auth_config or {},
        last_validated_at=account.last_validated_at,
        validation_error=account.validation_error,
        extra=account.extra or {},
        created=account.gmt_created,
        updated=account.gmt_updated,
    )


def _provider_to_schema(provider) -> ModelProvider:
    return ModelProvider(
        id=provider.id,
        provider_type=provider.provider_type,
        display_name=provider.display_name,
        description=provider.description,
        supported_capabilities=provider.supported_capabilities or [],
        account_schema=provider.account_schema or {},
        default_base_url=provider.default_base_url,
        enabled=provider.enabled,
        sort_order=provider.sort_order,
        extra=provider.extra or {},
        created=provider.gmt_created,
        updated=provider.gmt_updated,
    )


def _model_to_schema(model) -> Model:
    return Model(
        id=model.id,
        account_id=model.account_id,
        provider_model_id=model.provider_model_id,
        display_name=model.display_name,
        capability=model.capability,
        runner_type=model.runner_type,
        runner_config=model.runner_config or {},
        context_window=model.context_window,
        max_input_tokens=model.max_input_tokens,
        max_output_tokens=model.max_output_tokens,
        embedding_dimensions=model.embedding_dimensions,
        supports_vision=model.supports_vision,
        supports_tool_calling=model.supports_tool_calling,
        supports_multimodal_embedding=getattr(model, "supports_multimodal_embedding", False) or False,
        status=model.status,
        allowed_scenarios=allowed_scenarios_for_model(model),
        extra=model.extra or {},
        created=model.gmt_created,
        updated=model.gmt_updated,
    )


def _model_use_to_schema(model_use) -> ModelUse:
    return ModelUse(
        id=model_use.id,
        user_id=model_use.user_id,
        scenario=model_use.scenario,
        capability=model_use.capability,
        strategy=model_use.strategy,
        primary_model_id=model_use.primary_model_id,
        fallback_model_ids=model_use.fallback_model_ids or [],
        enabled=model_use.enabled,
        extra=model_use.extra or {},
        created=model_use.gmt_created,
        updated=model_use.gmt_updated,
    )


# Mapping from legacy provider names (used by the pre-#1697
# ``model_service_provider`` / ``custom_llm_provider`` fields) to the
# new ``model_provider.provider_type`` values. Same set the migration
# helper uses; kept duplicated here so neither file imports the other.
_LEGACY_PROVIDER_TO_TYPE = {
    "openai": "openai",
    "openrouter": "openai_compatible",
    "alibabacloud": "dashscope",
    "dashscope": "dashscope",
    "jina": "jina",
    "jina_ai": "jina",
    "openai_compatible": "openai_compatible",
}


def _normalise_provider_type(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.strip().lower()
    return _LEGACY_PROVIDER_TO_TYPE.get(lowered, lowered)


class ModelPlatformService:
    async def ensure_model_allowed_for_scenario(
        self, user_id: str, model_id: str | None, scenario: ModelUseScenario | str
    ) -> None:
        if not model_id:
            return
        model = await async_db_ops.query_model(model_id, user_id)
        if model is None:
            raise ResourceNotFoundException("Model", model_id)
        expected_capability = SCENARIO_CAPABILITY[_normalise_scenario(scenario)]
        model_capability = _normalise_capability(model.capability)
        if model_capability != expected_capability:
            raise ValidationException(
                f"Model '{model_id}' capability '{model_capability.value}' cannot be used for scenario "
                f"'{_scenario_value(scenario)}'"
            )
        allowed = allowed_scenarios_for_model(model)
        normalised_scenario = _normalise_scenario(scenario)
        if normalised_scenario not in allowed:
            raise ValidationException(f"Model '{model_id}' is not allowed for scenario '{normalised_scenario.value}'")

    async def list_providers(self) -> ModelProviderList:
        providers = await async_db_ops.query_model_providers()
        return ModelProviderList(items=[_provider_to_schema(provider) for provider in providers] or BUILTIN_PROVIDERS)

    async def resolve_legacy_model_id(
        self,
        user_id: str,
        *,
        provider_name: str | None,
        provider_model_name: str | None,
        custom_llm_provider: str | None = None,
        capability: ModelCapability | None = None,
    ) -> str | None:
        """Look up the new-schema ``model_id`` for a legacy
        ``{provider, model, custom_llm_provider}`` triple.

        Returns ``None`` when no matching ``Model`` row exists for the
        user (or for ``user_id="public"`` system models). Used by the
        permanent OpenAI-compat ``/api/v1/embeddings`` route
        (Weston msg=80e873c1 / Blocker A) to keep pre-#1697
        callers working after the model-platform refactor without
        touching v3.

        ``custom_llm_provider`` is treated as a tiebreak hint when the
        ``provider_name`` itself is empty / unknown; its primary role
        in the legacy schema was to disambiguate dialect within a
        single ``llm_provider`` row.
        """
        provider_type = _normalise_provider_type(provider_name) or _normalise_provider_type(custom_llm_provider)
        if not provider_type or not provider_model_name:
            return None
        models = await async_db_ops.query_models(user_id=user_id)
        candidates = [model for model in models if model.provider_model_id == provider_model_name]
        if not candidates:
            return None
        # Filter by provider_type via the joined ModelAccount row.
        # Prefer user-owned over public when both match.
        matched_user: list = []
        matched_public: list = []
        for model in candidates:
            if capability is not None and model.capability != capability.value:
                continue
            account = await async_db_ops.query_model_account(model.account_id, user_id)
            if account is None:
                continue
            if account.provider_type != provider_type:
                continue
            if account.user_id == user_id:
                matched_user.append(model)
            else:
                matched_public.append(model)
        chosen = matched_user or matched_public
        return chosen[0].id if chosen else None

    async def list_accounts(self, user_id: str) -> ModelAccountList:
        accounts = await async_db_ops.query_model_accounts(user_id)
        return ModelAccountList(items=[_account_to_schema(account) for account in accounts])

    async def get_user_provider_api_key(
        self,
        user_id: str,
        provider_type: str,
        *,
        fallback_to_public: bool = False,
    ) -> str | None:
        """Return the API key the user has configured for ``provider_type``.

        When ``fallback_to_public`` is ``True``, fall back to a
        system-shared (``public``) account if the user has no personal
        account configured. Returns ``None`` when no account is configured.

        This is the canonical surface for non-model-platform callers
        (web_access, knowledge_base) that need a raw provider API key
        without owning a ``ModelAccount`` id; it replaces the
        legacy ``llm_provider.api_key`` lookup the model-platform refactor
        removed.
        """
        return await async_db_ops.query_model_account_api_key(
            provider_type=provider_type,
            user_id=user_id,
            fallback_to_public=fallback_to_public,
        )

    async def create_account(self, user_id: str, request: ModelAccountCreate) -> ModelAccount:
        account = await async_db_ops.create_model_account(user_id=user_id, **request.model_dump())
        return _account_to_schema(account)

    async def update_account(self, account_id: str, user_id: str, request: ModelAccountUpdate) -> ModelAccount:
        account = await async_db_ops.update_model_account(account_id, user_id, request.model_dump(exclude_unset=True))
        if account is None:
            raise ResourceNotFoundException("ModelAccount", account_id)
        return _account_to_schema(account)

    async def validate_account(self, account_id: str, user_id: str) -> ModelValidationResponse:
        account = await async_db_ops.query_model_account(account_id, user_id)
        if account is None:
            raise ResourceNotFoundException("ModelAccount", account_id)
        return ModelValidationResponse(ok=bool(account), message=None if account else "Model account not found")

    async def list_models(
        self,
        user_id: str,
        account_id: str | None = None,
        capability: ModelCapability | None = None,
        scenario: ModelUseScenario | None = None,
    ) -> ModelList:
        models = await async_db_ops.query_models(user_id=user_id, account_id=account_id)
        if scenario is not None:
            capability = SCENARIO_CAPABILITY[scenario]
        if capability is not None:
            models = [model for model in models if _normalise_capability(model.capability) == capability]
        if scenario is not None:
            models = [model for model in models if scenario in allowed_scenarios_for_model(model)]
        return ModelList(items=[_model_to_schema(model) for model in models])

    async def create_model(self, user_id: str, request: ModelCreate) -> Model:
        account = await async_db_ops.query_model_account(request.account_id, user_id)
        if account is None:
            raise ResourceNotFoundException("ModelAccount", request.account_id)
        if account.user_id != user_id:
            raise PermissionDeniedError("Cannot create models under a shared system account")
        runner_type = request.runner_type or infer_runner_type(
            provider_type=account.provider_type,
            capability=request.capability.value,
        )
        payload = request.model_dump()
        payload.pop("runner_type", None)
        allowed_scenarios = payload.pop("allowed_scenarios", None)
        if allowed_scenarios is None and ALLOWED_SCENARIOS_EXTRA_KEY in payload.get("extra", {}):
            allowed_scenarios = payload["extra"][ALLOWED_SCENARIOS_EXTRA_KEY]
        payload["extra"] = _model_extra_with_allowed_scenarios(
            payload.get("extra"),
            request.capability,
            allowed_scenarios,
        )
        model = await async_db_ops.create_model(user_id=user_id, runner_type=runner_type, **payload)
        return _model_to_schema(model)

    async def update_model(self, model_id: str, user_id: str, request: ModelUpdate) -> Model:
        payload = request.model_dump(exclude_unset=True)
        if "allowed_scenarios" in payload:
            existing = await async_db_ops.query_model(model_id, user_id)
            if existing is None:
                raise ResourceNotFoundException("Model", model_id)
            capability = payload.get("capability") or existing.capability
            extra = dict(existing.extra or {})
            if payload.get("extra"):
                extra.update(payload["extra"])
            payload["extra"] = _model_extra_with_allowed_scenarios(
                extra,
                capability,
                payload.pop("allowed_scenarios"),
            )
        elif "capability" in payload:
            extra = payload.get("extra")
            if extra and ALLOWED_SCENARIOS_EXTRA_KEY in extra:
                payload["extra"] = _model_extra_with_allowed_scenarios(
                    extra,
                    payload["capability"],
                    extra[ALLOWED_SCENARIOS_EXTRA_KEY],
                )
        elif payload.get("extra") and ALLOWED_SCENARIOS_EXTRA_KEY in payload["extra"]:
            existing = await async_db_ops.query_model(model_id, user_id)
            if existing is None:
                raise ResourceNotFoundException("Model", model_id)
            payload["extra"] = _model_extra_with_allowed_scenarios(
                payload["extra"],
                existing.capability,
                payload["extra"][ALLOWED_SCENARIOS_EXTRA_KEY],
            )
        model = await async_db_ops.update_model(model_id, user_id, payload)
        if model is None:
            raise ResourceNotFoundException("Model", model_id)
        return _model_to_schema(model)

    async def validate_model(self, model_id: str, user_id: str) -> ModelValidationResponse:
        model = await async_db_ops.query_model(model_id, user_id)
        return ModelValidationResponse(ok=bool(model), message=None if model else "Model not found")

    async def list_model_uses(self, user_id: str) -> ModelUseList:
        uses = await async_db_ops.query_model_uses(user_id)
        return ModelUseList(items=[_model_use_to_schema(item) for item in uses])

    async def update_model_use(self, user_id: str, scenario: ModelUseScenario, request: ModelUseUpdate) -> ModelUse:
        for model_id in [request.primary_model_id, *request.fallback_model_ids]:
            await self.ensure_model_allowed_for_scenario(user_id, model_id, scenario)
        payload = request.model_dump()
        payload["capability"] = SCENARIO_CAPABILITY[scenario].value
        item = await async_db_ops.upsert_model_use(user_id, scenario.value, payload)
        return _model_use_to_schema(item)


model_platform_service = ModelPlatformService()
