from aperag.llm.runtime.resolver import resolve_model_invocation_from_records
from aperag.llm.runtime.types import ModelCapability, ResolvedModelInvocation


def test_resolver_builds_runner_config_from_model_and_account_records():
    account = {
        "id": "acct_1",
        "provider_type": "dashscope",
        "base_url": "https://dashscope.aliyuncs.com",
        "api_key": "sk-test",
    }
    model = {
        "id": "model_1",
        "account_id": "acct_1",
        "provider_model_id": "gte-rerank-v2",
        "capability": "rerank",
        "runner_type": "dashscope_rerank",
        "runner_config": {"path": "/api/v1/services/rerank/text-rerank/text-rerank"},
    }

    resolved = resolve_model_invocation_from_records(model=model, account=account)

    assert isinstance(resolved, ResolvedModelInvocation)
    assert resolved.model_id == "model_1"
    assert resolved.provider_model_id == "gte-rerank-v2"
    assert resolved.capability == ModelCapability.RERANK
    assert resolved.runner_type == "dashscope_rerank"
    assert resolved.base_url == "https://dashscope.aliyuncs.com"
    assert resolved.api_key == "sk-test"
    assert resolved.runner_config["path"].endswith("text-rerank")


def test_resolver_defaults_runner_for_openai_compatible_chat():
    account = {
        "id": "acct_2",
        "provider_type": "openai_compatible",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local-key",
    }
    model = {
        "id": "model_2",
        "account_id": "acct_2",
        "provider_model_id": "qwen3",
        "capability": "chat",
    }

    resolved = resolve_model_invocation_from_records(model=model, account=account)

    assert resolved.capability == ModelCapability.CHAT
    assert resolved.runner_type == "openai_compatible"
