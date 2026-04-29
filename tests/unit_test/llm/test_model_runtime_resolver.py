from aperag.llm.runtime.resolver import resolve_model_invocation_from_records
from aperag.llm.runtime.types import ModelCapability


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
