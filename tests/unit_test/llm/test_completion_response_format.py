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

"""task #14 / issue #1861: ``response_format`` provider-side strong constraint.

钉死 6 条契约 (fix-forward 后, 跟 huangzhangshu task #3 类同 audit msg=1c06455d
+ Weston msg=ef033106 + earayu2 msg=9fddd42d "只对 graph index 生效" 产品锁
对齐):

1. ``response_format=None`` (默认) 时 ``litellm.acompletion`` 调用 **不含**
   ``response_format`` kwarg — 保留 chat / summary / agent-runtime 等老
   调用的 prompt-text-only 行为.
2. ``response_format={"type":"json_object"}`` 时 kwarg 透传到
   ``litellm.acompletion`` (sync + async + stream 三条路径都透传).
3. cache key 包含 ``response_format`` — 同 prompt 但不同 mode 不复用缓存.
4. ``build_collection_llm_callable`` **默认 ``response_format=None``** —
   shared builder 给 graph_curation / collection_regen / evaluation worker /
   dataset_generator / summary worker 等非 graph extractor 模块复用,
   不能默认注入 JSON 模式.
5. ``build_collection_llm_callable`` 显式传 ``response_format`` 时透传给
   ``CompletionService``.
6. graph extractor builder 在调用入口显式传
   ``response_format={"type":"json_object"}`` — 这是 "只对 graph index
   生效" 落点, 其他共享 builder caller 不受影响.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_response_format_default_none_does_not_pass_kwarg_to_litellm():
    """task #14 契约 1: 老调用方 (chat / summary / agent-runtime) 默认
    ``response_format=None``, ``litellm.acompletion`` 调用 **不应** 含
    ``response_format`` kwarg, 行为跟 issue #1861 改造前一致.
    """
    from aperag.llm.completion import completion_service as module
    from aperag.llm.completion.completion_service import CompletionService

    service = CompletionService("openai", "gpt-test", "https://example.invalid", "sk", caching=False)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    with patch.object(module.litellm, "acompletion", new_callable=AsyncMock) as mocked:
        mocked.return_value = response
        await service.agenerate([], "hello")

    assert mocked.call_count == 1
    assert "response_format" not in mocked.call_args.kwargs


@pytest.mark.asyncio
async def test_response_format_json_object_passed_through_to_litellm():
    """task #14 契约 2: 显式传 ``response_format={"type":"json_object"}``
    时 kwarg 透传到 ``litellm.acompletion``, 让 OpenAI / DeepSeek / Qwen
    等 provider 走 JSON 模式强约束.
    """
    from aperag.llm.completion import completion_service as module
    from aperag.llm.completion.completion_service import CompletionService

    service = CompletionService(
        "openai",
        "gpt-test",
        "https://example.invalid",
        "sk",
        caching=False,
        response_format={"type": "json_object"},
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"x":1}'))])

    with patch.object(module.litellm, "acompletion", new_callable=AsyncMock) as mocked:
        mocked.return_value = response
        await service.agenerate([], "extract entities")

    assert mocked.call_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_response_format_propagates_to_sync_and_stream_paths():
    """task #14 契约 2 扩展: sync ``generate`` + streaming 都透传.
    避免 graph extractor 偶尔走错路径丢失约束.
    """
    from aperag.llm.completion import completion_service as module
    from aperag.llm.completion.completion_service import CompletionService

    service = CompletionService(
        "openai",
        "gpt-test",
        "https://example.invalid",
        "sk",
        caching=False,
        response_format={"type": "json_object"},
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"x":1}'))])

    # Sync path
    with patch.object(module.litellm, "completion") as mocked_sync:
        mocked_sync.return_value = response
        service.generate([], "p")
    assert mocked_sync.call_args.kwargs["response_format"] == {"type": "json_object"}

    # Streaming path — _acompletion_stream_raw 也用 _litellm_kwargs.

    async def _empty_stream():
        if False:  # pragma: no cover — empty async generator
            yield None

    with patch.object(module.litellm, "acompletion", new_callable=AsyncMock) as mocked_stream:
        mocked_stream.return_value = _empty_stream()
        async for _ in service.agenerate_stream([], "p"):
            pass
    assert mocked_stream.call_args.kwargs["response_format"] == {"type": "json_object"}
    assert mocked_stream.call_args.kwargs["stream"] is True


def test_cache_key_includes_response_format():
    """task #14 契约 3: ``response_format`` 影响 LLM 输出语义, 必须进
    cache key. 同 prompt 不同 response_format 不能命中同一个 cache.
    """
    from aperag.llm.completion.completion_service import CompletionService

    plain = CompletionService("openai", "gpt-test", "https://example.invalid", "sk")
    json_mode = CompletionService(
        "openai",
        "gpt-test",
        "https://example.invalid",
        "sk",
        response_format={"type": "json_object"},
    )

    messages = [{"role": "user", "content": "hello"}]
    plain_key = plain._cache_key_data(messages=messages, stream=False)
    json_key = json_mode._cache_key_data(messages=messages, stream=False)

    assert plain_key["response_format"] is None
    assert json_key["response_format"] == {"type": "json_object"}
    assert plain_key != json_key


def test_litellm_kwargs_shape_completeness():
    """task #14 / huangheng CR NIT-B: 钉 ``_litellm_kwargs`` 返回的字典 shape
    是 LiteLLM 调用的 stable contract — 漏一个 key 会让 provider 收不到必要
    参数. 防 future refactor 静默 break.
    """
    from aperag.llm.completion.completion_service import CompletionService

    service = CompletionService(
        "openai",
        "gpt-test",
        "https://example.invalid",
        "sk",
        temperature=0.5,
        max_tokens=512,
    )
    messages = [{"role": "user", "content": "hello"}]

    # response_format=None 时 (老语义), 不含 response_format key, 但其他必备 key 都齐.
    plain = service._litellm_kwargs(messages=messages, stream=False)
    expected_keys = {
        "custom_llm_provider",
        "model",
        "base_url",
        "api_key",
        "temperature",
        "max_tokens",
        "messages",
        "stream",
        "caching",
    }
    assert set(plain.keys()) == expected_keys, f"plain shape drift, got {set(plain.keys())}"
    assert plain["custom_llm_provider"] == "openai"
    assert plain["model"] == "gpt-test"
    assert plain["base_url"] == "https://example.invalid"
    assert plain["api_key"] == "sk"
    assert plain["temperature"] == 0.5
    assert plain["max_tokens"] == 512
    assert plain["messages"] == messages
    assert plain["stream"] is False
    assert plain["caching"] is False  # service-level caching 跟 LiteLLM-level caching 是两层

    # stream=True 路径 shape 跟 non-stream 一致, 只是 stream 字段反转.
    streaming = service._litellm_kwargs(messages=messages, stream=True)
    assert set(streaming.keys()) == expected_keys
    assert streaming["stream"] is True

    # response_format 显式注入时, key 集合扩到 +1.
    service.response_format = {"type": "json_object"}
    json_mode = service._litellm_kwargs(messages=messages, stream=False)
    assert set(json_mode.keys()) == expected_keys | {"response_format"}
    assert json_mode["response_format"] == {"type": "json_object"}


def _stub_build_callable_capture(call_kwargs: dict) -> tuple:
    """fixture helper: 调用 ``build_collection_llm_callable`` 把 ``CompletionService``
    构造 kwargs 截下来, 不真发 LLM 调用. 返回 (collection, patches).
    """
    from unittest.mock import MagicMock

    fake_collection = MagicMock()
    fake_collection.id = "c-1"
    fake_collection.user = "u-1"
    fake_collection.config = '{"completion": {"model_id": "m-1"}}'

    fake_invocation = SimpleNamespace(
        runner_config={"provider": "openai"},
        runner_type="openai_compatible",
        provider_type="openai",
        provider_model_id="gpt-test",
        base_url="https://example.invalid",
        api_key="sk",
    )

    class _StubCompletionService:
        def __init__(self, **kwargs):
            call_kwargs.update(kwargs)

    return fake_collection, fake_invocation, _StubCompletionService


def test_build_collection_llm_callable_defaults_to_response_format_none():
    """task #14 契约 4 (fix-forward 修订, huangzhangshu msg=1c06455d / Weston
    msg=ef033106 grep 实证 + earayu2 msg=9fddd42d 产品确认):
    ``build_collection_llm_callable`` 是 **shared builder**, 还被 graph_curation /
    collection_regen / evaluation worker / dataset_generator / summary worker
    复用. 默认 **不能** 注入 ``response_format`` — 不然这些 prose-output 模块
    会被强制 JSON-only. 默认值必须是 None, 由 caller 显式选择.
    """
    from aperag.indexing import llm as indexing_llm

    captured: dict = {}
    fake_collection, fake_invocation, stub_svc = _stub_build_callable_capture(captured)

    with (
        patch.object(indexing_llm.db_ops, "query_model_runtime", return_value=("model_row", "account_row")),
        patch(
            "aperag.llm.runtime.resolver.resolve_model_invocation_from_records",
            return_value=fake_invocation,
        ),
        patch("aperag.llm.completion.completion_service.CompletionService", new=stub_svc),
    ):
        # 不传 response_format
        indexing_llm.build_collection_llm_callable(fake_collection)

    assert captured.get("response_format") is None, (
        f"build_collection_llm_callable 默认 response_format 必须是 None (保护 prose 输出模块), "
        f"实测 captured={captured}"
    )


def test_build_collection_llm_callable_passes_through_explicit_response_format():
    """task #14 契约 5: caller 显式传 ``response_format`` 时, builder 透传给
    ``CompletionService``. 这是 graph extractor 入口启用 JSON 强约束的机制.
    """
    from aperag.indexing import llm as indexing_llm

    captured: dict = {}
    fake_collection, fake_invocation, stub_svc = _stub_build_callable_capture(captured)

    with (
        patch.object(indexing_llm.db_ops, "query_model_runtime", return_value=("model_row", "account_row")),
        patch(
            "aperag.llm.runtime.resolver.resolve_model_invocation_from_records",
            return_value=fake_invocation,
        ),
        patch("aperag.llm.completion.completion_service.CompletionService", new=stub_svc),
    ):
        indexing_llm.build_collection_llm_callable(
            fake_collection,
            response_format={"type": "json_object"},
        )

    assert captured.get("response_format") == {"type": "json_object"}


def test_graph_extractor_builder_explicitly_opts_in_json_object_mode():
    """task #14 契约 6 (huangheng msg=363109be sediment): graph extractor
    builder 调 ``build_collection_llm_callable`` 时必须显式传
    ``response_format={"type":"json_object"}``. 这是「只对 graph index 生效」
    invariant 的具体落点 — earayu2 msg=9fddd42d 产品锁.

    通过 monkey-patch ``build_collection_llm_callable`` 截 kwarg, 验证
    ``build_collection_graph_extractor`` 真传 json_object.
    """
    from unittest.mock import MagicMock

    from aperag.indexing import graph_extractor as ext_module

    captured: dict = {}

    async def _fake_llm(_prompt: str) -> str:
        return "{}"

    def _fake_builder(collection, *, response_format=None):
        captured["response_format"] = response_format
        return _fake_llm

    fake_collection = MagicMock()
    fake_collection.id = "c-1"
    fake_collection.config = "{}"

    with (
        patch.object(ext_module, "_resolve_language", return_value="english"),
        patch.object(ext_module, "_resolve_entity_types", return_value=()),
        patch.object(ext_module, "_resolve_int_kg_config", return_value=10),
        patch.object(ext_module, "_resolve_float_kg_config", return_value=30.0),
        patch("aperag.indexing.llm.build_collection_llm_callable", _fake_builder),
    ):
        ext_module.build_collection_graph_extractor(fake_collection)

    assert captured.get("response_format") == {"type": "json_object"}, (
        f"graph extractor builder 必须显式传 json_object response_format, 实测 captured={captured}"
    )
