import litellm
from litellm.caching.caching import Cache
from litellm.litellm_core_utils.model_param_helper import ModelParamHelper


def test_get_kwargs_for_cache_key():
    _cache = litellm.Cache()
    relevant_kwargs = ModelParamHelper._get_all_llm_api_params()
    print(relevant_kwargs)


def test_get_cache_key_chat_completion():
    cache = Cache()
    kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "temperature": 0.7,
    }
    cache_key_1 = cache.get_cache_key(**kwargs)
    assert isinstance(cache_key_1, str)
    assert len(cache_key_1) > 0

    kwargs_2 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_completion_tokens": 100,
    }
    cache_key_2 = cache.get_cache_key(**kwargs_2)
    assert cache_key_1 != cache_key_2

    kwargs_3 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_completion_tokens": 100,
    }
    cache_key_3 = cache.get_cache_key(**kwargs_3)
    assert cache_key_2 == cache_key_3

    kwargs_4 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_completion_tokens": 100,
        "max_tokens": 200,
    }
    cache_key_4 = cache.get_cache_key(**kwargs_4)

    kwargs_5 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_completion_tokens": 100,
        "max_tokens": None,
    }
    cache_key_5 = cache.get_cache_key(**kwargs_5)
    assert cache_key_4 != cache_key_5


def test_get_cache_key_embedding():
    cache = Cache()
    kwargs = {
        "model": "text-embedding-3-small",
        "input": "Hello, world!",
        "dimensions": 1536,
    }
    cache_key_1 = cache.get_cache_key(**kwargs)
    assert isinstance(cache_key_1, str)
    assert len(cache_key_1) > 0

    kwargs_2 = {
        "model": "text-embedding-3-small",
        "input": "Hello, world!",
        "dimensions": 1539,
    }
    cache_key_2 = cache.get_cache_key(**kwargs_2)
    assert cache_key_1 != cache_key_2

    kwargs_3 = {
        "model": "text-embedding-3-small",
        "input": "Hello, world!",
        "dimensions": 1539,
    }
    cache_key_3 = cache.get_cache_key(**kwargs_3)
    assert cache_key_2 == cache_key_3


def test_get_cache_key_text_completion():
    cache = Cache()
    kwargs = {
        "model": "gpt-3.5-turbo",
        "prompt": "Hello, world! here is a second line",
        "best_of": 3,
        "logit_bias": {"123": 1},
        "seed": 42,
    }
    cache_key_1 = cache.get_cache_key(**kwargs)
    assert isinstance(cache_key_1, str)
    assert len(cache_key_1) > 0

    kwargs_2 = {
        "model": "gpt-3.5-turbo",
        "prompt": "Hello, world! here is a second line",
        "best_of": 30,
    }
    cache_key_2 = cache.get_cache_key(**kwargs_2)
    assert cache_key_1 != cache_key_2

    kwargs_3 = {
        "model": "gpt-3.5-turbo",
        "prompt": "Hello, world! here is a second line",
        "best_of": 30,
    }
    cache_key_3 = cache.get_cache_key(**kwargs_3)
    assert cache_key_2 == cache_key_3


def test_get_hashed_cache_key():
    Cache()
    cache_key = "model:gpt-3.5-turbo,messages:Hello world"
    hashed_key = Cache._get_hashed_cache_key(cache_key)
    assert len(hashed_key) == 64  # SHA-256 produces a 64-character hex string


def test_add_namespace_to_cache_key():
    cache = Cache(namespace="test_namespace")
    hashed_key = "abcdef1234567890"

    # Test with class-level namespace
    result = cache._add_namespace_to_cache_key(hashed_key)
    assert result == "test_namespace:abcdef1234567890"

    # Test with metadata namespace
    kwargs = {"metadata": {"redis_namespace": "custom_namespace"}}
    result = cache._add_namespace_to_cache_key(hashed_key, **kwargs)
    assert result == "custom_namespace:abcdef1234567890"

    # Test with cache control namespace
    kwargs = {"cache": {"namespace": "cache_control_namespace"}}
    result = cache._add_namespace_to_cache_key(hashed_key, **kwargs)
    assert result == "cache_control_namespace:abcdef1234567890"

    kwargs = {"cache": {"namespace": "cache_control_namespace-2"}}
    result = cache._add_namespace_to_cache_key(hashed_key, **kwargs)
    assert result == "cache_control_namespace-2:abcdef1234567890"


def test_get_model_param_value():
    cache = Cache()

    # Test with regular model
    kwargs = {"model": "gpt-3.5-turbo"}
    assert cache._get_model_param_value(kwargs) == "gpt-3.5-turbo"

    # Test with model_group
    kwargs = {"model": "gpt-3.5-turbo", "metadata": {"model_group": "gpt-group"}}
    assert cache._get_model_param_value(kwargs) == "gpt-group"

    # Test with caching_group
    kwargs = {
        "model": "gpt-3.5-turbo",
        "metadata": {
            "model_group": "openai-gpt-3.5-turbo",
            "caching_groups": [("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")],
        },
    }
    assert cache._get_model_param_value(kwargs) == "('openai-gpt-3.5-turbo', 'azure-gpt-3.5-turbo')"

    kwargs = {
        "model": "gpt-3.5-turbo",
        "metadata": {
            "model_group": "azure-gpt-3.5-turbo",
            "caching_groups": [("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")],
        },
    }
    assert cache._get_model_param_value(kwargs) == "('openai-gpt-3.5-turbo', 'azure-gpt-3.5-turbo')"

    kwargs = {
        "model": "gpt-3.5-turbo",
        "metadata": {
            "model_group": "not-in-caching-group-gpt-3.5-turbo",
            "caching_groups": [("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")],
        },
    }
    assert cache._get_model_param_value(kwargs) == "not-in-caching-group-gpt-3.5-turbo"


def test_preset_cache_key():
    """
    Test that the preset cache key is used if it is set in kwargs["litellm_params"]
    """
    cache = Cache()
    kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "temperature": 0.7,
        "litellm_params": {"preset_cache_key": "preset-cache-key"},
    }

    assert cache.get_cache_key(**kwargs) == "preset-cache-key"


def test_generate_streaming_content():
    cache = Cache()
    content = "Hello, this is a test message."
    generator = cache.generate_streaming_content(content)

    full_response = ""
    chunk_count = 0

    for chunk in generator:
        chunk_count += 1
        assert "choices" in chunk
        assert len(chunk["choices"]) == 1
        assert "delta" in chunk["choices"][0]
        assert "role" in chunk["choices"][0]["delta"]
        assert chunk["choices"][0]["delta"]["role"] == "assistant"
        assert "content" in chunk["choices"][0]["delta"]

        chunk_content = chunk["choices"][0]["delta"]["content"]
        full_response += chunk_content

        # Check that each chunk is no longer than 5 characters
        assert len(chunk_content) <= 5
    print("full_response from generate_streaming_content", full_response)
    # Check that the full content is reconstructed correctly
    assert full_response == content
    # Check that there were multiple chunks
    assert chunk_count > 1

    print(f"Number of chunks: {chunk_count}")
