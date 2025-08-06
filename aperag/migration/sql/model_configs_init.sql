-- Model configuration initialization SQL script
-- Generated directly from configuration data on 2025-08-06 12:57:51
-- This script populates llm_provider and llm_provider_models tables

BEGIN;

-- Insert/Update LLM Providers

-- Provider: openai
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'openai', 'public', 'OpenAI', 'openai', 'openai', 'jina_ai',
    FALSE, 'https://api.openai.com/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: anthropic
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'anthropic', 'public', 'Anthropic', 'anthropic', 'openai', 'jina_ai',
    FALSE, 'https://api.anthropic.com', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: gemini
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'gemini', 'public', 'Google Gemini', 'google', 'openai', 'jina_ai',
    FALSE, 'https://generativelanguage.googleapis.com', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: xai
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'xai', 'public', 'xAI', 'openai', 'openai', 'jina_ai',
    FALSE, 'https://api.xai.com/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: deepseek
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'deepseek', 'public', 'DeepSeek', 'openai', 'openai', 'jina_ai',
    FALSE, 'https://api.deepseek.com/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: alibabacloud
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'public', 'AlibabaCloud', 'openai', 'openai', 'jina_ai',
    FALSE, 'https://dashscope.aliyuncs.com/compatible-mode/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: siliconflow
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'public', 'SiliconFlow', 'openai', 'openai', 'jina_ai',
    FALSE, 'https://api.siliconflow.cn/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: jina
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'jina', 'public', 'Jina AI', 'openai', 'jina_ai', 'jina_ai',
    FALSE, 'https://api.jina.ai/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: openrouter
INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect,
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'openrouter', 'public', 'OpenRouter', 'openai', 'openai', 'jina_ai',
    FALSE, 'https://openrouter.ai/api/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Insert/Update Provider Models

-- Completion models for openai
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'chatgpt-4o-latest', 'openai', 128000, 128000, 4096, '["vision", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo', 'openai', 16385, 16385, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo-0125', 'openai', 16385, 16385, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo-0301', 'openai', 4097, 4097, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo-0613', 'openai', 4097, 4097, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo-1106', 'openai', 16385, 16385, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo-16k', 'openai', 16385, 16385, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-3.5-turbo-16k-0613', 'openai', 16385, 16385, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4', 'openai', 8192, 8192, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-0125-preview', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-0314', 'openai', 8192, 8192, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-0613', 'openai', 8192, 8192, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-1106-preview', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-1106-vision-preview', 'openai', 128000, 128000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-32k', 'openai', 32768, 32768, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-32k-0314', 'openai', 32768, 32768, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-32k-0613', 'openai', 32768, 32768, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-turbo', 'openai', 128000, 128000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-turbo-2024-04-09', 'openai', 128000, 128000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-turbo-preview', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-vision-preview', 'openai', 128000, 128000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1', 'openai', 1047576, 1047576, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-2025-04-14', 'openai', 1047576, 1047576, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-mini', 'openai', 1047576, 1047576, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-mini-2025-04-14', 'openai', 1047576, 1047576, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-nano', 'openai', 1047576, 1047576, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-nano-2025-04-14', 'openai', 1047576, 1047576, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.5-preview', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.5-preview-2025-02-27', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-2024-05-13', 'openai', 128000, 128000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-2024-08-06', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-2024-11-20', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-audio-preview', 'openai', 128000, 128000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-audio-preview-2024-10-01', 'openai', 128000, 128000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-audio-preview-2024-12-17', 'openai', 128000, 128000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-audio-preview-2025-06-03', 'openai', 128000, 128000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini', 'openai', 128000, 128000, 16384, '["vision", "enable_for_collection", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-2024-07-18', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-audio-preview', 'openai', 128000, 128000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-audio-preview-2024-12-17', 'openai', 128000, 128000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-realtime-preview', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-realtime-preview-2024-12-17', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-search-preview', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini-search-preview-2025-03-11', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-realtime-preview', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-realtime-preview-2024-10-01', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-realtime-preview-2024-12-17', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-realtime-preview-2025-06-03', 'openai', 128000, 128000, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-search-preview', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-search-preview-2025-03-11', 'openai', 128000, 128000, 16384, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1', 'openai', 200000, 200000, 100000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1-2024-12-17', 'openai', 200000, 200000, 100000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1-mini', 'openai', 128000, 128000, 65536, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1-mini-2024-09-12', 'openai', 128000, 128000, 65536, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1-preview', 'openai', 128000, 128000, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1-preview-2024-09-12', 'openai', 128000, 128000, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o3', 'openai', 200000, 200000, 100000, '["vision", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o3-2025-04-16', 'openai', 200000, 200000, 100000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o3-mini', 'openai', 200000, 200000, 100000, '["enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o3-mini-2025-01-31', 'openai', 200000, 200000, 100000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o4-mini', 'openai', 200000, 200000, 100000, '["vision", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o4-mini-2025-04-16', 'openai', 200000, 200000, 100000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for anthropic
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-5-haiku-20241022', 'anthropic', 200000, 200000, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-5-haiku-latest', 'anthropic', 200000, 200000, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-5-sonnet-20240620', 'anthropic', 200000, 200000, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-5-sonnet-20241022', 'anthropic', 200000, 200000, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-5-sonnet-latest', 'anthropic', 200000, 200000, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-7-sonnet-20250219', 'anthropic', 200000, 200000, 128000, '["vision", "enable_for_collection", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-7-sonnet-latest', 'anthropic', 200000, 200000, 128000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-haiku-20240307', 'anthropic', 200000, 200000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-opus-20240229', 'anthropic', 200000, 200000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-opus-latest', 'anthropic', 200000, 200000, 4096, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-4-opus-20250514', 'anthropic', 200000, 200000, 32000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-4-sonnet-20250514', 'anthropic', 200000, 200000, 64000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-opus-4-1-20250805', 'anthropic', 200000, 200000, 32000, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-opus-4-20250514', 'anthropic', 200000, 200000, 32000, '["vision", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-sonnet-4-20250514', 'anthropic', 200000, 200000, 64000, '["vision", "enable_for_collection", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for gemini
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-001', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-002', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-8b', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-8b-exp-0827', 'gemini', 1000000, 1000000, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-8b-exp-0924', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-exp-0827', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-flash-latest', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-pro', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-pro-001', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-pro-002', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-pro-exp-0801', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-pro-exp-0827', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-1.5-pro-latest', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-001', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-exp', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-lite-preview-02-05', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-live-001', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-preview-image-generation', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-thinking-exp', 'gemini', 1048576, 1048576, 65536, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-flash-thinking-exp-01-21', 'gemini', 1048576, 1048576, 65536, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.0-pro-exp-02-05', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash-lite', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash-lite-preview-06-17', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash-preview-04-17', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash-preview-05-20', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash-preview-tts', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro-exp-03-25', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro-preview-03-25', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro-preview-05-06', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro-preview-06-05', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro-preview-tts', 'gemini', 1048576, 1048576, 65535, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-exp-1114', 'gemini', 1048576, 1048576, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-exp-1206', 'gemini', 2097152, 2097152, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-pro', 'gemini', 32760, 32760, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-pro-vision', 'gemini', 30720, 30720, 2048, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemma-3-27b-it', 'gemini', 131072, 131072, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/learnlm-1.5-pro-experimental', 'gemini', 32767, 32767, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for xai
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-2', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-2-1212', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-2-latest', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-2-vision', 'xai', 32768, 32768, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-2-vision-1212', 'xai', 32768, 32768, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-2-vision-latest', 'xai', 32768, 32768, 32768, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3', 'xai', 131072, 131072, 131072, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-beta', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-fast-beta', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-fast-latest', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-latest', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini', 'xai', 131072, 131072, 131072, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-beta', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-fast', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-fast-beta', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-fast-latest', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-latest', 'xai', 131072, 131072, 131072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-4', 'xai', 256000, 256000, 256000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-4-0709', 'xai', 256000, 256000, 256000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-4-latest', 'xai', 256000, 256000, 256000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-beta', 'xai', 131072, 131072, 131072, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-vision-beta', 'xai', 8192, 8192, 8192, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for deepseek
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'deepseek', 'completion', 'deepseek-r1', 'openai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'deepseek', 'completion', 'deepseek-v3', 'openai', NULL, NULL, NULL, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for alibabacloud
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'abab6.5g-chat', 'openai', 8192, 8000, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'abab6.5s-chat', 'openai', 245760, 245000, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'abab6.5t-chat', 'openai', 8192, 8000, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'baichuan2-turbo', 'openai', 32000, 32000, 2048, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1-distill-llama-70b', 'openai', 32768, 32768, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1-distill-llama-8b', 'openai', 32768, 32768, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1-distill-qwen-1.5b', 'openai', 32768, 32768, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1-distill-qwen-14b', 'openai', 32768, 32768, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1-distill-qwen-32b', 'openai', 32768, 32768, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1-distill-qwen-7b', 'openai', 32768, 32768, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-v3', 'openai', 65536, 57344, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'farui-plus', 'openai', 12000, 12000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'llama-4-maverick-17b-128e-instruct', 'openai', 430000, 421808, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'llama-4-scout-17b-16e-instruct', 'openai', 1000000, 991808, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'opennlu-v1', 'openai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-coder-plus', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-coder-plus-1106', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-coder-plus-latest', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-coder-turbo', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-coder-turbo-0919', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-coder-turbo-latest', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-long', 'openai', 10000000, 10000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-long-2025-01-25', 'openai', 10000000, 10000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-long-latest', 'openai', 10000000, 10000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-plus', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-plus-0816', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-plus-0919', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-plus-latest', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-turbo', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-turbo-0919', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-math-turbo-latest', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max-0403', 'openai', 8000, 6000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max-0428', 'openai', 8000, 6000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max-0919', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max-2025-01-25', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max-latest', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-mt-plus', 'openai', 2048, 1024, 1024, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-mt-turbo', 'openai', 2048, 1024, 1024, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus', 'openai', 131072, 129024, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-0112', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-0723', 'openai', 32000, 30000, 8000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-0806', 'openai', 131072, 128000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-0919', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-1125', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-1127', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-1220', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-2025-01-25', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-2025-04-28', 'openai', 131072, 129024, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-character', 'openai', 32768, 30720, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-latest', 'openai', 131072, 129024, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo', 'openai', 1000000, 1000000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo-0624', 'openai', 8000, 6000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo-0919', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo-1101', 'openai', 1000000, 1000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo-2025-02-11', 'openai', 1000000, 1000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo-2025-04-28', 'openai', 1000000, 1000000, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo-latest', 'openai', 1000000, 1000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-0.5b-chat', 'openai', 32000, 30000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-1.8b-chat', 'openai', 32000, 30000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-110b-chat', 'openai', 32000, 30000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-14b-chat', 'openai', 8000, 6000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-32b-chat', 'openai', 32000, 30000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-72b-chat', 'openai', 32000, 30000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen1.5-7b-chat', 'openai', 8000, 6000, 2000, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2-0.5b-instruct', 'openai', 32768, 30720, 6144, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2-1.5b-instruct', 'openai', 32768, 30720, 6144, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2-57b-a14b-instruct', 'openai', 65536, 63488, 6144, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2-72b-instruct', 'openai', 131072, 128000, 6144, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2-7b-instruct', 'openai', 131072, 128000, 6144, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-0.5b-instruct', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-1.5b-instruct', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-14b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-14b-instruct-1m', 'openai', 1000000, 1000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-32b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-3b-instruct', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-72b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-7b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-7b-instruct-1m', 'openai', 1000000, 1000000, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-coder-0.5b-instruct', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-coder-1.5b-instruct', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-coder-14b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-coder-32b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-coder-3b-instruct', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-coder-7b-instruct', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-math-1.5b-instruct', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-math-72b-instruct', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen2.5-math-7b-instruct', 'openai', 4096, 3072, 3072, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-0.6b', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-1.7b', 'openai', 32768, 30720, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-14b', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-235b-a22b', 'openai', 131072, 129024, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-30b-a3b', 'openai', 131072, 129024, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-32b', 'openai', 131072, 129024, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-4b', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen3-8b', 'openai', 131072, 129024, 8192, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwq-32b-preview', 'openai', 32768, 30720, 16384, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'tongyi-intent-detect-v3', 'openai', 8192, 8192, 4096, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Embedding models for alibabacloud
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'embedding', 'multimodal-embedding-v1', 'openai', NULL, 512, NULL, '["multimodal", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'embedding', 'text-embedding-v3', 'openai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'embedding', 'text-embedding-v4', 'openai', NULL, NULL, NULL, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Rerank models for alibabacloud
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'rerank', 'gte-rerank-v2', 'alibabacloud', 30000, 30000, NULL, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for siliconflow
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'Pro/Qwen/Qwen2.5-VL-7B-Instruct', 'openai', NULL, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'Qwen/QVQ-72B-Preview', 'openai', NULL, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'Qwen/Qwen2.5-VL-32B-Instruct', 'openai', NULL, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'Qwen/Qwen2.5-VL-72B-Instruct', 'openai', NULL, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'Qwen/Qwen3-8B', 'openai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'THUDM/GLM-4.1V-9B-Thinking', 'openai', NULL, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'deepseek-ai/DeepSeek-R1', 'openai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'deepseek-ai/DeepSeek-V3', 'openai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'deepseek-ai/deepseek-vl2', 'openai', NULL, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Embedding models for siliconflow
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'embedding', 'BAAI/bge-m3', 'openai', NULL, NULL, NULL, '["free", "enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Rerank models for siliconflow
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'rerank', 'BAAI/bge-reranker-v2-m3', 'jina_ai', NULL, NULL, NULL, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'rerank', 'netease-youdao/bce-reranker-base_v1', 'jina_ai', NULL, NULL, NULL, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Embedding models for jina
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'jina', 'embedding', 'jina-embeddings-v4', 'jina_ai', NULL, NULL, NULL, '["multimodal", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Rerank models for jina
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'jina', 'rerank', 'jina-reranker-m0', 'jina_ai', NULL, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for openrouter
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'agentica-org/deepcoder-14b-preview', 'openrouter', 96000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'agentica-org/deepcoder-14b-preview:free', 'openrouter', 96000, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'ai21/jamba-1.6-large', 'openrouter', 256000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'ai21/jamba-1.6-mini', 'openrouter', 256000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'aion-labs/aion-1.0', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'aion-labs/aion-1.0-mini', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'aion-labs/aion-rp-llama-3.1-8b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'alfredpros/codellama-7b-instruct-solidity', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'alpindale/goliath-120b', 'openrouter', 6144, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'amazon/nova-lite-v1', 'openrouter', 300000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'amazon/nova-micro-v1', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'amazon/nova-pro-v1', 'openrouter', 300000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthracite-org/magnum-v2-72b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthracite-org/magnum-v4-72b', 'openrouter', 16384, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3-haiku', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3-haiku:beta', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3-opus', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3-opus:beta', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-haiku', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-haiku-20241022', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-haiku:beta', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-sonnet', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-sonnet-20240620', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-sonnet-20240620:beta', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.5-sonnet:beta', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.7-sonnet', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.7-sonnet:beta', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-3.7-sonnet:thinking', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-opus-4', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-opus-4.1', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'anthropic/claude-sonnet-4', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arcee-ai/coder-large', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arcee-ai/maestro-reasoning', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arcee-ai/spotlight', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arcee-ai/virtuoso-large', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arliai/qwq-32b-arliai-rpr-v1', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arliai/qwq-32b-arliai-rpr-v1:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'baidu/ernie-4.5-300b-a47b', 'openrouter', 123000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'bytedance/ui-tars-1.5-7b', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin-mistral-24b-venice-edition:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin-mixtral-8x22b', 'openrouter', 16000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin3.0-mistral-24b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin3.0-mistral-24b:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin3.0-r1-mistral-24b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin3.0-r1-mistral-24b:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-a', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r-03-2024', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r-08-2024', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r-plus', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r-plus-04-2024', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r-plus-08-2024', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cohere/command-r7b-12-2024', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-chat', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-chat-v3-0324', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-chat-v3-0324:free', 'openrouter', 163840, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-prover-v2', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-0528', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-0528-qwen3-8b', 'openrouter', 32000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-0528-qwen3-8b:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-0528:free', 'openrouter', 163840, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-llama-70b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-llama-70b:free', 'openrouter', 8192, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-llama-8b', 'openrouter', 32000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-1.5b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-14b', 'openrouter', 64000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-14b:free', 'openrouter', 64000, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-32b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-7b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1:free', 'openrouter', 163840, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-v3-base', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'eleutherai/llemma_7b', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'featherless/qwerky-72b:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.0-flash-001', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.0-flash-exp:free', 'openrouter', 1048576, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.0-flash-lite-001', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-flash', 'openrouter', 1048576, NULL, NULL, '["vision", "enable_for_collection", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-flash-lite', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-flash-lite-preview-06-17', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-pro', 'openrouter', 1048576, NULL, NULL, '["vision", "enable_for_agent", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-pro-exp-03-25', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-pro-preview', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.5-pro-preview-05-06', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-flash-1.5', 'openrouter', 1000000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-flash-1.5-8b', 'openrouter', 1000000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-pro-1.5', 'openrouter', 2000000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-2-27b-it', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-2-9b-it', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-2-9b-it:free', 'openrouter', 8192, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-12b-it', 'openrouter', 96000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-12b-it:free', 'openrouter', 96000, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-27b-it', 'openrouter', 96000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-27b-it:free', 'openrouter', 96000, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-4b-it', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-4b-it:free', 'openrouter', 32768, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3n-e2b-it:free', 'openrouter', 8192, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3n-e4b-it', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3n-e4b-it:free', 'openrouter', 8192, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'gryphe/mythomax-l2-13b', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'inception/mercury', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'inception/mercury-coder', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'infermatic/mn-inferor-12b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'inflection/inflection-3-pi', 'openrouter', 8000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'inflection/inflection-3-productivity', 'openrouter', 8000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'liquid/lfm-3b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'liquid/lfm-40b', 'openrouter', 65536, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'liquid/lfm-7b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mancer/weaver', 'openrouter', 8000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3-70b-instruct', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3-8b-instruct', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-405b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-405b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-405b-instruct:free', 'openrouter', 65536, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-70b-instruct', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-8b-instruct', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-11b-vision-instruct', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-11b-vision-instruct:free', 'openrouter', 131072, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-1b-instruct', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-3b-instruct', 'openrouter', 20000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-3b-instruct:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-90b-vision-instruct', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.3-70b-instruct', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.3-70b-instruct:free', 'openrouter', 65536, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-4-maverick', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-4-scout', 'openrouter', 1048576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-guard-2-8b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-guard-3-8b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-guard-4-12b', 'openrouter', 163840, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/mai-ds-r1', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/mai-ds-r1:free', 'openrouter', 163840, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-3-medium-128k-instruct', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-3-mini-128k-instruct', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-3.5-mini-128k-instruct', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-4', 'openrouter', 16384, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-4-multimodal-instruct', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-4-reasoning-plus', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/wizardlm-2-8x22b', 'openrouter', 65536, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'minimax/minimax-01', 'openrouter', 1000192, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'minimax/minimax-m1', 'openrouter', 1000000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/codestral-2501', 'openrouter', 262144, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/codestral-2508', 'openrouter', 256000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/devstral-medium', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/devstral-small', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/devstral-small-2505', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/devstral-small-2505:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/magistral-medium-2506', 'openrouter', 40960, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/magistral-medium-2506:thinking', 'openrouter', 40960, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/magistral-small-2506', 'openrouter', 40000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/ministral-3b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/ministral-8b', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-7b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-7b-instruct-v0.1', 'openrouter', 2824, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-7b-instruct-v0.2', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-7b-instruct-v0.3', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-7b-instruct:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-large', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-large-2407', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-large-2411', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-medium-3', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-nemo', 'openrouter', 32000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-nemo:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-saba', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-24b-instruct-2501', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-24b-instruct-2501:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-3.1-24b-instruct', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-3.1-24b-instruct:free', 'openrouter', 128000, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-3.2-24b-instruct', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-3.2-24b-instruct:free', 'openrouter', 131072, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-tiny', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mixtral-8x22b-instruct', 'openrouter', 65536, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mixtral-8x7b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/pixtral-12b', 'openrouter', 32768, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/pixtral-large-2411', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/kimi-dev-72b:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/kimi-k2', 'openrouter', 63000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/kimi-k2:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/kimi-vl-a3b-thinking', 'openrouter', 131072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/kimi-vl-a3b-thinking:free', 'openrouter', 131072, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'morph/morph-v3-fast', 'openrouter', 81920, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'morph/morph-v3-large', 'openrouter', 81920, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'neversleep/llama-3-lumimaid-70b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'neversleep/llama-3.1-lumimaid-8b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'neversleep/noromaid-20b', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/deephermes-3-llama-3-8b-preview:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/deephermes-3-mistral-24b-preview', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/hermes-2-pro-llama-3-8b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/hermes-3-llama-3.1-405b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/hermes-3-llama-3.1-70b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/nous-hermes-2-mixtral-8x7b-dpo', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nvidia/llama-3.1-nemotron-70b-instruct', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nvidia/llama-3.1-nemotron-ultra-253b-v1', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nvidia/llama-3.3-nemotron-super-49b-v1', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/chatgpt-4o-latest', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/codex-mini', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-3.5-turbo', 'openrouter', 16385, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-3.5-turbo-0613', 'openrouter', 4095, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-3.5-turbo-16k', 'openrouter', 16385, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-3.5-turbo-instruct', 'openrouter', 4095, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4', 'openrouter', 8191, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4-0314', 'openrouter', 8191, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4-1106-preview', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4-turbo', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4-turbo-preview', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4.1', 'openrouter', 1047576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4.1-mini', 'openrouter', 1047576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4.1-nano', 'openrouter', 1047576, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-2024-05-13', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-2024-08-06', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-2024-11-20', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-mini', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-mini-2024-07-18', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-mini-search-preview', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o-search-preview', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-4o:extended', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-oss-120b', 'openrouter', 131072, NULL, NULL, '["enable_for_collection", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/gpt-oss-20b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o1', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o1-mini', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o1-mini-2024-09-12', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o1-pro', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o3', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o3-mini', 'openrouter', 200000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o3-mini-high', 'openrouter', 200000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o3-pro', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o4-mini', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openai/o4-mini-high', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'opengvlab/internvl3-14b', 'openrouter', 12288, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openrouter/auto', 'openrouter', 2000000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'openrouter/horizon-beta', 'openrouter', 256000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'perplexity/r1-1776', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'perplexity/sonar', 'openrouter', 127072, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'perplexity/sonar-deep-research', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'perplexity/sonar-pro', 'openrouter', 200000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'perplexity/sonar-reasoning', 'openrouter', 127000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'perplexity/sonar-reasoning-pro', 'openrouter', 128000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'pygmalionai/mythalion-13b', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2-72b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-72b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-72b-instruct:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-7b-instruct', 'openrouter', 65536, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-coder-32b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-coder-32b-instruct:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-vl-7b-instruct', 'openrouter', 32768, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-max', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-plus', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-turbo', 'openrouter', 1000000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-vl-max', 'openrouter', 7500, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-vl-plus', 'openrouter', 7500, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-32b-instruct', 'openrouter', 16384, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-32b-instruct:free', 'openrouter', 8192, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-72b-instruct', 'openrouter', 32000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-72b-instruct:free', 'openrouter', 32768, NULL, NULL, '["vision", "free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-14b', 'openrouter', 40960, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-14b:free', 'openrouter', 40960, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-235b-a22b', 'openrouter', 40960, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-235b-a22b-2507', 'openrouter', 262144, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-235b-a22b-thinking-2507', 'openrouter', 262144, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-235b-a22b:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-30b-a3b', 'openrouter', 40960, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-30b-a3b-instruct-2507', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-30b-a3b:free', 'openrouter', 40960, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-32b', 'openrouter', 40960, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-4b:free', 'openrouter', 40960, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-8b', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-8b:free', 'openrouter', 40960, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-coder', 'openrouter', 262144, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwq-32b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwq-32b-preview', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwq-32b:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'raifle/sorcererlm-8x22b', 'openrouter', 16000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'rekaai/reka-flash-3:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sao10k/fimbulvetr-11b-v2', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sao10k/l3-euryale-70b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sao10k/l3-lunaris-8b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sao10k/l3.1-euryale-70b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sao10k/l3.3-euryale-70b', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sarvamai/sarvam-m:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'scb10x/llama3.1-typhoon2-70b-instruct', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'shisa-ai/shisa-v2-llama3.3-70b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'shisa-ai/shisa-v2-llama3.3-70b:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'sophosympatheia/midnight-rose-70b', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'switchpoint/router', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'tencent/hunyuan-a13b-instruct', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'tencent/hunyuan-a13b-instruct:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thedrummer/anubis-70b-v1.1', 'openrouter', 16384, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thedrummer/anubis-pro-105b-v1', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thedrummer/rocinante-12b', 'openrouter', 8192, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thedrummer/skyfall-36b-v2', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thedrummer/unslopnemo-12b', 'openrouter', 32768, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thedrummer/valkyrie-49b-v1', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-4-32b', 'openrouter', 32000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-4.1v-9b-thinking', 'openrouter', 65536, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-z1-32b:free', 'openrouter', 32768, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'tngtech/deepseek-r1t-chimera', 'openrouter', 163840, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'tngtech/deepseek-r1t-chimera:free', 'openrouter', 163840, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'tngtech/deepseek-r1t2-chimera:free', 'openrouter', 163840, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'undi95/remm-slerp-l2-13b', 'openrouter', 6144, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'undi95/toppy-m-7b', 'openrouter', 4096, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-2-1212', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-2-vision-1212', 'openrouter', 32768, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-3', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-3-beta', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-3-mini', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-3-mini-beta', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-4', 'openrouter', 256000, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'x-ai/grok-vision-beta', 'openrouter', 8192, NULL, NULL, '["vision", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'z-ai/glm-4-32b', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'z-ai/glm-4.5', 'openrouter', 131072, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'z-ai/glm-4.5-air', 'openrouter', 128000, NULL, NULL, '["__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'z-ai/glm-4.5-air:free', 'openrouter', 131072, NULL, NULL, '["free", "__autogen__"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

COMMIT;

-- Script completed. Generated on 2025-08-06 12:57:51
-- Total providers: 9
-- Total models: 574