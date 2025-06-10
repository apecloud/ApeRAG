-- Model configuration initialization SQL script
-- Generated from model_configs.json on 2025-06-04 23:14:20
-- This script populates llm_provider and llm_provider_models tables

BEGIN;

-- Insert/Update LLM Providers

-- Provider: openai
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'openai', 'OpenAI', 'openai', 'openai', 'jina_ai', 
    FALSE, 'https://api.openai.com/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: anthropic
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'anthropic', 'Anthropic', 'anthropic', 'openai', 'jina_ai', 
    FALSE, 'https://api.anthropic.com', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: gemini
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'gemini', 'Google Gemini', 'google', 'openai', 'jina_ai', 
    FALSE, 'https://generativelanguage.googleapis.com', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: xai
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'xai', 'xAI', 'openai', 'openai', 'jina_ai', 
    FALSE, 'https://api.xai.com/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: deepseek
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'deepseek', 'DeepSeek', 'openai', 'openai', 'jina_ai', 
    FALSE, 'https://api.deepseek.com/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: alibabacloud
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'AlibabaCloud', 'openai', 'openai', 'jina_ai', 
    FALSE, 'https://dashscope.aliyuncs.com/compatible-mode/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: siliconflow
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'SiliconFlow', 'openai', 'openai', 'jina_ai', 
    FALSE, 'https://api.siliconflow.cn/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();

-- Provider: openrouter
INSERT INTO llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    'openrouter', 'OpenRouter', 'openai', 'openai', 'jina_ai', 
    FALSE, 'https://openrouter.ai/api/v1', NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
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
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4', 'openai', 4096, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4-turbo', 'openai', 4096, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1', 'openai', 32768, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-mini', 'openai', 32768, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4.1-nano', 'openai', 32768, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o', 'openai', 16384, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'gpt-4o-mini', 'openai', 16384, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1', 'openai', 100000, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o1-mini', 'openai', 65536, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o3', 'openai', 100000, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o3-mini', 'openai', 100000, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'completion', 'o4-mini', 'openai', 100000, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Embedding models for openai
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'embedding', 'text-embedding-3-large', 'openai', 8191, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'embedding', 'text-embedding-3-small', 'openai', 8191, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'embedding', 'text-embedding-ada-002', 'openai', 8191, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openai', 'embedding', 'text-embedding-ada-002-v2', 'openai', 8191, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for anthropic
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-5-sonnet-latest', 'anthropic', 8192, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'anthropic', 'completion', 'claude-3-7-sonnet-latest', 'anthropic', 128000, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for gemini
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-flash-preview-04-17', 'gemini', 65535, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'gemini', 'completion', 'gemini/gemini-2.5-pro-preview-03-25', 'gemini', 65535, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for xai
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-beta', 'xai', 131072, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-fast-latest', 'xai', 131072, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-beta', 'xai', 131072, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'xai', 'completion', 'xai/grok-3-mini-fast-latest', 'xai', 131072, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for deepseek
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'deepseek', 'completion', 'deepseek-r1', 'openai', NULL, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'deepseek', 'completion', 'deepseek-v3', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for alibabacloud
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-r1', 'openai', NULL, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'deepseek-v3', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-long', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-max', 'openai', NULL, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-plus-latest', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-turbo', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-vl-max', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwen-vl-plus', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwq-32b', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwq-plus', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'completion', 'qwq-plus-latest', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Embedding models for alibabacloud
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'embedding', 'text-embedding-v1', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'embedding', 'text-embedding-v2', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'alibabacloud', 'embedding', 'text-embedding-v3', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for siliconflow
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'Qwen/QwQ-32B', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'deepseek-ai/Deepseek-R1', 'openai', NULL, '["recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'completion', 'deepseek-ai/Deepseek-V3', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Embedding models for siliconflow
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'embedding', 'BAAI/bge-large-en-v1.5', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'embedding', 'BAAI/bge-large-zh-v1.5', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'embedding', 'BAAI/bge-m3', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'embedding', 'Pro/BAAI/bge-m3', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'embedding', 'netease-youdao/bce-embedding-base_v1', 'openai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Rerank models for siliconflow
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'rerank', 'BAAI/bge-reranker-v2-m3', 'jina_ai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'rerank', 'Pro/BAAI/bge-reranker-v2-m3', 'jina_ai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'siliconflow', 'rerank', 'netease-youdao/bce-reranker-base_v1', 'jina_ai', NULL, '[]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

-- Completion models for openrouter
INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'agentica-org/deepcoder-14b-preview:free', 'openrouter', 96000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'allenai/molmo-7b-d:free', 'openrouter', 4096, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'arliai/qwq-32b-arliai-rpr-v1:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'bytedance-research/ui-tars-72b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin3.0-mistral-24b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'cognitivecomputations/dolphin3.0-r1-mistral-24b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-chat-v3-0324:free', 'openrouter', 163840, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-chat:free', 'openrouter', 163840, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-prover-v2:free', 'openrouter', 163840, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-llama-70b:free', 'openrouter', 8192, '["free", "recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-14b:free', 'openrouter', 64000, '["free", "recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-distill-qwen-32b:free', 'openrouter', 16000, '["free", "recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1-zero:free', 'openrouter', 163840, '["free", "recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-r1:free', 'openrouter', 163840, '["free", "recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'deepseek/deepseek-v3-base:free', 'openrouter', 163840, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'featherless/qwerky-72b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemini-2.0-flash-exp:free', 'openrouter', 1048576, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-2-9b-it:free', 'openrouter', 8192, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-12b-it:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-1b-it:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-27b-it:free', 'openrouter', 96000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/gemma-3-4b-it:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'google/learnlm-1.5-pro-experimental:free', 'openrouter', 40960, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-405b:free', 'openrouter', 64000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.1-8b-instruct:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-11b-vision-instruct:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-1b-instruct:free', 'openrouter', 131000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.2-3b-instruct:free', 'openrouter', 20000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-3.3-70b-instruct:free', 'openrouter', 8000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-4-maverick:free', 'openrouter', 256000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'meta-llama/llama-4-scout:free', 'openrouter', 512000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/mai-ds-r1:free', 'openrouter', 163840, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-4-reasoning-plus:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'microsoft/phi-4-reasoning:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-7b-instruct:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-nemo:free', 'openrouter', 128000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-24b-instruct-2501:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'mistralai/mistral-small-3.1-24b-instruct:free', 'openrouter', 96000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/kimi-vl-a3b-thinking:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'moonshotai/moonlight-16b-a3b-instruct:free', 'openrouter', 8192, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nousresearch/deephermes-3-llama-3-8b-preview:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'nvidia/llama-3.3-nemotron-super-49b-v1:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'open-r1/olympiccoder-32b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'opengvlab/internvl3-14b:free', 'openrouter', 32000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'opengvlab/internvl3-2b:free', 'openrouter', 32000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-72b-instruct:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-7b-instruct:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-coder-32b-instruct:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen-2.5-vl-7b-instruct:free', 'openrouter', 64000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-32b-instruct:free', 'openrouter', 8192, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-3b-instruct:free', 'openrouter', 64000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen2.5-vl-72b-instruct:free', 'openrouter', 131072, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-0.6b-04-28:free', 'openrouter', 32000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-1.7b:free', 'openrouter', 32000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-14b:free', 'openrouter', 40960, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-235b-a22b:free', 'openrouter', 40960, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-30b-a3b:free', 'openrouter', 40960, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-32b:free', 'openrouter', 40960, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-4b:free', 'openrouter', 128000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwen3-8b:free', 'openrouter', 40960, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwq-32b-preview:free', 'openrouter', 16384, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'qwen/qwq-32b:free', 'openrouter', 40000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'rekaai/reka-flash-3:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'shisa-ai/shisa-v2-llama3.3-70b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-4-32b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-4-9b:free', 'openrouter', 32000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-z1-32b:free', 'openrouter', 32768, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'thudm/glm-z1-9b:free', 'openrouter', 32000, '["free"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    'openrouter', 'completion', 'tngtech/deepseek-r1t-chimera:free', 'openrouter', 163840, '["free", "recommend"]'::jsonb,
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();

COMMIT;

-- Script completed. Generated on 2025-06-04 23:14:20
-- Total providers: 8
-- Total models: 122