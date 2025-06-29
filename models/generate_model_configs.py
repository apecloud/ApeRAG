#!/usr/bin/env python3
"""
Script to generate model_configs_init.sql directly from model configuration data

This script generates PostgreSQL upsert statements to populate the llm_provider 
and llm_provider_models tables directly, without using an intermediate JSON file.

Usage:
    python generate_model_configs.py

Output:
    SQL script with upsert statements written to ../aperag/sql/model_configs_init.sql
"""

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
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import litellm
import requests
from litellm import model_cost

# --- Manual Override for Context Windows ---
# For models where litellm's data might be ambiguous or incorrect,
# we define the correct context window size here.
MODEL_CONTEXT_OVERRIDE = {
    "gpt-4o": 128000,
    "chatgpt-4o-latest": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4o-mini": 128000,
    "o4-mini": 128000,
}


# SQL Generation Helper Functions
def escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL strings"""
    if value is None:
        return 'NULL'
    return f"'{value.replace(chr(39), chr(39) + chr(39))}'"


def format_boolean(value: bool) -> str:
    """Format boolean for PostgreSQL"""
    return 'TRUE' if value else 'FALSE'


def format_nullable_int(value: Optional[int]) -> str:
    """Format nullable integer for PostgreSQL"""
    return str(value) if value is not None else 'NULL'


def format_json_array(value: List[str]) -> str:
    """Format list as JSON array for PostgreSQL"""
    if not value:
        return "'[]'::jsonb"
    # Convert list to JSON string and escape it for SQL
    json_str = json.dumps(value)
    return f"'{json_str}'::jsonb"


def generate_provider_upsert(provider: Dict[str, Any]) -> str:
    """Generate upsert statement for llm_provider table"""
    
    name = escape_sql_string(provider['name'])
    user_id = escape_sql_string(provider.get('user_id', 'public'))  # Default to 'public' for global providers
    label = escape_sql_string(provider['label'])
    completion_dialect = escape_sql_string(provider['completion_dialect'])
    embedding_dialect = escape_sql_string(provider['embedding_dialect'])
    rerank_dialect = escape_sql_string(provider['rerank_dialect'])
    allow_custom_base_url = format_boolean(provider['allow_custom_base_url'])
    base_url = escape_sql_string(provider['base_url'])
    
    return f"""INSERT INTO llm_provider (
    name, user_id, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    {name}, {user_id}, {label}, {completion_dialect}, {embedding_dialect}, {rerank_dialect}, 
    {allow_custom_base_url}, {base_url}, NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();"""


def generate_model_upserts(provider_name: str, api_type: str, models: List[Dict[str, Any]]) -> List[str]:
    """Generate upsert statements for llm_provider_models table"""
    
    upserts = []
    for model in models:
        provider_name_sql = escape_sql_string(provider_name)
        api_sql = escape_sql_string(api_type)
        model_name_sql = escape_sql_string(model['model'])
        custom_llm_provider_sql = escape_sql_string(model['custom_llm_provider'])
        
        # Extract the three token-related fields
        context_window_sql = format_nullable_int(model.get('context_window'))
        max_input_tokens_sql = format_nullable_int(model.get('max_input_tokens'))
        max_output_tokens_sql = format_nullable_int(model.get('max_output_tokens'))
        
        # 向后兼容性处理：如果旧数据中有max_tokens字段但没有context_window，
        # 将max_tokens作为context_window使用（因为max_tokens通常表示总的上下文窗口大小）
        if 'max_tokens' in model and context_window_sql == 'NULL':
            context_window_sql = format_nullable_int(model.get('max_tokens'))
        
        # Use tags from model specification
        tags = model.get('tags', [])
        tags_sql = format_json_array(tags)
        
        upsert = f"""INSERT INTO llm_provider_models (
    provider_name, api, model, custom_llm_provider, context_window, max_input_tokens, max_output_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    {provider_name_sql}, {api_sql}, {model_name_sql}, {custom_llm_provider_sql}, {context_window_sql}, {max_input_tokens_sql}, {max_output_tokens_sql}, {tags_sql},
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    context_window = EXCLUDED.context_window,
    max_input_tokens = EXCLUDED.max_input_tokens,
    max_output_tokens = EXCLUDED.max_output_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();"""
        
        upserts.append(upsert)
    
    return upserts


# Model Configuration Functions

def filter_models_by_blocklist(models: List[str], blocklist: List[str]) -> List[str]:
    """Filter models by removing those in the blocklist"""
    if not blocklist:
        return models
    
    filtered = [model for model in models if model not in blocklist]
    blocked_count = len(models) - len(filtered)
    if blocked_count > 0:
        print(f"  📋 Blocked {blocked_count} models from blocklist")
    return filtered


def apply_model_tags(model_specs: List[Dict[str, Any]], tag_rules: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Apply tags to model specifications based on tag rules
    
    Args:
        model_specs: List of model specifications
        tag_rules: Dict mapping tag names to lists of model name patterns
    
    Returns:
        Model specifications with tags applied
    """
    if not tag_rules:
        return model_specs
    
    for spec in model_specs:
        model_name = spec['model']
        tags = spec.get('tags', [])
        
        # Apply tag rules
        for tag, model_patterns in tag_rules.items():
            for pattern in model_patterns:
                # Special handling for :free suffix (OpenRouter specific)
                if pattern == ':free' and model_name.endswith(':free'):
                    if tag not in tags:
                        tags.append(tag)
                    break
                # Wildcard pattern to match all models
                elif pattern == '*':
                    if tag not in tags:
                        tags.append(tag)
                    break
                # Exact matching for all other patterns (especially recommend tags)
                elif pattern != ':free' and pattern != '*' and pattern == model_name:
                    if tag not in tags:
                        tags.append(tag)
                    break
        
        spec['tags'] = tags
    
    return model_specs


def generate_model_specs(models, provider, mode, blocklist=None, tag_rules=None):
    """
    Generate model specifications for a provider and mode, with optional blocklist filtering and tagging
    
    Args:
        models: List of model names
        provider: Provider name
        mode: Model mode ("chat", "embedding", "rerank")
        blocklist: Optional list of model names to exclude
        tag_rules: Optional dict mapping tag names to model name patterns
        
    Returns:
        List of model specifications
    """
    specs = []
    
    # Apply blocklist filtering
    filtered_models = filter_models_by_blocklist(models, blocklist or [])
    
    for model in filtered_models:
        try:
            info = litellm.get_model_info(model, provider)
            if info.get('mode') != mode:
                continue

            # Per user request, only process models with explicitly defined input and output tokens.
            max_input = info.get('max_input_tokens')
            max_output = info.get('max_output_tokens')
            if max_input is None or max_output is None:
                continue
            
            # Also require the legacy max_tokens for the context_window calculation.
            max_tokens = info.get('max_tokens')
            if max_tokens is None:
                continue

            spec = {
                "model": model,
                "custom_llm_provider": provider,
                "max_input_tokens": max_input,
                "max_output_tokens": max_output,
                "tags": []
            }
            
            # Derive context_window based on the new heuristic.
            if max_tokens <= max_input:
                spec['context_window'] = max_input
            else:  # max_tokens > max_input
                spec['context_window'] = max_tokens
            
            # Add other general properties
            if info.get('temperature'):
                spec["temperature"] = info['temperature']
            if info.get('timeout'):
                spec["timeout"] = info['timeout']
            if info.get('top_n'):
                spec["top_n"] = info['top_n']

            specs.append(spec)
        except Exception as e:
            print(f"Error processing {model}: {str(e)}")
            continue
    
    # Apply tag rules
    specs = apply_model_tags(specs, tag_rules or {})
    
    # Sort by model name
    specs.sort(key=lambda x: x["model"])
    return specs


def create_openai_config():
    provider = "openai"
    
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['chatgpt-4o-latest', 'gpt-4o-mini', 'o3', 'o3-mini', 'o4-mini'],
        'free': []  # No free models for OpenAI
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    config = {
        "name": provider,
        "label": "OpenAI",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://api.openai.com/v1"
    }
    
    provider_models = litellm.models_by_provider.get(provider, [])
    
    completion_models = generate_model_specs(provider_models, provider, "chat", completion_blocklist, completion_tag_rules)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", embedding_blocklist, embedding_tag_rules)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", rerank_blocklist, rerank_tag_rules)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_anthropic_config():
    provider = "anthropic"
    
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['claude-3-7-sonnet-20250219', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514']
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    config = {
        "name": provider,
        "label": "Anthropic",
        "completion_dialect": "anthropic",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://api.anthropic.com"
    }
    
    provider_models = litellm.models_by_provider.get(provider, [])
    
    completion_models = generate_model_specs(provider_models, provider, "chat", completion_blocklist, completion_tag_rules)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", embedding_blocklist, embedding_tag_rules)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", rerank_blocklist, rerank_tag_rules)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_deepseek_config():
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['deepseek-r1', 'deepseek-v3']
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    config = {
        "name": "deepseek",
        "label": "DeepSeek",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://api.deepseek.com/v1",
        "embedding": [],
        "completion": [
            {
                "model": "deepseek-r1",
                "custom_llm_provider": "openai",
                "tags": []
            },
            {
                "model": "deepseek-v3",
                "custom_llm_provider": "openai",
                "tags": []
            }
        ],
        "rerank": []
    }
    
    # Apply blocklist filtering
    config["completion"] = [m for m in config["completion"] if m["model"] not in completion_blocklist]
    config["embedding"] = [m for m in config["embedding"] if m["model"] not in embedding_blocklist]
    config["rerank"] = [m for m in config["rerank"] if m["model"] not in rerank_blocklist]
    
    # Apply tag rules
    config["completion"] = apply_model_tags(config["completion"], completion_tag_rules)
    config["embedding"] = apply_model_tags(config["embedding"], embedding_tag_rules)
    config["rerank"] = apply_model_tags(config["rerank"], rerank_tag_rules)
    
    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])
    
    return config


def create_gemini_config():
    provider = "gemini"
    
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['gemini/gemini-2.0-flash', 'gemini/gemini-2.5-flash-preview-05-20', 'gemini/gemini-2.5-pro-preview-06-05']
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    config = {
        "name": provider,
        "label": "Google Gemini",
        "completion_dialect": "google",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://generativelanguage.googleapis.com"
    }
    
    provider_models = litellm.models_by_provider.get(provider, [])
    
    completion_models = generate_model_specs(provider_models, provider, "chat", completion_blocklist, completion_tag_rules)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", embedding_blocklist, embedding_tag_rules)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", rerank_blocklist, rerank_tag_rules)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_xai_config():
    provider = "xai"
    
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['xai/grok-3', 'xai/grok-3-mini']
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    config = {
        "name": provider,
        "label": "xAI",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://api.xai.com/v1"
    }
    
    provider_models = litellm.models_by_provider.get(provider, [])
    
    completion_models = generate_model_specs(provider_models, provider, "chat", completion_blocklist, completion_tag_rules)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", embedding_blocklist, embedding_tag_rules)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", rerank_blocklist, rerank_tag_rules)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def parse_bailian_models(file_path: str, default_custom_llm_provider) -> List[Dict[str, Any]]:
    """
    Parse Alibaba Bailian models from JSON file, extracting contextWindow,
    maxInputTokens, and maxOutputTokens directly.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = []
    
    # Navigate through the nested JSON structure
    model_list_path = data.get("data", {}).get("DataV2", {}).get("data", {}).get("data", {})
    model_groups = model_list_path.get("list", [])
    
    processed_models = set()
    
    for group in model_groups:
        for model_info in group.get("items", []):
            model_id = model_info.get("model")
            if not model_id or model_id in processed_models:
                continue
            
            # Only include models that support inference
            if not model_info.get("supports", {}).get("inference", False):
                continue
            
            processed_models.add(model_id)
            
            spec = {
                "model": model_id,
                "custom_llm_provider": default_custom_llm_provider
            }
            
            # Extract contextWindow, maxInputTokens, and maxOutputTokens directly from JSON data
            context_window = model_info.get("contextWindow")
            if context_window:
                spec["context_window"] = context_window

            max_input = model_info.get("maxInputTokens")
            if max_input:
                spec["max_input_tokens"] = max_input

            max_output = model_info.get("maxOutputTokens")
            if max_output:
                spec["max_output_tokens"] = max_output
            
            models.append(spec)
    
    return models


def create_alibabacloud_config():
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = ["multimodal-embedding-v1", "text-embedding-async-v1", "text-embedding-async-v2", "text-embedding-v1", "text-embedding-v2"]
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['deepseek-r1-distill-qwen-32b', 'qwen-max']
    }
    embedding_tag_rules = {
        'recommend': ['*'],  # All embedding models get recommend tag
        'default_for_embedding': ['text-embedding-v4'],
    }
    rerank_tag_rules = {
        'recommend': ['*'],  # All rerank models get recommend tag
        'default_for_rerank': ['gte-rerank-v2'],
    }
    
    # Setup file paths - now that the script is in models directory, use current directory
    models_dir = os.path.dirname(__file__)
    completion_file = os.path.join(models_dir, "alibaba_bailian_models_completion.json")
    embedding_file = os.path.join(models_dir, "alibaba_bailian_models_embedding.json")
    rerank_file = os.path.join(models_dir, "alibaba_bailian_models_rerank.json")
    
    print(f"📁 Reading Alibaba Bailian models from multiple files...")
    
    # Parse completion models
    print(f"  - Reading completion models from: {completion_file}")
    completion_models = parse_bailian_models(completion_file, "openai")
    
    # Parse embedding models
    print(f"  - Reading embedding models from: {embedding_file}")
    embedding_models = parse_bailian_models(embedding_file, "openai")
    
    # Parse rerank models
    print(f"  - Reading rerank models from: {rerank_file}")
    rerank_models = parse_bailian_models(rerank_file, "alibabacloud")
    
    # Apply blocklist filtering
    completion_models = [m for m in completion_models if m["model"] not in completion_blocklist]
    embedding_models = [m for m in embedding_models if m["model"] not in embedding_blocklist]
    rerank_models = [m for m in rerank_models if m["model"] not in rerank_blocklist]
    
    # Initialize tags for all models
    for model in completion_models + embedding_models + rerank_models:
        model['tags'] = []
    
    # Apply tag rules
    completion_models = apply_model_tags(completion_models, completion_tag_rules)
    embedding_models = apply_model_tags(embedding_models, embedding_tag_rules)
    rerank_models = apply_model_tags(rerank_models, rerank_tag_rules)
    
    # Sort model lists
    completion_models.sort(key=lambda x: x["model"])
    embedding_models.sort(key=lambda x: x["model"])
    rerank_models.sort(key=lambda x: x["model"])
    
    print(f"✅ Found {len(completion_models)} completion, {len(embedding_models)} embedding, and {len(rerank_models)} rerank models from Alibaba Bailian")
    
    config = {
        "name": "alibabacloud",
        "label": "AlibabaCloud",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding": embedding_models,
        "completion": completion_models,
        "rerank": rerank_models
    }
    
    return config


def create_siliconflow_config():
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules
    completion_tag_rules = {
        'recommend': ['deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-V3']
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    config = {
        "name": "siliconflow",
        "label": "SiliconFlow",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://api.siliconflow.cn/v1",
        "embedding": [
            # {
            #     "model": "BAAI/bge-large-en-v1.5",
            #     "custom_llm_provider": "openai",
            #     "tags": []
            # },
            # {
            #     "model": "BAAI/bge-large-zh-v1.5",
            #     "custom_llm_provider": "openai",
            #     "tags": []
            # },
            {
                "model": "BAAI/bge-m3",
                "custom_llm_provider": "openai",
                "tags": []
            },
            # {
            #     "model": "netease-youdao/bce-embedding-base_v1",
            #     "custom_llm_provider": "openai",
            #     "tags": []
            # }
        ],
        "completion": [
            {
                "model": "Qwen/Qwen3-8B",
                "custom_llm_provider": "openai",
                "tags": []
            },
            {
                "model": "deepseek-ai/DeepSeek-R1",
                "custom_llm_provider": "openai",
                "tags": []
            },
            {
                "model": "deepseek-ai/DeepSeek-V3",
                "custom_llm_provider": "openai",
                "tags": []
            }
        ],
        "rerank": [
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "custom_llm_provider": "jina_ai",
                "tags": []
            },
            {
                "model": "netease-youdao/bce-reranker-base_v1",
                "custom_llm_provider": "jina_ai",
                "tags": []
            }
        ]
    }

    # Apply blocklist filtering
    config["completion"] = [m for m in config["completion"] if m["model"] not in completion_blocklist]
    config["embedding"] = [m for m in config["embedding"] if m["model"] not in embedding_blocklist]
    config["rerank"] = [m for m in config["rerank"] if m["model"] not in rerank_blocklist]

    # Apply tag rules
    config["completion"] = apply_model_tags(config["completion"], completion_tag_rules)
    config["embedding"] = apply_model_tags(config["embedding"], embedding_tag_rules)
    config["rerank"] = apply_model_tags(config["rerank"], rerank_tag_rules)

    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])

    return config


def create_openrouter_config():
    # Define blocklists
    completion_blocklist = []
    embedding_blocklist = []
    rerank_blocklist = []
    
    # Define tag rules - OpenRouter models with :free are free
    completion_tag_rules = {
        'free': [':free'],  # Models ending with :free
        'recommend': [
            'anthropic/claude-3.7-sonnet',
            'anthropic/claude-sonnet-4',
            'deepseek/deepseek-chat-v3-0324:free',
            'deepseek/deepseek-r1:free',
            'deepseek/deepseek-r1-distill-qwen-32b:free',
            'google/gemini-2.0-flash-exp:free',
            'google/gemini-2.5-flash-lite-preview-06-17',
            'google/gemini-2.5-flash',
            'google/gemini-2.5-pro',
            'openai/gpt-4o',
            'openai/gpt-4o-mini',
            'openai/o3',
            'openai/o3-mini',
            'openai/o4-mini',
            'qwen/qwen-max',
            'qwen/qwen-plus',
            'x-ai/grok-3-beta',
            'x-ai/grok-3-mini-beta'
        ],
        'default_for_indexing': ['google/gemini-2.5-flash'],
        'default_for_generation': ['google/gemini-2.5-flash'],
    }
    embedding_tag_rules = {
        'recommend': ['*']  # All embedding models get recommend tag
    }
    rerank_tag_rules = {
        'recommend': ['*']  # All rerank models get recommend tag
    }
    
    # Setup file paths - now that the script is in models directory, use current directory
    models_dir = os.path.dirname(__file__)
    openrouter_file = os.path.join(models_dir, "openrouter_models.json")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    data = None
    downloaded_data = None
    
    # First try to download from API (but don't save yet)
    try:
        print("Downloading OpenRouter models from API...")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={},
            timeout=10  # Add timeout
        )
        
        if response.status_code == 200:
            downloaded_data = response.json()
            # Validate downloaded data before using it
            if downloaded_data and isinstance(downloaded_data, dict) and "data" in downloaded_data:
                data = downloaded_data
                print("✅ Successfully downloaded OpenRouter models from API")
            else:
                print("❌ Downloaded data is invalid or missing 'data' field")
                downloaded_data = None
        else:
            print(f"❌ Error fetching OpenRouter models: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Network error downloading OpenRouter models: {str(e)}")
    
    # If download failed, try to read from local file 
    if data is None:
        try:
            if os.path.exists(openrouter_file):
                print(f"📁 Reading OpenRouter models from local file: {openrouter_file}")
                with open(openrouter_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print("✅ Successfully loaded OpenRouter models from local file")
            else:
                print(f"❌ Local file {openrouter_file} not found")
                print("💡 Tip: You can manually download the JSON from https://openrouter.ai/api/v1/models")
                print(f"💡 and save it as {openrouter_file}")
                return None
        except Exception as e:
            print(f"❌ Error reading local OpenRouter file: {str(e)}")
            return None
    
    if data is None:
        return None
    
    try:
        # Get all OpenRouter models (not just free ones)
        all_models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            # Include all models, not just free ones
            all_models.append({
                "model": model_id,
                "custom_llm_provider": "openrouter",
                "context_window": model.get("context_length"),
                "max_input_tokens": model.get("max_input_tokens"),
                "max_output_tokens": model.get("max_output_tokens"),
                "tags": []
            })
        
        # Apply blocklist filtering
        all_models = [m for m in all_models if m["model"] not in completion_blocklist]
        
        # Apply tag rules
        all_models = apply_model_tags(all_models, completion_tag_rules)
        
        # Sort by model name
        all_models.sort(key=lambda x: x["model"])
        
        print(f"✅ Found {len(all_models)} OpenRouter models")
        
        # Only save to file if we successfully processed downloaded data
        if downloaded_data is not None and len(all_models) > 0:
            try:
                with open(openrouter_file, 'w', encoding='utf-8') as f:
                    json.dump(downloaded_data, f, indent=2, ensure_ascii=False, sort_keys=True)
                print(f"✅ OpenRouter models saved to {openrouter_file}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save downloaded models to file: {str(e)}")
                print("💡 But we can continue with the downloaded data in memory")
        
        # Create OpenRouter configuration
        config = {
            "name": "openrouter",
            "label": "OpenRouter",
            "completion_dialect": "openai",
            "embedding_dialect": "openai",
            "rerank_dialect": "jina_ai",
            "allow_custom_base_url": False,
            "base_url": "https://openrouter.ai/api/v1",
            "embedding": [],
            "completion": all_models,
            "rerank": []
        }
        
        return config
        
    except Exception as e:
        print(f"❌ Error processing OpenRouter data: {str(e)}")
        return None


def create_provider_config():
    """
    Create provider configuration with internal block list filtering and tagging
    
    Each provider function now defines its own block lists and tag rules internally.
    No need to pass parameters - block lists and tag rules are defined in each provider function.
    """
    
    print("\n📋 Block List Usage:")
    print("- Block lists and tag rules are now defined internally in each provider function")
    print("- To modify block lists or tag rules, edit the provider functions directly")
    print("- Each provider supports completion_blocklist, embedding_blocklist, and rerank_blocklist")
    print("- Tag rules map tag names to model name patterns")
    print()
    
    # Generate provider configurations
    result = [
        create_openai_config(),
        create_anthropic_config(),
        create_gemini_config(),
        create_xai_config(),
        create_deepseek_config(),
        create_alibabacloud_config(),
        create_siliconflow_config()
    ]
    
    # Add OpenRouter configuration
    openrouter_config = create_openrouter_config()
    if openrouter_config:
        result.append(openrouter_config)
    
    return result


def generate_white_list(models_by_provider, provider_list=None):
    """Generate whitelist code and output to console
    
    Args:
        models_by_provider: Dictionary of models for all providers
        provider_list: List of providers to process, if None process all
    """
    print("\n=== Generated Whitelist Code ===\n")
    
    providers_to_process = provider_list if provider_list else models_by_provider.keys()
    
    for provider in providers_to_process:
        if provider not in models_by_provider:
            print(f"Skipping unknown provider: {provider}")
            continue
            
        models = models_by_provider[provider]
        
        modes = {}
        for model in models:
            try:
                info = litellm.get_model_info(model, provider)
                mode = info.get('mode', 'unknown')
                if mode not in modes:
                    modes[mode] = []
                modes[mode].append(model)
            except Exception:
                continue
        
        if not modes:
            continue
        
        whitelist_str = f"{provider}_whitelist = ["
        
        if "chat" in modes and modes["chat"]:
            whitelist_str += "\n    # chat models"
            chat_models = sorted(modes["chat"])  # Sort chat models
            for i in range(0, len(chat_models), 4):
                chunk = chat_models[i:i+4]
                line = ", ".join([f'"{model}"' for model in chunk])
                whitelist_str += f"\n    {line},"
        
        if "embedding" in modes and modes["embedding"]:
            whitelist_str += "\n    # embedding models"
            embedding_models = sorted(modes["embedding"])  # Sort embedding models
            for i in range(0, len(embedding_models), 4):
                chunk = embedding_models[i:i+4]
                line = ", ".join([f'"{model}"' for model in chunk])
                whitelist_str += f"\n    {line},"
                
        if "rerank" in modes and modes["rerank"]:
            whitelist_str += "\n    # rerank models"
            rerank_models = sorted(modes["rerank"])  # Sort rerank models
            for i in range(0, len(rerank_models), 4):
                chunk = rerank_models[i:i+4]
                line = ", ".join([f'"{model}"' for model in chunk])
                whitelist_str += f"\n    {line},"
        
        whitelist_str += "\n]\n"
        
        print(whitelist_str)


def generate_providers_whitelist(providers=None):
    models_by_provider = litellm.models_by_provider
    generate_white_list(models_by_provider, providers)


def generate_sql_script(providers_data: List[Dict[str, Any]]) -> str:
    """Generate complete SQL script from provider configuration data"""
    
    # Generate SQL header
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sql_lines = [
        "-- Model configuration initialization SQL script",
        f"-- Generated directly from configuration data on {timestamp}",
        "-- This script populates llm_provider and llm_provider_models tables",
        "",
        "BEGIN;",
        "",
        "-- Insert/Update LLM Providers",
        ""
    ]
    
    # Generate provider upserts
    for provider in providers_data:
        sql_lines.append(f"-- Provider: {provider['name']}")
        sql_lines.append(generate_provider_upsert(provider))
        sql_lines.append("")
    
    sql_lines.extend([
        "-- Insert/Update Provider Models",
        ""
    ])
    
    # Generate model upserts
    for provider in providers_data:
        provider_name = provider['name']
        
        # Process completion models
        if 'completion' in provider and provider['completion']:
            sql_lines.append(f"-- Completion models for {provider_name}")
            for upsert in generate_model_upserts(provider_name, 'completion', provider['completion']):
                sql_lines.append(upsert)
                sql_lines.append("")
        
        # Process embedding models
        if 'embedding' in provider and provider['embedding']:
            sql_lines.append(f"-- Embedding models for {provider_name}")
            for upsert in generate_model_upserts(provider_name, 'embedding', provider['embedding']):
                sql_lines.append(upsert)
                sql_lines.append("")
        
        # Process rerank models
        if 'rerank' in provider and provider['rerank']:
            sql_lines.append(f"-- Rerank models for {provider_name}")
            for upsert in generate_model_upserts(provider_name, 'rerank', provider['rerank']):
                sql_lines.append(upsert)
                sql_lines.append("")
    
    sql_lines.extend([
        "COMMIT;",
        "",
        f"-- Script completed. Generated on {timestamp}",
        f"-- Total providers: {len(providers_data)}",
        f"-- Total models: {sum(len(p.get('completion', [])) + len(p.get('embedding', [])) + len(p.get('rerank', [])) for p in providers_data)}"
    ])
    
    return "\n".join(sql_lines)


def save_sql_to_file(sql_content: str):
    """Save SQL content to model_configs_init.sql file"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_file = os.path.join(project_root, "aperag", "migration", "sql", "model_configs_init.sql")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sql_content)

    print(f"\nSQL script saved to: {output_file}")


def main():
    """Main function to generate model configuration and SQL script"""
    try:
        print("Generating model configuration data...")
        print("\n📋 Block List Usage:")
        print("- Block lists and tag rules are now defined internally in each provider function")
        print("- To modify block lists or tag rules, edit the provider functions directly")
        print("- Each provider supports completion_blocklist, embedding_blocklist, and rerank_blocklist")
        print("- Tag rules map tag names to model name patterns")
        print()
        
        providers_data = create_provider_config()
        
        print("Generating SQL script...")
        sql_script = generate_sql_script(providers_data)
        
        save_sql_to_file(sql_script)
        
        print("✅ Model configuration SQL script generated successfully!")
        print("\nTo execute the script:")
        print("  psql -h <host> -U <user> -d <database> -f aperag/sql/model_configs_init.sql")
        print("\nOr copy the contents and run in your PostgreSQL client.")
        
        print("\n🔧 Usage Examples:")
        print("1. To customize block lists or tag rules:")
        print("   Edit the block list and tag rule variables in each provider function")
        print("2. To add/remove models from block lists:")
        print("   Modify the completion_blocklist, embedding_blocklist, or rerank_blocklist in provider functions")
        print("3. To add/modify model tags:")
        print("   Update the tag_rules dictionaries in provider functions")
        print("   Example: completion_tag_rules = {'recommend': ['model1', 'model2'], 'free': ['model3']}")
        
    except Exception as e:
        print(f"❌ Error generating SQL script: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())