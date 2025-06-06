#!/usr/bin/env python3
"""
Script to generate model_configs_init.sql directly from model configuration data

This script generates PostgreSQL upsert statements to populate the aperag_llm_provider 
and aperag_llm_provider_models tables directly, without using an intermediate JSON file.

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
    """Generate upsert statement for aperag_llm_provider table"""
    
    name = escape_sql_string(provider['name'])
    label = escape_sql_string(provider['label'])
    completion_dialect = escape_sql_string(provider['completion_dialect'])
    embedding_dialect = escape_sql_string(provider['embedding_dialect'])
    rerank_dialect = escape_sql_string(provider['rerank_dialect'])
    allow_custom_base_url = format_boolean(provider['allow_custom_base_url'])
    base_url = escape_sql_string(provider['base_url'])
    
    return f"""INSERT INTO aperag_llm_provider (
    name, label, completion_dialect, embedding_dialect, rerank_dialect, 
    allow_custom_base_url, base_url, gmt_created, gmt_updated
) VALUES (
    {name}, {label}, {completion_dialect}, {embedding_dialect}, {rerank_dialect}, 
    {allow_custom_base_url}, {base_url}, NOW(), NOW()
)
ON CONFLICT (name) DO UPDATE SET
    label = EXCLUDED.label,
    completion_dialect = EXCLUDED.completion_dialect,
    embedding_dialect = EXCLUDED.embedding_dialect,
    rerank_dialect = EXCLUDED.rerank_dialect,
    allow_custom_base_url = EXCLUDED.allow_custom_base_url,
    base_url = EXCLUDED.base_url,
    gmt_updated = NOW();"""


def generate_model_upserts(provider_name: str, api_type: str, models: List[Dict[str, Any]]) -> List[str]:
    """Generate upsert statements for aperag_llm_provider_models table"""
    
    upserts = []
    for model in models:
        provider_name_sql = escape_sql_string(provider_name)
        api_sql = escape_sql_string(api_type)
        model_name_sql = escape_sql_string(model['model'])
        custom_llm_provider_sql = escape_sql_string(model['custom_llm_provider'])
        max_tokens_sql = format_nullable_int(model.get('max_tokens'))
        
        # Handle tags - for now, we'll set reasonable defaults based on provider and model
        tags = model.get('tags', [])
        if not tags:
            # Set default tags based on model characteristics
            tags = []
            model_name = model['model'].lower()
            
            # Add 'free' tag for free models
            if ':free' in model_name or provider_name in ['openrouter']:
                tags.append('free')
            
            # Add 'recommend' tag for popular models
            if any(keyword in model_name for keyword in ['gpt-4o', 'claude-3-5', 'gemini-2.5', 'qwen-max', 'deepseek-r1']):
                tags.append('recommend')
        
        tags_sql = format_json_array(tags)
        
        upsert = f"""INSERT INTO aperag_llm_provider_models (
    provider_name, api, model, custom_llm_provider, max_tokens, tags,
    gmt_created, gmt_updated
) VALUES (
    {provider_name_sql}, {api_sql}, {model_name_sql}, {custom_llm_provider_sql}, {max_tokens_sql}, {tags_sql},
    NOW(), NOW()
)
ON CONFLICT (provider_name, api, model) DO UPDATE SET
    custom_llm_provider = EXCLUDED.custom_llm_provider,
    max_tokens = EXCLUDED.max_tokens,
    tags = EXCLUDED.tags,
    gmt_updated = NOW();"""
        
        upserts.append(upsert)
    
    return upserts


# Model Configuration Functions
def generate_model_specs(models, provider, mode, enable_whitelist=False, model_whitelist=None):
    specs = []
    
    filtered_models = models
    if enable_whitelist and model_whitelist:
        filtered_models = [model for model in models if model in model_whitelist]
    
    for model in filtered_models:
        try:
            info = litellm.get_model_info(model, provider)
            spec = {
                "model": model,
                "custom_llm_provider": provider
            }

            if info.get('mode') != mode:
                continue

            if info.get('temperature'):
                spec["temperature"] = info['temperature']

            if info.get('max_tokens'):
                spec["max_tokens"] = info['max_tokens']

            if info.get('max_completion_tokens'):
                spec["max_completion_tokens"] = info['max_completion_tokens']

            if info.get('timeout'):
                spec["timeout"] = info['timeout']

            if info.get('top_n'):
                spec["top_n"] = info['top_n']

            specs.append(spec)
        except Exception as e:
            print(f"Error processing {model}: {str(e)}")
            continue
    
    # Sort by model name
    specs.sort(key=lambda x: x["model"])
    return specs


def create_openai_config(enable_whitelist=False, model_whitelist=None):
    provider = "openai"
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", enable_whitelist, model_whitelist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", enable_whitelist, model_whitelist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", enable_whitelist, model_whitelist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_anthropic_config(enable_whitelist=False, model_whitelist=None):
    provider = "anthropic"
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", enable_whitelist, model_whitelist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", enable_whitelist, model_whitelist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", enable_whitelist, model_whitelist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_deepseek_config():
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
                "custom_llm_provider": "openai"
            },
            {
                "model": "deepseek-v3",
                "custom_llm_provider": "openai"
            }
        ],
        "rerank": []
    }
    
    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])
    
    return config


def create_gemini_config(enable_whitelist=False, model_whitelist=None):
    provider = "gemini"
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", enable_whitelist, model_whitelist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", enable_whitelist, model_whitelist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", enable_whitelist, model_whitelist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_xai_config(enable_whitelist=False, model_whitelist=None):
    provider = "xai"
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", enable_whitelist, model_whitelist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", enable_whitelist, model_whitelist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", enable_whitelist, model_whitelist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_alibabacloud_config():
    config = {
        "name": "alibabacloud",
        "label": "AlibabaCloud",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding": [
            {
                "model": "text-embedding-v1",
                "custom_llm_provider": "openai"
            },
            {
                "model": "text-embedding-v2",
                "custom_llm_provider": "openai"
            },
            {
                "model": "text-embedding-v3",
                "custom_llm_provider": "openai"
            }
        ],
        "completion": [
            {
                "model": "deepseek-r1",
                "custom_llm_provider": "openai"
            },
            {
                "model": "deepseek-v3",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-max",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-long",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-plus",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-plus-latest",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-turbo",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwq-32b",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwq-plus",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwq-plus-latest",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-vl-max",
                "custom_llm_provider": "openai"
            },
            {
                "model": "qwen-vl-plus",
                "custom_llm_provider": "openai"
            }
        ],
        "rerank": []
    }

    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])

    return config


def create_siliconflow_config():
    config = {
        "name": "siliconflow",
        "label": "SiliconFlow",
        "completion_dialect": "openai",
        "embedding_dialect": "openai",
        "rerank_dialect": "jina_ai",
        "allow_custom_base_url": False,
        "base_url": "https://api.siliconflow.cn/v1",
        "embedding": [
            {
                "model": "BAAI/bge-large-en-v1.5",
                "custom_llm_provider": "openai"
            },
            {
                "model": "BAAI/bge-large-zh-v1.5",
                "custom_llm_provider": "openai"
            },
            {
                "model": "BAAI/bge-m3",
                "custom_llm_provider": "openai"
            },
            {
                "model": "Pro/BAAI/bge-m3",
                "custom_llm_provider": "openai"
            },
            {
                "model": "netease-youdao/bce-embedding-base_v1",
                "custom_llm_provider": "openai"
            }
        ],
        "completion": [
            {
                "model": "Qwen/QwQ-32B",
                "custom_llm_provider": "openai"
            },
            {
                "model": "deepseek-ai/Deepseek-R1",
                "custom_llm_provider": "openai"
            },
            {
                "model": "deepseek-ai/Deepseek-V3",
                "custom_llm_provider": "openai"
            }
        ],
        "rerank": [
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "custom_llm_provider": "jina_ai"
            },
            {
                "model": "Pro/BAAI/bge-reranker-v2-m3",
                "custom_llm_provider": "jina_ai"
            },
            {
                "model": "netease-youdao/bce-reranker-base_v1",
                "custom_llm_provider": "jina_ai"
            }
        ]
    }

    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])

    return config


def create_openrouter_config():
    try:
        # Request OpenRouter API to get model information
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={},
            proxies={"http": "http://127.0.0.1:8118", "https": "http://127.0.0.1:8118"}
        )
        
        if response.status_code != 200:
            print(f"Error fetching OpenRouter models: {response.status_code}")
            return None
        
        data = response.json()
        
        # Filter models ending with ":free"
        free_models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            context_length = model.get("context_length")
            if model_id.endswith(":free"):
                free_models.append({
                    "model": model_id,
                    "custom_llm_provider": "openrouter",
                    "max_tokens": context_length,
                })
        
        # Sort by model name
        free_models.sort(key=lambda x: x["model"])
        
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
            "completion": free_models,
            "rerank": []
        }
        
        return config
    except Exception as e:
        print(f"Error creating OpenRouter config: {str(e)}")
        return None


def create_provider_config():
    openai_whitelist = [
        # chat models
        "gpt-4", "gpt-4-turbo",
        "gpt-4o-mini", "gpt-4o",
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "o1", "o1-mini",
        "o3", "o3-mini",
        "o4-mini",
        # embedding models
        "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002", "text-embedding-ada-002-v2",
    ]
    # Sort openai whitelist
    openai_whitelist.sort()

    anthropic_whitelist = [
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
    ]
    # Sort anthropic whitelist
    anthropic_whitelist.sort()

    gemini_whitelist = [
        "gemini/gemini-2.5-pro-preview-03-25",
        "gemini/gemini-2.5-flash-preview-04-17",
    ]
    # Sort gemini whitelist
    gemini_whitelist.sort()

    xai_whitelist = [
        "xai/grok-3-beta",
        "xai/grok-3-fast-latest",
        "xai/grok-3-mini-beta",
        "xai/grok-3-mini-fast-latest",
    ]
    # Sort xai whitelist
    xai_whitelist.sort()

    enable_whitelist = True
    
    result = [
        create_openai_config(enable_whitelist, openai_whitelist),
        create_anthropic_config(enable_whitelist, anthropic_whitelist),
        create_gemini_config(enable_whitelist, gemini_whitelist),
        create_xai_config(enable_whitelist, xai_whitelist),
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
        "-- This script populates aperag_llm_provider and aperag_llm_provider_models tables",
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
    output_file = os.path.join(project_root, "aperag", "sql", "model_configs_init.sql")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sql_content)

    print(f"\nSQL script saved to: {output_file}")


def main():
    """Main function to generate model configuration and SQL script"""
    try:
        print("Generating model configuration data...")
        providers_data = create_provider_config()
        
        print("Generating SQL script...")
        sql_script = generate_sql_script(providers_data)
        
        save_sql_to_file(sql_script)
        
        print("✅ Model configuration SQL script generated successfully!")
        print("\nTo execute the script:")
        print("  psql -h <host> -U <user> -d <database> -f aperag/sql/model_configs_init.sql")
        print("\nOr copy the contents and run in your PostgreSQL client.")
        
    except Exception as e:
        print(f"❌ Error generating SQL script: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())