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
    label = escape_sql_string(provider['label'])
    completion_dialect = escape_sql_string(provider['completion_dialect'])
    embedding_dialect = escape_sql_string(provider['embedding_dialect'])
    rerank_dialect = escape_sql_string(provider['rerank_dialect'])
    allow_custom_base_url = format_boolean(provider['allow_custom_base_url'])
    base_url = escape_sql_string(provider['base_url'])
    
    return f"""INSERT INTO llm_provider (
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
    """Generate upsert statements for llm_provider_models table"""
    
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
        
        upsert = f"""INSERT INTO llm_provider_models (
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

class ModelBlockList:
    """
    Model filtering system using block lists for different API types (completion, embedding, rerank)
    """
    
    def __init__(self, 
                 completion_blocklist=None, 
                 embedding_blocklist=None, 
                 rerank_blocklist=None):
        """
        Initialize model filter with block lists for each API type
        
        Args:
            completion_blocklist: List of forbidden completion models (these models will be excluded)
            embedding_blocklist: List of forbidden embedding models (these models will be excluded)
            rerank_blocklist: List of forbidden rerank models (these models will be excluded)
        """
        self.completion_blocklist = completion_blocklist or []
        self.embedding_blocklist = embedding_blocklist or []
        self.rerank_blocklist = rerank_blocklist or []
    
    def filter_models(self, models, api_type):
        """
        Filter models based on block list for the given API type
        
        Args:
            models: List of model names to filter
            api_type: API type ("completion", "embedding", "rerank")
            
        Returns:
            Filtered list of model names (with blocked models removed)
        """
        # Select appropriate block list based on API type
        if api_type == "completion":
            blocklist = self.completion_blocklist
        elif api_type == "embedding":
            blocklist = self.embedding_blocklist
        elif api_type == "rerank":
            blocklist = self.rerank_blocklist
        else:
            return models
        
        # Apply block list filter (exclude blocked models)
        if blocklist:
            filtered = [model for model in models if model not in blocklist]
            blocked_count = len(models) - len(filtered)
            if blocked_count > 0:
                print(f"  📋 Blocked {blocked_count} {api_type} models from block list")
            return filtered
        
        return models

    def filter_model_specs(self, model_specs, api_type):
        """
        Filter model specifications based on model names
        
        Args:
            model_specs: List of model specification dicts with 'model' key
            api_type: API type ("completion", "embedding", "rerank")
            
        Returns:
            Filtered list of model specifications (with blocked models removed)
        """
        model_names = [spec['model'] for spec in model_specs]
        filtered_names = self.filter_models(model_names, api_type)
        filtered_names_set = set(filtered_names)
        
        return [spec for spec in model_specs if spec['model'] in filtered_names_set]


def generate_model_specs(models, provider, mode, model_blocklist=None):
    """
    Generate model specifications for a provider and mode, with optional block list filtering
    
    Args:
        models: List of model names
        provider: Provider name
        mode: Model mode ("chat", "embedding", "rerank")
        model_blocklist: Optional ModelBlockList instance for filtering
        
    Returns:
        List of model specifications
    """
    specs = []
    
    # Apply model block list if provided
    filtered_models = models
    if model_blocklist:
        # Map mode to API type
        api_type_map = {"chat": "completion", "embedding": "embedding", "rerank": "rerank"}
        api_type = api_type_map.get(mode, mode)
        filtered_models = model_blocklist.filter_models(models, api_type)
    
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


def create_openai_config(model_blocklist=None):
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", model_blocklist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", model_blocklist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", model_blocklist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_anthropic_config(model_blocklist=None):
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", model_blocklist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", model_blocklist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", model_blocklist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_deepseek_config(model_blocklist=None):
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
    
    # Apply model block list if provided
    if model_blocklist:
        config["completion"] = model_blocklist.filter_model_specs(config["completion"], "completion")
        config["embedding"] = model_blocklist.filter_model_specs(config["embedding"], "embedding")
        config["rerank"] = model_blocklist.filter_model_specs(config["rerank"], "rerank")
    
    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])
    
    return config


def create_gemini_config(model_blocklist=None):
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", model_blocklist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", model_blocklist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", model_blocklist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def create_xai_config(model_blocklist=None):
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
    
    completion_models = generate_model_specs(provider_models, provider, "chat", model_blocklist)
    embedding_models = generate_model_specs(provider_models, provider, "embedding", model_blocklist)
    rerank_models = generate_model_specs(provider_models, provider, "rerank", model_blocklist)

    config["completion"] = completion_models
    config["embedding"] = embedding_models
    config["rerank"] = rerank_models
    
    return config


def parse_bailian_models(file_path: str) -> List[Dict[str, Any]]:
    """Parse Alibaba Bailian models from JSON file"""
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
                "custom_llm_provider": "openai"
            }
            
            # Add context window as max_tokens if available
            context_window = model_info.get("contextWindow")
            if context_window:
                spec["max_tokens"] = context_window
            
            models.append(spec)
    
    return models


def create_alibabacloud_config(model_blocklist=None):
    # Setup file paths - now that the script is in models directory, use current directory
    models_dir = os.path.dirname(__file__)
    completion_file = os.path.join(models_dir, "alibaba_bailian_models_completion.json")
    embedding_file = os.path.join(models_dir, "alibaba_bailian_models_embedding.json")
    rerank_file = os.path.join(models_dir, "alibaba_bailian_models_rerank.json")
    
    print(f"📁 Reading Alibaba Bailian models from multiple files...")
    
    # Parse completion models
    print(f"  - Reading completion models from: {completion_file}")
    completion_models = parse_bailian_models(completion_file)
    
    # Parse embedding models
    print(f"  - Reading embedding models from: {embedding_file}")
    embedding_models = parse_bailian_models(embedding_file)
    
    # Parse rerank models
    print(f"  - Reading rerank models from: {rerank_file}")
    rerank_models = parse_bailian_models(rerank_file)
    
    # Apply model block list if provided
    if model_blocklist:
        completion_models = model_blocklist.filter_model_specs(completion_models, "completion")
        embedding_models = model_blocklist.filter_model_specs(embedding_models, "embedding")
        rerank_models = model_blocklist.filter_model_specs(rerank_models, "rerank")
    
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


def create_siliconflow_config(model_blocklist=None):
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
            #     "custom_llm_provider": "openai"
            # },
            # {
            #     "model": "BAAI/bge-large-zh-v1.5",
            #     "custom_llm_provider": "openai"
            # },
            {
                "model": "BAAI/bge-m3",
                "custom_llm_provider": "openai"
            },
            # {
            #     "model": "netease-youdao/bce-embedding-base_v1",
            #     "custom_llm_provider": "openai"
            # }
        ],
        "completion": [
            {
                "model": "Qwen/Qwen3-8B",
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
                "model": "netease-youdao/bce-reranker-base_v1",
                "custom_llm_provider": "jina_ai"
            }
        ]
    }

    # Apply model block list if provided
    if model_blocklist:
        config["completion"] = model_blocklist.filter_model_specs(config["completion"], "completion")
        config["embedding"] = model_blocklist.filter_model_specs(config["embedding"], "embedding")
        config["rerank"] = model_blocklist.filter_model_specs(config["rerank"], "rerank")

    # Sort model lists
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])

    return config


def create_openrouter_config(model_blocklist=None):
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
            context_length = model.get("context_length")
            # Include all models, not just free ones
            all_models.append({
                "model": model_id,
                "custom_llm_provider": "openrouter",
                "max_tokens": context_length,
            })
        
        # Apply model block list if provided
        if model_blocklist:
            all_models = model_blocklist.filter_model_specs(all_models, "completion")
        
        # Sort by model name
        all_models.sort(key=lambda x: x["model"])
        
        print(f"✅ Found {len(all_models)} OpenRouter models")
        
        # Only save to file if we successfully processed downloaded data
        if downloaded_data is not None and len(all_models) > 0:
            try:
                with open(openrouter_file, 'w', encoding='utf-8') as f:
                    json.dump(downloaded_data, f, indent=2, ensure_ascii=False)
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
    Create provider configuration with block list filtering
    
    To block specific models, you can add them to the appropriate block lists below.
    Example block lists (uncomment and modify as needed):
    """
    
    # Example block lists - uncomment and modify as needed
    # openai_completion_blocklist = ["gpt-3.5-turbo-instruct", "davinci-002"]
    # openai_embedding_blocklist = ["text-embedding-ada-002-v2"]
    
    # anthropic_completion_blocklist = ["claude-instant-1", "claude-instant-1.2"]
    
    # gemini_completion_blocklist = ["gemini-pro", "gemini-pro-vision"]
    
    # xai_completion_blocklist = ["grok-beta"]
    
    # deepseek_completion_blocklist = ["deepseek-coder"]
    
    # alibabacloud_completion_blocklist = ["qwen-turbo", "qwen-plus"]
    # alibabacloud_embedding_blocklist = ["text-embedding-v1"]
    
    # siliconflow_completion_blocklist = ["Qwen/Qwen2-7B-Instruct"]
    # siliconflow_embedding_blocklist = ["BAAI/bge-small-en-v1.5"]
    
    # openrouter_completion_blocklist = ["openai/gpt-3.5-turbo-instruct"]
    
    # Create ModelBlockList instances for each provider (None = no filtering)
    openai_blocklist = None
    anthropic_blocklist = None  
    gemini_blocklist = None
    xai_blocklist = None
    deepseek_blocklist = None
    alibabacloud_blocklist = ModelBlockList(
        embedding_blocklist=["multimodal-embedding-v1", "text-embedding-async-v1", "text-embedding-async-v2", "text-embedding-v1", "text-embedding-v2"]
    )
    siliconflow_blocklist = None
    openrouter_blocklist = None
    
    # Example of creating a block list (uncomment to use):
    # openai_blocklist = ModelBlockList(
    #     completion_blocklist=openai_completion_blocklist,
    #     embedding_blocklist=openai_embedding_blocklist
    # )
    
    # Generate provider configurations
    result = [
        create_openai_config(openai_blocklist),
        create_anthropic_config(anthropic_blocklist),
        create_gemini_config(gemini_blocklist),
        create_xai_config(xai_blocklist),
        create_deepseek_config(deepseek_blocklist),
        create_alibabacloud_config(alibabacloud_blocklist),
        create_siliconflow_config(siliconflow_blocklist)
    ]
    
    # Add OpenRouter configuration
    openrouter_config = create_openrouter_config(openrouter_blocklist)
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
        print("- To block specific models, modify the create_provider_config() function")
        print("- Uncomment the example block lists and add model names you want to exclude")
        print("- Each provider supports completion_blocklist, embedding_blocklist, and rerank_blocklist")
        print("- Example: ModelBlockList(completion_blocklist=['model1', 'model2'])")
        print("\n📚 For a working example, see create_provider_config_with_blocklist_example() function")
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
        print("1. To generate config with example block lists:")
        print("   Modify main() to call create_provider_config_with_blocklist_example()")
        print("2. To customize block lists:")
        print("   Edit the block list variables in create_provider_config()")
        
    except Exception as e:
        print(f"❌ Error generating SQL script: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())