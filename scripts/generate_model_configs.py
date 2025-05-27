import json
import os

import litellm
import requests


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
    
    # 按照模型名称排序
    specs.sort(key=lambda x: x["model"])
    return specs


def create_openai_config(enable_whitelist=False, model_whitelist=None):
    provider = "openai"
    config = {
        "name": provider,
        "label": "OpenAI",
        "dialect": "openai",
        "allow_custom_base_url": True,
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
        "dialect": "anthropic",
        "allow_custom_base_url": True,
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
        "dialect": "openai",
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
    
    # 对模型列表进行排序
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])
    
    return config


def create_gemini_config(enable_whitelist=False, model_whitelist=None):
    provider = "gemini"
    config = {
        "name": provider,
        "label": "Google Gemini",
        "dialect": "google",
        "allow_custom_base_url": True,
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
        "dialect": "openai",
        "allow_custom_base_url": True,
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
            chat_models = sorted(modes["chat"])  # 对聊天模型进行排序
            for i in range(0, len(chat_models), 4):
                chunk = chat_models[i:i+4]
                line = ", ".join([f'"{model}"' for model in chunk])
                whitelist_str += f"\n    {line},"
        
        if "embedding" in modes and modes["embedding"]:
            whitelist_str += "\n    # embedding models"
            embedding_models = sorted(modes["embedding"])  # 对嵌入模型进行排序
            for i in range(0, len(embedding_models), 4):
                chunk = embedding_models[i:i+4]
                line = ", ".join([f'"{model}"' for model in chunk])
                whitelist_str += f"\n    {line},"
                
        if "rerank" in modes and modes["rerank"]:
            whitelist_str += "\n    # rerank models"
            rerank_models = sorted(modes["rerank"])  # 对重排模型进行排序
            for i in range(0, len(rerank_models), 4):
                chunk = rerank_models[i:i+4]
                line = ", ".join([f'"{model}"' for model in chunk])
                whitelist_str += f"\n    {line},"
        
        whitelist_str += "\n]\n"
        
        print(whitelist_str)


def generate_providers_whitelist(providers=None):
    models_by_provider = litellm.models_by_provider
    generate_white_list(models_by_provider, providers)


def create_openrouter_config():
    try:
        # 请求OpenRouter API获取模型信息
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={},
            proxies={"http": "http://127.0.0.1:8118", "https": "http://127.0.0.1:8118"}
        )
        
        if response.status_code != 200:
            print(f"Error fetching OpenRouter models: {response.status_code}")
            return None
        
        data = response.json()
        
        # 筛选以":free"结尾的模型
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
        
        # 按照模型名称排序
        free_models.sort(key=lambda x: x["model"])
        
        # 创建OpenRouter配置
        config = {
            "name": "openrouter",
            "label": "OpenRouter",
            "dialect": "openai",
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
    # 对openai白名单进行排序
    openai_whitelist.sort()

    anthropic_whitelist = [
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
    ]
    # 对anthropic白名单进行排序
    anthropic_whitelist.sort()

    gemini_whitelist = [
        "gemini/gemini-2.5-pro-preview-03-25",
        "gemini/gemini-2.5-flash-preview-04-17",
    ]
    # 对gemini白名单进行排序
    gemini_whitelist.sort()

    xai_whitelist = [
        "xai/grok-3-beta",
        "xai/grok-3-fast-latest",
        "xai/grok-3-mini-beta",
        "xai/grok-3-mini-fast-latest",
    ]
    # 对xai白名单进行排序
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
    
    # 添加OpenRouter配置
    openrouter_config = create_openrouter_config()
    if openrouter_config:
        result.append(openrouter_config)
    
    return result


def save_json_to_file(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_file = os.path.join(os.getcwd(), f"{project_root}/model_configs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, indent=2, ensure_ascii=False, fp=f)

    print(f"\nConfiguration saved to: {output_file}")


def create_alibabacloud_config():
    config = {
        "name": "alibabacloud",
        "label": "AlibabaCloud",
        "dialect": "openai",
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

    # 对模型列表进行排序
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])

    return config


def create_siliconflow_config():
    config = {
        "name": "siliconflow",
        "label": "SiliconFlow",
        "dialect": "openai",
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
    
    # 对模型列表进行排序
    config["completion"].sort(key=lambda x: x["model"])
    config["embedding"].sort(key=lambda x: x["model"])
    config["rerank"].sort(key=lambda x: x["model"])
    
    return config


if __name__ == "__main__":
    generate_providers_whitelist(["openai", "anthropic", "gemini", "xai"])
    
    config = create_provider_config()

    save_json_to_file(config)