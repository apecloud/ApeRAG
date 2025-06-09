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

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Base URLs for API testing
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000/api/v1")

# Please specify the model service provider name and API key using environment variables
EMBEDDING_MODEL_PROVIDER = os.getenv("EMBEDDING_MODEL_PROVIDER", "siliconflow")
EMBEDDING_MODEL_PROVIDER_URL = os.getenv("EMBEDDING_MODEL_PROVIDER_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL_PROVIDER_API_KEY = os.getenv("EMBEDDING_MODEL_PROVIDER_API_KEY", "")

COMPLETION_MODEL_PROVIDER = os.getenv("COMPLETION_MODEL_PROVIDER", "openrouter")
COMPLETION_MODEL_PROVIDER_URL = os.getenv("COMPLETION_MODEL_PROVIDER_URL", "https://openrouter.ai/api/v1")
COMPLETION_MODEL_PROVIDER_API_KEY = os.getenv("COMPLETION_MODEL_PROVIDER_API_KEY", "")

RERANK_MODEL_PROVIDER = os.getenv("RERANK_MODEL_PROVIDER", "siliconflow")
RERANK_MODEL_PROVIDER_URL = os.getenv("RERANK_MODEL_PROVIDER_URL", "https://api.siliconflow.cn/v1")
RERANK_MODEL_PROVIDER_API_KEY = os.getenv("RERANK_MODEL_PROVIDER_API_KEY", "")

# The following model names are used for testing, please specify the model name using environment variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
EMBEDDING_MODEL_CUSTOM_PROVIDER = os.getenv("EMBEDDING_MODEL_CUSTOM_PROVIDER", "openai")

COMPLETION_MODEL_NAME = os.getenv("COMPLETION_MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")
COMPLETION_MODEL_CUSTOM_PROVIDER = os.getenv("COMPLETION_MODEL_CUSTOM_PROVIDER", "openrouter")

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-large-zh-1.5")


MAX_DOCUMENT_SIZE_MB = 100
