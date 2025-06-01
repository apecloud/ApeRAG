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

# Please set the API base URL to the Aperag backend API endpoint
API_BASE_URL = "http://localhost:8000"
# Please create an API key in the Aperag UI and set it here
API_KEY = "sk-aa5229cc5e0a4fa59e075c9f1183885a"

# Please setup the related model service provider key in the Aperag UI
EMBEDDING_MODEL_PROVIDER = "siliconflow"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EMBEDDING_MODEL_CUSTOM_PROVIDER = "openai"

COMPLETION_MODEL_PROVIDER = "openrouter"
COMPLETION_MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
COMPLETION_MODEL_CUSTOM_PROVIDER = "openrouter"

RERANK_MODEL_PROVIDER = "siliconflow"
RERANK_MODEL_NAME = "BAAI/bge-large-zh-1.5"

MAX_DOCUMENT_SIZE_MB = 100
