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

import logging

from aperag.db.repositories.api_key import AsyncApiKeyRepositoryMixin
from aperag.db.repositories.base import AsyncBaseRepository, SyncBaseRepository
from aperag.db.repositories.bot import AsyncBotRepositoryMixin
from aperag.db.repositories.chat import AsyncChatRepositoryMixin
from aperag.db.repositories.collection import (
    AsyncCollectionRepositoryMixin,
    CollectionRepositoryMixin,
)
from aperag.db.repositories.document import (
    AsyncDocumentRepositoryMixin,
    DocumentRepositoryMixin,
)
from aperag.db.repositories.document_index import AsyncDocumentIndexRepositoryMixin
from aperag.db.repositories.graph import GraphRepositoryMixin
from aperag.db.repositories.lightrag import LightragRepositoryMixin
from aperag.db.repositories.llm_provider import (
    AsyncLlmProviderRepositoryMixin,
    LlmProviderRepositoryMixin,
)
from aperag.db.repositories.marketplace import AsyncMarketplaceRepositoryMixin
from aperag.db.repositories.marketplace_collection import AsyncMarketplaceCollectionRepositoryMixin
from aperag.db.repositories.merge_suggestion import MergeSuggestionRepository
from aperag.db.repositories.search import AsyncSearchRepositoryMixin
from aperag.db.repositories.setting import (
    AsyncSettingRepositoryMixin,
    SettingRepositoryMixin,
)
from aperag.db.repositories.system import AsyncSystemRepositoryMixin
from aperag.db.repositories.user import AsyncUserRepositoryMixin

logger = logging.getLogger(__name__)


class DatabaseOps(
    SyncBaseRepository,
    CollectionRepositoryMixin,
    DocumentRepositoryMixin,
    LlmProviderRepositoryMixin,
    LightragRepositoryMixin,
    GraphRepositoryMixin,
    SettingRepositoryMixin,
):
    pass


class AsyncDatabaseOps(
    AsyncBaseRepository,
    AsyncApiKeyRepositoryMixin,
    AsyncCollectionRepositoryMixin,
    AsyncDocumentRepositoryMixin,
    AsyncBotRepositoryMixin,
    AsyncChatRepositoryMixin,
    AsyncUserRepositoryMixin,
    AsyncLlmProviderRepositoryMixin,
    AsyncMarketplaceRepositoryMixin,
    AsyncMarketplaceCollectionRepositoryMixin,
    AsyncSystemRepositoryMixin,
    AsyncSearchRepositoryMixin,
    MergeSuggestionRepository,
    AsyncDocumentIndexRepositoryMixin,
    AsyncSettingRepositoryMixin,
):
    pass


async_db_ops = AsyncDatabaseOps()
db_ops = DatabaseOps()
