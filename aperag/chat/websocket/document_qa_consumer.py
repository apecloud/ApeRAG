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

from asgiref.sync import sync_to_async

from aperag.pipeline.knowledge_pipeline import create_knowledge_pipeline

from .base_consumer import BaseConsumer

logger = logging.getLogger(__name__)


class DocumentQAConsumer(BaseConsumer):
    async def connect(self):
        logging.info("DocumentQAConsumer connect")
        await super().connect()
        self.collection = await sync_to_async(self.bot.collections.first)()
        self.collection_id = self.collection.id
        self.pipeline = await create_knowledge_pipeline(bot=self.bot, collection=self.collection, history=self.history)
        self.free_tier = self.pipeline.predictor.trial

    async def predict(self, query, **kwargs):
        async for msg in self.pipeline.run(query, gen_references=True, **kwargs):
            yield msg
