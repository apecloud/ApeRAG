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
import os

from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from .base_consumer import BaseConsumer

logger = logging.getLogger(__name__)


class FlowConsumer(BaseConsumer):
    async def connect(self):
        logging.info("FlowConsumer connect")
        await super().connect()
        self.collection = (await self.bot.collections())[0]
        self.collection_id = self.collection.id
        from django.conf import settings

        yaml_path = os.path.join(settings.BASE_DIR, 'aperag/flow/examples/rag_flow.yaml')
        
        # Load flow configuration
        self.flow = FlowParser.load_from_file(yaml_path)
        
        # Create execution engine
        self.engine = FlowEngine()

    async def predict(self, query, **kwargs):
        initial_data = {
            "query": query,
            "bot": self.bot,
            "user": self.user,
            "collection": self.collection,
            "history": self.history,
            "message_id": kwargs.get("message_id")
        }
        try:
            result = await self.engine.execute_flow(self.flow, initial_data)
            logger.info("Flow executed successfully!")
        except Exception as e:
            logger.exception(e)
            raise e
        
        if result is None:
            raise ValueError("No output node found")

        async_generator = None
        nodes = self.engine.find_output_nodes(self.flow)
        for node in nodes:
            async_generator = result[node].get("async_generator")
            if async_generator:
                break
        if not async_generator:
            raise ValueError("No output node found")
        async for chunk in async_generator():
            yield chunk