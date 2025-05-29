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
import logging

from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser

from .base_consumer import BaseConsumer

logger = logging.getLogger(__name__)


class FlowConsumer(BaseConsumer):
    async def connect(self):
        logging.info("FlowConsumer connect")
        await super().connect()
        collections = await self.bot.collections()
        if collections:
            self.collection = collections[0]
            self.collection_id = self.collection.id
        else:
            self.collection = None
            self.collection_id = None

        # Load flow configuration
        config = json.loads(self.bot.config)
        flow = config.get("flow")
        self.flow = FlowParser.parse(flow)

    async def predict(self, query, **kwargs):
        engine = FlowEngine()
        initial_data = {
            "query": query,
            "user": self.user,
            "history": self.history,
            "message_id": kwargs.get("message_id"),
        }

        _, system_outputs = await engine.execute_flow(self.flow, initial_data)
        if system_outputs is None:
            raise ValueError("No system outputs found")

        async_generator = None
        nodes = engine.find_end_nodes(self.flow)
        for node in nodes:
            async_generator = system_outputs[node].get("async_generator")
            if async_generator:
                break
        if not async_generator:
            raise ValueError("No generator found on the end node")
        async for chunk in async_generator():
            yield chunk
