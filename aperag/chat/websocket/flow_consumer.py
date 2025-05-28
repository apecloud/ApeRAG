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
        self.collection = (await self.bot.collections())[0]
        self.collection_id = self.collection.id

        # FIXME: get flow from the bot config when the frontend is ready
        # Load flow configuration

        config = json.loads(self.bot.config)
        flow = config.get("flow")
        self.flow = FlowParser.parse(flow)

    async def predict(self, query, **kwargs):
        engine = FlowEngine()
        initial_data = {
            "query": query,
            "bot": self.bot,
            "user": self.user,
            "history": self.history,
            "message_id": kwargs.get("message_id"),
        }
        try:
            _, system_outputs = await engine.execute_flow(self.flow, initial_data)
            logger.info("Flow executed successfully!")
        except Exception as e:
            logger.exception(e)
            raise e

        if system_outputs is None:
            raise ValueError("No output node found")

        async_generator = None
        nodes = engine.find_end_nodes(self.flow)
        for node in nodes:
            async_generator = system_outputs[node].get("async_generator")
            if async_generator:
                break
        if not async_generator:
            raise ValueError("No output node found")
        async for chunk in async_generator():
            yield chunk
