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

import asyncio
import json
import logging
from datetime import datetime

from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import ResourceNotFoundException
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from aperag.schema import view_models

logger = logging.getLogger(__name__)


class FlowService:
    """Flow service that handles business logic for bot flows"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def _convert_to_serializable(self, obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._convert_to_serializable(obj.__dict__)
        return obj

    async def stream_flow_events(self, flow_generator, flow_task, engine, flow):
        # event stream
        async for event in flow_generator:
            serializable_event = self._convert_to_serializable(event)
            yield f"data: {json.dumps(serializable_event)}\n\n"
            event_type = event.get("event_type")
            if event_type == "flow_end":
                break
            if event_type == "flow_error":
                return

        _, system_outputs = await flow_task
        node_id = ""
        nodes = engine.find_end_nodes(flow)
        async_generator = None
        for node in nodes:
            async_generator = system_outputs[node].get("async_generator")
            if async_generator:
                node_id = node
                break
        if not async_generator:
            yield "data: {'event_type': 'flow_error', 'error': 'No generator found on the end node'}\n\n"
            return

        # llm message chunk stream
        async for chunk in async_generator():
            data = {
                "event_type": "output_chunk",
                "node_id": node_id,
                "execution_id": engine.execution_id,
                "timestamp": datetime.now().isoformat(),
                "data": {"chunk": self._convert_to_serializable(chunk)},
            }
            yield f"data: {json.dumps(data)}\n\n"

    async def debug_flow_stream(self, user: str, bot_id: str, debug: view_models.DebugFlowRequest):
        """Stream debug flow events as SSE using FastAPI StreamingResponse."""
        bot = await self.db_ops.query_bot(user, bot_id)
        if not bot:
            raise ResourceNotFoundException("Bot", bot_id)

        bot_config = json.loads(bot.config)
        flow_config = bot_config.get("flow")
        if not flow_config:
            raise ValueError("Bot flow config not found")

        flow = FlowParser.parse(flow_config)
        engine = FlowEngine()
        initial_data = {"query": debug.query, "user": user}
        task = asyncio.create_task(engine.execute_flow(flow, initial_data))

        return StreamingResponse(
            self.stream_flow_events(engine.get_events(), task, engine, flow),
            media_type="text/event-stream",
        )

    async def get_flow(self, user: str, bot_id: str) -> dict:
        """Get flow config for a bot"""
        bot = await self.db_ops.query_bot(user, bot_id)
        if not bot:
            raise ResourceNotFoundException("Bot", bot_id)

        config = json.loads(bot.config or "{}")
        flow = config.get("flow")

        # If no flow config exists, return an empty dict
        if not flow:
            return {}

        return flow

    async def update_flow(self, user: str, bot_id: str, data: view_models.WorkflowDefinition) -> dict:
        """Update flow config for a bot"""
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if not bot:
            raise ResourceNotFoundException("Bot", bot_id)

        # Direct operation without nested transaction
        config = json.loads(bot.config or "{}")
        flow = data.model_dump(exclude_unset=True, by_alias=True)
        config["flow"] = flow

        # Update only bot config to avoid overwriting concurrently updated metadata
        updated_bot = await self.db_ops.update_bot_config_by_id(
            user=user,
            bot_id=bot_id,
            config=json.dumps(config, ensure_ascii=False),
        )

        if not updated_bot:
            raise ResourceNotFoundException("Bot", bot_id)

        return flow


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
flow_service_global = FlowService()
