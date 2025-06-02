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
from http import HTTPStatus

from fastapi.responses import StreamingResponse

from aperag.config import SessionDep
from aperag.db.ops import query_bot
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from aperag.schema import view_models
from aperag.views.utils import fail, success

logger = logging.getLogger(__name__)


def _convert_to_serializable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return _convert_to_serializable(obj.__dict__)
    return obj


async def stream_flow_events(flow_generator, flow_task, engine, flow):
    # event stream
    async for event in flow_generator:
        serializable_event = _convert_to_serializable(event)
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
            "data": {"chunk": _convert_to_serializable(chunk)},
        }
        yield f"data: {json.dumps(data)}\n\n"


async def debug_flow_stream(session: SessionDep, user: str, bot_id: str, debug: view_models.DebugFlowRequest):
    """Stream debug flow events as SSE using FastAPI StreamingResponse."""
    try:
        bot = await query_bot(session, user, bot_id)
        if not bot:
            return {"error": "Bot not found"}
        bot_config = json.loads(bot.config)
        flow_config = bot_config.get("flow")
        if not flow_config:
            return {"error": "Bot flow config not found"}
        flow = FlowParser.parse(flow_config)
        engine = FlowEngine()
        initial_data = {"query": debug.query, "user": user}
        task = asyncio.create_task(engine.execute_flow(flow, initial_data))
        return StreamingResponse(
            stream_flow_events(engine.get_events(), task, engine, flow),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception("Error in debug flow stream: %s", e)
        return {"error": str(e)}


async def get_flow(session: SessionDep, user: str, bot_id: str) -> view_models.WorkflowDefinition:
    """Get flow config for a bot"""
    bot = await query_bot(session, user, bot_id)
    if not bot:
        return fail(HTTPStatus.NOT_FOUND, message="Bot not found")
    try:
        config = json.loads(bot.config or "{}")
        flow = config.get("flow")
        if not flow:
            return success({})
        return success(flow)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))


async def update_flow(session: SessionDep, user: str, bot_id: str, data: view_models.WorkflowDefinition):
    """Update flow config for a bot"""
    bot = await query_bot(session, user, bot_id)
    if not bot:
        return fail(HTTPStatus.NOT_FOUND, message="Bot not found")
    try:
        config = json.loads(bot.config or "{}")
        flow = data.model_dump(exclude_unset=True, by_alias=True)
        config["flow"] = flow
        bot.config = json.dumps(config, ensure_ascii=False)
        session.add(bot)
        await session.commit()
        await session.refresh(bot)
        return success(flow)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))
