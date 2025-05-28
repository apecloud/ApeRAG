import asyncio
import json
import logging
import os
import uuid
from typing import AsyncGenerator

from aperag.chat.sse.base import APIRequest
from aperag.chat.sse.openai_consumer import OpenAIFormatter
from aperag.db.ops import query_bot
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from config import settings

logger = logging.getLogger(__name__)


async def stream_openai_sse_response(generator: AsyncGenerator[str, None], formatter, msg_id: str):
    # Stream SSE response for OpenAI API format
    yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"
    async for chunk in generator:
        await asyncio.sleep(0.001)
        yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"
    yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"


async def openai_chat_completions(user, body_data, query_params):
    bot_id = query_params.get("bot_id") or query_params.get("app_id")
    if not bot_id:
        return None, OpenAIFormatter.format_error("bot_id is required")
    api_request = APIRequest(
        user=user,
        bot_id=bot_id,
        msg_id=str(uuid.uuid4()),
        stream=body_data.get("stream", False),
        messages=body_data.get("messages", []),
    )
    bot = await query_bot(api_request.user, api_request.bot_id)
    if not bot:
        return None, OpenAIFormatter.format_error("Bot not found")
    formatter = OpenAIFormatter()
    yaml_path = os.path.join(settings.BASE_DIR, "aperag/flow/examples/rag_flow2.yaml")
    flow = FlowParser.load_from_file(yaml_path)
    engine = FlowEngine()
    initial_data = {
        "query": api_request.messages[-1]["content"],
        "bot": bot,
        "user": api_request.user,
        "formatter": formatter,
        "message_id": api_request.msg_id,
    }
    try:
        _, system_outputs = await engine.execute_flow(flow, initial_data)
        logger.info("Flow executed successfully!")
    except Exception as e:
        logger.exception(e)
        return None, OpenAIFormatter.format_error(str(e))
    async_generator = None
    nodes = engine.find_end_nodes(flow)
    for node in nodes:
        async_generator = system_outputs[node].get("async_generator")
        if async_generator:
            break
    if not async_generator:
        return None, OpenAIFormatter.format_error("No output node found")
    return (api_request, formatter, async_generator), None
