import logging
import os
from django.http import HttpRequest, StreamingHttpResponse
from aperag.chat.sse.base import APIRequest, MessageProcessor
from aperag.chat.sse.openai_consumer import OpenAIFormatter
from aperag.db.ops import query_bot
from ninja import Router
import json
import uuid
import asyncio
from typing import AsyncGenerator
from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser
from aperag.utils.request import get_user
from config import settings

logger = logging.getLogger(__name__)

router = Router()


async def stream_openai_sse_response(generator: AsyncGenerator[str, None], formatter, msg_id: str):
    """Stream SSE response for OpenAI API format"""
    # Send start event with role
    yield f"data: {json.dumps(formatter.format_stream_start(msg_id))}\n\n"
    
    # Send content chunks
    async for chunk in generator:
        # FIXME: This is a hack to ensure the response is sent immediately
        # https://github.com/django/channels/issues/1761
        await asyncio.sleep(0.001)
        yield f"data: {json.dumps(formatter.format_stream_content(msg_id, chunk))}\n\n"
    
    # Send end event with finish reason
    yield f"data: {json.dumps(formatter.format_stream_end(msg_id))}\n\n"


@router.post("/chat/completions")
async def openai_chat_completions(request: HttpRequest):
    try:
        # Get user ID from request
        user = get_user(request)
        
        # Parse request parameters
        body_data = json.loads(request.body.decode("utf-8"))
        
        # Get bot_id from query parameters
        query_params = dict(request.GET.items())
        bot_id = query_params.get('bot_id') or query_params.get('app_id')
        if not bot_id:
            return StreamingHttpResponse(
                json.dumps(OpenAIFormatter.format_error("bot_id is required")),
                content_type="application/json"
            )
        
        # Create API request
        api_request = APIRequest(
            user=user,
            bot_id=bot_id,
            msg_id=str(uuid.uuid4()),
            stream=body_data.get("stream", False),
            messages=body_data.get("messages", [])
        )
        
        # Get bot
        bot = await query_bot(api_request.user, api_request.bot_id)
        if not bot:
            return StreamingHttpResponse(
                json.dumps(OpenAIFormatter.format_error("Bot not found")),
                content_type="application/json"
            )

        collections = await bot.collections()
        if len(collections) > 0:
            collection = collections[0]
        else:
            raise ValueError("No collection found for bot")
        
        # Initialize message processor
        processor = MessageProcessor(bot)
        formatter = OpenAIFormatter()

        """Test the RAG flow execution"""
        # FIXME: get flow from the bot config when the frontend is ready
        yaml_path = os.path.join(settings.BASE_DIR, 'aperag/flow/examples/rag_flow.yaml')
        
        # Load flow configuration
        flow = FlowParser.load_from_file(yaml_path)
        
        # Create execution engine
        engine = FlowEngine()
        
        # Execute flow with initial data
        initial_data = {
            "query": api_request.messages[-1]["content"],
            "bot": bot,
            "user": api_request.user,
            "formatter": formatter,
            "message_id": api_request.msg_id
        }
        
        try:
            result = await engine.execute_flow(flow, initial_data)
            logger.info("Flow executed successfully!")
        except Exception as e:
            logger.exception(e)
            raise e

        async_generator = None
        nodes = engine.find_output_nodes(flow)
        for node in nodes:
            async_generator = result[node].get("async_generator")
            if async_generator:
                break
        if not async_generator:
            raise ValueError("No output node found")
        
        # Process message and send response based on stream mode
        if api_request.stream:
            return StreamingHttpResponse(
                stream_openai_sse_response(
                    async_generator(),
                    formatter,
                    api_request.msg_id
                ),
                content_type="text/event-stream",
            )
        else:
            # Collect all content for non-streaming mode
            full_content = ""
            async for chunk in async_generator():
                full_content += chunk
            
            return StreamingHttpResponse(
                json.dumps(formatter.format_complete_response(api_request.msg_id, full_content)),
                content_type="application/json"
            )
            
    except Exception as e:
        logger.exception(e)
        return StreamingHttpResponse(
            json.dumps(OpenAIFormatter.format_error(str(e))),
            content_type="application/json"
        )


