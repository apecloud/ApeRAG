import asyncio
import json
import logging
from datetime import datetime
from http import HTTPStatus

from django.http import StreamingHttpResponse

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
    try:
        async for event in flow_generator:
            try:
                serializable_event = _convert_to_serializable(event)
                yield f"data: {json.dumps(serializable_event)}\n\n"
            except Exception as e:
                logger.exception(f"Error sending event {event}")
                raise e
            event_type = event.get("event_type")
            if event_type == "flow_end":
                break
            if event_type == "flow_error":
                raise Exception(str(event))
        try:
            _, system_outputs = await flow_task
            if not system_outputs:
                raise Exception("Flow execution failed")
            node_id = ""
            nodes = engine.find_end_nodes(flow)
            for node in nodes:
                async_generator = system_outputs[node].get("async_generator")
                if async_generator:
                    node_id = node
                    break
            if not async_generator:
                raise Exception("No generator found on the end node")
            try:
                async for chunk in async_generator():
                    data = {
                        "event_type": "output_chunk",
                        "node_id": node_id,
                        "execution_id": engine.execution_id,
                        "timestamp": datetime.now().isoformat(),
                        "data": {"chunk": _convert_to_serializable(chunk)},
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                logger.exception(f"Error streaming output from node {node_id}")
                raise e
        except Exception as e:
            logger.exception(f"Error waiting for flow execution {engine.execution_id}")
            raise e
    except asyncio.CancelledError:
        logger.info(f"Flow event stream cancelled for execution {engine.execution_id}")
    except Exception as e:
        logger.exception(f"Error in flow event stream for execution {engine.execution_id}")
        raise e


async def debug_flow_stream(user: str, bot_id: str, debug: view_models.DebugFlowRequest) -> StreamingHttpResponse:
    try:
        bot = await query_bot(user, bot_id)
        if not bot:
            return StreamingHttpResponse(json.dumps({"error": "Bot not found"}), content_type="application/json")
        flow_config = debug.flow
        if not flow_config:
            flow_config = json.loads(bot.config)["flow"]
        flow = FlowParser.parse(flow_config)
        engine = FlowEngine()
        initial_data = {"query": debug.query, "user": user}
        task = asyncio.create_task(engine.execute_flow(flow, initial_data))
        return StreamingHttpResponse(
            stream_flow_events(engine.get_events(), task, engine, flow), content_type="text/event-stream"
        )
    except Exception as e:
        logger.exception("Error in debug flow stream")
        return StreamingHttpResponse(json.dumps({"error": str(e)}), content_type="application/json")


async def get_flow(user: str, bot_id: str) -> view_models.WorkflowDefinition:
    """Get flow config for a bot"""
    bot = await query_bot(user, bot_id)
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


async def update_flow(user: str, bot_id: str, data: view_models.WorkflowDefinition):
    """Update flow config for a bot"""
    bot = await query_bot(user, bot_id)
    if not bot:
        return fail(HTTPStatus.NOT_FOUND, message="Bot not found")
    try:
        config = json.loads(bot.config or "{}")
        flow = data.model_dump(exclude_unset=True, by_alias=True)
        config["flow"] = flow
        bot.config = json.dumps(config, ensure_ascii=False)
        await bot.asave()
        return success(flow)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))
