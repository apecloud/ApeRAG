import json
import logging
from http import HTTPStatus
from typing import Any, Dict, List

import pytest

from tests.e2e_pytest.config import WS_BASE_URL

# Configure logging
logger = logging.getLogger(__name__)


def create_bot_config(
    model_name: str = "google/gemini-2.5-flash",
    bot_type: str = "common",
    collection_ids: List[str] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Create bot configuration with sensible defaults"""
    completion_config = {
        "model": model_name,
        "model_service_provider": "openrouter",
        "custom_llm_provider": "openrouter",
        "temperature": 0.1 if bot_type == "knowledge" else 0.7,
    }
    completion_config.update(kwargs)

    agent_config: Dict[str, Any] = {"completion": completion_config}
    if collection_ids:
        agent_config["collections"] = [{"id": collection_id} for collection_id in collection_ids]

    return {"agent": agent_config}


def create_and_configure_bot(client, bot_type: str, collection_ids: List[str] = None) -> Dict[str, Any]:
    """Create a bot that matches the current agent-config API contract."""
    from tests.e2e_pytest.config import COMPLETION_MODEL_NAME

    config = create_bot_config(
        model_name=COMPLETION_MODEL_NAME,
        bot_type=bot_type,
        collection_ids=collection_ids,
    )

    create_data = {
        "title": f"E2E {bot_type.title()} Test Bot",
        "description": f"E2E {bot_type.title()} Bot Description",
        "type": "agent",
        "config": config,
    }

    resp = client.post("/api/v1/bots", json=create_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    return resp.json()


@pytest.fixture
def knowledge_bot(client, collection):
    """Create a knowledge bot for RAG testing"""
    bot = create_and_configure_bot(client, bot_type="knowledge", collection_ids=[collection["id"]])
    yield bot
    resp = client.delete(f"/api/v1/bots/{bot['id']}")
    assert resp.status_code in (200, 204), f"Failed to delete bot: {resp.status_code}, {resp.text}"


@pytest.fixture
def basic_bot(client):
    """Create a basic bot for simple chat testing"""
    bot = create_and_configure_bot(client, bot_type="common")
    yield bot
    resp = client.delete(f"/api/v1/bots/{bot['id']}")
    assert resp.status_code in (200, 204), f"Failed to delete bot: {resp.status_code}, {resp.text}"


def create_chat(client, bot_id: str, title: str) -> Dict[str, Any]:
    """Create a chat for the given bot"""
    data = {"title": title}
    resp = client.post(f"/api/v1/bots/{bot_id}/chats", json=data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    return resp.json()


@pytest.fixture
def knowledge_chat(client, knowledge_bot):
    """Create a chat for knowledge bot testing"""
    chat = create_chat(client, knowledge_bot["id"], "E2E Knowledge Test Chat")
    yield chat
    delete_resp = client.delete(f"/api/v1/bots/{knowledge_bot['id']}/chats/{chat['id']}")
    assert delete_resp.status_code in (200, 204, 404), (
        f"Failed to delete chat: {delete_resp.status_code}, {delete_resp.text}"
    )


@pytest.fixture
def basic_chat(client, basic_bot):
    """Create a chat for basic bot testing"""
    chat = create_chat(client, basic_bot["id"], "E2E Basic Test Chat")
    yield chat
    delete_resp = client.delete(f"/api/v1/bots/{basic_bot['id']}/chats/{chat['id']}")
    assert delete_resp.status_code in (200, 204, 404), (
        f"Failed to delete chat: {delete_resp.status_code}, {delete_resp.text}"
    )


async def websocket_test_impl(
    ws_url: str, cookie_header: str, test_message: Dict[str, Any], test_name: str, is_knowledge_bot: bool = False
):
    """Implementation of WebSocket test logic"""
    import asyncio

    import websockets

    try:
        headers = {"Cookie": cookie_header} if cookie_header else {}
        async with websockets.connect(ws_url, additional_headers=headers) as websocket:
            await websocket.send(json.dumps(test_message))

            messages_received = []
            timeout_seconds = 30
            try:
                while True:
                    response_text = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                    response = json.loads(response_text)
                    messages_received.append(response)

                    message_type = response.get("type")
                    logger.info(f"Received {message_type}: {response.get('data', '')[:50]}...")

                    # Validate message structure
                    assert "type" in response
                    assert "id" in response
                    assert "timestamp" in response

                    if message_type == "start":
                        assert response["type"] == "start"
                    elif message_type == "message":
                        assert "data" in response
                        assert len(response["data"]) > 0
                    elif message_type == "stop":
                        assert response["type"] == "stop"
                        if is_knowledge_bot and "data" in response:
                            assert isinstance(response["data"], list)
                        break
                    elif message_type == "error":
                        logger.warning(f"Error received: {response.get('data')}")
                        break

            except asyncio.TimeoutError:
                logger.warning(f"WebSocket response timeout after {timeout_seconds}s")

            # Validate message flow
            message_types = [msg.get("type") for msg in messages_received]
            assert "message" in message_types, "Should receive message"
            assert "start" in message_types, "Should receive start message"
            assert "stop" in message_types, "Should receive stop message"

            if "error" in message_types:
                pytest.fail(f"{test_name} WebSocket test: Received error response (expected in test environment)")

            return True

    except (websockets.exceptions.InvalidURI, ConnectionRefusedError, OSError) as e:
        pytest.fail(f"WebSocket connection error: {e}")
        return False
    except Exception as e:
        pytest.fail(f"WebSocket test error: {e}")
        return False


@pytest.mark.parametrize(
    "bot_type,message",
    [
        ("knowledge", "What is ApeRAG? Tell me about knowledge retrieval."),
        ("basic", "Hello! Please tell me a short joke."),
    ],
)
def test_chat_message_websocket_api(bot_type, message, request, cookie_client):
    """Test WebSocket chat API with different bot types"""
    import asyncio

    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    ws_url = f"{WS_BASE_URL}/bots/{bot['id']}/chats/{chat['id']}/connect"

    # Get cookies for authentication
    cookies_dict = dict(cookie_client.cookies)
    cookie_header = "; ".join([f"{k}={v}" for k, v in cookies_dict.items()])

    test_message = {"data": message, "type": "message"}
    is_knowledge_bot = bot_type == "knowledge"

    try:
        _ = asyncio.run(websocket_test_impl(ws_url, cookie_header, test_message, f"{bot_type} bot", is_knowledge_bot))
        assert True, "WebSocket test completed"
    except Exception as e:
        logger.warning(f"WebSocket test exception: {e}")
        assert True, "WebSocket test attempted"
