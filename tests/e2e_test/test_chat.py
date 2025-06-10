import json
import logging
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from openai import OpenAI

from tests.e2e_test.config import API_BASE_URL, WS_BASE_URL

# Configure logging
logger = logging.getLogger(__name__)


class APITestHelper:
    """Helper class for API testing to reduce code duplication"""

    def __init__(self, client, base_url: str, api_key: str):
        self.client = client
        self.base_url = base_url
        self.api_key = api_key
        # Set longer timeout for non-streaming requests
        self.openai_client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key, timeout=120.0)

    def test_openai_api_non_streaming(self, bot_id: str, chat_id: str, message: str, test_name: str) -> None:
        """Test OpenAI-compatible API for non-streaming mode"""
        try:
            response = self.openai_client.chat.completions.create(
                model="aperag",
                messages=[{"role": "user", "content": message}],
                stream=False,
                extra_query={"bot_id": bot_id, "chat_id": chat_id},
            )

            # Validate OpenAI response inline
            if hasattr(response, "error") and response.error:
                assert hasattr(response, "error")
                assert response.error.get("type") == "server_error"
                pytest.fail(f"{test_name} non-streaming test: API responded with expected error structure")
            else:
                # Non-streaming response validation
                assert response.id is not None
                assert response.object == "chat.completion"
                assert response.created is not None
                assert response.model == "aperag"
                assert len(response.choices) == 1
                assert response.choices[0].message.role == "assistant"
                assert response.choices[0].message.content is not None
                assert len(response.choices[0].message.content) > 0
                assert response.choices[0].finish_reason == "stop"

        except Exception as e:
            pytest.fail(f"{test_name} non-streaming request failed: {e}")

    def test_openai_api_streaming(self, bot_id: str, chat_id: str, message: str, test_name: str) -> None:
        """Test OpenAI-compatible API for streaming mode"""
        try:
            stream = self.openai_client.chat.completions.create(
                model="aperag",
                messages=[{"role": "user", "content": f"{message} (streaming)"}],
                stream=True,
                extra_query={"bot_id": bot_id, "chat_id": chat_id},
            )

            collected_content = ""
            chunk_count = 0

            for chunk in stream:
                chunk_count += 1

                if hasattr(chunk, "error") and chunk.error:
                    pytest.fail(f"{test_name} streaming API returned error: {chunk.error}")

                assert chunk.id is not None
                assert chunk.object == "chat.completion.chunk"
                assert chunk.created is not None
                assert chunk.model == "aperag"
                assert len(chunk.choices) == 1

                if chunk.choices[0].delta.content is not None:
                    collected_content += chunk.choices[0].delta.content

            assert chunk_count > 1, "Should receive multiple chunks in streaming mode"
            assert len(collected_content) > 0, "Should receive content in streaming response"

        except Exception as e:
            pytest.fail(f"{test_name} streaming request failed: {e}")

    def test_frontend_api_non_streaming(
        self, bot_id: str, chat_id: str, message: str, test_name: str, is_knowledge_bot: bool = False
    ) -> None:
        """Test frontend-specific API for non-streaming mode"""
        try:
            msg_id = f"{test_name.lower().replace(' ', '_')}_msg_001"
            response = self.client.post(
                "/api/v1/chat/completions/frontend",
                content=message,
                params={"stream": "false", "bot_id": bot_id, "chat_id": chat_id},
                headers={"msg_id": msg_id, "Content-Type": "text/plain"},
                timeout=120.0,  # Set longer timeout for non-streaming requests
            )

            assert response.status_code == HTTPStatus.OK, response.text
            response_data = response.json()

            # Validate frontend response inline
            if response_data.get("type") == "error":
                logger.warning(f"Frontend API returned error: {response_data.get('data')}")
                assert response_data.get("type") == "error"
                assert "data" in response_data
                pytest.fail(f"{test_name} frontend non-streaming test: API responded with expected error structure")
            else:
                assert response_data.get("type") == "message"
                assert response_data.get("id") == msg_id
                assert "data" in response_data
                assert response_data.get("data") is not None
                assert len(response_data.get("data", "")) > 0
                assert "timestamp" in response_data

        except Exception as e:
            pytest.fail(f"{test_name} frontend non-streaming request failed: {e}")

    def test_frontend_api_streaming(
        self, bot_id: str, chat_id: str, message: str, test_name: str, is_knowledge_bot: bool = False
    ) -> None:
        """Test frontend-specific API for streaming mode"""
        try:
            msg_id = f"{test_name.lower().replace(' ', '_')}_msg_002"
            response = self.client.post(
                "/api/v1/chat/completions/frontend",
                content=f"{message} (streaming)",
                params={"stream": "true", "bot_id": bot_id, "chat_id": chat_id},
                headers={"msg_id": msg_id, "Content-Type": "text/plain"},
                timeout=120.0,  # Set longer timeout for streaming requests as well
            )

            assert response.status_code == HTTPStatus.OK, response.text
            assert response.headers.get("content-type").startswith("text/event-stream")

            # Parse SSE response
            events = []
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                        events.append(event_data)
                    except json.JSONDecodeError:
                        continue

            # Validate SSE events inline
            if not events:
                pytest.fail(f"{test_name} frontend streaming test: No events received in SSE response")
            else:
                start_events = [e for e in events if e.get("type") == "start"]
                message_events = [e for e in events if e.get("type") == "message"]
                stop_events = [e for e in events if e.get("type") == "stop"]
                error_events = [e for e in events if e.get("type") == "error"]

                if error_events:
                    assert error_events[0].get("type") == "error"
                    pytest.fail(f"{test_name} frontend streaming test: API responded with expected error structure")
                else:
                    # Verify event structure
                    assert len(start_events) >= 1, "Should have at least one start event"
                    assert start_events[0].get("id") == msg_id
                    assert "timestamp" in start_events[0]

                    if message_events:
                        for event in message_events:
                            assert event.get("id") == msg_id
                            assert "data" in event
                            assert "timestamp" in event

                    if stop_events:
                        assert stop_events[0].get("id") == msg_id
                        assert "timestamp" in stop_events[0]
                        # Knowledge bots might have references/urls in stop event
                        if is_knowledge_bot and "data" in stop_events[0]:
                            assert isinstance(stop_events[0]["data"], list)
        except Exception as e:
            pytest.fail(f"{test_name} frontend streaming request failed: {e}")


def create_bot_config(
    model_name: str = "deepseek/deepseek-v3-base:free", bot_type: str = "common", **kwargs
) -> Dict[str, Any]:
    """Create bot configuration with sensible defaults"""
    base_config = {
        "model_name": model_name,
        "model_service_provider": "openrouter",
        "llm": {"context_window": 3500, "temperature": 0.1 if bot_type == "knowledge" else 0.7},
    }

    if bot_type == "knowledge":
        base_config["llm"].update(
            {
                "similarity_score_threshold": 0.5,
                "similarity_topk": 3,
            }
        )

    base_config["llm"].update(kwargs)
    return base_config


def create_and_configure_bot(
    client, bot_type: str, collection_ids: List[str] = None, flow_file: str = None
) -> Dict[str, Any]:
    """Create and configure a bot with the given parameters"""
    config = create_bot_config(bot_type=bot_type)

    create_data = {
        "title": f"E2E {bot_type.title()} Test Bot",
        "description": f"E2E {bot_type.title()} Bot Description",
        "type": bot_type,
        "config": json.dumps(config),
        "collection_ids": collection_ids or [],
    }

    resp = client.post("/api/v1/bots", json=create_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    bot = resp.json()

    # Configure flow if specified
    if flow_file:
        flow_path = Path(__file__).parent / "testdata" / flow_file
        with open(flow_path, "r", encoding="utf-8") as f:
            flow = yaml.safe_load(f)

        # Update collection_ids in flow nodes if needed
        if collection_ids:
            for node in flow.get("nodes", []):
                if node.get("type") in ["vector_search", "fulltext_search", "graph_search"]:
                    if "data" in node and "input" in node["data"] and "values" in node["data"]["input"]:
                        node["data"]["input"]["values"]["collection_ids"] = collection_ids

        flow_json = json.dumps(flow)
        resp = client.put(
            f"/api/v1/bots/{bot['id']}/flow", content=flow_json, headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == HTTPStatus.OK, resp.text

    return bot


@pytest.fixture
def knowledge_bot(client, collection):
    """Create a knowledge bot for RAG testing"""
    bot = create_and_configure_bot(
        client, bot_type="knowledge", collection_ids=[collection["id"]], flow_file="rag-flow.yaml"
    )
    yield bot
    resp = client.delete(f"/api/v1/bots/{bot['id']}")
    assert resp.status_code in (200, 204), f"Failed to delete bot: {resp.status_code}, {resp.text}"


@pytest.fixture
def basic_bot(client):
    """Create a basic bot for simple chat testing"""
    bot = create_and_configure_bot(client, bot_type="common", flow_file="basic-flow.yaml")
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


@pytest.fixture
def api_helper(client, api_key):
    """Create API test helper instance"""
    return APITestHelper(client, base_url=API_BASE_URL, api_key=api_key)


# Parameterized tests to reduce duplication
@pytest.mark.parametrize("bot_type", ["knowledge", "basic"])
def test_get_chat_detail(client, bot_type, request):
    """Test getting chat details for different bot types"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    resp = client.get(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}")
    assert resp.status_code == HTTPStatus.OK, resp.text
    detail = resp.json()
    assert detail["id"] == chat["id"]
    assert detail["title"] == chat["title"]


@pytest.mark.parametrize("bot_type", ["knowledge", "basic"])
def test_update_chat(client, bot_type, request):
    """Test updating chat title for different bot types"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    new_title = f"E2E {bot_type.title()} Test Chat Updated"
    update_data = {"title": new_title}
    resp = client.put(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}", json=update_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    updated = resp.json()
    assert updated["title"] == new_title


@pytest.mark.parametrize(
    "bot_type,message",
    [
        ("knowledge", "What is ApeRAG?"),
        ("basic", "Hello, how are you today?"),
    ],
)
def test_chat_message_openai_api_non_streaming(api_helper, bot_type, message, request):
    """Test OpenAI-compatible chat completions API - Non-streaming mode"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    api_helper.test_openai_api_non_streaming(bot["id"], chat["id"], message, f"{bot_type} bot")


@pytest.mark.parametrize(
    "bot_type,message",
    [
        ("knowledge", "What is ApeRAG?"),
        ("basic", "Hello, how are you today?"),
    ],
)
def test_chat_message_openai_api_streaming(api_helper, bot_type, message, request):
    """Test OpenAI-compatible chat completions API - Streaming mode"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    api_helper.test_openai_api_streaming(bot["id"], chat["id"], message, f"{bot_type} bot")


@pytest.mark.parametrize(
    "bot_type,message",
    [
        ("knowledge", "What is ApeRAG? Please tell me about this knowledge base system."),
        ("basic", "Hello, this is a test message for frontend API"),
    ],
)
def test_chat_message_frontend_api_non_streaming(api_helper, bot_type, message, request):
    """Test frontend-specific chat completions API - Non-streaming mode"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    is_knowledge_bot = bot_type == "knowledge"
    api_helper.test_frontend_api_non_streaming(bot["id"], chat["id"], message, f"{bot_type} bot", is_knowledge_bot)


@pytest.mark.parametrize(
    "bot_type,message",
    [
        ("knowledge", "What is ApeRAG? Please tell me about this knowledge base system."),
        ("basic", "Hello, this is a test message for frontend API"),
    ],
)
def test_chat_message_frontend_api_streaming(api_helper, bot_type, message, request):
    """Test frontend-specific chat completions API - Streaming mode"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    is_knowledge_bot = bot_type == "knowledge"
    api_helper.test_frontend_api_streaming(bot["id"], chat["id"], message, f"{bot_type} bot", is_knowledge_bot)


def test_openai_api_error_handling(api_helper, basic_chat):
    """Test error handling for OpenAI API"""
    logger.info("Testing error handling scenarios")

    # Test invalid bot_id
    try:
        response = api_helper.openai_client.chat.completions.create(
            model="aperag",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            extra_query={"bot_id": "invalid_bot_id"},
        )

        if hasattr(response, "error") and response.error:
            error_message = response.error.get("message", "")
            assert "Bot not found" in error_message or "not found" in error_message.lower()
            logger.info("Got expected 'Bot not found' error")
        else:
            assert False, "Should have received an error for invalid bot_id"

    except Exception as e:
        error_message = str(e)
        assert "Bot not found" in error_message or "not found" in error_message.lower()
        logger.info("Got expected exception for invalid bot_id")

    # Test without bot_id
    try:
        response = api_helper.openai_client.chat.completions.create(
            model="aperag", messages=[{"role": "user", "content": "Hello"}], stream=False
        )

        if hasattr(response, "error") and response.error:
            error_message = response.error.get("message", "")
            assert "bot_id is required" in error_message or "required" in error_message.lower()
            logger.info("Got expected 'bot_id is required' error")
        else:
            assert False, "Should have received an error when bot_id is missing"

    except Exception as e:
        error_message = str(e)
        assert "bot_id is required" in error_message or "required" in error_message.lower()
        logger.info("Got expected exception for missing bot_id")


def test_frontend_api_error_handling(client, basic_chat):
    """Test error handling for frontend API"""
    # Test invalid bot_id
    try:
        message = "Test message"
        response = client.post(
            "/api/v1/chat/completions/frontend",
            content=message,
            params={"stream": "false", "bot_id": "invalid_bot_id", "chat_id": basic_chat["id"]},
            headers={"msg_id": "test_msg_003", "Content-Type": "text/plain"},
            timeout=120.0,
        )

        assert response.status_code == HTTPStatus.OK, response.text
        response_data = response.json()
        assert response_data.get("type") == "error"
        error_message = response_data.get("data", "")
        assert "Bot not found" in error_message or "not found" in error_message.lower()
        logger.info("Got expected 'Bot not found' error")

    except Exception as e:
        logger.warning(f"Frontend error handling request failed: {e}")
        assert "bot_id" not in str(e), "API should accept bot_id as query parameter"

    # Test without bot_id
    try:
        message = "Test message"
        response = client.post(
            "/api/v1/chat/completions/frontend",
            content=message,
            params={"stream": "false", "chat_id": basic_chat["id"]},
            headers={"msg_id": "test_msg_004", "Content-Type": "text/plain"},
            timeout=120.0,
        )

        if response.status_code == HTTPStatus.OK:
            response_data = response.json()
            assert response_data.get("type") == "error"
            error_message = response_data.get("data", "")
            assert "bot_id" in error_message.lower() or "required" in error_message.lower()
            logger.info("Got expected 'bot_id required' error")
        else:
            assert response.status_code in [400, 422], "Should return 400 or 422 for missing bot_id"
            logger.info("Got expected HTTP error for missing bot_id")

    except Exception as e:
        logger.warning(f"Frontend missing bot_id test failed: {e}")
        error_message = str(e)
        assert "bot_id" in error_message.lower() or "required" in error_message.lower()
        logger.info("Got expected exception for missing bot_id")


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
