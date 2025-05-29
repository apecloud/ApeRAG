import pytest
import time
import json
import httpx
import asyncio
import websockets

@pytest.fixture
def chat(client, bot):
    # Create a chat for testing
    data = {"title": "E2E Test Chat", "bot_id": bot["id"]}
    resp = client.post(f"/api/v1/bots/{bot['id']}/chats", json=data)
    assert resp.status_code in (200, 201)
    chat = resp.json()
    yield chat
    # Cleanup: delete chat after test
    client.delete(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}")
    # Ensure chat is deleted
    resp = client.get(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}")
    assert resp.status_code == 404


def test_get_chat_detail(client, bot, chat):
    # Test getting chat details
    resp = client.get(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == chat["id"]
    assert detail["title"] == chat["title"]


def test_update_chat(client, bot, chat):
    # Test updating chat title
    update_data = {"title": "E2E Test Chat Updated"}
    resp = client.put(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}", json=update_data)
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["title"] == "E2E Test Chat Updated"


def test_chat_message_and_feedback(client, bot, collection, chat):
    # Test sending a message in chat
    msg_data = {"content": "What is ApeRAG?", "role": "user"}
    resp = client.post(f"/api/v1/collections/{collection['id']}/chats/{chat['id']}/messages", json=msg_data)
    assert resp.status_code in (200, 201)
    message = resp.json()
    assert message["id"]
    assert message["content"] == msg_data["content"]
    assert message["role"] == "user"
    # Test getting message list
    resp = client.get(f"/api/v1/collections/{collection['id']}/chats/{chat['id']}/messages")
    assert resp.status_code == 200
    msg_list = resp.json()["items"]
    assert any(m["id"] == message["id"] for m in msg_list)
    # Test feedback for a message
    feedback_data = {"type": "good", "tag": "Other", "message": "Great answer"}
    resp = client.post(f"/api/v1/bots/{bot['id']}/chats/{chat['id']}/messages/{message['id']}", json=feedback_data)
    assert resp.status_code == 200


def test_websocket_chat_and_feedback(client, bot, chat):
    # Test websocket chat and feedback, wait for stop message, feedback via HTTP API, and check chat history
    import websockets
    import os
    import json
    ws_url = f"ws://localhost:8000/api/v1/bots/{bot['id']}/chats/{chat['id']}/connect"
    message = {"content": "What is ApeRAG?", "role": "user"}
    received_stop = False
    received_message_id = None
    async def ws_chat():
        nonlocal received_stop, received_message_id
        async with websockets.connect(ws_url, extra_headers={"Authorization": f"Bearer {os.environ.get('API_KEY', '')}"}) as ws:
            await ws.send(json.dumps(message))
            while True:
                reply = await ws.recv()
                data = json.loads(reply)
                if data.get("type") == "stop":
                    received_stop = True
                    received_message_id = data.get("id")
                    break
    import asyncio
    asyncio.get_event_loop().run_until_complete(ws_chat())
    assert received_stop
    # Feedback via HTTP API after stop
    feedback_data = {"type": "good", "tag": "Other", "message": "Great answer"}
    message_id = received_message_id or "1"
    feedback_url = f"/api/v1/bots/{bot['id']}/chats/{chat['id']}/messages/{message_id}"
    resp = client.post(feedback_url, json=feedback_data)
    assert resp.status_code == 200
    # Check chat history contains the sent message
    history_url = f"/api/v1/bots/{bot['id']}/chats/{chat['id']}"
    resp = client.get(history_url)
    assert resp.status_code == 200
    detail = resp.json()
    assert "history" in detail
    assert any(m.get("data", "").find("ApeRAG") != -1 for m in detail["history"] if m.get("type") == "message")


def test_openai_sse_chat(client, bot):
    # Test OpenAI-compatible SSE chat (no feedback, no chat_id required)
    url = "/api/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "aperag",
        "messages": [{"role": "user", "content": "What is ApeRAG?"}],
        "stream": False,
        "bot_id": bot["id"]
    }
    resp = client.post(url, headers=headers, json=data)
    assert resp.status_code == 200
    result = resp.json()
    assert "choices" in result
    assert result["choices"][0]["message"]["content"]


def test_frontend_sse_chat_and_feedback(client, bot, chat):
    # Test frontend custom SSE chat and feedback, wait for stop message, and check chat history
    import sseclient
    url = f"/api/v1/chat/completions/frontend?bot_id={bot['id']}&chat_id={chat['id']}"
    headers = {"Content-Type": "text/plain"}
    message = "What is ApeRAG?"
    received_stop = False
    received_message_id = None
    # Send SSE request
    resp = client.post(url, headers=headers, content=message, stream=True)
    assert resp.status_code == 200 or resp.status_code == 206
    sse = sseclient.SSEClient(resp)
    for event in sse.events():
        if event.data:
            data = json.loads(event.data)
            if data.get("type") == "stop":
                received_stop = True
                received_message_id = data.get("id")
                break
    assert received_stop
    # Feedback after stop
    feedback_data = {"type": "good", "tag": "Other", "message": "Great answer"}
    message_id = received_message_id or "1"
    feedback_url = f"/api/v1/bots/{bot['id']}/chats/{chat['id']}/messages/{message_id}"
    resp = client.post(feedback_url, json=feedback_data)
    assert resp.status_code == 200
    # Check chat history contains the sent message
    history_url = f"/api/v1/bots/{bot['id']}/chats/{chat['id']}"
    resp = client.get(history_url)
    assert resp.status_code == 200
    detail = resp.json()
    assert "history" in detail
    assert any(m.get("data", "").find("ApeRAG") != -1 for m in detail["history"] if m.get("type") == "message")
