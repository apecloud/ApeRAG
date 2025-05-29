import pytest
import time

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
