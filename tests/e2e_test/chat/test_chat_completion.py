import asyncio
import json

import pytest
from openai import OpenAI


@pytest.mark.asyncio
async def test_openai_chat_completions():
    # Initialize OpenAI client with custom base URL and API key
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-3b26ff8792444d65a7075a31dd0c438a"
    )
    
    # Test non-streaming mode
    response = client.chat.completions.create(
        model="aperag",  # Use your model name
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        stream=False,
        bot_id="test_bot_id"  # Add bot_id as a custom parameter
    )
    
    # Verify response structure
    assert response.id is not None
    assert response.object == "chat.completion"
    assert response.created is not None
    assert response.model == "aperag"
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content is not None
    assert response.choices[0].finish_reason == "stop"
    
    # Test streaming mode
    stream = client.chat.completions.create(
        model="aperag",
        messages=[
            {"role": "user", "content": "Tell me a short story."}
        ],
        stream=True,
        bot_id="test_bot_id"
    )
    
    # Collect and verify streaming response
    collected_content = ""
    for chunk in stream:
        assert chunk.id is not None
        assert chunk.object == "chat.completion.chunk"
        assert chunk.created is not None
        assert chunk.model == "aperag"
        assert len(chunk.choices) == 1
        
        if chunk.choices[0].delta.content is not None:
            collected_content += chunk.choices[0].delta.content
    
    # Verify final content
    assert len(collected_content) > 0


@pytest.mark.asyncio
async def test_openai_chat_completions_error_handling():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-3b26ff8792444d65a7075a31dd0c438a"
    )
    
    # Test with invalid bot_id
    try:
        client.chat.completions.create(
            model="aperag",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            stream=False,
            bot_id="invalid_bot_id"
        )
        assert False, "Should have raised an error"
    except Exception as e:
        error_data = json.loads(str(e))
        assert "error" in error_data
        assert error_data["error"]["message"] == "Bot not found"
    
    # Test with invalid messages format
    try:
        client.chat.completions.create(
            model="aperag",
            messages="invalid_messages",  # Should be a list
            stream=False,
            bot_id="test_bot_id"
        )
        assert False, "Should have raised an error"
    except Exception as e:
        assert "Invalid request" in str(e)


if __name__ == "__main__":
    asyncio.run(test_openai_chat_completions())
    asyncio.run(test_openai_chat_completions_error_handling()) 