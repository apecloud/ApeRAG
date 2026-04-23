from aperag.app import app


def _operation(path: str, method: str) -> dict:
    return app.openapi()["paths"][path][method]


def _json_schema(operation: dict, status: str = "200") -> dict:
    return operation["responses"][status]["content"]["application/json"]["schema"]


def test_agent_runtime_v2_openapi_contract_exposes_turn_event_and_artifact_models():
    create_turn = _operation("/api/v2/agent/chats/{chat_id}/turns", "post")
    assert create_turn["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CreateTurnRequest"
    }
    assert _json_schema(create_turn) == {"$ref": "#/components/schemas/CreateTurnResponse"}

    snapshot = _operation("/api/v2/agent/chats/{chat_id}/turns/{turn_id}", "get")
    assert _json_schema(snapshot) == {"$ref": "#/components/schemas/AgentTurnSnapshot"}

    cancel = _operation("/api/v2/agent/chats/{chat_id}/turns/{turn_id}/cancel", "post")
    assert _json_schema(cancel) == {"$ref": "#/components/schemas/CancelTurnResponse"}

    artifact = _operation("/api/v2/agent/artifacts/{artifact_id}", "get")
    assert _json_schema(artifact) == {"$ref": "#/components/schemas/AgentArtifactEnvelope"}

    events = _operation("/api/v2/agent/chats/{chat_id}/turns/{turn_id}/events", "get")
    event_stream = events["responses"]["200"]["content"]["text/event-stream"]["schema"]
    assert event_stream["type"] == "string"
    assert "AgentTimelineEventEnvelope" in event_stream["description"]


def test_openai_compatible_chat_completions_openapi_contract_is_explicit():
    operation = _operation("/v1/chat/completions", "post")
    parameter_names = {parameter["name"] for parameter in operation["parameters"]}
    assert {"bot_id", "chat_id", "language"}.issubset(parameter_names)

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["title"] == "OpenAIChatCompletionRequest"
    assert {"messages", "stream", "model"}.issubset(request_schema["properties"])

    json_any_of = _json_schema(operation)["anyOf"]
    assert {"$ref": "#/components/schemas/OpenAIChatCompletionResponse"} in json_any_of
    assert {"$ref": "#/components/schemas/OpenAIErrorResponse"} in json_any_of

    stream_schema = operation["responses"]["200"]["content"]["text/event-stream"]["schema"]
    assert stream_schema["type"] == "string"
    assert "chat.completion.chunk" in stream_schema["description"]
