from fastapi import FastAPI

from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi
from aperag.views.bots_v2 import router


def _bot_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


def _request_schema(spec: dict, path: str, method: str) -> dict:
    request_ref = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    return spec["components"]["schemas"][request_ref.removeprefix("#/components/schemas/")]


def test_bot_v2_routes_are_public_and_typed():
    spec = _bot_v2_spec()
    paths = spec["paths"]

    required_paths = {
        "/api/v2/bots",
        "/api/v2/bots/{bot_id}",
        "/api/v2/bots/{bot_id}/chats",
        "/api/v2/bots/{bot_id}/chats/{chat_id}",
        "/api/v2/bots/{bot_id}/chats/{chat_id}/title",
    }

    assert required_paths <= set(paths)

    assert _json_ref(spec, "/api/v2/bots", "post") == "#/components/schemas/Bot"
    assert _json_ref(spec, "/api/v2/bots", "get") == "#/components/schemas/BotList"
    assert _json_ref(spec, "/api/v2/bots/{bot_id}", "get") == "#/components/schemas/Bot"
    assert _json_ref(spec, "/api/v2/bots/{bot_id}", "put") == "#/components/schemas/Bot"
    assert _json_ref(spec, "/api/v2/bots/{bot_id}/chats", "post") == "#/components/schemas/Chat"
    assert _json_ref(spec, "/api/v2/bots/{bot_id}/chats", "get") == "#/components/schemas/ChatList"
    assert _json_ref(spec, "/api/v2/bots/{bot_id}/chats/{chat_id}", "get") == (
        "#/components/schemas/ChatDetails"
    )
    assert _json_ref(spec, "/api/v2/bots/{bot_id}/chats/{chat_id}", "put") == "#/components/schemas/Chat"
    assert _json_ref(spec, "/api/v2/bots/{bot_id}/chats/{chat_id}/title", "post") == (
        "#/components/schemas/TitleGenerateResponse"
    )


def test_bot_v2_delete_routes_return_204_without_body():
    spec = _bot_v2_spec()

    for path in ("/api/v2/bots/{bot_id}", "/api/v2/bots/{bot_id}/chats/{chat_id}"):
        responses = spec["paths"][path]["delete"]["responses"]
        assert "204" in responses, f"DELETE {path} must declare 204 response"
        assert "content" not in responses["204"], f"DELETE {path} 204 response must not carry a JSON body"
        assert "200" not in responses, f"DELETE {path} must not mix 204 with a 200 JSON response"


def test_bot_v2_write_bodies_use_path_ids_as_canonical():
    spec = _bot_v2_spec()

    bot_update_schema = _request_schema(spec, "/api/v2/bots/{bot_id}", "put")
    assert bot_update_schema["properties"].keys().isdisjoint({"id", "bot_id"})

    chat_update_schema = _request_schema(spec, "/api/v2/bots/{bot_id}/chats/{chat_id}", "put")
    assert chat_update_schema["properties"].keys().isdisjoint({"id", "bot_id", "chat_id"})

    title_schema = _request_schema(spec, "/api/v2/bots/{bot_id}/chats/{chat_id}/title", "post")
    assert title_schema["properties"].keys().isdisjoint({"bot_id", "chat_id"})

    assert "requestBody" not in spec["paths"]["/api/v2/bots/{bot_id}/chats"]["post"]


def test_bot_v2_scope_is_bot_crud_and_bot_chat_shell_only():
    spec = _bot_v2_spec()

    for path in spec["paths"]:
        assert path.startswith("/api/v2/bots"), f"bots_v2 must not own standalone chat or other domain path: {path}"
        assert "/documents" not in path, f"bots_v2 must not own chat document path: {path}"
        assert "/feedback" not in path, f"bots_v2 must not own turn feedback path: {path}"
        assert "/search" not in path, f"bots_v2 must not own chat search path: {path}"


def test_bot_v2_operation_ids_are_unique():
    spec = _bot_v2_spec()

    operation_ids = [
        operation["operationId"]
        for path_item in spec["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]

    assert len(operation_ids) == len(set(operation_ids))
