import re

from fastapi import FastAPI

from aperag.domains.conversation.api.routes import bots_router as router
from aperag.openapi_spec import build_full_openapi_spec, custom_generate_unique_id, filter_public_openapi


def _bot_v2_spec():
    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)
    app.include_router(router, prefix="/api/v2")
    return filter_public_openapi(build_full_openapi_spec(app))


def _json_ref(spec: dict, path: str, method: str, status: str = "200") -> str:
    return spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]


def _request_schema(spec: dict, path: str, method: str) -> dict:
    request_ref = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    return spec["components"]["schemas"][request_ref.removeprefix("#/components/schemas/")]


# v1 bot routes still coexist with bots_v2 as a parallel path on main (#21 added parallel v2
# without deleting v1). A v1-absence assertion is intentionally NOT added at this time; it will
# be introduced by the #26 final sweep when the v1 bot surface is removed.


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
    assert _json_ref(spec, "/api/v2/bots/{bot_id}/chats/{chat_id}", "get") == ("#/components/schemas/ChatDetails")
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


def test_bot_v2_delete_routes_contract():
    """Every DELETE route under bots_v2 must respect the command-vs-report contract:

    - must not declare both 200 and 204 success responses
    - if 204 is declared it must have no application/json body (pure command)

    Generalizes ``test_bot_v2_delete_routes_return_204_without_body`` to cover every DELETE
    route (new ones appear automatically).
    """
    spec = _bot_v2_spec()
    checked = 0
    for path, operations in spec["paths"].items():
        op = operations.get("delete")
        if not op:
            continue
        responses = op.get("responses") or {}
        assert not ("200" in responses and "204" in responses), (
            f"DELETE {path} must not mix 200 and 204 success responses; pick one"
        )
        if "204" in responses:
            assert "content" not in (responses["204"] or {}), (
                f"DELETE {path} 204 response must not carry an application/json body"
            )
        else:
            assert "200" in responses, f"DELETE {path} must declare a 200 or 204 success response"
        checked += 1
    assert checked >= 1, "bots_v2 should expose at least one DELETE route to exercise this contract"


def test_bot_v2_all_write_request_bodies_omit_path_params():
    """Every POST/PUT/PATCH request body under bots_v2 must not redeclare path params.

    Generalizes ``test_bot_v2_write_bodies_use_path_ids_as_canonical`` to all path params so
    new write routes under ``/api/v2/bots/...`` are covered automatically.
    """
    spec = _bot_v2_spec()
    components = spec["components"]["schemas"]
    path_param_re = re.compile(r"\{([^{}]+)\}")

    checked = 0
    for path, methods in spec["paths"].items():
        path_params = set(path_param_re.findall(path))
        if not path_params:
            continue
        for method, operation in methods.items():
            if method not in {"post", "put", "patch"}:
                continue
            request_body = (operation or {}).get("requestBody") or {}
            json_schema = request_body.get("content", {}).get("application/json", {}).get("schema") or {}
            ref = json_schema.get("$ref")
            if not ref:
                continue
            schema_name = ref.removeprefix("#/components/schemas/")
            properties = set(components[schema_name].get("properties", {}).keys())
            overlap = path_params & properties
            assert not overlap, (
                f"{method.upper()} {path} request body {schema_name} duplicates path param(s) {sorted(overlap)}"
            )
            checked += 1

    assert checked >= 1, (
        f"Expected at least 1 write route with request body under bots_v2, but only inspected {checked}"
    )
