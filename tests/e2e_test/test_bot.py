import json
from pathlib import Path

import pytest
import yaml

from tests.e2e_test.config import COMPLETION_MODEL_NAME, COMPLETION_MODEL_PROVIDER


@pytest.fixture
def bot(client, document, collection):
    config = {
        "model_name": f"{COMPLETION_MODEL_NAME}",
        "model_service_provider": COMPLETION_MODEL_PROVIDER,
        "llm": {"context_window": 3500, "similarity_score_threshold": 0.5, "similarity_topk": 3, "temperature": 0.1},
    }
    create_data = {
        "title": "E2E Test Bot",
        "description": "E2E Bot Description",
        "type": "knowledge",
        "config": json.dumps(config),
        "collection_ids": [collection["id"]],
    }
    resp = client.post("/api/v1/bots", json=create_data)
    assert resp.status_code == 200
    bot = resp.json()
    yield bot
    resp = client.delete(f"/api/v1/bots/{bot['id']}")
    assert resp.status_code in (200, 204)


def test_list_bots(client, bot):
    resp = client.get("/api/v1/bots?page=1&page_size=10")
    assert resp.status_code == 200
    bots = resp.json()["items"]
    assert any(b["id"] == bot["id"] for b in bots)


def test_get_bot_detail(client, bot):
    resp = client.get(f"/api/v1/bots/{bot['id']}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == bot["id"]
    assert detail["title"] == bot["title"]


def test_update_bot(client, collection, bot):
    config = json.dumps(
        {
            "model_name": f"{COMPLETION_MODEL_NAME}",
            "model_service_provider": COMPLETION_MODEL_PROVIDER,
            "llm": {
                "context_window": 3500,
                "similarity_score_threshold": 0.5,
                "similarity_topk": 5,
                "temperature": 0.2,
            },
        }
    )
    update_data = {
        "title": "E2E Test Bot Updated",
        "description": "E2E Bot Description Updated",
        "type": "knowledge",
        "config": config,
        "collection_ids": [collection["id"]],
    }
    resp = client.put(f"/api/v1/bots/{bot['id']}", json=update_data)
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["title"] == "E2E Test Bot Updated"
    assert updated["description"] == "E2E Bot Description Updated"
    assert updated["config"] == config


def test_update_flow(client, bot):
    flow_path = Path(__file__).parent / "testdata" / "flow_for_bot.yaml"
    with open(flow_path, "r", encoding="utf-8") as f:
        flow = yaml.safe_load(f)
    flow_json = json.dumps(flow)
    resp = client.put(f"/api/v1/bots/{bot['id']}/flow", data=flow_json, headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    resp = client.get(f"/api/v1/bots/{bot['id']}/flow")
    assert resp.status_code == 200
    result = resp.json()
    assert result is not None
    assert result.get("name") == "test_rag_flow"
    assert result.get("version") == "1.0.0"
    assert result.get("schema") is not None
    assert result.get("nodes") is not None
    assert result.get("edges") is not None
    assert result.get("execution") is not None


def test_get_flow(client, bot):
    resp = client.get(f"/api/v1/bots/{bot['id']}/flow")
    assert resp.status_code == 200
    flow = resp.json()
    assert not flow
