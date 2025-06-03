import json
from http import HTTPStatus
from pathlib import Path

import yaml

from tests.e2e_test.config import COMPLETION_MODEL_NAME, COMPLETION_MODEL_PROVIDER


def test_list_bots(benchmark, client, bot):
    resp = benchmark(client.get, "/api/v1/bots?page=1&page_size=10")
    assert resp.status_code == HTTPStatus.OK
    bots = resp.json()["items"]
    assert any(b["id"] == bot["id"] for b in bots)


def test_get_bot_detail(benchmark, client, bot):
    resp = benchmark(client.get, f"/api/v1/bots/{bot['id']}")
    assert resp.status_code == HTTPStatus.OK
    detail = resp.json()
    assert detail["id"] == bot["id"]
    assert detail["title"] == bot["title"]


def test_update_bot(benchmark, client, collection, bot):
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
    resp = benchmark(client.put, f"/api/v1/bots/{bot['id']}", json=update_data)
    assert resp.status_code == HTTPStatus.OK
    updated = resp.json()
    assert updated["title"] == "E2E Test Bot Updated"
    assert updated["description"] == "E2E Bot Description Updated"
    assert updated["config"] == config


def test_update_flow(benchmark, client, bot):
    flow_path = Path(__file__).parent / "testdata" / "flow_for_bot.yaml"
    with open(flow_path, "r", encoding="utf-8") as f:
        flow = yaml.safe_load(f)
    flow_json = json.dumps(flow)
    resp = benchmark(
        client.put, f"/api/v1/bots/{bot['id']}/flow", data=flow_json, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == HTTPStatus.OK
    resp = client.get(f"/api/v1/bots/{bot['id']}/flow")
    assert resp.status_code == HTTPStatus.OK
    result = resp.json()
    assert result is not None
    assert result.get("name") == "rag_flow"
    assert result.get("version") == "1.0.0"
    assert result.get("schema") is not None
    assert result.get("nodes") is not None
    assert result.get("edges") is not None
    assert result.get("execution") is not None


def test_get_flow(benchmark, client, bot):
    resp = benchmark(client.get, f"/api/v1/bots/{bot['id']}/flow")
    assert resp.status_code == HTTPStatus.OK
    flow = resp.json()
    assert not flow
