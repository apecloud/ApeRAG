import json

from tests.benchmarks.graph_extraction import runner


def test_parse_extraction_accepts_fenced_json():
    raw = """```json
{"entities": [{"name": "GB 12158-2024"}], "relations": []}
```"""

    json_ok, parse_error, entities, relations = runner.parse_extraction(raw)

    assert json_ok is True
    assert parse_error is None
    assert entities == [{"name": "GB 12158-2024"}]
    assert relations == []


def test_score_result_matches_fuzzy_entity_and_relation_names():
    sample = {
        "expected_entities": ["GB 12158-2024", "静电防护"],
        "expected_relations": [["GB 12158-2024", "静电防护"]],
    }
    entities = [
        {"name": "GB12158-2024"},
        {"name": "静电 防护"},
    ]
    relations = [
        {"source": "GB12158-2024", "target": "静电防护", "relation_type": "regulates"},
    ]

    score = runner.score_result(sample, entities, relations)

    assert score["entity_hits"] == 2
    assert score["relation_hits"] == 1


def test_sample_fixtures_are_valid():
    samples = runner.load_samples(runner.SAMPLES_DIR)

    assert {sample["id"] for sample in samples} >= {"asf_cn", "esd_cn", "vendor_esd_en"}
    for sample in samples:
        json.dumps(sample, ensure_ascii=False)
        assert sample["text"].strip()
        assert sample["expected_entities"]
        assert sample["expected_relations"]
