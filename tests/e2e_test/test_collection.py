# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from http import HTTPStatus

import pytest

from tests.e2e_test.utils import assert_search_result


def run_search_test(client, collection, document, search_data):
    resp = client.post(f"/api/v1/collections/{collection['id']}/searches", json=search_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    result = resp.json()
    assert_search_result(search_data, result)

    resp = client.get(f"/api/v1/collections/{collection['id']}/searches")
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == result["id"]

    test_id = result["id"]
    resp = client.delete(f"/api/v1/collections/{collection['id']}/searches/{test_id}")
    assert resp.status_code == HTTPStatus.OK, resp.text


@pytest.mark.parametrize(
    "search_data",
    [
        {"query": "test", "vector_search": {"topk": 10, "similarity": 0.1}},
        {"query": "test", "fulltext_search": {"topk": 10}},
        {"query": "test", "graph_search": {"topk": 10}},
        {
            "query": "test",
            "vector_search": {"topk": 10, "similarity": 0.1},
            "fulltext_search": {"topk": 10},
            "graph_search": {"topk": 10},
        },
    ],
    ids=["vector", "fulltext", "graph", "hybrid"],
)
def test_search_types(benchmark, client, collection, document, search_data):
    benchmark(run_search_test, client, collection, document, search_data)
