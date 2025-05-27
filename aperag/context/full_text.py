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

import logging
from typing import List

from elasticsearch import AsyncElasticsearch, Elasticsearch, NotFoundError

from config import settings

logger = logging.getLogger(__name__)


if settings.ENABLE_KEYWORD_SEARCH:
    es = Elasticsearch(settings.ES_HOST)
    async_es = AsyncElasticsearch(settings.ES_HOST)


# import redis
# from redis.commands.search.query import Query
# from redis.commands.json.path import Path
# from redis.commands.search.field import TextField
# from redis.commands.search.indexDefinition import IndexDefinition, IndexType


# def get_redis_client() -> redis.Redis:
#     return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD, decode_responses=True)
#

# def insert_document(collection_id, doc_id, doc_name, contentgg):
#     r = get_redis_client()
#     rs = r.ft(f"collection:fulltext:{collection_id}")
#     try:
#         schema = (
#             TextField("$.name", as_name="name"),
#             TextField("$.content", as_name="content"),
#         )
#         rs.create_index(schema, definition=IndexDefinition(prefix=["document:"], index_type=IndexType.JSON))
#     except redis.ResponseError as e:
#         if "Index already exists" not in str(e):
#             raise e
#     doc = {
#         "name": doc_name,
#         "content": content,
#     }
#     r.json().set(f"document:{doc_id}", Path.root_path(), doc)

# def search_document(collection_id: str, keywords: List[str]):
#     r = get_redis_client()
#     rs = r.ft(f"collection:fulltext:{collection_id}")
#     query = Query(" ".join(keywords)).language("chinese")
#     docs = rs.search(query).docs
#     return docs

# def search_document(collection_id: str, keywords: List[str]):
#     client = Client(f"collection:fulltext:{collection_id}")
#     docs = client.search(Query(' '.join(keywords))).docs
#     return docs
#
#
# def insert_document(collection_id, doc_id, doc_name, content):
#     client = Client(f"collection:fulltext:{collection_id}")
#     try:
#         definition = IndexDefinition(prefix=[f"collection:{collection_id}:document:"])
#         client.create_index([TextField('name'), TextField('content')],
#                             definition=definition, stopwords=",<>{}[]\"':;!@#$%^&*()-+=~")
#     except redis.ResponseError as e:
#         if "Index already exists" not in str(e):
#             raise e
#
#     client.add_document(f"collection:{collection_id}:document:{doc_id}", language="chinese", name=doc_name, content=content)


def delete_index(index):
    if es.indices.exists(index=index).body:
        es.indices.delete(index=index)


def create_index(index):
    if not es.indices.exists(index=index).body:
        mapping = {
            "properties": {"content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}}
        }
        es.indices.create(index=index, body={"mappings": mapping})
    else:
        logger.warning("index %s already exists", index)


# insert document into elasticsearch
def insert_document(index, doc_id, doc_name, content):
    if es.indices.exists(index=index).body:
        doc = {
            "name": doc_name,
            "content": content,
        }
        es.index(index=index, id=f"{doc_id}", document=doc)
    else:
        logger.warning("index %s not exists", index)


def remove_document(index, doc_id):
    if es.indices.exists(index=index).body:
        try:
            es.delete(index=index, id=f"{doc_id}")
        except NotFoundError:
            logger.warning("document %s not found in index %s", doc_id, index)
    else:
        logger.warning("index %s not exists", index)


async def search_document(index: str, keywords: List[str], topk=3):
    resp = await async_es.indices.exists(index=index)
    if not resp.body:
        return []

    if not keywords:
        return []

    query = {
        "bool": {
            "should": [{"match": {"content": keyword}} for keyword in keywords],
            "minimum_should_match": "80%",
            # "minimum_should_match": "-1",
        },
    }
    sort = [{"_score": {"order": "desc"}}]
    resp = await async_es.search(index=index, query=query, sort=sort, size=topk)
    hits = resp.body["hits"]
    result = []
    for hit in hits["hits"]:
        result.append({
            "content": hit["_source"]["content"],
            "name": hit["_source"]["name"],
            "score": hit["_score"],
        })
    return result
