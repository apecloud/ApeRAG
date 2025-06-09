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
import os
from pathlib import Path
from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch, Elasticsearch, NotFoundError

from aperag.config import settings
from aperag.query.query import DocumentWithScore

logger = logging.getLogger(__name__)


if settings.enable_fulltext_search:
    es = Elasticsearch(settings.es_host)
    async_es = AsyncElasticsearch(settings.es_host)


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


async def search_document(index: str, keywords: List[str], topk=3) -> List[DocumentWithScore]:
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
        result.append(
            DocumentWithScore(
                text=hit["_source"]["content"],
                score=hit["_score"],
                metadata={
                    "source": hit["_source"]["name"],
                },
            )
        )
    return result


class KeywordExtractor(object):
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    async def extract(self, text):
        raise NotImplementedError


class IKExtractor(KeywordExtractor):
    """
    Extract keywords from text using IK
    """

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.client = AsyncElasticsearch(ctx.get("es_host", "http://127.0.0.1:9200"))
        self.index_name = ctx["index_name"]
        # TODO move stop words to global
        stop_words_path = ctx.get("stop_words_path", Path(__file__).parent / "stopwords.txt")
        if os.path.exists(stop_words_path):
            with open(stop_words_path) as f:
                self.stop_words = set(f.read().splitlines())
        else:
            self.stop_words = set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

    async def extract(self, text):
        resp = await self.client.indices.exists(index=self.index_name)
        if not resp.body:
            logger.warning("index %s not exists", self.index_name)
            return []

        resp = await self.client.indices.analyze(index=self.index_name, body={"text": text}, analyzer="ik_smart")
        tokens = {}
        for item in resp.body["tokens"]:
            token = item["token"]
            if token in self.stop_words:
                continue
            tokens[token] = True
        return tokens.keys()
