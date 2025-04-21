#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from threading import Lock
from typing import List
import aiohttp
from aperag.query.query import DocumentWithScore
from config.settings import (
    RERANK_BACKEND,
    RERANK_SERVICE_MODEL,
    RERANK_SERVICE_TOKEN_API_KEY,
    RERANK_SERVICE_URL,
)

# Mutex and synchronized decorator
mutex = Lock()

def synchronized(func):
    def wrapper(*args, **kwargs):
        with mutex:
            return func(*args, **kwargs)
    return wrapper

class Ranker(ABC):
    @abstractmethod
    async def rank(self, query, results: List[DocumentWithScore]):
        pass

class XinferenceRanker(Ranker):
    def __init__(self):
        self.url = f"{RERANK_SERVICE_URL}/v1/rerank"
        self.model_uid = RERANK_SERVICE_MODEL

    async def rank(self, query, results: List[DocumentWithScore]):
        documents = [doc.text for doc in results]
        body = {
            "model": self.model_uid,
            "documents": documents,
            "query": query,
            "return_documents": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=body) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"Failed to rerank, detail: {data['detail']}")
                indices = [r["index"] for r in data["results"]]
                return [results[i] for i in indices]

class JinaRanker(Ranker):
    def __init__(self):
        self.url = RERANK_SERVICE_URL
        self.model = RERANK_SERVICE_MODEL
        self.auth_token = RERANK_SERVICE_TOKEN_API_KEY

    async def rank(self, query, results: List[DocumentWithScore]):
        documents = [doc.text for doc in results]
        body = {
            "model": self.model,
            "query": query,
            "top_n": len(documents),
            "documents": documents,
            "return_documents": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=headers, json=body) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"Failed to rerank, detail: {data}")
                indices = [r["index"] for r in data["results"]]
                return [results[i] for i in indices]

class SiliconflowRanker(Ranker):
    def __init__(self):
        self.url = RERANK_SERVICE_URL
        self.model = RERANK_SERVICE_MODEL
        self.auth_token = RERANK_SERVICE_TOKEN_API_KEY

    async def rank(self, query, results: List[DocumentWithScore]):
        documents = [doc.text for doc in results]
        body = {
            "model": self.model,
            "query": query,
            "top_n": len(documents),
            "documents": documents,
            "return_documents": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=headers, json=body) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"Failed to rerank, detail: {data}")
                indices = [r["index"] for r in data["results"]]
                return [results[i] for i in indices]

class RankerService(Ranker):
    def __init__(self):
        if RERANK_BACKEND == "xinference":
            self.ranker = XinferenceRanker()
        elif RERANK_BACKEND == "jina":
            self.ranker = JinaRanker()
        elif RERANK_BACKEND == "siliconflow":
            self.ranker = SiliconflowRanker()
        else:
            raise Exception("Unsupported backend")

    async def rank(self, query, results: List[DocumentWithScore]):
        return await self.ranker.rank(query, results)

@synchronized
def get_rerank_model():
    return RankerService()

async def rerank(message, results):
    model = get_rerank_model()
    return await model.rank(message, results)