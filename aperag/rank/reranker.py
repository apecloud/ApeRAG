#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, List

import aiohttp

from aperag.query.query import DocumentWithScore
from config.settings import (
    RERANK_BACKEND,
    RERANK_SERVICE_MODEL_UID,
    RERANK_SERVICE_MODEL,
    RERANK_SERVICE_TOKEN,
    RERANK_SERVICE_URL,
)

default_rerank_model_path = "/data/models/bge-reranker-large"

# Mutex and synchronized decorator
mutex = Lock()
rerank_model_cache = {}

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
        self.model_uid = RERANK_SERVICE_MODEL_UID

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
        self.url = RERANK_SERVICE_URL # "https://api.jina.ai/v1/rerank"
        self.model = RERANK_SERVICE_MODEL # "jina-reranker-v2-base-multilingual"
        self.auth_token = RERANK_SERVICE_TOKEN # "Bearer YOUR_JINA_TOKEN"

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

class ContentRatioRanker(Ranker):
    def __init__(self, query):
        self.query = query

    async def rank(self, query, results: List[DocumentWithScore]):
        results.sort(key=lambda x: (x.metadata.get("content_ratio", 1), x.score), reverse=True)
        return results

class AutoCrossEncoderRanker(Ranker):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_path = os.environ.get("RERANK_MODEL_PATH", default_rerank_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    async def rank(self, query, results: List[DocumentWithScore]):
        import torch
        pairs = []
        for idx, doc in enumerate(results):
            pairs.append((query, doc.text))
            doc.rank_before = idx
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
            if isinstance(scores, torch.Tensor):
                scores = scores.tolist()
            ranked = sorted(zip(scores, results), key=lambda k: k[0], reverse=True)
        return [x for _, x in ranked]

class FlagCrossEncoderRanker(Ranker):
    def __init__(self):
        from FlagEmbedding import FlagReranker
        model_path = os.environ.get("RERANK_MODEL_PATH", default_rerank_model_path)
        self.reranker = FlagReranker(model_path)

    async def rank(self, query, results: List[DocumentWithScore]):
        import torch
        pairs = []
        max_length = 512
        for idx, doc in enumerate(results):
            pairs.append((query[:max_length], doc.text[:max_length]))
            doc.rank_before = idx
        if not pairs:
            return []
        with torch.no_grad():
            scores = self.reranker.compute_score(pairs, max_length=max_length)
            if isinstance(scores, float):
                scores = [scores]
        ranked = sorted(zip(scores, results), key=lambda k: k[0], reverse=True)
        return [x for _, x in ranked]

class RankerService(Ranker):
    def __init__(self):
        if RERANK_BACKEND == "xinference":
            self.ranker = XinferenceRanker()
        elif RERANK_BACKEND == "local":
            self.ranker = FlagCrossEncoderRanker()
        elif RERANK_BACKEND == "jina":
            self.ranker = JinaRanker()
        else:
            raise Exception("Unsupported backend")

    async def rank(self, query, results: List[DocumentWithScore]):
        return await self.ranker.rank(query, results)

@synchronized
def get_rerank_model(model_type: str = "bge-reranker-large"):
    if model_type in rerank_model_cache:
        return rerank_model_cache[model_type]
    model = RankerService()
    rerank_model_cache[model_type] = model
    return model

async def rerank(message, results):
    model = get_rerank_model()
    return await model.rank(message, results)