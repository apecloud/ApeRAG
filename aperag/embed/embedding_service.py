import json
from typing import List

import requests
from langchain.embeddings.base import Embeddings

import logging

class EmbeddingService(Embeddings):
    def __init__(self, embedding_backend, embedding_model, embedding_dim,
                 embedding_service_url, embedding_service_api_key, embedding_max_chunks_in_batch):
        if embedding_backend == "xinference":
            self.model = XinferenceEmbedding(embedding_model, embedding_service_url)
        elif embedding_backend == "openai":
            self.model = OpenAIEmbedding(embedding_model, embedding_service_url, embedding_service_api_key, embedding_max_chunks_in_batch)
        elif embedding_backend == "alibabacloud":
            self.model = OpenAIEmbedding(embedding_model, embedding_service_url, embedding_service_api_key, embedding_max_chunks_in_batch)
        elif embedding_backend == "siliconflow":
            self.model = OpenAIEmbedding(embedding_model, embedding_service_url, embedding_service_api_key, embedding_max_chunks_in_batch)
        else:
            raise Exception("Unsupported embedding backend")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class XinferenceEmbedding(Embeddings):
    def __init__(self, embedding_model, embedding_service_url):
        self.url = f"{embedding_service_url}/embeddings"
        self.model_uid = embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        response = requests.post(url=self.url, json={"model": self.model_uid, "input": texts})

        results = (json.loads(response.content))["data"]
        embeddings = [result["embedding"] for result in results]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedding(Embeddings):
    def __init__(self, embedding_model, embedding_service_url, embedding_service_api_key, embedding_max_chunks_in_batch):
        self.url = f"{embedding_service_url}/embeddings"
        self.model = f"{embedding_model}"
        self.openai_api_key = f"{embedding_service_api_key}"
        self.max_chunks = embedding_max_chunks_in_batch

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        max_chunks = self.max_chunks if self.max_chunks and self.max_chunks > 0 else len(texts)
        embeddings = []
        for i in range(0, len(texts), max_chunks):
            batch = texts[i:i + max_chunks]
            response = requests.post(url=self.url, headers=headers, json={"model": self.model, "input": batch})
            results = (json.loads(response.content))["data"]
            embeddings.extend([result["embedding"] for result in results])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
