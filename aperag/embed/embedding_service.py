import json
from typing import List

import requests
from langchain.embeddings.base import Embeddings


class EmbeddingService(Embeddings):
    def __init__(self, embedding_backend, embedding_model, embedding_dim,
                 embedding_service_url, embedding_service_api_key, embedding_service_batch_size):
        if embedding_backend == "xinference":
            self.model = XinferenceEmbedding(embedding_model, embedding_service_url)
        elif embedding_backend == "openai":
            self.model = OpenAIEmbedding(embedding_model, embedding_service_url, embedding_service_api_key, embedding_service_batch_size)
        else:
            raise Exception("Unsupported embedding backend")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class XinferenceEmbedding(Embeddings):
    def __init__(self, embedding_model, embedding_service_url):
        self.url = f"{embedding_service_url}/v1/embeddings"
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
    def __init__(self, embedding_model, embedding_service_url, embedding_service_api_key, embedding_service_batch_size):
        self.url = f"{embedding_service_url}/v1/embeddings"
        self.model = f"{embedding_model}"
        self.openai_api_key = f"{embedding_service_api_key}"
        self.batch_size = embedding_service_batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        batch_size = self.batch_size if self.batch_size and self.batch_size > 0 else len(texts)
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = requests.post(url=self.url, headers=headers, json={"model": self.model, "input": batch})
            results = (json.loads(response.content))["data"]
            embeddings.extend([result["embedding"] for result in results])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]