#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, List

import requests
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, MT5EncoderModel

from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from config.settings import (
    EMBEDDING_BACKEND,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_SERVICE_MODEL,
    EMBEDDING_SERVICE_MODEL_UID,
    EMBEDDING_SERVICE_TOKEN,
    EMBEDDING_SERVICE_URL,
)


class EmbeddingService(Embeddings):
    def __init__(self, embedding_backend, model_type):
        if embedding_backend == "local":
            self.model = EMBEDDING_MODEL_CLS.get(model_type)()
        elif embedding_backend == "xinference":
            self.model = XinferenceEmbedding()
        elif embedding_backend == "openai":
            self.model = OpenAIEmbedding()
        else:
            raise Exception("Unsupported embedding backend")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class XinferenceEmbedding(Embeddings):
    def __init__(self):
        self.url = f"{EMBEDDING_SERVICE_URL}/v1/embeddings"
        self.model_uid = EMBEDDING_SERVICE_MODEL_UID

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        response = requests.post(url=self.url, json={"model": self.model_uid, "input": texts})

        results = (json.loads(response.content))["data"]
        embeddings = [result["embedding"] for result in results]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedding(Embeddings):
    def __init__(self):
        self.url = f"{EMBEDDING_SERVICE_URL}/v1/embeddings"
        self.model = f"{EMBEDDING_SERVICE_MODEL}"
        self.openai_api_key = f"{EMBEDDING_SERVICE_TOKEN}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url=self.url, headers=headers, json={"model": self.model, "input": texts})

        results = (json.loads(response.content))["data"]
        embeddings = [result["embedding"] for result in results]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class Text2VecEmbedding(Embeddings):
    def __init__(self, **kwargs):
        from text2vec import SentenceModel
        self.model = SentenceModel('shibing624/text2vec-base-chinese', **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class MT5Embedding(Embeddings):
    def __init__(self):
        model_name = "csebuetnlp/mT5_m2o_chinese_simplified_crossSum"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MT5EncoderModel.from_pretrained(model_name)

    def _embed_text(self, text):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
        r = torch.mean(outputs.hidden_states[-1].squeeze(), dim=0)
        return r.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        result = []
        for text in texts:
            result.append(self._embed_text(text))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class PiccoloEmbedding(Embeddings):

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("https_proxy", "socks5://127.0.0.1:34001")
        self.model = SentenceTransformer('sensenova/piccolo-large-zh')

    def _embed_text(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        result = []
        for text in texts:
            text = text.replace("\n", " ")
            if len(text) > 100:
                text = "结果：" + text
            result.append(text)
        return self._embed_text(result)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(["查询：" + text])[0]


class MultilingualSentenceTransformer(Embeddings):

    def __init__(self):
        import torch.nn.functional as F

        self.F = F
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        result = []
        for text in texts:
            embeddings = self.model.encode([f"passage: {text}"], normalize_embeddings=True)
            result.append(embeddings.tolist()[0])
        return result

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.encode([f"query: {text}"], normalize_embeddings=True)
        return embeddings.tolist()[0]  # Corrected original code - should return list[0]


class BertEmbedding(Embeddings):
    def __init__(self):
        from transformers import BertForMaskedLM, BertTokenizer
        model_name = "IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)

    def _embed_text(self, text):
        import torch
        input_tokens = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors='pt')
        bert_outputs = self.model(**input_tokens, return_dict=True, output_hidden_states=True)
        r = torch.mean(bert_outputs.hidden_states[-1].squeeze(), dim=0)
        return r.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        result = []
        for text in texts:
            result.append(self._embed_text(text))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


mutex = Lock()
embedding_model_cache = {}

EMBEDDING_MODEL_CLS = {
    "huggingface": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': EMBEDDING_DEVICE},
    ),
    "text2vec": lambda: Text2VecEmbedding(device=EMBEDDING_DEVICE),
    "bge": lambda: HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )
}


def synchronized(func):
    def wrapper(*args, **kwargs):
        with mutex:
            return func(*args, **kwargs)

    return wrapper


@synchronized
def get_embedding_model(model_type: str = "bge", load=True, **kwargs) -> tuple[Embeddings | None, int]:
    if model_type in embedding_model_cache:
        return embedding_model_cache[model_type]

    embedding_model = None
    vector_size = get_embedding_dimension(EMBEDDING_BACKEND, model_type, EMBEDDING_SERVICE_MODEL)

    if load:
        embedding_model = EmbeddingService(EMBEDDING_BACKEND, model_type)
        embedding_model_cache[model_type] = (embedding_model, vector_size)
        return embedding_model, vector_size  # Return inside if load
    else:
        # Return tuple with None embedding if load is False
        return None, vector_size


def get_embedding_dimension(embedding_backend: str, model_type: str, service_model: str = None) -> int:
    rules = EMBEDDING_DIMENSIONS.get(embedding_backend, {})

    if embedding_backend == "openai":
        # Use service_model for openai, fallback to __default__ within openai rules
        return rules.get(service_model, rules.get("__default__", 1536))

    # Use model_type for others, fallback to __default__ for that backend
    return rules.get(model_type, rules.get("__default__", 768))


def get_collection_embedding_model(collection):
    config = json.loads(collection.config)
    model_name = config.get("embedding_model", "text2vec")
    return get_embedding_model(model_name)

class DocumentBaseEmbedding(ABC):
    def __init__(
            self,
            vector_store_adaptor: VectorStoreConnectorAdaptor,
            embedding_model: Embeddings = None,
            vector_size: int = None,
            **kwargs: Any,
    ) -> None:
        self.connector = vector_store_adaptor.connector
        # Improved logic to handle optional embedding_model/vector_size
        if embedding_model is None or vector_size is None:
            loaded_embedding, loaded_vector_size = get_embedding_model(EMBEDDING_MODEL)
            self.embedding = embedding_model if embedding_model is not None else loaded_embedding
            self.vector_size = vector_size if vector_size is not None else loaded_vector_size
        else:
            self.embedding, self.vector_size = embedding_model, vector_size

        # Ensure embedding is loaded if needed
        if self.embedding is None:
            raise ValueError("Embedding model could not be loaded or provided.")

        self.client = vector_store_adaptor.connector.client

    @abstractmethod
    def load_data(self, **kwargs: Any):
        pass