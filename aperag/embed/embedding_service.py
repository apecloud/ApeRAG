from __future__ import annotations

from typing import List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from langchain.embeddings.base import Embeddings
from tenacity import retry, wait_exponential, stop_after_attempt


class EmbeddingService(Embeddings):
    def __init__(
        self,
        embedding_backend: str,
        embedding_model: str,
        embedding_service_url: str,
        embedding_service_api_key: str,
        embedding_max_chunks_in_batch: int,
    ):
        if embedding_backend in ["siliconflow", "alibabacloud"]:
            embedding_backend = "openai"
        self.model = f"{embedding_backend}/{embedding_model}"
        self.api_base = embedding_service_url
        self.api_key = embedding_service_api_key
        self.max_chunks = embedding_max_chunks_in_batch
        self.max_workers = 8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [t.replace("\n", " ") for t in texts]
        chunk_size = self.max_chunks or len(clean_texts)

        batches: List[Tuple[int, Sequence[str]]] = [
            (start, clean_texts[start : start + chunk_size])
            for start in range(0, len(clean_texts), chunk_size)
        ]

        results: List[List[float]] = [None] * len(clean_texts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._embed_batch, batch): start
                for start, batch in batches
            }

            for future in as_completed(futures):
                start_idx = futures[future]
                batch_embeddings = future.result()
                results[start_idx : start_idx + len(batch_embeddings)] = batch_embeddings

        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        response = litellm.embedding(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            input=list(batch),
        )
        return [item["embedding"] for item in response["data"]]