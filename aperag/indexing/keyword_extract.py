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

"""Keyword extraction helper — celery T3.1.

Per architect msg=3890c9d7 commit-4 split, the search-time keyword
extraction subsystem (``extract_keywords`` + the IK / LLM extractors
that back it) is moved out of the soon-to-be-deleted
``aperag/domains/indexing/fulltext_index.py`` into the Wave 1+
``aperag/indexing/`` surface. The function shape is unchanged so the
two caller sites (``aperag/domains/retrieval/pipeline.py`` and
``aperag/service/search_pipeline_service.py``) only swap their import
path; their call shapes are untouched.

This module is a write-side helper for the SEARCH path, not the
indexing pipeline. It does not depend on any
``aperag/indexing/`` write-side surface (no orchestrator, no
modality workers, no DocumentIndex). It deliberately stays separate
from ``aperag/indexing/fulltext.py`` (Wave 1 fulltext modality
worker, write-side) because the two have orthogonal lifecycles —
the modality worker writes search rows on document index, this
helper extracts keywords on every search query.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch

from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.llm.completion.completion_service import CompletionService

logger = logging.getLogger(__name__)


def _es_client_config() -> Dict[str, Any]:
    """Common ES client configuration shared by every IK extractor instance."""
    return {
        "request_timeout": settings.es_timeout,
        "max_retries": settings.es_max_retries,
        "retry_on_timeout": True,
    }


# ---------------------------------------------------------------------
# Extractor base + IK (default fallback)
# ---------------------------------------------------------------------


class KeywordExtractor:
    """Base class for keyword extraction (kept for backwards-compat with
    callers that may type-annotate against the abstract base)."""

    def __init__(self, ctx: Dict[str, Any]) -> None:
        self.ctx = ctx

    async def extract(self, text: str) -> List[str]:
        raise NotImplementedError


class IKKeywordExtractor(KeywordExtractor):
    """Extract keywords from text using the Elasticsearch IK analyzer.

    The IK analyzer is the default tier-1 fallback — always available
    when ES is reachable. Every search-time keyword extraction request
    falls back to IK if a configured LLM extractor either fails or is
    not configured for the current user.
    """

    def __init__(self, ctx: Dict[str, Any]) -> None:
        super().__init__(ctx)
        config = _es_client_config()
        config.update(
            {
                "request_timeout": ctx.get("es_timeout", settings.es_timeout),
                "max_retries": ctx.get("es_max_retries", settings.es_max_retries),
            }
        )

        self.client = AsyncElasticsearch(ctx.get("es_host", settings.es_host), **config)
        self.index_name = ctx["index_name"]
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        """Load stop words from the same shared misc/stopwords.txt file
        that the legacy fulltext_index module read. Path is anchored to
        ``aperag/misc/stopwords.txt`` so the move does not break the
        relative resolution.
        """
        # ``aperag/indexing/keyword_extract.py`` → parent (indexing) →
        # parent (aperag) → ``misc/stopwords.txt``
        stop_words_path = Path(__file__).parent.parent / "misc" / "stopwords.txt"
        if os.path.exists(stop_words_path):
            with open(stop_words_path) as f:
                return set(f.read().splitlines())
        return set()

    async def __aenter__(self) -> "IKKeywordExtractor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.client.close()

    async def extract(self, text: str) -> List[str]:
        try:
            resp = await self.client.indices.exists(index=self.index_name)
            if not resp.body:
                logger.warning("index %s not exists", self.index_name)
                return []

            resp = await self.client.indices.analyze(
                index=self.index_name,
                body={"text": text, "analyzer": "ik_smart"},
            )

            tokens: set = set()
            for item in resp.body["tokens"]:
                token = item["token"]
                if token not in self.stop_words:
                    tokens.add(token)
            return list(tokens)

        except Exception as exc:  # noqa: BLE001 — fall back to caller's downstream extractor
            logger.error(
                "Failed to extract keywords for index %s: %s",
                self.index_name,
                exc,
            )
            return []


# ---------------------------------------------------------------------
# Optional LLM extractor (configured via settings)
# ---------------------------------------------------------------------


class LLMKeywordExtractor(KeywordExtractor):
    """Extract keywords from text using an LLM with structured JSON output."""

    def __init__(self, ctx: Dict[str, Any]) -> None:
        super().__init__(ctx)
        self.completion_service = self._create_completion_service()

    def _create_completion_service(self) -> Optional[CompletionService]:
        """Construct the per-user LLM completion service if configured.

        Returns ``None`` (not an error) when the runtime is not
        configured — :func:`extract_keywords` then falls back to the IK
        extractor instead of failing the whole search request.
        """
        try:
            if not settings.llm_keyword_extraction_model:
                return None

            user_id = self.ctx.get("user_id")
            if not user_id:
                logger.warning("User ID not available in context for LLM keyword extraction")
                return None
            row = db_ops.query_model_runtime(settings.llm_keyword_extraction_model, user_id)
            if not row:
                logger.warning(
                    "LLM keyword extraction model '%s' not found",
                    settings.llm_keyword_extraction_model,
                )
                return None
            model, account = row
            from aperag.llm.runtime.resolver import resolve_model_invocation_from_records

            invocation = resolve_model_invocation_from_records(model=model, account=account)
            provider = invocation.runner_config.get("provider")
            if not provider:
                provider = "openai" if invocation.runner_type == "openai_compatible" else invocation.provider_type

            return CompletionService(
                provider=provider,
                model=invocation.provider_model_id,
                base_url=invocation.base_url,
                api_key=invocation.api_key,
            )

        except Exception as exc:  # noqa: BLE001 — runtime config failures degrade to IK
            logger.warning("Failed to create LLM completion service: %s", exc)
            return None

    async def extract(self, text: str) -> List[str]:
        """Extract keywords using LLM with structured JSON output."""
        if not self.completion_service:
            raise RuntimeError("LLM completion service not available")

        prompt = f"""Extract the most important keywords from the following text. Focus on:
1. Nouns, verbs, and adjectives that capture the main concepts
2. Remove stop words and meaningless terms
3. Keywords should be in the same language as the input text

Text: {text}

Please respond with ONLY a JSON object in the following format:
{{"keywords": ["keyword1", "keyword2", "keyword3", ...]}}

Do not include any other text or explanation, just the JSON object."""

        try:
            response = await self.completion_service.agenerate([], prompt)

            keywords = self._parse_json_response(response)
            if keywords:
                return keywords[:10]

            logger.warning("JSON parsing failed, falling back to simple parsing")
            return self._parse_keywords_fallback(response)

        except Exception as exc:
            logger.error("LLM keyword extraction failed: %s", exc)
            raise

    def _parse_json_response(self, response: str) -> List[str]:
        """Parse JSON response to extract keywords."""
        response = response.strip()

        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "keywords" in data:
                    keywords = data["keywords"]
                    if isinstance(keywords, list):
                        filtered_keywords = [str(k).strip() for k in keywords if k and str(k).strip()]
                        return filtered_keywords[:10]
            except json.JSONDecodeError as exc:
                logger.warning("JSON decode error: %s, response: %s", exc, json_str)
        else:
            logger.warning("JSON object not found in response: %s", response)

        return []

    def _parse_keywords_fallback(self, response: str) -> List[str]:
        """Fallback keyword parsing method."""
        keywords: list = []
        for line in response.strip().split("\n"):
            keyword = line.strip()
            keyword = keyword.lstrip("- *•").strip()
            if keyword and not keyword.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
                keyword = keyword.strip("\"'")
                if keyword:
                    keywords.append(keyword)

        return keywords[:10]

    async def __aenter__(self) -> "LLMKeywordExtractor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # No cleanup needed for completion service
        pass


# ---------------------------------------------------------------------
# Public entry point — multi-extractor fallback
# ---------------------------------------------------------------------


async def extract_keywords(text: str, ctx: Dict[str, Any]) -> List[str]:
    """Extract keywords from text with LLM-then-IK fallback.

    Priority order:
      1. :class:`LLMKeywordExtractor` (only if configured + per-user
         model resolution succeeds)
      2. :class:`IKKeywordExtractor` (fallback — works whenever ES is
         reachable)

    ``ctx`` is the per-request context dict the caller passes through
    from the FastAPI handler:

      - ``index_name`` (required) — ES index to introspect for IK
        analysis
      - ``user_id`` (optional) — gates LLM extraction; without it the
        LLM extractor is skipped
      - ``es_host`` / ``es_timeout`` / ``es_max_retries`` (optional) —
        per-request ES client overrides

    Returns the merged keyword list (capped at 10 in extractor
    classes). Returns ``[]`` (not an error) if every extractor fails.
    """
    extractors: list[tuple[str, type[KeywordExtractor]]] = []

    if settings.llm_keyword_extraction_provider and settings.llm_keyword_extraction_model and ctx.get("user_id"):
        extractors.append(("LLM", LLMKeywordExtractor))

    extractors.append(("IK", IKKeywordExtractor))

    for extractor_name, extractor_class in extractors:
        try:
            logger.info("Trying %s keyword extractor", extractor_name)
            async with extractor_class(ctx) as extractor:
                keywords = await extractor.extract(text)
                if keywords:
                    logger.info(
                        "%s extractor succeeded, got %d keywords",
                        extractor_name,
                        len(keywords),
                    )
                    return keywords
                logger.warning("%s extractor returned no keywords", extractor_name)
        except Exception as exc:  # noqa: BLE001 — fall through to next extractor
            logger.warning("%s extractor failed: %s", extractor_name, exc)
            continue

    logger.error("All keyword extractors failed")
    return []


__all__ = [
    "KeywordExtractor",
    "IKKeywordExtractor",
    "LLMKeywordExtractor",
    "extract_keywords",
]
