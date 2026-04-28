# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wave 10 §K.13 — collection summary + description regen service.

Two-stage pipeline:

* **Stage 1** (``regen_summary``): agent-runtime free-explore over the
  collection produces the long-form ``Collection.summary`` (5000-10000
  chars). 3-tier fallback chain (per huangheng BLOCKER #2):
  agent → chunks.jsonl first-substantive-chunk + LLM call →
  transient skip (input not ready, retry next reconciler).
* **Stage 2** (``regen_description``): cheap LLM call derives the
  short ``Collection.description`` (200-500 chars) from the existing
  ``summary``. ~5K tokens / ~10s, much faster than Stage 1.

Both stages share a **single cluster-level lease** (``regen_lease_owner``
+ ``regen_lease_expires_at`` columns on ``Collection``) so a multi-
instance deployment cannot run summary + description in parallel
against the same row (description depends on summary).

Wave 10 follow-up note: the Stage 1 agent-runtime invocation goes
through a clearly-marked extension hook
(``_invoke_summary_agent``) that today returns ``None``,
forcing the chunks.jsonl Tier-2 fallback. Filling in the actual
``agent_runtime_manager.launch_turn`` integration (per design appendix
A — fake Turn/Chat/Bot ORM construction) is staged as a Wave 10.1
follow-up to avoid coupling this PR to agent-runtime headless API
formalisation (huangheng N2 sediment, Wave 11 candidate).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Awaitable, Callable

from sqlalchemy import and_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from aperag.config import get_async_session
from aperag.domains.knowledge_base.db.models import Collection
from aperag.domains.knowledge_base.service.regen_constants import (
    DESCRIPTION_DERIVE_PROMPT_EN,
    DESCRIPTION_DERIVE_PROMPT_ZH,
    DESCRIPTION_MIN_CHARS,
    DESCRIPTION_TIMEOUT_SECONDS,
    INVALID_OUTPUT_FRAGMENTS,
    LEASE_TTL,
    SUMMARY_MIN_CHARS,
)
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Lease primitives — Wave 10 §2.3 cluster-level concurrency control.
# ---------------------------------------------------------------------


async def _try_acquire_lease(
    session: AsyncSession,
    *,
    collection_id: str,
) -> str | None:
    """Atomically acquire the regen lease on ``collection_id``.

    Returns the lease token (UUID hex) on success, ``None`` if another
    instance already holds an unexpired lease. Reclaims expired
    leases via the same UPDATE so a crashed holder cannot block
    forever.
    """
    now = utc_now()
    expires_at = now + LEASE_TTL
    new_token = uuid.uuid4().hex

    stmt = (
        update(Collection)
        .where(
            and_(
                Collection.id == collection_id,
                # Acquire when (a) no current owner OR (b) existing
                # lease has expired. Atomic on the row — concurrent
                # instances race exactly one winner.
                (Collection.regen_lease_owner.is_(None)) | (Collection.regen_lease_expires_at < now),
            )
        )
        .values(
            regen_lease_owner=new_token,
            regen_lease_expires_at=expires_at,
        )
    )
    result = await session.execute(stmt)
    await session.commit()
    if result.rowcount:
        return new_token
    return None


async def _release_lease(
    session: AsyncSession,
    *,
    collection_id: str,
    lease_token: str,
) -> None:
    """Release the lease iff this caller still owns it. No-op if the
    token has already been reclaimed (crash recovery + stale renewer)."""
    stmt = (
        update(Collection)
        .where(
            and_(
                Collection.id == collection_id,
                Collection.regen_lease_owner == lease_token,
            )
        )
        .values(
            regen_lease_owner=None,
            regen_lease_expires_at=None,
        )
    )
    await session.execute(stmt)
    await session.commit()


# ---------------------------------------------------------------------
# Quality gates (per huangheng N4)
# ---------------------------------------------------------------------


_ALPHA_CHAR_RE = re.compile(r"[a-zA-Z一-鿿]")


def is_valid_summary(text: str | None) -> bool:
    """Length + keyword + alphabetic-char threshold.

    Catches agent/LLM error responses ("I cannot…", "tool error",
    "无法生成") and outputs that are technically non-empty but lack
    actual descriptive content.
    """
    if not text or len(text) < SUMMARY_MIN_CHARS:
        return False
    text_lower = text.lower()
    if any(bad in text_lower for bad in INVALID_OUTPUT_FRAGMENTS):
        return False
    # Require at least 50 alphabetic chars (latin or CJK), counted
    # across the whole string, to filter out responses that are
    # mostly punctuation / numbers.
    if len(_ALPHA_CHAR_RE.findall(text)) < 50:
        return False
    return True


def is_valid_description(text: str | None) -> bool:
    """Same shape as ``is_valid_summary`` but with description thresholds."""
    if not text or len(text) < DESCRIPTION_MIN_CHARS:
        return False
    text_lower = text.lower()
    if any(bad in text_lower for bad in INVALID_OUTPUT_FRAGMENTS):
        return False
    if len(_ALPHA_CHAR_RE.findall(text)) < 20:
        return False
    return True


# ---------------------------------------------------------------------
# Language detection (per huangheng N1 — language-aware prompts)
# ---------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Returns ``"zh"`` if CJK chars are the majority of alphabetic
    characters, else ``"en"``. Mixed collections fall back to the
    primary script."""
    if not text:
        return "en"
    cjk = len(re.findall(r"[一-鿿]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    if cjk > latin:
        return "zh"
    return "en"


# ---------------------------------------------------------------------
# Stage 1 — Summary regen
# ---------------------------------------------------------------------


# Type alias for the LLM callable contract used across stages.
LLMCall = Callable[[str], Awaitable[str]]


async def _invoke_summary_agent(collection: Collection) -> str | None:
    """Stage 1 Tier 1: agent-runtime free-explore.

    Wave 10 follow-up scaffold: returns ``None`` today so the caller
    falls through to Tier 2 (``_invoke_summary_chunks_fallback``).
    Filling this in requires the headless agent-runtime invocation
    pattern documented in design appendix A (fake Turn/Chat/Bot ORM
    construction via ``agent_runtime_manager.launch_turn``); that
    integration is sediment as Wave 10.1 follow-up + Wave 11 "agent
    runtime headless invocation API formalize" candidate.
    """
    return None


async def _invoke_summary_chunks_fallback(
    collection: Collection,
    *,
    llm: LLMCall,
) -> str | None:
    """Stage 1 Tier 2: read the first substantive chunk of an arbitrary
    indexed document in the collection, feed to LLM with a summary
    prompt, return the result.

    Scaffolded to demonstrate the contract; full chunks.jsonl wiring
    (object-store read + chunk filter + multi-doc aggregation) lands
    in Wave 10.1 follow-up alongside the agent-runtime upgrade.
    Today returns ``None`` so the caller falls through to Tier 3
    (transient skip) — the reconciler will retry on the next sweep
    once Wave 10.1 ships either tier.
    """
    return None


async def regen_summary(
    collection_id: str,
    *,
    llm_factory: Callable[[Collection], LLMCall] | None = None,
) -> bool:
    """Stage 1: regenerate ``Collection.summary`` for ``collection_id``.

    Returns ``True`` if a new summary was successfully written,
    ``False`` if the call was skipped (lease busy / collection deleted
    / all 3 tiers fell through to transient skip).

    Lease-protected: acquires the regen lease, runs the 3-tier
    fallback chain, writes ``summary`` + ``summary_updated_at``
    atomically, releases the lease in ``finally``.
    """
    async for session in get_async_session():
        # Step 1: acquire lease
        lease_token = await _try_acquire_lease(session, collection_id=collection_id)
        if lease_token is None:
            logger.debug("regen_summary skipped: lease busy for %s", collection_id)
            return False

        try:
            # Step 2: load collection (post-lease so we see latest state)
            stmt = select(Collection).where(Collection.id == collection_id)
            result = await session.execute(stmt)
            collection = result.scalar_one_or_none()
            if collection is None or collection.gmt_deleted is not None:
                logger.info(
                    "regen_summary skipped: collection %s missing or deleted",
                    collection_id,
                )
                return False

            # Step 3: 3-tier fallback chain
            summary: str | None = await _invoke_summary_agent(collection)
            tier_used = "agent"

            if not is_valid_summary(summary):
                # Build the LLM callable for Tier 2 fallback.
                llm = llm_factory(collection) if llm_factory else _default_llm_factory(collection)
                summary = await _invoke_summary_chunks_fallback(collection, llm=llm)
                tier_used = "chunks_fallback"

            if not is_valid_summary(summary):
                # Tier 3: transient skip — input not ready (e.g. doc
                # parse hasn't completed, chunks.jsonl absent).
                # We do NOT update summary_updated_at so the
                # reconciler picks the row up again next sweep.
                logger.info(
                    "regen_summary transient skip for %s: tier=%s, all tiers returned invalid",
                    collection_id,
                    tier_used,
                )
                return False

            # Step 4: atomic writeback
            now = utc_now()
            await session.execute(
                update(Collection)
                .where(Collection.id == collection_id)
                .values(
                    summary=summary,
                    summary_updated_at=now,
                    gmt_updated=now,
                )
            )
            await session.commit()
            logger.info(
                "regen_summary succeeded for %s via tier=%s (%d chars)",
                collection_id,
                tier_used,
                len(summary),
            )
            return True
        finally:
            await _release_lease(session, collection_id=collection_id, lease_token=lease_token)
    return False


# ---------------------------------------------------------------------
# Stage 2 — Description derive
# ---------------------------------------------------------------------


async def regen_description(
    collection_id: str,
    *,
    llm_factory: Callable[[Collection], LLMCall] | None = None,
) -> bool:
    """Stage 2: derive ``Collection.description`` from existing
    ``Collection.summary``. Returns ``True`` on success, ``False`` on
    skip (lease busy, summary missing, or LLM output invalid).

    Cheap path — single LLM call, no agent multi-turn. Caller MUST
    ensure ``summary`` is already populated; the OpenAPI handler returns
    400 if not.
    """
    async for session in get_async_session():
        lease_token = await _try_acquire_lease(session, collection_id=collection_id)
        if lease_token is None:
            logger.debug("regen_description skipped: lease busy for %s", collection_id)
            return False

        try:
            stmt = select(Collection).where(Collection.id == collection_id)
            result = await session.execute(stmt)
            collection = result.scalar_one_or_none()
            if collection is None or collection.gmt_deleted is not None:
                return False

            if not collection.summary:
                logger.info(
                    "regen_description skipped: collection %s has no summary yet",
                    collection_id,
                )
                return False

            language = _detect_language(collection.summary)
            template = DESCRIPTION_DERIVE_PROMPT_ZH if language == "zh" else DESCRIPTION_DERIVE_PROMPT_EN
            prompt = template.format(summary=collection.summary)

            llm = llm_factory(collection) if llm_factory else _default_llm_factory(collection)
            try:
                description = await asyncio.wait_for(llm(prompt), timeout=DESCRIPTION_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, Exception):
                logger.exception(
                    "regen_description LLM call failed for %s; transient skip",
                    collection_id,
                )
                return False

            if not is_valid_description(description):
                logger.info(
                    "regen_description quality gate failed for %s; transient skip",
                    collection_id,
                )
                return False

            now = utc_now()
            await session.execute(
                update(Collection)
                .where(Collection.id == collection_id)
                .values(
                    description=description.strip(),
                    description_updated_at=now,
                    gmt_updated=now,
                )
            )
            await session.commit()
            logger.info(
                "regen_description succeeded for %s (%d chars, lang=%s)",
                collection_id,
                len(description),
                language,
            )
            return True
        finally:
            await _release_lease(session, collection_id=collection_id, lease_token=lease_token)
    return False


# ---------------------------------------------------------------------
# Default LLM factory — uses the collection's configured completion model.
# ---------------------------------------------------------------------


def _default_llm_factory(collection: Collection) -> LLMCall:
    """Build an ``LLMCall`` closure from the collection's completion
    model config (mirror ``aperag.indexing.llm.build_collection_llm_callable``).

    Kept thin so unit tests can pass a stub factory and exercise the
    quality-gate / fallback / lease logic without an LLM dependency.
    """
    from aperag.indexing.llm import build_collection_llm_callable

    # ``build_collection_llm_callable`` returns a sync callable (no
    # awaitable wrapper); wrap it so the regen service can ``await``
    # uniformly across Stage 1 / Stage 2.
    sync_call = build_collection_llm_callable(collection)

    async def _async_llm(prompt: str) -> str:
        return await asyncio.to_thread(sync_call, prompt)

    return _async_llm


__all__ = [
    "is_valid_description",
    "is_valid_summary",
    "regen_description",
    "regen_summary",
]


# Silence unused imports for now — these will be wired in subsequent
# chunks (json: response envelope serialisation; Awaitable: typing).
_ = (Awaitable, json)
