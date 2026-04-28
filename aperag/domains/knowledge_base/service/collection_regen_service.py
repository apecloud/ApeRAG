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
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Awaitable, Callable

from sqlalchemy import and_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from aperag.config import get_async_session
from aperag.domains.knowledge_base.db.models import Collection, Document
from aperag.domains.knowledge_base.service.regen_constants import (
    CHUNKS_FALLBACK_MAX_CHARS,
    CHUNKS_FALLBACK_MAX_DOCUMENTS,
    CHUNKS_FALLBACK_PROMPT_EN,
    CHUNKS_FALLBACK_PROMPT_ZH,
    DESCRIPTION_DERIVE_PROMPT_EN,
    DESCRIPTION_DERIVE_PROMPT_ZH,
    DESCRIPTION_MIN_CHARS,
    DESCRIPTION_TIMEOUT_SECONDS,
    INVALID_OUTPUT_FRAGMENTS,
    LEASE_TTL,
    SUMMARY_AGENT_SYSTEM_PROMPT,
    SUMMARY_MIN_CHARS,
    SUMMARY_TIMEOUT_SECONDS,
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


async def get_or_create_summary_bot_for_user(user_id: str):
    """Wave 10 §K.13 — fetch (or lazy-create) the user's hidden
    summary bot.

    Main path: register-time creation in
    ``aperag.app._BotInitOpsAdapter.create_default_bot_for_user``
    (per c1-extend-hide design ratify). Lazy fallback here is
    defense-in-depth: the register hook only logs on failure
    (``user_manager.py:137`` does not roll back the user), so a
    successful registration could still leave a user without a
    summary bot.

    Returns the bot row (always — raises only on transient DB
    failure, in which case the caller's lease/transient-skip
    handling takes over).
    """
    from sqlalchemy import and_

    from aperag.config import get_async_session
    from aperag.domains.conversation.db.models import Bot, BotStatus, BotType

    async for session in get_async_session():
        stmt = select(Bot).where(
            and_(
                Bot.user == user_id,
                Bot.type == BotType.SUMMARY,
                Bot.is_system.is_(True),
                Bot.gmt_deleted.is_(None),
            )
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            return existing

        # Lazy create — register-time hook missed this user. Same
        # transaction; partial unique index guards against races.
        bot = Bot(
            user=user_id,
            title="Summary Generation Bot",
            type=BotType.SUMMARY,
            description="System-managed bot for collection summary regen (Wave 10 §K.13).",
            status=BotStatus.ACTIVE,
            config='{"agent": {"system_prompt_template": null}}',
            is_system=True,
        )
        session.add(bot)
        try:
            await session.commit()
            await session.refresh(bot)
            return bot
        except Exception:
            # Concurrent caller raced us through the lazy path — one
            # of us wins, the other fetches the winner's row.
            await session.rollback()
            result = await session.execute(stmt)
            return result.scalar_one()
    raise RuntimeError("collection_regen_service: failed to acquire DB session for summary bot lookup")


async def _invoke_summary_agent(collection: Collection) -> str | None:
    """Stage 1 Tier 1: agent-runtime free-explore.

    Mirrors ``aperag/domains/evaluation/worker.py:114-180`` —
    real ``Bot`` / ``Chat`` / ``AgentTurn`` ORMs, fire-and-forget
    ``launch_turn``, poll terminal status, extract UIMessage parts.

    Returns the summary text on success, ``None`` if the agent didn't
    produce a usable output (failed / cancelled / lease conflict /
    empty answer); the caller falls through to Tier 2.
    """
    from aperag.domains.agent_runtime.db.models import AgentTurnStatus
    from aperag.domains.agent_runtime.runtime import agent_runtime_manager
    from aperag.domains.agent_runtime.schemas import CreateTurnRequest
    from aperag.domains.conversation.service.chat_service import chat_service_global
    from aperag.domains.knowledge_base.schemas import Collection as CollectionSchema

    bot = await get_or_create_summary_bot_for_user(collection.user)
    if bot is None:  # pragma: no cover — get_or_create raises on transient failure
        return None

    user_id = collection.user
    chat_view = await chat_service_global.create_chat(user_id, bot.id)
    chat_id = chat_view.id

    title = collection.title or collection.id
    query = (
        f"请为 collection `{title}` (id={collection.id}) 生成一段详细丰富的 summary。"
        f" 用提供的 read-only 工具自由探索 collection 内容后, 输出最终 summary 文本。"
    )
    turn_request = CreateTurnRequest(
        query=query,
        collections=[CollectionSchema(id=collection.id, title=title)],
    )

    chat, bot_orm, turn, _created = await agent_runtime_manager.turn_service.create_or_get_turn(
        user_id, chat_id, turn_request
    )

    lease_owner = await agent_runtime_manager.claim_turn(turn.id)
    if not lease_owner:
        logger.info(
            "_invoke_summary_agent: could not claim agent turn %s for collection %s",
            turn.id,
            collection.id,
        )
        return None

    agent_runtime_manager.launch_turn(
        turn=turn,
        chat=chat,
        bot=bot_orm,
        user=user_id,
        request=turn_request,
        lease_owner=lease_owner,
    )

    terminal_statuses = {
        AgentTurnStatus.COMPLETED.value,
        AgentTurnStatus.FAILED.value,
        AgentTurnStatus.CANCELLED.value,
    }

    deadline = time.monotonic() + SUMMARY_TIMEOUT_SECONDS
    final_status: str | None = None
    while True:
        current = await agent_runtime_manager.turn_service.db_ops.query_agent_turn(user_id, chat_id, turn.id)
        status_value = (
            current.status.value if current and hasattr(current.status, "value") else (current and current.status)
        )
        if status_value in terminal_statuses:
            final_status = status_value
            break
        if time.monotonic() >= deadline:
            try:
                await agent_runtime_manager.cancel_turn(turn.id)
            except Exception:
                logger.exception("cancel_turn failed for summary turn %s", turn.id)
            logger.info(
                "_invoke_summary_agent timed out after %ss for collection %s",
                SUMMARY_TIMEOUT_SECONDS,
                collection.id,
            )
            return None
        await asyncio.sleep(2)

    if final_status != AgentTurnStatus.COMPLETED.value:
        logger.info(
            "_invoke_summary_agent terminal=%s for collection %s",
            final_status,
            collection.id,
        )
        return None

    persisted = await agent_runtime_manager.uimessage_store.read(turn.id)
    parts = list(persisted.parts) if persisted and persisted.parts else []
    return _extract_answer_text(parts) or None


def _extract_answer_text(parts) -> str:
    """Join the assistant's ``TextPart`` contents into a single string."""
    from aperag.domains.agent_runtime.uimessage import TextPart

    chunks = [part.text for part in parts if isinstance(part, TextPart) and part.text]
    return "".join(chunks).strip()


async def _invoke_summary_chunks_fallback(
    collection: Collection,
    *,
    llm: LLMCall,
) -> str | None:
    """Stage 1 Tier 2: read substantive chunks from active vector
    indexes for documents in the collection, stitch them, and feed
    to a single LLM call to produce a summary.

    Returns the LLM output on success, ``None`` if no chunks are
    available (parse hasn't completed) or the LLM call itself fails.
    The caller falls through to Tier 3 (transient skip) so the
    reconciler retries on the next sweep.
    """
    chunks_text = await _stitch_collection_chunks(collection)
    if not chunks_text:
        logger.info(
            "_invoke_summary_chunks_fallback: no chunks available yet for %s",
            collection.id,
        )
        return None

    language = _detect_language(chunks_text)
    template = CHUNKS_FALLBACK_PROMPT_ZH if language == "zh" else CHUNKS_FALLBACK_PROMPT_EN
    prompt = template.format(
        collection_title=collection.title or collection.id,
        chunks_text=chunks_text,
    )

    try:
        result = await asyncio.wait_for(llm(prompt), timeout=SUMMARY_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, Exception):
        logger.exception(
            "_invoke_summary_chunks_fallback LLM call failed for %s",
            collection.id,
        )
        return None
    return result.strip() if result else None


async def _stitch_collection_chunks(collection: Collection) -> str:
    """Read ``chunks.jsonl`` for documents in the collection and stitch
    a representative concatenation, capped at ``CHUNKS_FALLBACK_MAX_CHARS``.

    Picks documents in deterministic order (by ``Document.id``), reads
    each document's active vector ``DocumentIndex.source_path``, and
    pulls the first substantive chunk (length > 200 chars) from each.
    Returns ``""`` if no chunks are available.
    """
    from aperag.indexing.models import DocumentIndex, IndexStatus, Modality
    from aperag.indexing.parser import read_chunks
    from aperag.objectstore.base import get_object_store

    async for session in get_async_session():
        doc_stmt = (
            select(Document.id)
            .where(
                and_(
                    Document.collection_id == collection.id,
                    Document.gmt_deleted.is_(None),
                )
            )
            .order_by(Document.id)
            .limit(CHUNKS_FALLBACK_MAX_DOCUMENTS)
        )
        doc_rows = (await session.execute(doc_stmt)).all()
        if not doc_rows:
            return ""

        document_ids = [row[0] for row in doc_rows]
        index_stmt = select(DocumentIndex.document_id, DocumentIndex.source_path).where(
            and_(
                DocumentIndex.document_id.in_(document_ids),
                DocumentIndex.modality == Modality.VECTOR.value,
                DocumentIndex.status == IndexStatus.ACTIVE.value,
                DocumentIndex.is_serving.is_(True),
            )
        )
        index_rows = (await session.execute(index_stmt)).all()
        # Deterministic order: re-key by document_id so ordering matches
        # the doc_stmt sort.
        path_by_doc = {doc_id: src for doc_id, src in index_rows}
        if not path_by_doc:
            return ""

        store = get_object_store()
        collected: list[str] = []
        total = 0
        for doc_id in document_ids:
            chunks_path = path_by_doc.get(doc_id)
            if not chunks_path:
                continue
            try:
                chunks = await asyncio.to_thread(read_chunks, store, chunks_path)
            except Exception:
                logger.exception(
                    "_stitch_collection_chunks: read_chunks failed for %s (doc %s)",
                    chunks_path,
                    doc_id,
                )
                continue
            chunk_text = _pick_substantive_chunk_text(chunks)
            if not chunk_text:
                continue
            remaining = CHUNKS_FALLBACK_MAX_CHARS - total
            if remaining <= 0:
                break
            snippet = chunk_text[:remaining]
            collected.append(snippet)
            total += len(snippet)

        return "\n\n---\n\n".join(collected) if collected else ""
    return ""


def _pick_substantive_chunk_text(chunks: list[dict]) -> str | None:
    """Return the first chunk whose ``text`` is at least 200 chars."""
    for chunk in chunks:
        text = chunk.get("text")
        if isinstance(text, str) and len(text) >= 200:
            return text
    # Fall back to the longest available chunk if none cross the
    # threshold — a small collection still produces a usable signal.
    candidates = [c.get("text") for c in chunks if isinstance(c.get("text"), str)]
    candidates = [t for t in candidates if t]
    if not candidates:
        return None
    return max(candidates, key=len)


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
    """Return the collection's configured async LLM callable.

    ``build_collection_llm_callable`` already returns an async
    ``(prompt) -> str`` closure (per ``aperag/indexing/llm.py``);
    we surface it directly so the regen service can ``await`` it
    uniformly across Stage 1 / Stage 2.
    """
    from aperag.indexing.llm import build_collection_llm_callable

    return build_collection_llm_callable(collection)


__all__ = [
    "SUMMARY_AGENT_SYSTEM_PROMPT",
    "get_or_create_summary_bot_for_user",
    "is_valid_description",
    "is_valid_summary",
    "regen_description",
    "regen_summary",
]


# Silence unused imports flagged by ruff that are only used for typing.
_ = Awaitable
