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

import json
from datetime import datetime
from http import HTTPStatus

from sqlmodel import select

from aperag.apps import QuotaType
from aperag.config import SessionDep, settings
from aperag.db import models as db_models
from aperag.db.ops import (
    PagedQuery,
    query_bot,
    query_bots,
    query_bots_count,
    query_collection,
    query_msp_dict,
    query_user_quota,
)
from aperag.schema import view_models
from aperag.schema.view_models import Bot, BotList
from aperag.views.utils import fail, success, validate_bot_config


def build_bot_response(bot: db_models.Bot, collection_ids: list[str]) -> view_models.Bot:
    """Build Bot response object for API return."""
    return Bot(
        id=bot.id,
        title=bot.title,
        description=bot.description,
        type=bot.type,
        config=bot.config,
        collection_ids=collection_ids,
        created=bot.gmt_created.isoformat(),
        updated=bot.gmt_updated.isoformat(),
    )


async def create_bot(session: SessionDep, user, bot_in: view_models.BotCreate) -> view_models.Bot:
    if settings.max_bot_count:
        bot_limit = await query_user_quota(session, user, QuotaType.MAX_BOT_COUNT)
        if bot_limit is None:
            bot_limit = settings.max_bot_count
        if await query_bots_count(session, user) >= bot_limit:
            return fail(HTTPStatus.FORBIDDEN, f"bot number has reached the limit of {bot_limit}")
    bot = db_models.Bot(
        user=user,
        title=bot_in.title,
        type=bot_in.type,
        status=db_models.BotStatus.ACTIVE,
        description=bot_in.description,
        config="{}",
    )
    session.add(bot)
    await session.commit()
    await session.refresh(bot)
    collection_ids = []
    if bot_in.collection_ids is not None:
        for cid in bot_in.collection_ids:
            collection = await query_collection(session, user, cid)
            if not collection:
                return fail(HTTPStatus.NOT_FOUND, f"Collection {cid} not found")
            if collection.status == db_models.CollectionStatus.INACTIVE:
                return fail(HTTPStatus.BAD_REQUEST, f"Collection {cid} is inactive")
            relation = db_models.BotCollectionRelation(bot_id=bot.id, collection_id=cid)
            session.add(relation)
            await session.commit()
            collection_ids.append(cid)
    return success(build_bot_response(bot, collection_ids=collection_ids))


async def list_bots(session: SessionDep, user, pq: PagedQuery) -> view_models.BotList:
    pr = await query_bots(session, [user, settings.admin_user], pq)
    response = []
    for bot in pr.data:
        bot_config = json.loads(bot.config)
        model = bot_config.get("model", None)
        if model in ["chatgpt-3.5", "gpt-3.5-turbo-instruct"]:
            bot_config["model"] = "gpt-3.5-turbo"
        elif model == "chatgpt-4":
            bot_config["model"] = "gpt-4"
        elif model in ["gpt-4-vision-preview", "gpt-4-32k", "gpt-4-32k-0613"]:
            bot_config["model"] = "gpt-4-1106-preview"
        bot.config = json.dumps(bot_config)
        collection_ids = await bot.collections(session, only_ids=True)
        response.append(build_bot_response(bot, collection_ids=collection_ids))
    return success(BotList(items=response), pr=pr)


async def get_bot(session: SessionDep, user, bot_id) -> view_models.Bot:
    bot = await query_bot(session, user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    collection_ids = await bot.collections(session, only_ids=True)
    return success(build_bot_response(bot, collection_ids=collection_ids))


async def update_bot(session: SessionDep, user, bot_id, bot_in: view_models.BotUpdate) -> view_models.Bot:
    bot = await query_bot(session, user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    new_config = json.loads(bot_in.config)
    model_service_provider = new_config.get("model_service_provider")
    model_name = new_config.get("model_name")
    memory = new_config.get("memory", False)
    llm_config = new_config.get("llm")
    msp_dict = await query_msp_dict(session, user)
    if model_service_provider in msp_dict:
        msp = msp_dict[model_service_provider]
        base_url = msp.base_url
        api_key = msp.api_key
        valid, msg = validate_bot_config(
            model_service_provider, model_name, base_url, api_key, llm_config, bot_in.type, memory
        )
        if not valid:
            return fail(HTTPStatus.BAD_REQUEST, msg)
    else:
        return fail(HTTPStatus.BAD_REQUEST, "Model service provider not found")
    old_config = json.loads(bot.config)
    old_config.update(new_config)
    bot.config = json.dumps(old_config)
    bot.title = bot_in.title
    bot.type = bot_in.type
    bot.description = bot_in.description
    if bot_in.collection_ids is not None:
        # Soft delete old relations
        stmt = select(db_models.BotCollectionRelation).where(
            db_models.BotCollectionRelation.bot_id == bot.id, db_models.BotCollectionRelation.gmt_deleted is None
        )
        result = await session.exec(stmt)
        relations = result.all()
        for rel in relations:
            rel.gmt_deleted = datetime.utcnow()
            session.add(rel)
        await session.commit()
        # Add new relations
        for cid in bot_in.collection_ids:
            collection = await query_collection(session, user, cid)
            if not collection:
                return fail(HTTPStatus.NOT_FOUND, f"Collection {cid} not found")
            if collection.status == db_models.CollectionStatus.INACTIVE:
                return fail(HTTPStatus.BAD_REQUEST, f"Collection {cid} is inactive")
            relation = db_models.BotCollectionRelation(bot_id=bot.id, collection_id=cid)
            session.add(relation)
            await session.commit()
    session.add(bot)
    await session.commit()
    await session.refresh(bot)
    collection_ids = await bot.collections(session, only_ids=True)
    return success(build_bot_response(bot, collection_ids=collection_ids))


async def delete_bot(session: SessionDep, user, bot_id) -> view_models.Bot:
    bot = await query_bot(session, user, bot_id)
    if bot is None:
        return fail(HTTPStatus.NOT_FOUND, "Bot not found")
    bot.status = db_models.BotStatus.DELETED
    bot.gmt_deleted = datetime.utcnow()
    session.add(bot)
    await session.commit()
    # Soft delete all relations
    stmt = select(db_models.BotCollectionRelation).where(
        db_models.BotCollectionRelation.bot_id == bot.id, db_models.BotCollectionRelation.gmt_deleted is None
    )
    result = await session.exec(stmt)
    relations = result.all()
    for rel in relations:
        rel.gmt_deleted = datetime.utcnow()
        session.add(rel)
    await session.commit()
    collection_ids = await bot.collections(session, only_ids=True)
    return success(build_bot_response(bot, collection_ids=collection_ids))
