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
from http import HTTPStatus
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import settings
from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.schema import view_models
from aperag.schema.view_models import Bot, BotList
from aperag.utils.constant import QuotaType
from aperag.views.utils import fail, success, validate_bot_config


class BotService:
    """Bot service that handles business logic for bots"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def build_bot_response(self, bot: db_models.Bot, collection_ids: List[str]) -> view_models.Bot:
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

    async def create_bot(self, user: str, bot_in: view_models.BotCreate) -> view_models.Bot:
        # Check quota limit
        if settings.max_bot_count:
            bot_limit = await self.db_ops.query_user_quota(user, QuotaType.MAX_BOT_COUNT)
            if bot_limit is None:
                bot_limit = settings.max_bot_count
            if await self.db_ops.query_bots_count(user) >= bot_limit:
                return fail(HTTPStatus.FORBIDDEN, f"bot number has reached the limit of {bot_limit}")

        async def _create_operation(session):
            # Use DatabaseOps to create bot
            db_ops_session = AsyncDatabaseOps(session)
            bot = await db_ops_session.create_bot(
                user=user, title=bot_in.title, description=bot_in.description, bot_type=bot_in.type, config="{}"
            )

            collection_ids = []
            if bot_in.collection_ids is not None:
                for cid in bot_in.collection_ids:
                    collection = await self.db_ops.query_collection(user, cid)
                    if not collection:
                        raise ValueError(f"Collection {cid} not found")
                    if collection.status == db_models.CollectionStatus.INACTIVE:
                        raise ValueError(f"Collection {cid} is inactive")

                    await db_ops_session.create_bot_collection_relation(bot.id, cid)
                    collection_ids.append(cid)

            return self.build_bot_response(bot, collection_ids=collection_ids)

        try:
            result = await self.db_ops.execute_with_transaction(_create_operation)
            return success(result)
        except ValueError as e:
            status_code = HTTPStatus.NOT_FOUND if "not found" in str(e) else HTTPStatus.BAD_REQUEST
            return fail(status_code, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create bot: {str(e)}")

    async def list_bots(self, user: str) -> view_models.BotList:
        bots = await self.db_ops.query_bots([user])
        response = []

        # Use _execute_query pattern to get collection IDs for all bots safely
        async def _get_bot_collections_data(session):
            bot_responses = []
            for bot in bots:
                # Handle legacy model names
                bot_config = json.loads(bot.config)
                model = bot_config.get("model", None)
                if model in ["chatgpt-3.5", "gpt-3.5-turbo-instruct"]:
                    bot_config["model"] = "gpt-3.5-turbo"
                elif model == "chatgpt-4":
                    bot_config["model"] = "gpt-4"
                elif model in ["gpt-4-vision-preview", "gpt-4-32k", "gpt-4-32k-0613"]:
                    bot_config["model"] = "gpt-4-1106-preview"
                bot.config = json.dumps(bot_config)

                # Get collection IDs for this bot using the session
                collection_ids = await bot.collections(session, only_ids=True)
                bot_responses.append(self.build_bot_response(bot, collection_ids=collection_ids))
            return bot_responses

        response = await self.db_ops._execute_query(_get_bot_collections_data)
        return success(BotList(items=response))

    async def get_bot(self, user: str, bot_id: str) -> view_models.Bot:
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            return fail(HTTPStatus.NOT_FOUND, "Bot not found")

        # Use _execute_query pattern to get collection IDs safely
        async def _get_bot_collections(session):
            collection_ids = await bot.collections(session, only_ids=True)
            return collection_ids

        collection_ids = await self.db_ops._execute_query(_get_bot_collections)
        return success(self.build_bot_response(bot, collection_ids=collection_ids))

    async def update_bot(self, user: str, bot_id: str, bot_in: view_models.BotUpdate) -> view_models.Bot:
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            return fail(HTTPStatus.NOT_FOUND, "Bot not found")

        # Validate configuration
        new_config = json.loads(bot_in.config)
        model_service_provider = new_config.get("model_service_provider")
        model_name = new_config.get("model_name")
        memory = new_config.get("memory", False)
        llm_config = new_config.get("llm")

        msp_dict = await self.db_ops.query_msp_dict(user)
        if model_service_provider in msp_dict:
            msp = msp_dict[model_service_provider]
            api_key = msp.api_key
            # Get base_url from LLMProvider
            try:
                llm_provider = await async_db_ops.query_llm_provider_by_name(model_service_provider)
                base_url = llm_provider.base_url
            except Exception:
                return fail(HTTPStatus.BAD_REQUEST, f"LLMProvider {model_service_provider} not found")
            valid, msg = validate_bot_config(
                model_service_provider, model_name, base_url, api_key, llm_config, bot_in.type, memory
            )
            if not valid:
                return fail(HTTPStatus.BAD_REQUEST, msg)
        else:
            return fail(HTTPStatus.BAD_REQUEST, "Model service provider not found")

        async def _update_operation(session):
            # Use DatabaseOps to update bot
            db_ops_session = AsyncDatabaseOps(session)

            old_config = json.loads(bot.config)
            old_config.update(new_config)
            config_str = json.dumps(old_config)

            updated_bot = await db_ops_session.update_bot_by_id(
                user=user,
                bot_id=bot_id,
                title=bot_in.title,
                description=bot_in.description,
                bot_type=bot_in.type,
                config=config_str,
            )

            if not updated_bot:
                raise ValueError("Bot not found")

            # Handle collection relations update
            if bot_in.collection_ids is not None:
                # Delete old relations
                await db_ops_session.delete_bot_collection_relations(bot_id)

                # Add new relations
                for cid in bot_in.collection_ids:
                    collection = await self.db_ops.query_collection(user, cid)
                    if not collection:
                        raise ValueError(f"Collection {cid} not found")
                    if collection.status == db_models.CollectionStatus.INACTIVE:
                        raise ValueError(f"Collection {cid} is inactive")
                    await db_ops_session.create_bot_collection_relation(bot_id, cid)

            collection_ids = await updated_bot.collections(session, only_ids=True)
            return self.build_bot_response(updated_bot, collection_ids=collection_ids)

        try:
            result = await self.db_ops.execute_with_transaction(_update_operation)
            return success(result)
        except ValueError as e:
            status_code = HTTPStatus.NOT_FOUND if "not found" in str(e) else HTTPStatus.BAD_REQUEST
            return fail(status_code, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update bot: {str(e)}")

    async def delete_bot(self, user: str, bot_id: str) -> view_models.Bot:
        # First check if bot exists
        bot = await self.db_ops.query_bot(user, bot_id)
        if bot is None:
            return fail(HTTPStatus.NOT_FOUND, "Bot not found")

        async def _delete_operation(session):
            # Use DatabaseOps to delete bot
            db_ops_session = AsyncDatabaseOps(session)
            deleted_bot = await db_ops_session.delete_bot_by_id(user, bot_id)

            if not deleted_bot:
                raise ValueError("Bot not found")

            # Delete all relations
            await db_ops_session.delete_bot_collection_relations(bot_id)

            collection_ids = await deleted_bot.collections(session, only_ids=True)
            return self.build_bot_response(deleted_bot, collection_ids=collection_ids)

        try:
            result = await self.db_ops.execute_with_transaction(_delete_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete bot: {str(e)}")


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
bot_service = BotService()
