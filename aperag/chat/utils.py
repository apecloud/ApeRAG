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

from aperag.utils.utils import now_unix_milliseconds


def success_response(message_id, data, issql=False):
    return json.dumps(
        {
            "type": "message" if not issql else "sql",
            "id": message_id,
            "data": data,
            "timestamp": now_unix_milliseconds(),
        }
    )

def welcome_response(message_id, welcome_message):
    return json.dumps(
        {
            "type": "welcome",
            "id": message_id,
            "data": welcome_message,
            "timestamp": now_unix_milliseconds()
        }
    )
    
def fail_response(message_id, error):
    return json.dumps(
        {
            "type": "error",
            "id": message_id,
            "data": error,
            "timestamp": now_unix_milliseconds(),
        }
    )


def start_response(message_id):
    return json.dumps(
        {
            "type": "start",
            "id": message_id,
            "timestamp": now_unix_milliseconds(),
        }
    )

  
def stop_response(message_id, references, related_question=[], related_question_prompt='', memory_count=0, urls=[]):
    if references is None:
        references = []
    return json.dumps(
        {
            "type": "stop",
            "id": message_id,
            "data": references,
            "memoryCount": memory_count,
            "related_question_prompt": related_question_prompt,
            "related_question": related_question,
            "urls": urls,
            "timestamp": now_unix_milliseconds()
        }
    )


async def check_quota_usage(user, conversation_limit):
    key = "conversation_history:" + user
    redis_client = get_async_redis_client()

    if await redis_client.exists(key):
        if int(await redis_client.get(key)) >= conversation_limit:
            return False
    return True


async def manage_quota_usage(user, conversation_limit):
    key = "conversation_history:" + user
    redis_client = get_async_redis_client()

    # already used aperag today
    if await redis_client.exists(key):
        if int(await redis_client.get(key)) < conversation_limit:
            await redis_client.incr(key)
    # first time to use aperag today
    else:
        now = datetime.now()
        end_of_today = datetime(now.year, now.month, now.day, 23, 59, 59)
        await redis_client.set(key, 1)
        await redis_client.expireat(key, int(end_of_today.timestamp()))


async_redis_client = None
sync_redis_client = None


def get_async_redis_client():
    global async_redis_client
    if not async_redis_client:
        import redis.asyncio as redis

        from config.settings import MEMORY_REDIS_URL
        async_redis_client = redis.Redis.from_url(MEMORY_REDIS_URL)
    return async_redis_client


def get_sync_redis_client():
    global sync_redis_client
    if not sync_redis_client:
        import redis

        from config.settings import MEMORY_REDIS_URL
        sync_redis_client = redis.Redis.from_url(MEMORY_REDIS_URL)
    return sync_redis_client