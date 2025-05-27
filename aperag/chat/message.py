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

from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import get_async_redis_client
from aperag.db.models import MessageFeedback
from aperag.views.utils import fail


async def feedback_message(user, chat_id, message_id, feedback_type=None, feedback_tag=None, feedback_message=None):
    history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    msg = None
    for message in await history.messages:
        item = json.loads(message.content)
        if item["id"] != message_id:
            continue
        if message.additional_kwargs.get("role", "") != "ai":
            continue
        msg = item
    if msg is None:
        return fail(HTTPStatus.NOT_FOUND, "Message not found")

    # If feedback_type is None, delete the feedback record
    if feedback_type is None:
        try:
            feedback = await MessageFeedback.objects.aget(user=user, chat_id=chat_id, message_id=message_id)
            await feedback.adelete()
            return None
        except MessageFeedback.DoesNotExist:
            return None

    # Otherwise create or update the feedback record
    data = {
        "question": msg["query"],
        "original_answer": msg.get("response", ""),
        "type": feedback_type,
        "tag": feedback_tag,
        "message": feedback_message,
    }

    data["status"] = MessageFeedback.Status.PENDING
    collection_id = msg.get("collection_id", None)
    feedback, _ = await MessageFeedback.objects.aupdate_or_create(
        user=user, chat_id=chat_id, message_id=message_id, collection_id=collection_id, defaults=data
    )
    return feedback
