import json
from http import HTTPStatus

from deeprag.chat.history.redis import RedisChatMessageHistory
from deeprag.chat.utils import get_async_redis_client
from deeprag.db.models import MessageFeedback, MessageFeedbackStatus
from deeprag.views.utils import fail


async def feedback_message(user, chat_id, message_id, upvote, downvote, revised_answer=None):
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
    data = {
        "question": msg["query"],
        "original_answer": msg.get("response", ""),
    }
    if upvote is not None:
        data["upvote"] = upvote
    if downvote is not None:
        data["downvote"] = downvote

    if revised_answer is not None:
        data["revised_answer"] = revised_answer
    data["status"] = MessageFeedbackStatus.PENDING
    collection_id = msg.get("collection_id", None)
    feedback, _ = await MessageFeedback.objects.aupdate_or_create(
        user=user, chat_id=chat_id, message_id=message_id, collection_id=collection_id,
        defaults=data
    )
    return feedback
