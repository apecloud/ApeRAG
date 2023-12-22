import asyncio
import hashlib
import json
import time
import redis
import redis.asyncio as aredis
from datetime import datetime

from ninja import Router
from config import settings
import kubechat.chat.message

from kubechat.apps import QuotaType
from config.settings import MAX_CONVERSATION_COUNT
from kubechat.chat.history.redis import RedisChatMessageHistory
from kubechat.pipeline.keyword_pipeline import KeywordPipeline
from kubechat.db.ops import *
from kubechat.source.weixin.WXBizMsgCrypt import WXBizMsgCrypt
from kubechat.source.weixin.client import WeixinClient

from urllib.parse import unquote
import xml.etree.cElementTree as ET

logger = logging.getLogger(__name__)

router = Router()


async def check_quota_usage(user, conversation_limit):
    key = "conversation_history:" + user
    redis_client = aredis.Redis.from_url(settings.MEMORY_REDIS_URL)

    if await redis_client.exists(key):
        if int(await redis_client.get(key)) >= conversation_limit:
            return False
    return True


async def manage_quota_usage(user, conversation_limit):
    key = "conversation_history:" + user
    redis_client = aredis.Redis.from_url(settings.MEMORY_REDIS_URL)

    # already used kubechat today
    if await redis_client.exists(key):
        if int(await redis_client.get(key)) < conversation_limit:
            await redis_client.incr(key)
    # first time to use kubechat today
    else:
        now = datetime.now()
        end_of_today = datetime(now.year, now.month, now.day, 23, 59, 59)
        await redis_client.set(key, 1)
        await redis_client.expireat(key, int(end_of_today.timestamp()))


async def weixin_text_response(client, user, bot, query, msg_id):
    chat_id = user
    chat = await query_chat_by_peer(bot.user, ChatPeer.WEIXIN, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot=bot, peer_type=ChatPeer.WEIXIN, peer_id=chat_id)
        await chat.asave()

    history = RedisChatMessageHistory(session_id=str(chat.id), url=settings.MEMORY_REDIS_URL)
    collection = await sync_to_async(bot.collections.first)()
    response = ""

    pipeline = KeywordPipeline(bot=bot, collection=collection, history=history)
    use_default_token = pipeline.predictor.use_default_token

    conversation_limit = await query_user_quota(user, QuotaType.MAX_CONVERSATION_COUNT)
    if conversation_limit is None:
        conversation_limit = MAX_CONVERSATION_COUNT

    try:
        if use_default_token and conversation_limit:
            if not await check_quota_usage(bot.user, conversation_limit):
                error = f"conversation rounds have reached to the limit of {conversation_limit}"
                await client.send_message(error, user)
                return

        await client.send_message("KubeChat正在解答中，请稍候......", user)

        async for msg in pipeline.run(query, message_id=msg_id):
            response += msg

        max_length = 2048
        for i in range(0, len(response), max_length):
            message = response[i:i + max_length]
            await client.send_message(message, user)
    except Exception as e:
        logger.exception(e)
    finally:
        if use_default_token and conversation_limit:
            await manage_quota_usage(bot.user, conversation_limit)


async def weixin_card_response(client, user, bot, query, msg_id):
    chat_id = user
    chat = await query_chat_by_peer(bot.user, ChatPeer.WEIXIN, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot=bot, peer_type=ChatPeer.WEIXIN, peer_id=chat_id)
        await chat.asave()

    history = RedisChatMessageHistory(session_id=str(chat.id), url=settings.MEMORY_REDIS_URL)
    collection = await sync_to_async(bot.collections.first)()
    response = ""

    pipeline = KeywordPipeline(bot=bot, collection=collection, history=history)
    use_default_token = pipeline.predictor.use_default_token

    conversation_limit = await query_user_quota(user, QuotaType.MAX_CONVERSATION_COUNT)
    if conversation_limit is None:
        conversation_limit = MAX_CONVERSATION_COUNT

    try:
        if use_default_token and conversation_limit:
            if not await check_quota_usage(bot.user, conversation_limit):
                error = f"conversation rounds have reached to the limit of {conversation_limit}"
                await client.send_message(error, user)
                return

        task_id = int(time.time())
        resp, response_code = await client.send_card("KubeChat正在解答中，请稍候......", user, task_id)

        async for msg in KeywordPipeline(bot=bot, collection=collection, history=history).run(query, message_id=msg_id):
            response += msg

        await client.redis_client.set(f"{task_id}2message", response)
        await client.redis_client.set(f"{task_id}2msg_id", msg_id)
        await client.update_card(response, user, response_code)
    except Exception as e:
        logger.exception(e)
    finally:
        if use_default_token and conversation_limit:
            await manage_quota_usage(bot.user, conversation_limit)


async def weixin_feedback_response(client, user, bot, key, response_code, task_id):
    upvote = 1 if key == 1 else None
    downvote = 0 if key == 0 else None

    chat_id = user
    chat = await query_chat_by_peer(bot.user, ChatPeer.WEIXIN, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot=bot, peer_type=ChatPeer.WEIXIN, peer_id=chat_id)
        await chat.asave()

    msg_id = await client.redis_client.get(f"{task_id}2msg_id")
    await kubechat.chat.message.feedback_message(bot.user, chat.id, msg_id.decode(), upvote, downvote, "")

    message = await client.redis_client.get(f"{task_id}2message")
    await client.update_card(message.decode(), user, response_code, vote=key)


@router.get("/webhook/event")
async def callback(request, user, bot_id, msg_signature, timestamp, nonce, echostr):
    bot = await query_bot(user, bot_id)
    if bot is None:
        logger.warning("bot not found: %s", bot_id)
        return
    bot_config = json.loads(bot.config)
    weixin_config = bot_config.get("weixin")

    api_token = weixin_config.get("api_token")
    api_encoding_aes_key = weixin_config.get("api_encoding_aes_key")
    corp_id = weixin_config.get("corp_id")

    msg_signature = unquote(msg_signature)
    timestamp = unquote(timestamp)
    nonce = unquote(nonce)
    echostr = unquote(echostr)
    logger.info(f"receive echostr: {echostr}")

    wxcpt = WXBizMsgCrypt(api_token,
                          api_encoding_aes_key,
                          corp_id)
    ret, sEchoStr = wxcpt.VerifyURL(msg_signature, timestamp, nonce, echostr)
    if ret != 0:
        raise Exception(f"decrypt_messgae failed: {ret}")
    logger.info(f"decrypted echostr: {sEchoStr}")

    return int(sEchoStr)


@router.post("/webhook/event")
async def callback(request, user, bot_id, msg_signature, timestamp, nonce):
    bot = await query_bot(user, bot_id)
    if bot is None:
        logger.warning("bot not found: %s", bot_id)
        return
    bot_config = json.loads(bot.config)
    weixin_config = bot_config.get("weixin")

    agent_id = weixin_config.get("app_id")
    corp_secret = weixin_config.get("app_secret")
    corp_id = weixin_config.get("corp_id")
    api_token = weixin_config.get("api_token")
    api_encoding_aes_key = weixin_config.get("api_encoding_aes_key")

    msg_signature = unquote(msg_signature)
    timestamp = unquote(timestamp)
    nonce = unquote(nonce)

    message = request.body.decode()
    logger.info(f"receive message: {message}")

    # decrypt message
    wxcpt = WXBizMsgCrypt(sToken=api_token, sEncodingAESKey=api_encoding_aes_key, sReceiveId=corp_id)
    ret, decrypted_messgae = wxcpt.DecryptMsg(message, msg_signature, timestamp, nonce)
    decrypted_messgae = decrypted_messgae.decode()
    if ret != 0:
        raise Exception(f"decrypt_messgae failed: {message}")
    logger.info(f"decrypted message: {decrypted_messgae}")

    xml_tree = ET.fromstring(decrypted_messgae)
    user = xml_tree.find("FromUserName").text
    ctx = {
        "corpid": corp_id,
        "corpsecret": corp_secret,
        "agentid": agent_id,
    }
    client = WeixinClient(ctx)

    # answer the question from user
    if xml_tree.find("MsgType") is not None and xml_tree.find("MsgType").text == "text":
        # ignore duplicate messages
        msg_id = xml_tree.find("MsgId").text
        if await client.redis_client.exists(msg_id):
            logger.info(f"ignore duplicate messages: {msg_id}")
            return
        await client.redis_client.set(msg_id, decrypted_messgae)
        query = xml_tree.find("Content").text

        # asyncio.create_task(weixin_card_response(client, user, bot, query, msg_id))
        asyncio.create_task(weixin_text_response(client, user, bot, query, msg_id))

    # respond to the feedback from user
    elif xml_tree.find("MsgType") is not None and xml_tree.find("MsgType").text == "event":
        if xml_tree.find("Event") is not None and xml_tree.find("Event").text == "template_card_menu_event":
            key = xml_tree.find("EventKey").text
            response_code = xml_tree.find("ResponseCode").text
            task_id = xml_tree.find("TaskId").text

            asyncio.create_task(weixin_feedback_response(client, user, bot, int(key), response_code, task_id))

    return "success"


@router.get("/officaccount/webhook/event")
async def officaccount_callback(request, user, bot_id, signature, timestamp, nonce, echostr):
    bot = await query_bot(user, bot_id)
    if bot is None:
        logger.warning("bot not found: %s", bot_id)
        return
    bot_config = json.loads(bot.config)
    weixin_config = bot_config.get("weixin_official_account")
    api_token = weixin_config.get("api_token")

    signature = unquote(signature)
    timestamp = unquote(timestamp)
    nonce = unquote(nonce)
    logger.info(f"receive echostr: {echostr}")

    sortlist = [api_token, timestamp, nonce]
    sortlist.sort()
    sha = hashlib.sha1()
    sha.update("".join(sortlist).encode())
    signature2 = sha.hexdigest()

    if signature2 == signature:
        return int(echostr)


def generate_xml_response(to_user_name, from_user_name, create_time, msg_type, response):
    response = response.replace('\n', '&#xA;')

    resp = f"""<xml>\
                <ToUserName><![CDATA[{to_user_name}]]></ToUserName>\
                <FromUserName><![CDATA[{from_user_name}]]></FromUserName>\
                <CreateTime>{create_time}</CreateTime>\
                <MsgType><![CDATA[{msg_type}]]></MsgType>\
                <Content>{response}</Content>\
            </xml>"""

    return resp


async def weixin_officaccount_response(query, msg_id, to_user_name, bot):
    chat_id = to_user_name
    chat = await query_chat_by_peer(bot.user, ChatPeer.WEIXIN_OFFICIAL, chat_id)
    if chat is None:
        chat = Chat(user=bot.user, bot=bot, peer_type=ChatPeer.WEIXIN_OFFICIAL, peer_id=chat_id)
        await chat.asave()

    history = RedisChatMessageHistory(session_id=str(chat.id), url=settings.MEMORY_REDIS_URL)
    collection = await sync_to_async(bot.collections.first)()
    pipeline = KeywordPipeline(bot=bot, collection=collection, history=history)
    use_default_token = pipeline.predictor.use_default_token
    redis_client = aredis.Redis.from_url(settings.MEMORY_REDIS_URL)
    response = ""

    conversation_limit = await query_user_quota(to_user_name, QuotaType.MAX_CONVERSATION_COUNT)
    if conversation_limit is None:
        conversation_limit = MAX_CONVERSATION_COUNT

    try:
        if use_default_token and conversation_limit:
            if not await check_quota_usage(bot.user, conversation_limit):
                error = f"conversation rounds have reached to the limit of {conversation_limit}"
                logger.info(f"generate response failed, conversation rounds have reached to the limit")
                await redis_client.set(to_user_name + msg_id, error)
                return

        async for msg in pipeline.run(query, message_id=msg_id):
            response += msg
        logger.info(f"response:{response}")

        await redis_client.set(to_user_name + msg_id, response)
        logger.info(f"generate response success, restored in redis, key:{to_user_name + msg_id}")
    except Exception as e:
        logger.exception(e)
    finally:
        if use_default_token and conversation_limit:
            await manage_quota_usage(bot.user, conversation_limit)


@router.post("/officaccount/webhook/event")
async def officaccount_callback(request, user, bot_id, signature, timestamp, nonce, openid, encrypt_type, msg_signature):
    bot = await query_bot(user, bot_id)
    if bot is None:
        logger.warning("bot not found: %s", bot_id)
        return
    bot_config = json.loads(bot.config)
    weixin_config = bot_config.get("weixin_official_account")

    api_token = weixin_config.get("api_token")
    api_encoding_aes_key = weixin_config.get("api_encoding_aes_key")
    app_id = weixin_config.get("app_id")

    message = request.body.decode()
    logger.info(f"receive message: {message}")
    wxcpt = WXBizMsgCrypt(api_token, api_encoding_aes_key, app_id)
    ret, decrypted_messgae = wxcpt.DecryptMsg(message, msg_signature, timestamp, nonce)
    if ret != 0:
        raise Exception(f"decrypt_messgae failed: {message}")
    logger.info(f"decrypted message: {decrypted_messgae}")

    xml_tree = ET.fromstring(decrypted_messgae)
    user = xml_tree.find("FromUserName").text
    redis_client = redis.from_url(settings.MEMORY_REDIS_URL)

    # answer the question from user
    if xml_tree.find("MsgType") is not None and xml_tree.find("MsgType").text == "text":
        msg_id = xml_tree.find("MsgId").text
        if redis_client.exists(msg_id):
            logger.info(f"ignore duplicate messages: {msg_id}")
            return
        redis_client.set(msg_id, decrypted_messgae)

        to_user_name = xml_tree.find("FromUserName").text
        from_user_name = xml_tree.find("ToUserName").text
        create_time = int(time.time())
        query = xml_tree.find("Content").text
        logger.info(f"query: {query}")

        query_record = f"{user + query}_query"
        query_answer = user + query
        if redis_client.exists(query_record):
            if redis_client.exists(query_answer):
                response = redis_client.get(query_answer).decode()
            else:
                response = f"正在解答中，请稍后重新发送"
        else:
            asyncio.create_task(weixin_officaccount_response(query, msg_id, to_user_name, bot))
            redis_client.set(f"{to_user_name+msg_id}_query", 1)
            response = f"KubeChat已收到问题，请发送{msg_id}获取答案"

        resp = generate_xml_response(to_user_name, from_user_name, create_time, "text", response)
        return resp
