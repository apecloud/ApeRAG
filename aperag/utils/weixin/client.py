import logging
import time
from abc import ABC
from threading import Lock
from typing import Any, Dict

import aiohttp

from deeprag.chat.utils import get_async_redis_client
from deeprag.source.base import CustomSourceInitializationError

logger = logging.getLogger(__name__)


class WeixinClient(ABC):
    def __init__(self, ctx: Dict[str, Any]):
        self.corpid = ctx.get("corpid", None)
        if self.corpid is None:
            raise CustomSourceInitializationError("corpid is required")

        self.corpsecret = ctx.get("corpsecret")
        if self.corpsecret is None:
            raise CustomSourceInitializationError("corpsecret is required")

        self.agentid = ctx.get("agentid")
        if self.agentid is None:
            raise CustomSourceInitializationError("agentid is required")

        self.mutex = Lock()

        self.redis_client = get_async_redis_client()

    async def send_message_to_all(self, message):
        return await self.send_message(message, "@all")

    async def send_message(self, message, user):
        """
        https://developer.work.weixin.qq.com/document/path/90236#文本消息
        """
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send"
        data = {
            "touser": user,
            "msgtype": "text",
            "agentid": self.agentid,
            "text": {"content": message},
            "safe": 0,
        }

        return await self.post(url, data)

    async def send_card(self, message, user, task_id):
        """
        https://developer.work.weixin.qq.com/document/path/90236#模板卡片消息
        """
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send"
        template_card = await self.build_template_card(message, task_id)
        data = {
            "touser": user,
            "msgtype": "template_card",
            "agentid": self.agentid,
            "template_card": template_card
        }

        return await self.post(url, data)

    async def update_card(self, message, user, response_code, vote=None):
        """
        https://developer.work.weixin.qq.com/document/path/96459
        """
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/update_template_card"

        template_card = await self.build_template_card(message, update=True, vote=vote)
        data = {
            "userids": [user],
            "agentid": self.agentid,
            "response_code": response_code,
            "enable_id_trans": 1,
            "template_card": template_card
        }

        return await self.post(url, data)

    async def create_chat(self, name, owner, userlist: list[str], chat_id: str):
        """
        https://developer.work.weixin.qq.com/document/path/90245
        """
        url = "https://qyapi.weixin.qq.com/cgi-bin/appchat/create"
        data = {
            "name": name,
            "owner": owner,
            "userlist": userlist,
            "chatid": chat_id
        }

        return await self.post(url, data)

    async def update_chat(self, name, owner, add_userlist: list[str], del_userlist: list[str], chat_id: str):
        """
        https://developer.work.weixin.qq.com/document/path/98913
        """
        url = "https://qyapi.weixin.qq.com/cgi-bin/appchat/update"
        data = {
            "name": name,
            "owner": owner,
            "add_user_list": add_userlist,
            "del_user_list": del_userlist,
            "chatid": chat_id
        }

        return await self.post(url, data)

    async def send_chat(self, chat_id, messgae):
        """
        https://developer.work.weixin.qq.com/document/path/90248
        """
        url = "https://qyapi.weixin.qq.com/cgi-bin/appchat/send"
        data = {
            "chatid": chat_id,
            "msgtype": "text",
            "text": {"content": messgae},
            "safe": 0
        }

        return await self.post(url, data)

    async def post(self, url, data):
        with self.mutex:
            if await self.redis_client.exists("weixin_access_token"):
                self.access_token = await self.redis_client.get("weixin_access_token")
                self.access_token = self.access_token.decode()
            else:
                self.access_token = await self.get_access_token()

        timeout = aiohttp.ClientTimeout(connect=3)
        async with aiohttp.ClientSession(raise_for_status=True, timeout=timeout) as session:
            async with session.post(url=url, json=data, params={"access_token": self.access_token}) as r:
                if r.status != 200:
                    raise Exception(f"request failed: {r.text}")
                logger.info("send message to weixin success")

                r = await r.json()
                return r, r.get("response_code", None)

    async def build_template_card(self, message, task_id=None, update=False, vote=None):
        template_card = {
            "card_type": "text_notice",
            "source": {
                "icon_url": "https://cdn.kubeblocks.com/img/apecloud/favicon.png",
                "desc": "DeepRAG",
                "desc_color": 1
            },
            "action_menu": {
                "desc": "对此回答是否满意",
                "action_list": [
                    {"text": "满意", "key": "1"},
                    {"text": "不满意", "key": "0"}
                ]
            },
            "main_title": {
                "title": "欢迎使用 DeepRAG",
                "desc": "DeepRAG 是一款企业级智能知识库"
            },
            "sub_title_text": message,
            "horizontal_content_list": [],
            "jump_list": [],
            "card_action": {
                "type": 1,
                "url": "https://chat.kubeblocks.io"
            }
        }
        if not update:
            template_card.update({"task_id": task_id})
        else:
            if vote is None:
                template_card.update(
                    {
                        "horizontal_content_list":
                            [{
                                "keyname": "评价",
                                "value": "点击右上角可对本次回答进行评价"
                            }]
                    }
                )
            else:
                template_card.update(
                    {
                        "horizontal_content_list":
                            [{
                                "keyname": "评价",
                                "value": "感谢您对本次回答的认可" if vote == 1
                                else "感谢您的反馈，我们会努力改进的"
                            }]
                    }
                )

        return template_card

    async def get_access_token(self):
        """
        https://developer.work.weixin.qq.com/document/path/91039
        """
        params = {
            "corpid": self.corpid,
            "corpsecret": self.corpsecret,
        }
        url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"

        timeout = aiohttp.ClientTimeout(connect=3)
        async with aiohttp.ClientSession(raise_for_status=True, timeout=timeout) as session:
            async with session.get(url=url, params=params) as r:
                if r.status != 200:
                    raise Exception(f"request failed: {r.text}")
                logger.info("get access token success")

                resp = await r.json()
                if resp["errcode"] != 0:
                    raise Exception(f"request failed: {resp['errmsg']}")

                access_token = resp["access_token"]
                expires_in = resp["expires_in"]
                await self.redis_client.set("weixin_access_token", access_token)
                await self.redis_client.expireat("weixin_access_token", int(time.time()) + expires_in - 600)

                return access_token
