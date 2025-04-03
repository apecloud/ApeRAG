import datetime
import json
import logging
from abc import ABC
from threading import Lock
from typing import Any, Dict

import requests
from pydantic import BaseModel

from aperag.source.base import CustomSourceInitializationError

logger = logging.getLogger(__name__)


class Space(BaseModel):
    description: str
    name: str
    id: str
    visibility: str


class FeishuNoPermission(Exception):
    """
    raised when user has no permission to call the Feishu API
    """


class FeishuPermissionDenied(Exception):
    """
    raised when user has no permission to access the Feishu resource
    """


class FeishuClient(ABC):
    def __init__(self, ctx: Dict[str, Any]):
        self.app_id = ctx.get("app_id", None)
        if self.app_id is None:
            raise CustomSourceInitializationError("app_id is required")

        self.app_secret = ctx.get("app_secret")
        if self.app_secret is None:
            raise CustomSourceInitializationError("app_secret is required")

        self.mutex = Lock()
        self.space_id = ctx.get("space_id")
        self.tenant_access_token = ""
        self.expire_at = datetime.datetime.now()

    def request(self, method, path, **kwargs):
        resp = self.raw_request(method, path, **kwargs)
        resp = resp.json()
        if resp["code"] != 0:
            raise Exception(f"request failed: {resp['msg']}")
        return resp

    def raw_request(self, method, path, **kwargs):
        url = f"https://open.feishu.cn/open-apis/{path}"
        logger.info("request feishu api: %s %s", method, url)
        with self.mutex:
            if self.expire_at - datetime.datetime.now() < datetime.timedelta(minutes=15):
                self.tenant_access_token, self.expire_at = self.get_tenant_access_token()
        headers = {"Authorization": f"Bearer {self.tenant_access_token}"}
        resp = requests.request(method, url, headers=headers, **kwargs)
        if resp.status_code != 200:
            if "No permission" in resp.json()["msg"]:
                raise FeishuNoPermission()
            if "permission denied" in resp.json()["msg"]:
                raise FeishuPermissionDenied()
            raise Exception(f"request failed: {resp.text}")
        return resp

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

    def patch(self, path, **kwargs):
        return self.request("PATCH", path, **kwargs)

    def get_user_access_token(self, code, redirect_uri):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        params = {
            "code": f"{code}",
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }
        resp = requests.post("https://passport.feishu.cn/suite/passport/oauth/token", params=params, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"request failed: {resp.text}")
        resp = resp.json()
        return resp["access_token"]

    def get_tenant_access_token(self):
        """
        https://open.feishu.cn/document/server-docs/authentication-management/access-token/tenant_access_token_internal
        """
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }
        resp = requests.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal", json=data)
        if resp.status_code != 200:
            raise Exception(f"request failed: {resp.text}")
        resp = resp.json()
        if resp["code"] != 0:
            raise Exception(f"request failed: {resp['msg']}")
        expired_at = datetime.datetime.now() + datetime.timedelta(seconds=resp["expire"])
        return resp["tenant_access_token"], expired_at

    def reply_card_message(self, message_id, data):
        """
        https://open.feishu.cn/document/server-docs/im-v1/message/reply
        """
        resp = self.post(f"im/v1/messages/{message_id}/reply", json=data)
        return resp["data"]["message_id"]

    def delay_update_card_message(self, data):
        """
        https://open.feishu.cn/document/server-docs/im-v1/message-card/delay-update-message-card
        """
        self.post("interactive/v1/card/update", json=data)

    def update_card_message(self, message_id, data):
        """
        https://open.feishu.cn/document/server-docs/im-v1/message-card/patch
        """
        self.patch(f"im/v1/messages/{message_id}", json=data)

    def send_message(self, chat_id, message):
        """
        https://open.feishu.cn/document/server-docs/im-v1/message/create
        """
        params = {"receive_id_type": "chat_id"}
        content = {
            "config": {
                "wide_screen_mode": True
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": message,
                }
            ]
        }
        data = {
            "receive_id": f"{chat_id}",
            "msg_type": "interactive",
            "content": json.dumps(content),
        }
        resp = self.post("im/v1/messages", params=params, json=data)
        return resp["data"]["message_id"]

    def get_spaces(self):
        """
        https://open.feishu.cn/document/server-docs/docs/wiki-v2/space/list
        """
        spaces = []
        resp = self.get("wiki/v2/spaces")
        for item in resp["data"]["items"]:
            spaces.append(Space(
                description=item["description"],
                name=item["name"],
                id=item["space_id"],
                visibility=item["visibility"],
            ))
        return spaces

    def get_space_nodes(self, space_id, parent_node_token=""):
        """
        https://open.feishu.cn/document/server-docs/docs/wiki-v2/space-node/list
        """
        nodes = []
        page_token = None
        while True:
            params = {
                "parent_node_token": parent_node_token,
                "page_size": 1,
            }
            if page_token is not None:
                params["page_token"] = page_token
            resp = self.get(f"wiki/v2/spaces/{space_id}/nodes", params=params)
            for node in resp["data"]["items"]:
                nodes.append(node)
            if not resp["data"]["has_more"]:
                break
            page_token = resp["data"]["page_token"]
        return nodes

    def get_node(self, token, **kwargs):
        """
        https://open.feishu.cn/document/server-docs/docs/wiki-v2/space-node/get_node
        """
        params = {
            "token": token,
        }
        resp = self.get("wiki/v2/spaces/get_node", params=params, **kwargs)
        return resp["data"]["node"]

    def get_docx_plain_content(self, doc_id):
        """
        https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/raw_content
        """
        if doc_id is None:
            raise Exception("doc_id is None")
        resp = self.get(f"docx/v1/documents/{doc_id}/raw_content?lang=0")
        return resp["data"]["content"]

    def get_doc_plain_content(self, doc_id):
        """
        https://open.feishu.cn/document/server-docs/docs/docs/docs/content/obtain-document-content
        """
        if doc_id is None:
            raise Exception("doc_id is None")
        resp = self.get(f"doc/v2/{doc_id}/raw_content")
        return resp["data"]["content"]

    def get_docx_blocks(self, doc_id):
        """
        https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/list
        """
        if doc_id is None:
            raise Exception("doc_id is None")
        resp = self.get(f"docx/v1/documents/{doc_id}/blocks")
        return resp["data"]["items"]

    def get_doc_blocks(self, doc_id):
        """
        https://open.feishu.cn/document/server-docs/docs/docs/docs/content/get-document
        """
        if doc_id is None:
            raise Exception("doc_id is None")
        resp = self.get(f"doc/v2/{doc_id}/content")
        return resp["data"]["content"]

    def create_export_task(self, doc_id, doc_type="docx", extension="pdf"):
        """
        https://open.feishu.cn/document/server-docs/docs/drive-v1/export_task/create
        """
        if doc_id is None:
            raise Exception("doc_id is None")
        data = {
            "type": doc_type,
            "token": doc_id,
            "file_extension": extension,
        }
        resp = self.post("drive/v1/export_tasks", json=data)
        return resp["data"]["ticket"]

    def query_export_task(self, ticket, doc_id):
        """
        https://open.feishu.cn/document/server-docs/docs/drive-v1/export_task/get
        """
        if ticket is None:
            raise Exception("ticket is None")
        params = {
            "token": doc_id,
        }
        resp = self.get(f"drive/v1/export_tasks/{ticket}", params=params)
        return resp["data"]["result"]

    def download_doc(self, token):
        """
        https://open.feishu.cn/document/server-docs/docs/drive-v1/export_task/download
        """
        if token is None:
            raise Exception("token is None")
        resp = self.raw_request("GET", f"drive/v1/export_tasks/file/{token}/download")
        return resp.content
