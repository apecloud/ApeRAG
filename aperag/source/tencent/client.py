import time
from abc import ABC
from threading import Lock
from typing import Any, Dict, Iterator

import redis
import requests

from config import settings
from deeprag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument
from deeprag.source.utils import gen_temporary_file


class TencentClient(ABC):
    def __init__(self, ctx: Dict[str, Any]):
        self.client_id = ctx.get("client_id", None)
        if self.client_id is None:
            raise CustomSourceInitializationError("client_id is required")

        self.client_secret = ctx.get("client_secret", None)
        if self.client_secret is None:
            raise CustomSourceInitializationError("client_secret is required")

        self.folder_id = ctx.get("folder_id", None)

        self.redirect_uri = ctx.get("redirect_uri", None)
        if self.redirect_uri is None:
            raise CustomSourceInitializationError("redirect_uri is required")

        self.code = ctx.get("code", None)
        if self.code is None:
            raise CustomSourceInitializationError("code is required")

        self.mutex = Lock()
        self.redis_client = redis.from_url(settings.MEMORY_REDIS_URL)

    def scan_documents(self, folder_id=None, source="download") -> Iterator[RemoteDocument]:
        """
        https://docs.qq.com/open/document/app/openapi/v2/file/folders/list.html
        """
        while True:
            if folder_id is None:
                folder_id = self.folder_id
            url = f"/openapi/drive/v2/folders/{folder_id}"
            resp = self.request(method="get", url=url)

            if resp.get("ret") != 0:
                raise Exception(f"request failed: {resp.get('msg')}")

            data = resp.get("data")
            docs = data["list"]
            for doc in docs:
                if doc.type == "doc":
                    yield self.get_file_metadata(doc["ID"], source=source)
                elif doc.type == "folder":
                    yield self.scan_documents(doc["ID"])

            if data["next"] == 0:
                break

    def get_file_metadata(self, file_id, source="download") -> RemoteDocument:
        """
        https://docs.qq.com/open/document/app/openapi/v2/file/files/metadata.html
        """
        url = f"https://docs.qq.com/openapi/drive/v2/files/{file_id}/metadata"
        resp = self.request(method="get", url=url)
        if resp.get("ret") != 0:
            raise Exception(f"request failed: {resp.get('msg')}")

        data = resp.get("data")
        metadata = {
            "ID": data["ID"],
            "url": data["url"],
            "modified_time": data["lastModifyTime"]
        }

        doc = RemoteDocument(
            name=data['title'] + ".pdf" if source == "download" else data['title'] + ".txt",
            size=0,
            metadata=metadata
        )
        return doc

    def prepare_document(self, name: str, metadata: Dict[str, Any], source="download") -> LocalDocument:
        file_id = metadata["ID"]

        if source == "online":
            content = self.get_file_online(file_id)
            temp_file = gen_temporary_file(name, default_suffix="txt")
        elif source == "download":
            content = self.get_file_download(file_id)
            temp_file = gen_temporary_file(name, default_suffix="pdf")
        else:
            raise Exception("unsupported file source")

        temp_file.write(content)
        temp_file.close()
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file.name, metadata=metadata)

    def get_file_online(self, file_id):
        """
        https://docs.qq.com/open/document/app/openapi/v3/doc/get/get.html
        """
        url = f"https://docs.qq.com/openapi/doc/v3/{file_id}"
        resp = self.request(method="get", url=url)

        text_values = []

        # 递归函数来遍历JSON数据
        def recursive_extract(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "text":
                        text_values.append(value)
                    else:
                        recursive_extract(value)
            elif isinstance(obj, list):
                for item in obj:
                    recursive_extract(item)

        recursive_extract(resp["document"])
        result = "\n".join(text_values)
        return result

    def get_file_download(self, file_id):
        operation_id = self.get_operation_id(file_id)
        download_url = self.get_download_url(file_id, operation_id)

        response = requests.get(download_url)

        if response.status_code != 200:
            raise Exception(f"request failed: {response.text}")

        return response.content

    def get_operation_id(self, file_id):
        """
        https://docs.qq.com/open/document/app/openapi/v2/file/export/async_export.html
        """
        url = f"https://docs.qq.com/openapi/drive/v2/files/{file_id}/async-export"
        data = {"exportType": "pdf"}

        resp = self.request(method="post", url=url, json=data)

        if resp.get("ret") != 0:
            raise Exception(f"request failed: {resp.get('msg')}")

        operation_id = resp["data"]["operationID"]
        return operation_id

    def get_download_url(self, file_id, operation_id):
        """
        https://docs.qq.com/open/document/app/openapi/v2/file/export/export_progress.html
        """
        url = f"https://docs.qq.com/openapi/drive/v2/files/{file_id}/export-progress"
        params = {"operationID": operation_id}

        download_url = ""
        progress = 0

        while progress != 100:
            time.sleep(1.0)
            resp = self.request(method="get", url=url, params=params)
            if resp.get("ret") != 0:
                raise Exception(f"request failed: {resp.get('msg')}")
            download_url = resp["data"].get("url")
            progress = resp["data"]["progress"]

        return download_url

    def request(self, method, url, **kwargs):
        with self.mutex:
            if self.redis_client.exists("tencent_access_token"):
                self.access_token = self.redis_client.get("tencent_access_token")
                self.access_token = self.access_token.decode()

                self.open_id = self.redis_client.get("tencent_open_id")
                self.open_id = self.open_id.decode()
            else:
                self.access_token, self.open_id = self.get_access_token()

        headers = {
                "Access-Token": self.access_token,
                "Client-Id": self.client_id,
                "Open-Id": self.open_id
            }
        if method.lower() == "post":
            headers.update({
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "14"
            })

        r = requests.request(method=method, url=url, headers=headers, **kwargs)
        if r.status_code != 200:
            raise Exception(f"request failed: {r.text}")

        return r.json()

    def get_access_token(self):
        """
        https://docs.qq.com/open/document/app/get_started.html
        """
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
            "code": self.code
        }
        url = "https://docs.qq.com/oauth/v2/token"

        r = requests.get(url=url, params=params)
        if r.status_code != 200:
            raise Exception(f"request failed: {r.text}")

        resp = r.json()

        access_token = resp["access_token"]
        expires_in = resp["expires_in"]
        open_id = resp["user_id"]
        self.redis_client.set("tencent_access_token", access_token)
        self.redis_client.set("tencent_open_id", open_id)
        self.redis_client.expireat("tencent_access_token", int(time.time()) + expires_in - 600)

        return access_token, open_id