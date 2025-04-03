import hashlib
import json
import os
import time
import uuid

import aiohttp
import requests

from aperag.llm.base import LLMConfigError, Predictor, logger


class BaiChuanPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.url = kwargs.get("url", "https://api.baichuan-ai.com/v1/stream/chat")

        self.api_key = kwargs.get("api_key", os.environ.get("BAICHUAN_API_KEY", ""))
        if not self.api_key:
            raise LLMConfigError("Please specify the API Key")

        self.secret_key = kwargs.get("secret_key", os.environ.get("BAICHUAN_SECRET_KEY", ""))
        if not self.secret_key:
            raise LLMConfigError("Please specify the Secret Key")

        self.url = self.url.strip()
        self.api_key = self.api_key.strip()
        self.secret_key = self.secret_key.strip()


    @staticmethod
    def provide_default_token():
        return bool(os.environ.get("BAICHUAN_API_KEY", "")) and bool(os.environ.get("BAICHUAN_SECRET_KEY", ""))

    @staticmethod
    def calculate_md5(input_string):
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        encrypted = md5.hexdigest()
        return encrypted

    def build_request_data(self, history, prompt, memory=False):
        return {
            "model": "Baichuan2-53B",
            "messages": history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}],
            "parameters": {
                "temperature": self.temperature,
            }
        }

    def build_request_headers(self, data):
        json_data = json.dumps(data)
        time_stamp = int(time.time())
        signature = BaiChuanPredictor.calculate_md5(self.secret_key + json_data + str(time_stamp))
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "X-BC-Request-Id": str(uuid.uuid4()),
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }

    async def _agenerate_stream(self, history, prompt, memory=False):
        data = self.build_request_data(history, prompt, memory)
        headers = self.build_request_headers(data)
        timeout = aiohttp.ClientTimeout(connect=3)
        async with aiohttp.ClientSession(raise_for_status=True, timeout=timeout) as session:
            async with session.post(self.url, json=data, headers=headers, ssl=False) as r:
                async for line in r.content:
                    data = json.loads(line.decode("utf-8"))
                    if data["code"] != 0:
                        raise Exception("BaiChuan API error, code: %d msg: %s" % (data["code"], data["msg"]))

                    for msg in data["data"]["messages"]:
                        yield msg["content"]

    def _generate_stream(self, history, prompt, memory=False):
        data = self.build_request_data(history, prompt, memory)
        headers = self.build_request_headers(data)
        response = requests.post(self.url, data=data, headers=headers, stream=True)
        for chunk in response.iter_lines():
            if not chunk:
                continue
            data = json.loads(chunk.decode("utf-8"))
            if data["code"] != 0:
                logger.warning("BaiChuan API error, code: %d msg: %s" % (data["code"], data["msg"]))
                return

            for msg in data["data"]["messages"]:
                yield msg["content"]

    async def agenerate_stream(self, history, prompt, memory=False):
        async for tokens in self._agenerate_stream(history, prompt, memory):
            yield tokens

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(history, prompt, memory=False):
            yield tokens
