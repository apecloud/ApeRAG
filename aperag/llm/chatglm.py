import os
import time

import aiohttp
import jwt
import requests
from zhipuai.utils.sse_client import SSEClient

from deeprag.llm.base import LLMConfigError, Predictor, logger


class ChatGLMPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = kwargs.get("endpoint", "https://open.bigmodel.cn/api/paas/v3/model-api")
        self.model = kwargs.get("model", "chatglm_turbo")

        self.temperature = kwargs.get("temperature", 0.01)
        self.top_p = kwargs.get("top_p", 0.7)

        if self.model not in ["chatglm_lite", "chatglm_std", "chatglm_pro", "chatglm_turbo"]:
            raise LLMConfigError("Please specify the correct model")

        self.api_key = kwargs.get("api_key", os.environ.get("GLM_API_KEY", ""))
        if not self.api_key:
            raise LLMConfigError("Please specify the API KEY")
        parts = self.api_key.split('.')
        if not (len(parts) == 2 and all(parts)):
            raise LLMConfigError("Please specify the correct API KEY")


    @staticmethod
    def provide_default_token():
        return bool(os.environ.get("GLM_API_KEY", ""))

    @staticmethod
    def build_request_data(self, history, prompt, memory=False):
        return {
            "prompt": history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p
        }

    @staticmethod
    def generate_token(apikey: str, exp_seconds: int):
        try:
            id, secret = apikey.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)

        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

    def build_request_headers(self, api_key):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.generate_token(api_key, 60),
            "accept": "text/event-stream",
        }

    async def _agenerate_stream(self, history, prompt, memory=False):

        url = "%s/%s/sse-invoke" % (self.endpoint, self.model)
        data = self.build_request_data(self, history, prompt, memory)
        headers = self.build_request_headers(self.api_key)

        timeout = aiohttp.ClientTimeout(connect=3)
        async with aiohttp.ClientSession(raise_for_status=True, timeout=timeout) as session:
            async with session.post(url, json=data, headers=headers) as r:
                async for line in r.content:
                    if line == b'\n':
                        continue

                    key, value = line.decode("utf-8").split(":", 1)
                    if value[:-1] != "":     # 跳过回答中不该删的换行符
                        value = value[:-1]   # 删除格式里自带的换行符

                    if key == "event":
                        if value == "add":
                            continue
                        elif value == "finish":
                            return
                        else:
                            logger.warning("ChatGLM API error")
                    elif key == "data":
                        yield value

    def _generate_stream(self, history, prompt, memory=False):
        url = "%s/%s/sse-invoke" % (self.endpoint, self.model)
        data = self.build_request_data(self, history, prompt, memory)
        headers = self.build_request_headers(self.api_key)

        response = requests.post(url, json=data, headers=headers, stream=True)
        response = SSEClient(response)

        for event in response.events():
            if event.event == "add":
                yield event.data
            elif event.event == "finish":
                return
            elif event.event == "error" or event.event == "interrupted":
                logger.warning("ChatGLM API error:%s" % event.data)
                return

    async def agenerate_stream(self, history, prompt, memory=False):
        async for tokens in self._agenerate_stream(history, prompt, memory):
            yield tokens

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(history, prompt, memory):
            yield tokens
