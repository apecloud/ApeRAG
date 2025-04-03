import json

import aiohttp
import requests

from deeprag.llm.base import Predictor


class CustomLLMPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "baichuan-13b")
        endpoint = kwargs.get("endpoint", "http://localhost:18000")
        self.url = "%s/generate_stream" % endpoint
        self.free_tier = False

    @staticmethod
    def provide_default_token():
        return False

    async def _agenerate_stream(self, history, prompt, memory=False):
        data = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "model": self.model,
            "stop": "\nSQLResult:",
        }

        timeout = aiohttp.ClientTimeout(connect=3)
        async with aiohttp.ClientSession(raise_for_status=True, timeout=timeout) as session:
            async with session.post(self.url, json=data) as r:
                while True:
                    line = await r.content.readuntil(b"\0")
                    if not line:
                        break
                    line = line.strip(b"\0")
                    yield json.loads(line.decode("utf-8"))["text"]

    def _generate_stream(self, history, prompt, memory=False):
        data = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "model": self.model,
            "stop": "\nSQLResult:",
        }

        response = requests.post(self.url, json=data, stream=True)
        for chunk in response.iter_lines(chunk_size=2048, decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                yield data["text"]

    async def agenerate_stream(self, history, prompt, memory=False):
        async for tokens in self._agenerate_stream(history, prompt, memory):
            yield tokens

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(history, prompt, memory):
            yield tokens
