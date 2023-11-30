import json
import os
from http import HTTPStatus

import aiohttp

from kubechat.llm.base import Predictor, LLMConfigError, LLMAPIError
import dashscope
from dashscope import Generation


class QianWenPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = kwargs.get("model", "qwen-turbo")
        if self.model not in  ["qwen-turbo", "qwen-plus", "qwen-max"]:
            raise LLMConfigError("Please specify the correct model")

        self.api_key = kwargs.get("api_key", os.environ.get("QIANWEN_API_KEY", ""))
        if not self.api_key:
            raise LLMConfigError("Please specify the API KEY")

    @staticmethod
    def build_request_data(self, history, prompt, memory=False):
        return {
            "model": self.model,
            "input": {
                "messages": history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "result_format": "message",
                "incremental_output": "True"
            }
        }

    @staticmethod
    def build_request_headers(api_key):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "X-DashScope-SSE": "enable",
        }

    async def agenerate_stream(self, history, prompt, memory=False):
        data = self.build_request_data(self, history, prompt, memory)
        headers = self.build_request_headers(self.api_key)

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(self.url, json=data, headers=headers) as r:
                async for line in r.content:
                    if line == b'\n':
                        continue

                    key, value = line.decode("utf-8").split(":", 1)
                    if value[:-1] != "":     # 跳过回答中不该删的换行符
                        value = value[:-1]   # 删除格式里自带的换行符

                    if key == "data":
                        response = json.loads(value)
                        # if error occurs, output will return "code" "message" "request_id"
                        if "code" in response:
                            raise LLMAPIError('Request id: %s, error code: %s, error message: %s' % (
                                response['request_id'],  response['code'], response['message']))

                        output = response["output"]["choices"][0]
                        if output["finish_reason"] == "null":
                            yield output["message"]["content"]
                        elif output["finish_reason"] == "stop":
                            yield output["message"]["content"]
                            return

    def generate_stream(self, history, prompt, memory=False):
        responses = Generation.call(
            model=self.model,
            api_key= self.api_key,
            messages=history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}],
            result_format='message',  # set the result to be "message" format.
            stream=True,
            incremental_output=True  # get streaming output incrementally
        )
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                yield response.output.choices[0]['message']['content']
            else:
                raise LLMAPIError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,response.code, response.message))