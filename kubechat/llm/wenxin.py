import os

import qianfan

from kubechat.llm.base import Predictor, LLMConfigError


class BaiduQianFan(Predictor):
    """
    https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = "ERNIE-Bot-turbo"

        self.api_key = kwargs.get("api_key", os.environ.get("QIANFAN_API_KEY", ""))
        if not self.api_key:
            raise LLMConfigError("Please specify the API Key")

        self.secret_key = kwargs.get("secret_key", os.environ.get("QIANFAN_SECRET_KEY", ""))
        if not self.secret_key:
            raise LLMConfigError("Please specify the Secret Key")

        self.chat_comp = qianfan.ChatCompletion(ak=self.api_key, sk=self.secret_key)

    async def agenerate_stream(self, history, prompt, memory=False):
        resp = await self.chat_comp.ado(model=self.model,
                                        stream=True,
                                        temperature=self.temperature,
                                        messages=history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}]
                                        )
        async for chunk in resp:
            yield chunk["result"]

    def generate_stream(self, history, prompt, memory=False):
        resp = self.chat_comp.do(model=self.model,
                                 stream=True,
                                 temperature=self.temperature,
                                 messages=history + [{"role": "user", "content": prompt}] if memory else [{"role": "user", "content": prompt}],
                                 )
        yield resp['body']['result']
