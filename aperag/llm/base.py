import json
import logging
from abc import ABC, abstractmethod
from enum import Enum

import requests
import yaml

from config import settings

logger = logging.getLogger(__name__)


class PredictorType(Enum):
    KB_VLLM = "kb-vllm"
    CUSTOM_LLM = "custom-llm"


class LLMConfigError(Exception):
    """
    LLMConfigError is raised when the LLM config is invalid.
    """


class LLMAPIError(Exception):
    """
    LLMAPIError is raised when error occurs when calling LLM API.
    """


class Predictor(ABC):

    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.1)
        self.trial = kwargs.get("trial", False)

    @abstractmethod
    async def agenerate_stream(self, history, prompt, memory):
        pass

    @abstractmethod
    def generate_stream(self, history, prompt, memory):
        pass

    @staticmethod
    def get_model_context(model_name):
        model_families = yaml.safe_load(settings.MODEL_FAMILIES)
        for model_family in model_families:
            for model_server in model_family.get("models", []):
                if model_name == model_server["name"]:
                    return model_server
        return None

    @staticmethod
    def match_predictor(model_name, predictor_type, kwargs):
        match model_name:
            case "chatgpt-3.5":
                kwargs["model"] = "gpt-3.5-turbo"
                from deeprag.llm.openai import OpenAIPredictor
                return OpenAIPredictor
            case "gpt-3.5-turbo-1106" | "gpt-3.5-turbo" | "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-instruct":
                from deeprag.llm.openai import OpenAIPredictor
                return OpenAIPredictor
            case "chatgpt-4":
                kwargs["model"] = "gpt-4"
                from deeprag.llm.openai import OpenAIPredictor
                return OpenAIPredictor
            case "gpt-4-1106-preview" | "gpt-4-vision-preview" | "gpt-4" | "gpt-4-32k" | "gpt-4-0613" | "gpt-4-32k-0613":
                from deeprag.llm.openai import OpenAIPredictor
                return OpenAIPredictor
            case "azure-openai":
                from deeprag.llm.azure import AzureOpenAIPredictor
                return AzureOpenAIPredictor
            case "baichuan-53b":
                from deeprag.llm.baichuan import BaiChuanPredictor
                return BaiChuanPredictor
            case "ernie-bot-turbo":
                from deeprag.llm.wenxin import BaiduQianFan
                return BaiduQianFan
            case "chatglm-pro" | "chatglm-std" | "chatglm-lite" | "chatglm-turbo":
                kwargs["model"] = model_name.replace("-", "_")
                from deeprag.llm.chatglm import ChatGLMPredictor
                return ChatGLMPredictor
            case "qwen-turbo" | "qwen-plus" | "qwen-max":
                from deeprag.llm.qianwen import QianWenPredictor
                return QianWenPredictor

        endpoint = kwargs.get("endpoint", "")
        if not endpoint:
            ctx = Predictor.get_model_context(model_name)
            if not ctx:
                raise Exception("No model server available for model: %s" % model_name)
            endpoint = ctx.get("endpoint", "")
            kwargs["endpoint"] = endpoint
            predictor_type = ctx.get("type", PredictorType.CUSTOM_LLM)

        match predictor_type:
            case PredictorType.KB_VLLM:
                return KubeBlocksLLMPredictor
            case PredictorType.CUSTOM_LLM:
                from deeprag.llm.custom import CustomLLMPredictor
                return CustomLLMPredictor
            case _:
                raise Exception("Unsupported predictor type: %s" % predictor_type)

    @staticmethod
    def from_model(model_name, predictor_type="", **kwargs):
        predictor = Predictor.match_predictor(model_name, predictor_type, kwargs)
        return predictor(**kwargs)

    @staticmethod
    def check_default_token(model_name, predictor_type="", **kwargs):
        predictor = Predictor.match_predictor(model_name, predictor_type, kwargs)
        return predictor.provide_default_token()

    def get_latest_history(self, messages, limit_length, limit_count, use_ai_memory) -> str:
        latest_history = []
        length = 0
        count = 0

        for message in reversed(messages):
            if message.additional_kwargs["role"] == "human":
                history = {"role": "user", "content": json.loads(message.content)["query"] + "\n"}
            else:
                if not use_ai_memory:
                    continue
                history = {"role": "assistant", "content": json.loads(message.content)["response"] + "\n"}
            count += 1

            if count >= limit_count or length + len(history["content"]) > limit_length:
                break

            latest_history.append(history)
            length += len(history["content"])

        if len(latest_history) > 0 and latest_history[-1]["role"] == "assistant":
            latest_history = latest_history[:-1]

        return latest_history[::-1]


class KubeBlocksLLMPredictor(Predictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = kwargs.get("endpoint", "http://localhost:18000")
        self.use_deafult_token = False

    @staticmethod
    def provide_default_token():
        return False

    def _generate_stream(self, history, prompt, memory=False):
        input = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        response = requests.post("%s/generate" % self.endpoint, json=input, stream=True)
        output = prompt
        for chunk in response.iter_lines():
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                tokens = data["text"][0][len(output):]
                yield tokens
                output = data["text"][0]

    async def agenerate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(prompt):
            yield tokens

    def generate_stream(self, history, prompt, memory=False):
        for tokens in self._generate_stream(prompt):
            yield tokens


