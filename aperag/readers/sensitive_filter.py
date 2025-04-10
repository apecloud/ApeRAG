import json
import logging
import subprocess
from abc import ABC
from typing import Dict, Tuple

from langchain_core.prompts import PromptTemplate

from config import settings
from aperag.db.models import ProtectAction
from aperag.llm.base import Predictor, PredictorType
from aperag.llm.prompts import CLASSIFY_SENSITIVE_INFORMATION_TEMPLATE, SENSITIVE_INFORMATION_TEMPLATE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SensitiveFilter(ABC):

    def __init__(self, **kwargs):
        self.prompt_template = PromptTemplate(template=SENSITIVE_INFORMATION_TEMPLATE, input_variables=["context", "types"])
        sensitive_filter_model = settings.SENSITIVE_FILTER_MODEL
        self.sensitive_protect_llm = False
        if sensitive_filter_model != '':
            self.sensitive_protect_llm = True
            self.predictor = Predictor.from_model(self.sensitive_filter_model, PredictorType.CUSTOM_LLM, **kwargs)

    def sensitive_filter_llm(self, context, types=["密码", "API-KEY", "special token"]):
        prompt = self.prompt_template.format(context=context, types=types)
        response = ""
        for tokens in self.predictor.generate_stream([], prompt):
            response += tokens

        try:
            start = response.find('[')
            end = response.rfind(']')
            raw_results = json.loads(response[start:end + 1])

            # check raw results
            results = []
            for result in raw_results:
                text = result.get("text", "")
                text_type = result.get("type", "")
                if not text or (text not in context) or text_type not in types:
                    continue
                while text in context:
                    span = result.get("span", [0, 0])
                    if context[span[0]:span[1]] != text:
                        span[0] = context.find(text)
                        span[1] = span[0] + len(text)
                    context = context[:span[0]] + "*" * len(text) + context[span[1]:]
                    results.append({"text": text, "span": span, "type": text_type})
        except Exception:
            return context, []

        return context, results

    def sensitive_filter(self, text: str, sensitive_protect_method: str) -> Tuple[str, Dict]:
        output_sensitive_info = {}
        output_text = text
        try:
            result = subprocess.run(['dlptool', text], capture_output=True, text=True)
            output = result.stdout.split('\n')
            dlp_num = int(output[0])
            dlp_outputs = []
            for line in output[1:dlp_num + 1]:
                dlp_outputs.append(json.loads(line))
            dlp_masktext = '\n'.join(output[dlp_num + 2:])
            if dlp_num > 0:
                output_sensitive_info = {"chunk": text, "masked_chunk": dlp_masktext, "sensitive_info": dlp_outputs}
                if sensitive_protect_method == ProtectAction.REPLACE_WORDS:
                    output_text = dlp_masktext

                # llm check
                if self.sensitive_protect_llm:
                    llm_masktext, llm_outputs = self.sensitive_filter_llm(text)
                    if len(llm_outputs) > 0:
                        output_sensitive_info = {"chunk": text, "masked_chunk": llm_masktext, "sensitive_info": llm_outputs}
                        if sensitive_protect_method == ProtectAction.REPLACE_WORDS:
                            output_text = llm_masktext
        except Exception as e:
            logger.error(f"sensitive filter failed:{e}")

        return output_text, output_sensitive_info


class SensitiveFilterClassify(ABC):

    def __init__(self, **kwargs):
        self.prompt_template = PromptTemplate(template=CLASSIFY_SENSITIVE_INFORMATION_TEMPLATE, input_variables=["context", "types"])
        sensitive_filter_model = settings.SENSITIVE_FILTER_MODEL
        self.sensitive_protect_llm = False
        if sensitive_filter_model != '':
            self.sensitive_protect_llm = True
            self.predictor = Predictor.from_model(self.sensitive_filter_model, PredictorType.CUSTOM_LLM, **kwargs)
    def sensitive_filter_llm(self, context, types=["密码", "API-KEY", "special token"]):
        prompt = self.prompt_template.format(context=context, types=types)
        response = ""
        is_sensitive = False
        for tokens in self.predictor.generate_stream([], prompt):
            response += tokens
        if "是" in response and "不" not in response:
            is_sensitive = True

        return is_sensitive

    def sensitive_filter(self, text: str, sensitive_protect_method: str) -> Tuple[str, Dict]:
        output_sensitive_info = {}
        output_text = text
        try:
            result = subprocess.run(['dlptool', text], capture_output=True, text=True)
            output = result.stdout.split('\n')
            dlp_num = int(output[0])
            dlp_outputs = []
            for line in output[1:dlp_num + 1]:
                dlp_outputs.append(json.loads(line))
            dlp_masktext = '\n'.join(output[dlp_num + 2:])
            is_sensitive = True

            if dlp_num > 0:
                # llm check
                if self.sensitive_protect_llm:
                    is_sensitive = self.sensitive_filter_llm(text)
                if is_sensitive:
                    output_sensitive_info = {"chunk": text, "masked_chunk": dlp_masktext, "sensitive_info": dlp_outputs}
                    if sensitive_protect_method == ProtectAction.REPLACE_WORDS:
                        output_text = dlp_masktext
        except Exception as e:
            logger.error(f"sensitive filter failed:{e}")

        return output_text, output_sensitive_info
