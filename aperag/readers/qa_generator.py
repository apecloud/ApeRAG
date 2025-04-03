import logging
import re
from abc import ABC, abstractmethod

from langchain import PromptTemplate

from deeprag.llm.base import Predictor, PredictorType
from deeprag.llm.prompts import CHINESE_QA_EXTRACTION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QAGenerator(ABC):
    @abstractmethod
    def gen_qa_pairs(self, text):
        pass


class AlgoletQAGenerator(QAGenerator):

    def __init__(self):
        from question_generation import pipeline
        self.qg = pipeline("question-generation", device="cpu")
        self.qa = pipeline("question-answering", device="cpu")

    def gen_qa_pairs(self, text):
        result = []
        questions = self.qg(text)
        if len(questions) == 0:
            return result

        answers = self.qa([
            {
                "question": question,
                "context": text
            }
            for question in questions
        ])
        for idx, q in enumerate(questions):
            if "answer" not in answers[idx]:
                logger.warning("no answer for question [%s]" % q)
                continue
            result.append((q, answers[idx]['answer']))
        return result


class BaiChuanQAGenerator(QAGenerator):
    def __init__(self, **kwargs):
        self.prompt_template = PromptTemplate(template=CHINESE_QA_EXTRACTION_PROMPT_TEMPLATE, input_variables=["context"])
        self.predictor = Predictor.from_model(model_name="baichuan-13b", predictor_type=PredictorType.CUSTOM_LLM, **kwargs)
        self.q_pattern = r"问题.*：(.*)"
        self.a_pattern = r"答.*?：(.+?)(?=问题|$)"

    def gen_qa_pairs(self, text):
        prompt = self.prompt_template.format(context=text)
        response = ""
        for tokens in self.predictor.generate_stream(prompt):
            response += tokens
        q_matches = re.findall(self.q_pattern, response)
        a_matches = re.findall(self.a_pattern, response, re.DOTALL)
        if len(q_matches) != len(a_matches):
            logger.warning("number of questions and answers not match")
            return []
        pairs = []
        for idx, q_match in enumerate(q_matches):
            pairs.append((q_match, a_matches[idx]))
        return pairs

