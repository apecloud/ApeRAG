import logging
import re
from abc import ABC, abstractmethod

from langchain_core.prompts import PromptTemplate

from aperag.llm.base import Predictor, PredictorType
from aperag.llm.prompts import CHINESE_QA_EXTRACTION_PROMPT_TEMPLATE

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
