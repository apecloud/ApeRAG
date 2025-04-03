import logging
import re
from abc import ABC

from langchain import PromptTemplate

from deeprag.llm.base import Predictor, PredictorType
from deeprag.llm.prompts import QUESTION_EXTRACTION_PROMPT_TEMPLATE_V2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QuestionGenerator(ABC):
    def __init__(self, **kwargs):
        self.prompt_template = PromptTemplate(template=QUESTION_EXTRACTION_PROMPT_TEMPLATE_V2, input_variables=["context"])
        self.predictor = Predictor.from_model(model_name="gpt-4-1106-preview", predictor_type=PredictorType.CUSTOM_LLM, **kwargs)
                
    def gen_questions(self, text):
        prompt = self.prompt_template.format(context=text)
        response = ""
        for tokens in self.predictor.generate_stream([], prompt):
            response += tokens
        questions = re.sub(r'\n+', '\n', response).split('\n')
        for i, question in enumerate(questions):
            if question.startswith(str(i + 1) + "."):
                questions[i] = questions[i][2:].strip()
        return questions
