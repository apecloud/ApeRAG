import json
import random
from pathlib import Path

from aperag.pipeline.base_pipeline import Pipeline

KUBE_CHAT_DOC_QA_REFERENCES = "|KUBE_CHAT_DOC_QA_REFERENCES|"


class FakePipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        words_path = kwargs.get("words_path", Path(__file__).parent / "words_dictionary.json")
        with open(words_path) as fd:
            self.words = list(json.loads(fd.read()).keys())

    def sentence_generator(self, batch=1, min_len=3, max_len=10):
        for i in range(batch):
            tokens = []
            for j in range(random.randint(min_len, max_len)):
                tokens.append(random.choice(self.words))
            yield " ".join(tokens)

    async def run(self, message, gen_references=False, message_id=""):
        await self.add_human_message(message, message_id)

        response = ""
        for sentence in self.sentence_generator(batch=5, min_len=10, max_len=30):
            if random.sample([True, False], 1):
                sentence += "\n\n"
            yield sentence
            response += sentence

        references = []
        for result in range(3):
            ref = ""
            for sentence in self.sentence_generator(batch=5, min_len=20, max_len=50):
                if random.sample([True, False], 1):
                    sentence += "\n\n"
                ref += sentence
            references.append({
                "score": round(random.uniform(0.5, 0.6), 2),
                "text": ref,
                "metadata": {"source": ref[:20]},
            })

        await self.add_ai_message(message, message_id, response, references, urls=[])

        if gen_references:
            yield KUBE_CHAT_DOC_QA_REFERENCES + json.dumps(references)
