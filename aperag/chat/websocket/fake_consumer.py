import logging

from deeprag.pipeline.fake_pipeline import FakePipeline

from .base_consumer import BaseConsumer

logger = logging.getLogger(__name__)


class FakeConsumer(BaseConsumer):
    async def predict(self, query, **kwargs):
        pipeline = FakePipeline(bot=self.bot, collection=self.collection, history=self.history)
        async for msg in pipeline.run(query, gen_references=True, **kwargs):
            yield msg
