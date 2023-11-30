import logging

from .base_consumer import BaseConsumer
from kubechat.pipeline.pipeline import FakePipeline

logger = logging.getLogger(__name__)


class FakeConsumer(BaseConsumer):
    async def predict(self, query, **kwargs):
        pipeline = FakePipeline(bot=self.bot, collection=self.collection, history=self.history)
        async for msg in pipeline.run(query, gen_references=True, **kwargs):
            yield msg
