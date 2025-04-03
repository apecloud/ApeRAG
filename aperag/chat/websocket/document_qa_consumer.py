import logging

from asgiref.sync import sync_to_async

from deeprag.pipeline.knowledge_pipeline import KnowledgePipeline
from deeprag.readers.base_embedding import get_collection_embedding_model

from .base_consumer import BaseConsumer

logger = logging.getLogger(__name__)


class DocumentQAConsumer(BaseConsumer):
    async def connect(self):
        await super().connect()
        self.collection = await sync_to_async(self.bot.collections.first)()
        self.collection_id = self.collection.id
        self.embedding_model, self.vector_size = get_collection_embedding_model(self.collection)
        self.pipeline = KnowledgePipeline(bot=self.bot, collection=self.collection, history=self.history)
        self.free_tier = self.pipeline.predictor.trial

    async def predict(self, query, **kwargs):
        async for msg in self.pipeline.run(query, gen_references=True, **kwargs):
            yield msg
