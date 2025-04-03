import asyncio
import json
import logging
import random
import string

from config import settings
from deeprag.context.context import ContextManager
from deeprag.utils.utils import generate_vector_db_collection_name

from .base_consumer import BaseConsumer

logger = logging.getLogger(__name__)


class EmbeddingConsumer(BaseConsumer):

    async def predict(self, query, **kwargs):
        bot_config = json.loads(self.bot.config)

        llm_config = bot_config.get("llm", {})
        score_threshold = llm_config.get("similarity_score_threshold", 0.5)
        topk = llm_config.get("similarity_topk", 3)

        vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        vector_db_collection_id = generate_vector_db_collection_name(
            self.collection_id
        )
        vectordb_ctx["collection"] = vector_db_collection_id

        ctx_manager = ContextManager(vector_db_collection_id, self.embedding_model, settings.VECTOR_DB_TYPE, vectordb_ctx)
        results = ctx_manager.query(query, score_threshold, topk)
        for result in results:
            yield result.text
            await asyncio.sleep(random.uniform(0.1, 0.5))

        yield "" + ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(100))


