
from llama_index import LangchainEmbedding
from llama_index.data_structs import Node
from llama_index.schema import NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.types import NodeWithEmbedding

from aperag.readers.base_embedding import DocumentBaseEmbedding
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor


class QAEmbedding(DocumentBaseEmbedding):

    def __init__(
            self,
            question: str,
            answer: str,
            vector_store_adaptor: VectorStoreConnectorAdaptor,
            embedding_model: LangchainEmbedding = None,
            vector_size: int = 0,
    ):
        super().__init__(vector_store_adaptor, embedding_model, vector_size)
        self.question = question
        self.answer = answer

    def load_data(self):
        vector = self.embedding.embed_query(self.question)
        node = Node(
            text=self.answer,
        )
        node.relationships = {
            NodeRelationship.SOURCE: RelatedNodeInfo(
                node_id=node.node_id, metadata={"source": f"{self.question}"}
            )
        }
        return self.connector.store.add([NodeWithEmbedding(node=node, embedding=vector)])

