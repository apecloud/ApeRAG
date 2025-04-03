import time
from typing import cast

from llama_index.schema import TextNode
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStoreQuery
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    utility,
)
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from aperag.vectorstore.connector import VectorStoreConnectorAdaptor


def test_qdrant_connector():
    ctx = {"url": "http://localhost", "port": 6333, "collection": "test"}
    c = VectorStoreConnectorAdaptor("qdrant", ctx)
    client = cast(QdrantClient, c.connector.client)

    try:
        client.get_collection("test")
    except UnexpectedResponse:
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=100, distance=Distance.COSINE),
        )

    print(c.connector.client.get_collections())


def test_opensearch_connector():
    ctx = {
        "url": "https://localhost",
        "port": 9200,
        "index": "test",
        "vector_size": 300,
    }
    auth = {
        "verify": False,  # Disable SSL verification
        "basic_auth": (
            "admin",
            "admin",
        ),  # The default username/password for the OpenSearch docker image
    }
    # auth = {}
    c = VectorStoreConnectorAdaptor("opensearch", ctx, auth=auth)
    client = cast(OpensearchVectorClient, c.connector.client)

    # Initialize an OpensearchVectorStore instance
    vector_store = OpensearchVectorStore(client=client)

    # Create some sample embedding results
    embedding_results = [
        NodeWithEmbedding(
            node=TextNode(text="Sample text 1", id_="1"),
            embedding=[0.2 + i / 10.0 for i in range(300)],
        ),
        NodeWithEmbedding(
            node=TextNode(text="Sample text 2", id_="2"),
            embedding=[0.3 + i / 10.0 for i in range(300)],
        ),
        # Add more embedding results as needed
    ]

    # Test adding embedding results to the index
    vector_store.add(embedding_results)

    # Test querying the index to ensure the added embeddings are present
    query_embedding = [0.2 + i / 10.0 for i in range(300)]
    query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
    query_result = vector_store.query(query)
    assert len(query_result.ids) == len(
        embedding_results
    )  # Ensure all added embeddings are retrieved

    # Test deleting a document from the index
    ref_doc_id = "1"
    vector_store.delete(ref_doc_id)
    time.sleep(5)  # Wait until the deletion is ok

    # Test querying the index again to ensure the deleted document is no longer present
    query_result = vector_store.query(query)
    assert (
        ref_doc_id not in query_result.ids
    )  # Ensure the deleted document is not retrieved


def test_milvus_connector():
    ctx = {"url": "http://localhost", "port": 19530, "collection": "test"}
    c = VectorStoreConnectorAdaptor("milvus", ctx)
    client = cast(MilvusClient, c.connector.client)
    fields = [
        FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=100,
        ),
        FieldSchema(name="random", dtype=DataType.DOUBLE),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8),
    ]

    schema = CollectionSchema(
        fields, "hello_milvus is the simplest demo to introduce the APIs"
    )
    print("Create collection `hello_milvus`")
    hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")
    has = utility.has_collection("hello_milvus")
    print(f"Does collection hello_milvus exist in Milvus: {has}")
    utility.drop_collection("hello_milvus")
    has = utility.has_collection("hello_milvus")
    print(
        f"drop collection \nDoes collection hello_milvus exist in Milvus after drop: {has}"
    )


def test_chroma_connector():
    ctx = {"collection_name": "test"}
    c = VectorStoreConnectorAdaptor("chroma", ctx)
    client = cast(ChromaClient, c.connector.client)

    collection = client.get_or_create_collection(name="test")

    print(c.connector.client.get_collection(name="test"))


def test_weaviate_connector():
    ctx = {"url": "http://localhost:8080"}
    c = VectorStoreConnectorAdaptor("weaviate", ctx)
    client = cast(WeaviateClient, c.connector.client)

    print(c.connector.client.schema.get())


if __name__ == "__main__":
    test_weaviate_connector()
