import json

from config.settings import VECTOR_DB_CONTEXT, VECTOR_DB_TYPE
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor


def get_vector_db_connector(collection: str) -> VectorStoreConnectorAdaptor:
    # todo: specify the collection for different user
    # one person one collection
    ctx = json.loads(VECTOR_DB_CONTEXT)
    ctx["collection"] = collection
    return VectorStoreConnectorAdaptor(VECTOR_DB_TYPE, ctx=ctx)


# vector_db_connector = get_local_vector_db_connector(VECTOR_DB_TYPE)
