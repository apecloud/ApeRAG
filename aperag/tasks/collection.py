import json

from config import settings
from config.celery import app
from config.vector_db import get_vector_db_connector
from aperag.context.full_text import create_index, delete_index
from aperag.db.models import Collection, CollectionStatus
from aperag.readers.base_embedding import get_embedding_model
from aperag.source.base import get_source
from aperag.tasks.sync_documents_task import sync_documents
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
)


@app.task
def init_collection_task(collection_id, document_user_quota):
    collection = Collection.objects.get(id=collection_id)
    if collection.status == CollectionStatus.DELETED:
        return

    vector_db_conn = get_vector_db_connector(
        collection=generate_vector_db_collection_name(collection_id=collection_id)
    )
    config = json.loads(collection.config)
    
    embedding_model = config.get("embedding_model", "")
    if not embedding_model:
        _, size = get_embedding_model(settings.EMBEDDING_MODEL, load=False)
        config["embedding_model"] = settings.EMBEDDING_MODEL
        collection.config = json.dumps(config)
    else:
        _, size = get_embedding_model(embedding_model, load=False)
    # pre-create collection in vector db
    vector_db_conn.connector.create_collection(vector_size=size)
    
    qa_vector_db_conn = get_vector_db_connector(
        collection=generate_qa_vector_db_collection_name(collection=collection_id)
    )
    qa_vector_db_conn.connector.create_collection(vector_size=size)

    index_name = generate_fulltext_index_name(collection_id)
    create_index(index_name)

    collection.status = CollectionStatus.ACTIVE
    collection.save()
    
    source = get_source(json.loads(collection.config))
    if source.sync_enabled():
        sync_documents.delay(collection_id=collection_id, document_user_quota=document_user_quota)


@app.task
def delete_collection_task(collection_id):
    collection = Collection.objects.get(id=collection_id)

    # TODO remove the related collection in the vector db
    index_name = generate_fulltext_index_name(collection.id)
    delete_index(index_name)

    vector_db_conn = get_vector_db_connector(
        collection=generate_vector_db_collection_name(collection_id=collection_id)
    )
    vector_db_conn.connector.delete_collection()

    qa_vector_db_conn = get_vector_db_connector(
        collection=generate_qa_vector_db_collection_name(collection=collection_id)
    )
    qa_vector_db_conn.connector.delete_collection()
