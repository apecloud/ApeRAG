# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime

from sqlalchemy import select

from aperag.db.models import (
    LightRAGDocChunksModel,
    LightRAGDocFullModel,
    LightRAGDocStatus,
    LightRAGDocStatusModel,
    LightRAGLLMCacheModel,
    LightRAGVDBEntityModel,
    LightRAGVDBRelationModel,
)
from aperag.db.repositories.base import SyncRepositoryProtocol


class LightragRepositoryMixin(SyncRepositoryProtocol):
    # LightRAG Document Status Operations
    def upsert_lightrag_doc_status(self, workspace: str, doc_status_data: dict):
        """Upsert LightRAG document status records"""

        def _operation(session):
            from sqlalchemy.dialects.postgresql import insert

            for doc_id, status_data in doc_status_data.items():
                # Convert datetime strings to datetime objects if needed
                created_at = status_data.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                elif created_at is None:
                    created_at = datetime.utcnow()

                updated_at = status_data.get("updated_at")
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                elif updated_at is None:
                    updated_at = datetime.utcnow()

                stmt = insert(LightRAGDocStatusModel).values(
                    workspace=workspace,
                    id=doc_id,
                    content=status_data.get("content"),
                    content_summary=status_data.get("content_summary"),
                    content_length=status_data.get("content_length"),
                    chunks_count=status_data.get("chunks_count", -1),
                    status=status_data.get("status"),
                    file_path=status_data.get("file_path"),
                    created_at=created_at,
                    updated_at=updated_at,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["workspace", "id"],
                    set_={
                        "content": stmt.excluded.content,
                        "content_summary": stmt.excluded.content_summary,
                        "content_length": stmt.excluded.content_length,
                        "chunks_count": stmt.excluded.chunks_count,
                        "status": stmt.excluded.status,
                        "file_path": stmt.excluded.file_path,
                        "created_at": stmt.excluded.created_at,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                session.execute(stmt)
            session.commit()

        return self._execute_transaction(_operation)

    def query_lightrag_docs_by_status(self, workspace: str, status: LightRAGDocStatus):
        """Query LightRAG documents by status"""

        def _query(session):
            stmt = select(LightRAGDocStatusModel).where(
                LightRAGDocStatusModel.workspace == workspace, LightRAGDocStatusModel.status == status
            )
            result = session.execute(stmt)
            return {doc.id: doc for doc in result.scalars().all()}

        return self._execute_query(_query)

    def query_lightrag_doc_status_by_id(self, workspace: str, doc_id: str):
        """Query LightRAG document status by ID"""

        def _query(session):
            stmt = select(LightRAGDocStatusModel).where(
                LightRAGDocStatusModel.workspace == workspace, LightRAGDocStatusModel.id == doc_id
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def delete_lightrag_doc_status(self, workspace: str, doc_ids: list):
        """Delete LightRAG document status records"""

        def _operation(session):
            stmt = select(LightRAGDocStatusModel).where(
                LightRAGDocStatusModel.workspace == workspace, LightRAGDocStatusModel.id.in_(doc_ids)
            )
            result = session.execute(stmt)
            docs = result.scalars().all()

            for doc in docs:
                session.delete(doc)
            session.commit()
            return len(docs)

        return self._execute_transaction(_operation)

    # LightRAG Doc Full Operations
    def query_lightrag_doc_full_by_id(self, workspace: str, doc_id: str):
        """Query LightRAG document full by ID"""

        def _query(session):
            stmt = select(LightRAGDocFullModel).where(
                LightRAGDocFullModel.workspace == workspace, LightRAGDocFullModel.id == doc_id
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def query_lightrag_doc_full_by_ids(self, workspace: str, doc_ids: list):
        """Query LightRAG document full by IDs"""

        def _query(session):
            if not doc_ids:
                return []
            stmt = select(LightRAGDocFullModel).where(
                LightRAGDocFullModel.workspace == workspace, LightRAGDocFullModel.id.in_(doc_ids)
            )
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)

    def query_lightrag_doc_full_all(self, workspace: str):
        """Query all LightRAG document full records for workspace"""

        def _query(session):
            stmt = select(LightRAGDocFullModel).where(LightRAGDocFullModel.workspace == workspace)
            result = session.execute(stmt)
            return {doc.id: doc for doc in result.scalars().all()}

        return self._execute_query(_query)

    def filter_lightrag_doc_full_keys(self, workspace: str, keys: list):
        """Filter existing keys for LightRAG document full"""

        def _query(session):
            if not keys:
                return []
            stmt = select(LightRAGDocFullModel.id).where(
                LightRAGDocFullModel.workspace == workspace, LightRAGDocFullModel.id.in_(keys)
            )
            result = session.execute(stmt)
            return [row[0] for row in result.fetchall()]

        return self._execute_query(_query)

    def upsert_lightrag_doc_full(self, workspace: str, doc_data: dict):
        """Upsert LightRAG document full records"""

        def _operation(session):
            for doc_id, doc_content in doc_data.items():
                # Check if record exists
                stmt = select(LightRAGDocFullModel).where(
                    LightRAGDocFullModel.workspace == workspace, LightRAGDocFullModel.id == doc_id
                )
                result = session.execute(stmt)
                existing = result.scalars().first()

                if existing:
                    # Update existing record
                    existing.doc_name = doc_content.get("doc_name")
                    existing.content = doc_content.get("content", "")
                    existing.meta = doc_content.get("meta")
                    existing.update_time = datetime.utcnow()
                    session.add(existing)
                else:
                    # Create new record
                    new_doc = LightRAGDocFullModel(
                        workspace=workspace,
                        id=doc_id,
                        doc_name=doc_content.get("doc_name"),
                        content=doc_content.get("content", ""),
                        meta=doc_content.get("meta"),
                        create_time=datetime.utcnow(),
                        update_time=datetime.utcnow(),
                    )
                    session.add(new_doc)

            session.commit()

        return self._execute_transaction(_operation)

    def delete_lightrag_doc_full(self, workspace: str, doc_ids: list):
        """Delete LightRAG document full records"""

        def _operation(session):
            stmt = select(LightRAGDocFullModel).where(
                LightRAGDocFullModel.workspace == workspace, LightRAGDocFullModel.id.in_(doc_ids)
            )
            result = session.execute(stmt)
            docs = result.scalars().all()

            for doc in docs:
                session.delete(doc)
            session.commit()
            return len(docs)

        return self._execute_transaction(_operation)

    # LightRAG Doc Chunks Operations
    def query_lightrag_doc_chunks_by_id(self, workspace: str, chunk_id: str):
        """Query LightRAG document chunks by ID"""

        def _query(session):
            stmt = select(LightRAGDocChunksModel).where(
                LightRAGDocChunksModel.workspace == workspace, LightRAGDocChunksModel.id == chunk_id
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def query_lightrag_doc_chunks_by_ids(self, workspace: str, chunk_ids: list):
        """Query LightRAG document chunks by IDs"""

        def _query(session):
            if not chunk_ids:
                return []
            stmt = select(LightRAGDocChunksModel).where(
                LightRAGDocChunksModel.workspace == workspace, LightRAGDocChunksModel.id.in_(chunk_ids)
            )
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)

    def query_lightrag_doc_chunks_all(self, workspace: str):
        """Query all LightRAG document chunks records for workspace"""

        def _query(session):
            stmt = select(LightRAGDocChunksModel).where(LightRAGDocChunksModel.workspace == workspace)
            result = session.execute(stmt)
            return {chunk.id: chunk for chunk in result.scalars().all()}

        return self._execute_query(_query)

    def filter_lightrag_doc_chunks_keys(self, workspace: str, keys: list):
        """Filter existing keys for LightRAG document chunks"""

        def _query(session):
            if not keys:
                return []
            stmt = select(LightRAGDocChunksModel.id).where(
                LightRAGDocChunksModel.workspace == workspace, LightRAGDocChunksModel.id.in_(keys)
            )
            result = session.execute(stmt)
            return [row[0] for row in result.fetchall()]

        return self._execute_query(_query)

    def upsert_lightrag_doc_chunks(self, workspace: str, chunks_data: dict):
        """Upsert LightRAG document chunks records using PostgreSQL UPSERT"""

        def _operation(session):
            for chunk_id, chunk_data in chunks_data.items():
                # Prepare vector data - convert from JSON string if needed
                vector_data = chunk_data.get("content_vector")
                if isinstance(vector_data, str):
                    import json

                    vector_data = json.loads(vector_data)

                # Use raw SQL UPSERT to avoid race conditions
                sql = """
                INSERT INTO lightrag_doc_chunks (workspace, id, tokens, chunk_order_index, full_doc_id, content, content_vector, file_path, create_time, update_time)
                VALUES (:workspace, :id, :tokens, :chunk_order_index, :full_doc_id, :content, :content_vector, :file_path, :create_time, :update_time)
                ON CONFLICT (workspace, id) DO UPDATE SET
                    tokens = EXCLUDED.tokens,
                    chunk_order_index = EXCLUDED.chunk_order_index,
                    full_doc_id = EXCLUDED.full_doc_id,
                    content = EXCLUDED.content,
                    content_vector = CASE 
                        WHEN EXCLUDED.content_vector IS NOT NULL THEN EXCLUDED.content_vector 
                        ELSE lightrag_doc_chunks.content_vector 
                    END,
                    file_path = EXCLUDED.file_path,
                    update_time = EXCLUDED.update_time
                """

                from sqlalchemy import text

                session.execute(
                    text(sql),
                    {
                        "workspace": workspace,
                        "id": chunk_id,
                        "tokens": chunk_data.get("tokens"),
                        "chunk_order_index": chunk_data.get("chunk_order_index"),
                        "full_doc_id": chunk_data.get("full_doc_id"),
                        "content": chunk_data.get("content", ""),
                        "content_vector": vector_data,
                        "file_path": chunk_data.get("file_path"),
                        "create_time": datetime.utcnow(),
                        "update_time": datetime.utcnow(),
                    },
                )

            session.commit()

        return self._execute_transaction(_operation)

    def delete_lightrag_doc_chunks(self, workspace: str, chunk_ids: list):
        """Delete LightRAG document chunks records"""

        def _operation(session):
            stmt = select(LightRAGDocChunksModel).where(
                LightRAGDocChunksModel.workspace == workspace, LightRAGDocChunksModel.id.in_(chunk_ids)
            )
            result = session.execute(stmt)
            chunks = result.scalars().all()

            for chunk in chunks:
                session.delete(chunk)
            session.commit()
            return len(chunks)

        return self._execute_transaction(_operation)

    # LightRAG VDB Entity Operations
    def query_lightrag_vdb_entity_by_id(self, workspace: str, entity_id: str):
        """Query LightRAG VDB Entity by ID"""

        def _query(session):
            stmt = select(LightRAGVDBEntityModel).where(
                LightRAGVDBEntityModel.workspace == workspace, LightRAGVDBEntityModel.id == entity_id
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def upsert_lightrag_vdb_entity(self, workspace: str, entity_data: dict):
        """Upsert LightRAG VDB Entity records using PostgreSQL UPSERT"""

        def _operation(session):
            for entity_id, entity_info in entity_data.items():
                # Prepare vector data - convert from JSON string if needed
                vector_data = entity_info.get("content_vector")
                if isinstance(vector_data, str):
                    import json

                    vector_data = json.loads(vector_data)

                # Use raw SQL UPSERT to avoid race conditions
                sql = """
                INSERT INTO lightrag_vdb_entity (workspace, id, entity_name, content, content_vector, chunk_ids, file_path, create_time, update_time)
                VALUES (:workspace, :id, :entity_name, :content, :content_vector, :chunk_ids, :file_path, :create_time, :update_time)
                ON CONFLICT (workspace, id) DO UPDATE SET
                    entity_name = EXCLUDED.entity_name,
                    content = EXCLUDED.content,
                    content_vector = CASE 
                        WHEN EXCLUDED.content_vector IS NOT NULL THEN EXCLUDED.content_vector 
                        ELSE lightrag_vdb_entity.content_vector 
                    END,
                    chunk_ids = EXCLUDED.chunk_ids,
                    file_path = EXCLUDED.file_path,
                    update_time = EXCLUDED.update_time
                """

                from sqlalchemy import text

                session.execute(
                    text(sql),
                    {
                        "workspace": workspace,
                        "id": entity_id,
                        "entity_name": entity_info.get("entity_name"),
                        "content": entity_info.get("content", ""),
                        "content_vector": vector_data,
                        "chunk_ids": entity_info.get("chunk_ids"),
                        "file_path": entity_info.get("file_path"),
                        "create_time": datetime.utcnow(),
                        "update_time": datetime.utcnow(),
                    },
                )

            session.commit()

        return self._execute_transaction(_operation)

    def delete_lightrag_vdb_entity(self, workspace: str, entity_ids: list):
        """Delete LightRAG VDB Entity records"""

        def _operation(session):
            stmt = select(LightRAGVDBEntityModel).where(
                LightRAGVDBEntityModel.workspace == workspace, LightRAGVDBEntityModel.id.in_(entity_ids)
            )
            result = session.execute(stmt)
            entities = result.scalars().all()

            for entity in entities:
                session.delete(entity)
            session.commit()
            return len(entities)

        return self._execute_transaction(_operation)

    # LightRAG VDB Relation Operations
    def query_lightrag_vdb_relation_by_id(self, workspace: str, relation_id: str):
        """Query LightRAG VDB Relation by ID"""

        def _query(session):
            stmt = select(LightRAGVDBRelationModel).where(
                LightRAGVDBRelationModel.workspace == workspace, LightRAGVDBRelationModel.id == relation_id
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def upsert_lightrag_vdb_relation(self, workspace: str, relation_data: dict):
        """Upsert LightRAG VDB Relation records using PostgreSQL UPSERT"""

        def _operation(session):
            for relation_id, relation_info in relation_data.items():
                # Prepare vector data - convert from JSON string if needed
                vector_data = relation_info.get("content_vector")
                if isinstance(vector_data, str):
                    import json

                    vector_data = json.loads(vector_data)

                # Use raw SQL UPSERT to avoid race conditions
                sql = """
                INSERT INTO lightrag_vdb_relation (workspace, id, source_id, target_id, content, content_vector, chunk_ids, file_path, create_time, update_time)
                VALUES (:workspace, :id, :source_id, :target_id, :content, :content_vector, :chunk_ids, :file_path, :create_time, :update_time)
                ON CONFLICT (workspace, id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    content = EXCLUDED.content,
                    content_vector = CASE 
                        WHEN EXCLUDED.content_vector IS NOT NULL THEN EXCLUDED.content_vector 
                        ELSE lightrag_vdb_relation.content_vector 
                    END,
                    chunk_ids = EXCLUDED.chunk_ids,
                    file_path = EXCLUDED.file_path,
                    update_time = EXCLUDED.update_time
                """

                from sqlalchemy import text

                session.execute(
                    text(sql),
                    {
                        "workspace": workspace,
                        "id": relation_id,
                        "source_id": relation_info.get("source_id"),
                        "target_id": relation_info.get("target_id"),
                        "content": relation_info.get("content", ""),
                        "content_vector": vector_data,
                        "chunk_ids": relation_info.get("chunk_ids"),
                        "file_path": relation_info.get("file_path"),
                        "create_time": datetime.utcnow(),
                        "update_time": datetime.utcnow(),
                    },
                )

            session.commit()

        return self._execute_transaction(_operation)

    def delete_lightrag_vdb_relation(self, workspace: str, relation_ids: list):
        """Delete LightRAG VDB Relation records"""

        def _operation(session):
            stmt = select(LightRAGVDBRelationModel).where(
                LightRAGVDBRelationModel.workspace == workspace, LightRAGVDBRelationModel.id.in_(relation_ids)
            )
            result = session.execute(stmt)
            relations = result.scalars().all()

            for relation in relations:
                session.delete(relation)
            session.commit()
            return len(relations)

        return self._execute_transaction(_operation)

    # LightRAG LLM Cache Operations
    def query_lightrag_llm_cache(self, workspace: str, cache_id: str, mode: str):
        """Query LightRAG LLM Cache by ID and mode"""

        def _query(session):
            stmt = select(LightRAGLLMCacheModel).where(
                LightRAGLLMCacheModel.workspace == workspace,
                LightRAGLLMCacheModel.id == cache_id,
                LightRAGLLMCacheModel.mode == mode,
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def upsert_lightrag_llm_cache(self, workspace: str, cache_data: dict):
        """Upsert LightRAG LLM Cache records"""

        def _operation(session):
            for cache_key, cache_info in cache_data.items():
                # Extract ID and mode from cache_key or cache_info
                cache_id = cache_info.get("id")
                mode = cache_info.get("mode")

                if not cache_id or not mode:
                    continue

                # Check if record exists
                stmt = select(LightRAGLLMCacheModel).where(
                    LightRAGLLMCacheModel.workspace == workspace,
                    LightRAGLLMCacheModel.id == cache_id,
                    LightRAGLLMCacheModel.mode == mode,
                )
                result = session.execute(stmt)
                existing = result.scalars().first()

                if existing:
                    # Update existing record
                    existing.original_prompt = cache_info.get("original_prompt")
                    existing.return_value = cache_info.get("return_value")
                    existing.update_time = datetime.utcnow()
                    session.add(existing)
                else:
                    # Create new record
                    new_cache = LightRAGLLMCacheModel(
                        workspace=workspace,
                        id=cache_id,
                        mode=mode,
                        original_prompt=cache_info.get("original_prompt"),
                        return_value=cache_info.get("return_value"),
                        create_time=datetime.utcnow(),
                        update_time=datetime.utcnow(),
                    )
                    session.add(new_cache)

            session.commit()

        return self._execute_transaction(_operation)

    def delete_lightrag_llm_cache(self, workspace: str, cache_keys: list):
        """Delete LightRAG LLM Cache records"""

        def _operation(session):
            deleted_count = 0
            for cache_key in cache_keys:
                # Extract ID and mode from cache_key
                if isinstance(cache_key, dict):
                    cache_id = cache_key.get("id")
                    mode = cache_key.get("mode")
                elif isinstance(cache_key, tuple) and len(cache_key) >= 2:
                    cache_id, mode = cache_key[0], cache_key[1]
                else:
                    continue

                stmt = select(LightRAGLLMCacheModel).where(
                    LightRAGLLMCacheModel.workspace == workspace,
                    LightRAGLLMCacheModel.id == cache_id,
                    LightRAGLLMCacheModel.mode == mode,
                )
                result = session.execute(stmt)
                cache_item = result.scalars().first()

                if cache_item:
                    session.delete(cache_item)
                    deleted_count += 1

            session.commit()
            return deleted_count

        return self._execute_transaction(_operation)

    # Add vector similarity search methods
    def query_lightrag_doc_chunks_similarity(
        self, workspace: str, embedding: list, top_k: int, doc_ids: list = None, threshold: float = 0.2
    ):
        """Query similar document chunks using vector similarity"""

        def _query(session):
            from sqlalchemy import text

            # Convert embedding to PostgreSQL array format
            embedding_string = ",".join(map(str, embedding))

            if doc_ids:
                # Query with document ID filter
                sql = text(
                    """
                    WITH relevant_chunks AS (
                        SELECT id as chunk_id
                        FROM lightrag_doc_chunks
                        WHERE workspace = :workspace AND full_doc_id = ANY(:doc_ids)
                    )
                    SELECT id, content, file_path, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at,
                           1 - (content_vector <=> '[:embedding]'::vector) as distance
                    FROM lightrag_doc_chunks
                    WHERE workspace = :workspace
                    AND id IN (SELECT chunk_id FROM relevant_chunks)
                    AND 1 - (content_vector <=> '[:embedding]'::vector) > :threshold
                    ORDER BY distance DESC
                    LIMIT :top_k
                """.replace(":embedding", embedding_string)
                )

                result = session.execute(
                    sql, {"workspace": workspace, "doc_ids": doc_ids, "threshold": threshold, "top_k": top_k}
                )
            else:
                # Query without document ID filter
                sql = text(
                    """
                    SELECT id, content, file_path, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at,
                           1 - (content_vector <=> '[:embedding]'::vector) as distance
                    FROM lightrag_doc_chunks
                    WHERE workspace = :workspace
                    AND 1 - (content_vector <=> '[:embedding]'::vector) > :threshold
                    ORDER BY distance DESC
                    LIMIT :top_k
                """.replace(":embedding", embedding_string)
                )

                result = session.execute(sql, {"workspace": workspace, "threshold": threshold, "top_k": top_k})

            # Properly convert SQLAlchemy Row objects to dictionaries
            return [dict(row._mapping) for row in result]

        return self._execute_query(_query)

    def query_lightrag_vdb_entity_similarity(
        self, workspace: str, embedding: list, top_k: int, doc_ids: list = None, threshold: float = 0.2
    ):
        """Query similar entities using vector similarity"""

        def _query(session):
            from sqlalchemy import text

            # Convert embedding to PostgreSQL array format
            embedding_string = ",".join(map(str, embedding))

            if doc_ids:
                # Query with document ID filter
                sql = text(
                    """
                    WITH relevant_chunks AS (
                        SELECT id as chunk_id
                        FROM lightrag_doc_chunks
                        WHERE workspace = :workspace AND full_doc_id = ANY(:doc_ids)
                    )
                    SELECT entity_name, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at,
                           1 - (content_vector <=> '[:embedding]'::vector) as distance
                    FROM lightrag_vdb_entity e
                    WHERE e.workspace = :workspace
                    AND EXISTS (
                        SELECT 1 FROM relevant_chunks rc 
                        WHERE rc.chunk_id = ANY(e.chunk_ids)
                    )
                    AND 1 - (content_vector <=> '[:embedding]'::vector) > :threshold
                    ORDER BY distance DESC
                    LIMIT :top_k
                """.replace(":embedding", embedding_string)
                )

                result = session.execute(
                    sql, {"workspace": workspace, "doc_ids": doc_ids, "threshold": threshold, "top_k": top_k}
                )
            else:
                # Query without document ID filter
                sql = text(
                    """
                    SELECT entity_name, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at,
                           1 - (content_vector <=> '[:embedding]'::vector) as distance
                    FROM lightrag_vdb_entity
                    WHERE workspace = :workspace
                    AND 1 - (content_vector <=> '[:embedding]'::vector) > :threshold
                    ORDER BY distance DESC
                    LIMIT :top_k
                """.replace(":embedding", embedding_string)
                )

                result = session.execute(sql, {"workspace": workspace, "threshold": threshold, "top_k": top_k})

            # Properly convert SQLAlchemy Row objects to dictionaries
            return [dict(row._mapping) for row in result]

        return self._execute_query(_query)

    def query_lightrag_vdb_relation_similarity(
        self, workspace: str, embedding: list, top_k: int, doc_ids: list = None, threshold: float = 0.2
    ):
        """Query similar relations using vector similarity"""

        def _query(session):
            from sqlalchemy import text

            # Convert embedding to PostgreSQL array format
            embedding_string = ",".join(map(str, embedding))

            if doc_ids:
                # Query with document ID filter
                sql = text(
                    """
                    WITH relevant_chunks AS (
                        SELECT id as chunk_id
                        FROM lightrag_doc_chunks
                        WHERE workspace = :workspace AND full_doc_id = ANY(:doc_ids)
                    )
                    SELECT source_id as src_id, target_id as tgt_id, 
                           EXTRACT(EPOCH FROM create_time)::BIGINT as created_at,
                           1 - (content_vector <=> '[:embedding]'::vector) as distance
                    FROM lightrag_vdb_relation r
                    WHERE r.workspace = :workspace
                    AND EXISTS (
                        SELECT 1 FROM relevant_chunks rc 
                        WHERE rc.chunk_id = ANY(r.chunk_ids)
                    )
                    AND 1 - (content_vector <=> '[:embedding]'::vector) > :threshold
                    ORDER BY distance DESC
                    LIMIT :top_k
                """.replace(":embedding", embedding_string)
                )

                result = session.execute(
                    sql, {"workspace": workspace, "doc_ids": doc_ids, "threshold": threshold, "top_k": top_k}
                )
            else:
                # Query without document ID filter
                sql = text(
                    """
                    SELECT source_id as src_id, target_id as tgt_id,
                           EXTRACT(EPOCH FROM create_time)::BIGINT as created_at,
                           1 - (content_vector <=> '[:embedding]'::vector) as distance
                    FROM lightrag_vdb_relation
                    WHERE workspace = :workspace
                    AND 1 - (content_vector <=> '[:embedding]'::vector) > :threshold
                    ORDER BY distance DESC
                    LIMIT :top_k
                """.replace(":embedding", embedding_string)
                )

                result = session.execute(sql, {"workspace": workspace, "threshold": threshold, "top_k": top_k})

            # Properly convert SQLAlchemy Row objects to dictionaries
            return [dict(row._mapping) for row in result]

        return self._execute_query(_query)

    # Additional entity and relation operations
    def query_lightrag_vdb_entity_by_name(self, workspace: str, entity_name: str):
        """Query entity by entity name"""

        def _query(session):
            stmt = select(LightRAGVDBEntityModel).where(
                LightRAGVDBEntityModel.workspace == workspace, LightRAGVDBEntityModel.entity_name == entity_name
            )
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def delete_lightrag_vdb_entity_by_name(self, workspace: str, entity_name: str):
        """Delete entity by entity name"""

        def _operation(session):
            stmt = select(LightRAGVDBEntityModel).where(
                LightRAGVDBEntityModel.workspace == workspace, LightRAGVDBEntityModel.entity_name == entity_name
            )
            result = session.execute(stmt)
            entity = result.scalars().first()

            if entity:
                session.delete(entity)
                session.commit()
                return 1
            return 0

        return self._execute_transaction(_operation)

    def delete_lightrag_vdb_relation_by_entity(self, workspace: str, entity_name: str):
        """Delete all relations where entity is source or target"""

        def _operation(session):
            from sqlalchemy import or_

            stmt = select(LightRAGVDBRelationModel).where(
                LightRAGVDBRelationModel.workspace == workspace,
                or_(
                    LightRAGVDBRelationModel.source_id == entity_name, LightRAGVDBRelationModel.target_id == entity_name
                ),
            )
            result = session.execute(stmt)
            relations = result.scalars().all()

            for relation in relations:
                session.delete(relation)
            session.commit()
            return len(relations)

        return self._execute_transaction(_operation)

    def query_lightrag_vdb_entity_by_ids(self, workspace: str, entity_ids: list):
        """Query entities by IDs"""

        def _query(session):
            if not entity_ids:
                return []
            stmt = select(LightRAGVDBEntityModel).where(
                LightRAGVDBEntityModel.workspace == workspace, LightRAGVDBEntityModel.id.in_(entity_ids)
            )
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)

    def query_lightrag_vdb_relation_by_ids(self, workspace: str, relation_ids: list):
        """Query relations by IDs"""

        def _query(session):
            if not relation_ids:
                return []
            stmt = select(LightRAGVDBRelationModel).where(
                LightRAGVDBRelationModel.workspace == workspace, LightRAGVDBRelationModel.id.in_(relation_ids)
            )
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)

    def query_lightrag_doc_status_by_ids(self, workspace: str, doc_ids: list):
        """Query document status by IDs"""

        def _query(session):
            if not doc_ids:
                return []
            stmt = select(LightRAGDocStatusModel).where(
                LightRAGDocStatusModel.workspace == workspace, LightRAGDocStatusModel.id.in_(doc_ids)
            )
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)

    def filter_lightrag_doc_status_keys(self, workspace: str, keys: list):
        """Filter existing keys for document status"""

        def _query(session):
            if not keys:
                return []
            stmt = select(LightRAGDocStatusModel.id).where(
                LightRAGDocStatusModel.workspace == workspace, LightRAGDocStatusModel.id.in_(keys)
            )
            result = session.execute(stmt)
            return [row[0] for row in result.fetchall()]

        return self._execute_query(_query)
