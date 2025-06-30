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

from sqlalchemy import select

from aperag.db.models import (
    LightRAGDocChunksModel,
    LightRAGVDBEntityModel,
    LightRAGVDBRelationModel,
)
from aperag.db.repositories.base import SyncRepositoryProtocol
from aperag.utils.utils import utc_now


class LightragRepositoryMixin(SyncRepositoryProtocol):
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
                        "create_time": utc_now(),
                        "update_time": utc_now(),
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
                        "create_time": utc_now(),
                        "update_time": utc_now(),
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
                        "create_time": utc_now(),
                        "update_time": utc_now(),
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

    def query_lightrag_vdb_entity_all(self, workspace: str):
        """Query all LightRAG VDB Entity records for workspace"""

        def _query(session):
            stmt = select(LightRAGVDBEntityModel).where(LightRAGVDBEntityModel.workspace == workspace)
            result = session.execute(stmt)
            return {entity.id: entity for entity in result.scalars().all()}

        return self._execute_query(_query)

    def query_lightrag_vdb_relation_all(self, workspace: str):
        """Query all LightRAG VDB Relation records for workspace"""

        def _query(session):
            stmt = select(LightRAGVDBRelationModel).where(LightRAGVDBRelationModel.workspace == workspace)
            result = session.execute(stmt)
            return {relation.id: relation for relation in result.scalars().all()}

        return self._execute_query(_query)
