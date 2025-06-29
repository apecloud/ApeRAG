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

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from psycopg.rows import namedtuple_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class PostgreSQLTransaction:
    """
    Transaction wrapper for PostgreSQL operations.
    Provides execute and query methods within a single transaction.
    """

    def __init__(self, connection):
        self.connection = connection

    async def execute(self, sql: str, params: Dict[str, Any]) -> None:
        """Execute a SQL statement with parameters within the transaction."""
        try:
            with self.connection.cursor() as cur:
                cur.execute(sql, params)
        except Exception as e:
            logger.error(f"Failed to execute SQL in transaction: {sql}, error: {e}")
            raise

    async def query(self, sql: str, params: Dict[str, Any], multirows: bool = False) -> Any:
        """Query the database and return results within the transaction."""
        try:
            with self.connection.cursor(row_factory=namedtuple_row) as cur:
                cur.execute(sql, params)
                if multirows:
                    results = cur.fetchall()
                    return [row._asdict() for row in results]
                else:
                    result = cur.fetchone()
                    return result._asdict() if result else None
        except Exception as e:
            logger.error(f"Failed to query SQL in transaction: {sql}, error: {e}")
            raise


class PostgreSQLGraphDB:
    """
    PostgreSQL Graph database client for graph operations.
    Workspace-agnostic - manages global connection pool, workspace handled as data column.
    """

    def __init__(self):
        self._connection_pool: Optional[ConnectionPool] = None
        self._config: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the database connection."""
        with self._lock:
            if self._connection_pool is None:
                # Use provided config or environment variables
                if config:
                    self._config = config
                else:
                    self._config = {
                        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
                        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
                        "user": os.environ.get("POSTGRES_USER", "postgres"),
                        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
                        "dbname": os.environ.get("POSTGRES_DB", "postgres"),
                        "max_size": int(os.environ.get("POSTGRES_MAX_CONNECTION_POOL_SIZE", "20")),
                        "min_size": int(os.environ.get("POSTGRES_MIN_CONNECTION_POOL_SIZE", "2")),
                        "timeout": int(os.environ.get("POSTGRES_CONNECTION_TIMEOUT", "30")),
                    }

                logger.info("Initializing PostgreSQL Graph client (global)")

                # Build connection string
                connection_string = (
                    f"host='{self._config['host']}' "
                    f"port={self._config['port']} "
                    f"user='{self._config['user']}' "
                    f"password='{self._config['password']}' "
                    f"dbname='{self._config['dbname']}'"
                )

                # Create connection pool
                self._connection_pool = ConnectionPool(
                    connection_string,
                    max_size=self._config["max_size"],
                    min_size=self._config["min_size"],
                    timeout=self._config["timeout"],
                    open=False,
                )

                # Open the pool
                self._connection_pool.open()

                # Initialize tables
                self._initialize_tables()

                logger.info("PostgreSQL Graph client initialized (global)")

    def _initialize_tables(self):
        """Initialize the graph tables with workspace-optimized indexes."""
        try:
            with self._connection_pool.getconn() as conn:
                with conn.cursor() as cur:
                    # Create nodes table - reference lightrag_vdb_entity with only source_id as TEXT
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS LIGHTRAG_GRAPH_NODES (
                            id BIGSERIAL PRIMARY KEY,
                            entity_id VARCHAR(256) NOT NULL,
                            entity_name VARCHAR(255),
                            entity_type VARCHAR(255),
                            description TEXT,
                            source_id TEXT,
                            file_path TEXT,
                            workspace VARCHAR(255) NOT NULL,
                            createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updatetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(workspace, entity_id)
                        )
                    """)

                    # Create edges table - reference lightrag_vdb_relation with only source_id as TEXT
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS LIGHTRAG_GRAPH_EDGES (
                            id BIGSERIAL PRIMARY KEY,
                            source_entity_id VARCHAR(255) NOT NULL,
                            target_entity_id VARCHAR(255) NOT NULL,
                            weight DECIMAL(10,6) DEFAULT 0.0,
                            keywords TEXT,
                            description TEXT,
                            source_id TEXT,
                            file_path TEXT,
                            workspace VARCHAR(255) NOT NULL,
                            createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updatetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(workspace, source_entity_id, target_entity_id)
                        )
                    """)

                    # Create performance-optimized indexes with workspace as first column for data isolation
                    # Nodes table indexes - workspace first for optimal partition pruning
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_nodes_workspace_entity ON LIGHTRAG_GRAPH_NODES(workspace, entity_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_nodes_entity_type ON LIGHTRAG_GRAPH_NODES(workspace, entity_type)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_nodes_entity_name ON LIGHTRAG_GRAPH_NODES(workspace, entity_name)"
                    )

                    # Edges table indexes - workspace first for optimal query performance
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_edges_workspace_source ON LIGHTRAG_GRAPH_EDGES(workspace, source_entity_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_edges_workspace_target ON LIGHTRAG_GRAPH_EDGES(workspace, target_entity_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_edges_workspace_source_target ON LIGHTRAG_GRAPH_EDGES(workspace, source_entity_id, target_entity_id)"
                    )
                    # Index for degree calculation (both source and target)
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_edges_workspace_degree ON LIGHTRAG_GRAPH_EDGES(workspace, source_entity_id, target_entity_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lightrag_edges_weight ON LIGHTRAG_GRAPH_EDGES(workspace, weight)"
                    )

                    conn.commit()
                    logger.info("PostgreSQL Graph tables and workspace-optimized indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise

    @asynccontextmanager
    async def get_transaction(self):
        """
        Get a database transaction context manager.

        Usage:
            async with db.get_transaction() as tx:
                await tx.execute(sql1, params1)
                result = await tx.query(sql2, params2)
                # All operations are in the same transaction
        """
        conn = None
        try:
            conn = self._connection_pool.getconn()
            # Disable autocommit to enable manual transaction control
            conn.autocommit = False

            # Create transaction wrapper
            tx = PostgreSQLTransaction(conn)
            yield tx

            # Commit if no exception occurred
            conn.commit()

        except Exception as e:
            # Rollback on any exception
            if conn:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
            logger.error(f"Transaction failed and rolled back: {e}")
            raise
        finally:
            # Always return connection to pool
            if conn:
                # Re-enable autocommit for next use
                conn.autocommit = True
                self._connection_pool.putconn(conn)

    # Legacy methods for backward compatibility (deprecated)
    async def execute(self, sql: str, data: Dict[str, Any]) -> None:
        """Execute a SQL statement with parameters. DEPRECATED: Use get_transaction() instead."""
        async with self.get_transaction() as tx:
            await tx.execute(sql, data)

    async def query(self, sql: str, params: Dict[str, Any], multirows: bool = False) -> Any:
        """Query the database and return results. DEPRECATED: Use get_transaction() instead."""
        async with self.get_transaction() as tx:
            return await tx.query(sql, params, multirows)

    def close(self):
        """Close the database connection."""
        if self._connection_pool:
            self._connection_pool.close()
            self._connection_pool = None
            logger.info("PostgreSQL Graph client closed (global)")


class PostgreSQLGraphClientManager:
    """
    Global client manager for PostgreSQL Graph database.
    Manages single connection pool for all workspaces - workspace isolation at data level.
    """

    _client: Optional[PostgreSQLGraphDB] = None
    _lock = threading.Lock()

    @classmethod
    async def get_client(cls) -> PostgreSQLGraphDB:
        """Get the global PostgreSQL Graph client."""
        with cls._lock:
            if cls._client is None:
                cls._client = PostgreSQLGraphDB()
                cls._client.initialize()
                logger.info("Created global PostgreSQL Graph client")
            return cls._client

    @classmethod
    async def release_client(cls, client: PostgreSQLGraphDB) -> None:
        """Release a client (for compatibility with TiDB interface)."""
        # In this implementation, we keep the global client alive for reuse
        # The actual cleanup happens in close_all()
        pass

    @classmethod
    def close_all(cls):
        """Close the global client."""
        with cls._lock:
            if cls._client:
                cls._client.close()
                cls._client = None
                logger.info("Closed global PostgreSQL Graph client")


# Celery signal handlers for worker lifecycle
def setup_worker_postgres_graph(**kwargs):
    """Initialize PostgreSQL Graph when worker starts."""
    logger.info(f"Worker {os.getpid()}: PostgreSQL Graph sync connection initialized")


def cleanup_worker_postgres_graph(**kwargs):
    """Cleanup PostgreSQL Graph when worker shuts down."""
    PostgreSQLGraphClientManager.close_all()
    logger.info(f"Worker {os.getpid()}: PostgreSQL Graph sync connection closed")
