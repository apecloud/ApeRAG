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
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import psycopg
from psycopg.rows import namedtuple_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class PostgreSQLAGESyncConnectionManager:
    """
    Worker-level PostgreSQL AGE connection manager using sync driver.
    This avoids event loop issues and provides true connection reuse across Celery tasks.
    """

    # Class-level storage for worker-scoped connection pool
    _connection_pool: Optional[ConnectionPool] = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None
    # Cache for prepared graphs to avoid repeated checks
    _prepared_graphs: set = set()

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """Initialize the connection manager with configuration."""
        with cls._lock:
            if cls._connection_pool is None:
                # Use provided config or environment variables
                if config:
                    cls._config = config
                else:
                    cls._config = {
                        "host": os.environ.get("AGE_POSTGRES_HOST", "127.0.0.1"),
                        "port": int(os.environ.get("AGE_POSTGRES_PORT", "5432")),
                        "user": os.environ.get("AGE_POSTGRES_USER", "postgres"),
                        "password": os.environ.get("AGE_POSTGRES_PASSWORD", "postgres"),
                        "dbname": os.environ.get("AGE_POSTGRES_DB", "postgres"),
                        "max_size": int(os.environ.get("AGE_MAX_CONNECTION_POOL_SIZE", "20")),
                        "min_size": int(os.environ.get("AGE_MIN_CONNECTION_POOL_SIZE", "2")),
                        "timeout": int(os.environ.get("AGE_CONNECTION_TIMEOUT", "30")),
                    }

                logger.info(f"Initializing PostgreSQL AGE sync connection pool for worker {os.getpid()}")

                # Build connection string
                connection_string = (
                    f"host='{cls._config['host']}' "
                    f"port={cls._config['port']} "
                    f"user='{cls._config['user']}' "
                    f"password='{cls._config['password']}' "
                    f"dbname='{cls._config['dbname']}'"
                )

                # Create connection pool
                cls._connection_pool = ConnectionPool(
                    connection_string,
                    max_size=cls._config["max_size"],
                    min_size=cls._config["min_size"],
                    timeout=cls._config["timeout"],
                    open=False,
                )

                # Open the pool
                cls._connection_pool.open()

                # Verify AGE extension availability
                cls._verify_age_extension()

                logger.info(f"PostgreSQL AGE sync connection pool initialized successfully for worker {os.getpid()}")

    @classmethod
    def _verify_age_extension(cls):
        """Verify that Apache AGE extension is available."""
        try:
            with cls.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if AGE extension exists
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'age'")
                    if not cur.fetchone():
                        # Try to create AGE extension
                        cur.execute("CREATE EXTENSION IF NOT EXISTS age")
                        logger.info("AGE extension created successfully")
                    else:
                        logger.debug("AGE extension already available")

                    # Load AGE into search path
                    cur.execute('SET search_path = ag_catalog, "$user", public')
                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to verify AGE extension: {e}")
            raise RuntimeError(f"Apache AGE extension is not available: {e}")

    @classmethod
    def get_pool(cls) -> ConnectionPool:
        """Get the shared connection pool instance."""
        if cls._connection_pool is None:
            cls.initialize()
        return cls._connection_pool

    @classmethod
    @contextmanager
    def get_connection(cls):
        """Get a connection from the shared connection pool."""
        pool = cls.get_pool()
        connection = None
        try:
            connection = pool.getconn()
            # Ensure AGE is in search path for this connection
            with connection.cursor() as cur:
                cur.execute('SET search_path = ag_catalog, "$user", public')
            connection.commit()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                pool.putconn(connection)

    @classmethod
    @contextmanager
    def get_cursor(cls, row_factory=None):
        """Get a cursor from a connection."""
        with cls.get_connection() as conn:
            cursor_kwargs = {}
            if row_factory:
                cursor_kwargs["row_factory"] = row_factory
            with conn.cursor(**cursor_kwargs) as cur:
                yield cur, conn

    @classmethod
    def prepare_graph(cls, workspace: str, max_wait: int = 30) -> str:
        """
        Prepare AGE graph and return graph name.

        Args:
            workspace: Workspace name to prepare
            max_wait: Maximum time to wait for graph readiness (seconds)

        Returns:
            Graph name
        """
        # Sanitize workspace name for AGE (only alphanumeric and underscore allowed)
        graph_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)

        # Ultra-fast path: If we've already prepared this graph, return immediately
        with cls._lock:
            if graph_name in cls._prepared_graphs:
                logger.debug(f"Graph {graph_name} already prepared (cached)")
                return graph_name

        # Fast path: Check if graph exists and is ready
        try:
            with cls.get_cursor() as (cur, conn):
                # Check if graph exists
                cur.execute(
                    "SELECT graph_name FROM ag_catalog.ag_graph WHERE graph_name = %s",
                    (graph_name,)
                )
                
                if cur.fetchone():
                    # Graph exists, do a quick test
                    try:
                        test_query = f"""
                        SELECT * FROM ag_catalog.cypher('{graph_name}', $$
                            MATCH (n) RETURN count(n) AS node_count LIMIT 1
                        $$) AS (node_count agtype)
                        """
                        cur.execute(test_query)
                        cur.fetchone()
                        logger.info(f"Graph {graph_name} already exists and ready (fast path)")
                        # Cache this graph as prepared
                        with cls._lock:
                            cls._prepared_graphs.add(graph_name)
                        return graph_name
                    except Exception:
                        logger.debug(f"Quick readiness test failed for graph {graph_name}, proceeding with setup")

        except Exception as e:
            logger.warning(f"Fast path check failed: {e}, falling back to normal creation")

        # Normal path: Create or ensure graph is properly set up
        with cls.get_cursor() as (cur, conn):
            try:
                # Create graph if it doesn't exist
                cur.execute(f"SELECT create_graph('{graph_name}')")
                logger.info(f"Graph {graph_name} created successfully")
                conn.commit()
            except psycopg.errors.UniqueViolation:
                # Graph already exists
                logger.debug(f"Graph {graph_name} already exists")
                conn.rollback()
            except Exception as e:
                logger.error(f"Failed to create graph {graph_name}: {e}")
                conn.rollback()
                raise

        # Test graph readiness
        start_time = time.time()
        graph_ready = False

        while time.time() - start_time < max_wait:
            try:
                with cls.get_cursor() as (cur, conn):
                    # Test basic cypher query
                    test_query = f"""
                    SELECT * FROM ag_catalog.cypher('{graph_name}', $$
                        RETURN 1 AS test_value
                    $$) AS (test_value agtype)
                    """
                    cur.execute(test_query)
                    result = cur.fetchone()
                    if result:
                        elapsed = time.time() - start_time
                        logger.info(f"Graph {graph_name} ready after {elapsed:.1f}s")
                        graph_ready = True
                        break
            except Exception as e:
                logger.debug(f"Graph readiness test failed: {e}")
            time.sleep(0.5)

        if not graph_ready:
            logger.warning(f"Graph {graph_name} may not be fully ready after {max_wait}s")

        logger.info(f"Graph {graph_name} prepared successfully")

        # Cache this graph as prepared
        with cls._lock:
            cls._prepared_graphs.add(graph_name)

        return graph_name

    @classmethod
    def close(cls):
        """Close the connection pool and clean up resources."""
        with cls._lock:
            if cls._connection_pool:
                logger.info(f"Closing PostgreSQL AGE connection pool for worker {os.getpid()}")
                cls._connection_pool.close()
                cls._connection_pool = None
                cls._config = None
                cls._prepared_graphs.clear()


# Celery signal handlers for worker lifecycle
def setup_worker_postgres_age(**kwargs):
    """Initialize PostgreSQL AGE when worker starts."""
    PostgreSQLAGESyncConnectionManager.initialize()
    logger.info(f"Worker {os.getpid()}: PostgreSQL AGE sync connection initialized")


def cleanup_worker_postgres_age(**kwargs):
    """Cleanup PostgreSQL AGE when worker shuts down."""
    PostgreSQLAGESyncConnectionManager.close()
    logger.info(f"Worker {os.getpid()}: PostgreSQL AGE sync connection closed") 