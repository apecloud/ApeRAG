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
from contextlib import contextmanager
from typing import Any, Dict, Optional

from neo4j import Driver, GraphDatabase, Session
from neo4j import exceptions as neo4jExceptions

logger = logging.getLogger(__name__)


class Neo4jSyncConnectionManager:
    """
    Neo4j connection manager using sync driver with lazy loading.

    This manager provides Worker/Process-level connection reuse through lazy initialization.
    Connections are created on-demand when first used, avoiding unnecessary resource allocation.

    Key features:
    - Lazy loading: connections created only when needed
    - Worker-level reuse: same connection pool shared across all tasks in a worker
    - Thread-safe: uses threading.Lock for initialization
    - Automatic cleanup: connections closed when process exits
    """

    # Class-level storage for worker-scoped driver
    _driver: Optional[Driver] = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """Initialize the connection manager with configuration."""
        with cls._lock:
            if cls._driver is None:
                # Use provided config or environment variables
                if config:
                    cls._config = config
                else:
                    cls._config = {
                        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                        "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                        "password": os.environ.get("NEO4J_PASSWORD", "neo4j"),
                        "max_connection_pool_size": int(os.environ.get("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")),
                        "connection_timeout": 30.0,
                        "max_transaction_retry_time": 30.0,
                    }

                logger.info(f"Initializing Neo4j sync driver for worker {os.getpid()}")
                cls._driver = GraphDatabase.driver(
                    cls._config["uri"],
                    auth=(cls._config["username"], cls._config["password"]),
                    max_connection_pool_size=cls._config["max_connection_pool_size"],
                    connection_timeout=cls._config["connection_timeout"],
                    max_transaction_retry_time=cls._config["max_transaction_retry_time"],
                )

                # Verify connectivity
                cls._driver.verify_connectivity()
                logger.info(f"Neo4j sync driver initialized successfully for worker {os.getpid()}")

    @classmethod
    def get_driver(cls) -> Driver:
        """Get the shared driver instance."""
        if cls._driver is None:
            cls.initialize()
        return cls._driver

    @classmethod
    @contextmanager
    def get_session(cls, database: Optional[str] = None) -> Session:
        """Get a session from the shared driver."""
        driver = cls.get_driver()
        session = driver.session(database=database)
        try:
            yield session
        finally:
            session.close()

    @classmethod
    def prepare_database(cls, workspace: str) -> str:
        """Prepare database and return database name."""
        DATABASE = os.environ.get("NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", workspace))

        driver = cls.get_driver()

        # Try to connect to the target database first
        try:
            with driver.session(database=DATABASE) as session:
                result = session.run("MATCH (n) RETURN n LIMIT 0")
                result.consume()
                logger.debug(f"Connected to existing database: {DATABASE}")

                # Create indexes
                try:
                    result = session.run("CREATE INDEX IF NOT EXISTS FOR (n:base) ON (n.entity_id)")
                    result.consume()
                    logger.debug(f"Ensured index exists in database: {DATABASE}")
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")

                return DATABASE

        except neo4jExceptions.ClientError as e:
            if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                logger.info(f"Database {DATABASE} not found, attempting to create")
                try:
                    with driver.session() as session:
                        result = session.run(f"CREATE DATABASE `{DATABASE}` IF NOT EXISTS")
                        result.consume()
                        logger.info(f"Database {DATABASE} created successfully")

                    # Create indexes in new database
                    with driver.session(database=DATABASE) as session:
                        try:
                            result = session.run("CREATE INDEX IF NOT EXISTS FOR (n:base) ON (n.entity_id)")
                            result.consume()
                        except Exception as e:
                            logger.warning(f"Could not create index: {e}")

                    return DATABASE

                except (neo4jExceptions.ClientError, neo4jExceptions.DatabaseError) as e:
                    if "UnsupportedAdministrationCommand" in str(e) or "ExecutionFailed" in str(e):
                        logger.warning("Database creation not supported, using default")
                        return "neo4j"
                    raise
            else:
                raise

    @classmethod
    def close(cls):
        """Close the driver and clean up resources."""
        with cls._lock:
            if cls._driver:
                logger.info(f"Closing Neo4j driver for worker {os.getpid()}")
                cls._driver.close()
                cls._driver = None
                cls._config = None


# Legacy Celery signal handlers (now unused with lazy loading)
# These functions are kept for backward compatibility but are no longer used
def setup_worker_neo4j(**kwargs):
    """Legacy function - Neo4j now uses lazy loading instead of worker signals."""
    logger.info(f"Worker {os.getpid()}: Neo4j connection will be initialized on-demand (lazy loading)")


def cleanup_worker_neo4j(**kwargs):
    """Legacy function - Neo4j cleanup happens automatically on process exit."""
    logger.info(f"Worker {os.getpid()}: Neo4j connections will be cleaned up automatically")
