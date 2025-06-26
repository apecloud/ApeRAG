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

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool, Session

# Set nebula3 logger level to WARNING to reduce connection pool noise
logging.getLogger("nebula3.logger").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class NebulaSyncConnectionManager:
    """
    Worker-level NebulaGraph connection manager using sync driver.
    This avoids event loop issues and provides true connection reuse across Celery tasks.
    """

    # Class-level storage for worker-scoped pool
    _connection_pool: ConnectionPool = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None
    _space_schemas_created: set = set()  # Track which spaces have schemas created

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
                        "host": os.environ.get("NEBULA_HOST", "localhost"),
                        "port": int(os.environ.get("NEBULA_PORT", "9669")),
                        "username": os.environ.get("NEBULA_USER", "root"),
                        "password": os.environ.get("NEBULA_PASSWORD", "nebula"),
                        "max_connection_pool_size": int(os.environ.get("NEBULA_MAX_CONNECTION_POOL_SIZE", "10")),
                        "timeout": int(os.environ.get("NEBULA_TIMEOUT", "60000")),  # milliseconds
                        "decode_type": "utf-8",
                    }

                logger.info(f"Initializing NebulaGraph sync connection pool for worker {os.getpid()}")

                # Create config for connection pool
                pool_config = Config()
                pool_config.max_connection_pool_size = cls._config["max_connection_pool_size"]
                pool_config.timeout = cls._config["timeout"]
                pool_config.decode_type = cls._config["decode_type"]

                # Create connection pool
                cls._connection_pool = ConnectionPool()

                # Initialize the pool
                addresses = [(cls._config["host"], cls._config["port"])]
                ok = cls._connection_pool.init(addresses, pool_config)
                if not ok:
                    raise RuntimeError("Failed to initialize NebulaGraph connection pool")

                logger.info(f"NebulaGraph sync connection pool initialized successfully for worker {os.getpid()}")

    @classmethod
    def get_connection_pool(cls) -> ConnectionPool:
        """Get the shared connection pool instance."""
        if cls._connection_pool is None:
            cls.initialize()
        return cls._connection_pool

    @classmethod
    @contextmanager
    def get_session(cls, space: Optional[str] = None) -> Session:
        """Get a session from the shared pool."""
        pool = cls.get_connection_pool()
        session = pool.get_session(cls._config["username"], cls._config["password"])

        if session is None:
            raise RuntimeError("Failed to get session from NebulaGraph pool")

        try:
            # Switch to space if provided
            if space:
                result = session.execute(f"USE `{space}`")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to use space {space}: {result.error_msg()}")

            yield session
        finally:
            # Return session to pool
            session.release()

    @classmethod
    def sanitize_space_name(cls, workspace: str) -> str:
        """Sanitize workspace name to be valid NebulaGraph space name."""
        # Replace non-alphanumeric characters with underscores
        space_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
        # Ensure it starts with a letter or underscore
        if space_name and not space_name[0].isalpha() and space_name[0] != "_":
            space_name = f"s_{space_name}"
        # Limit length to 64 characters (NebulaGraph limit)
        return space_name[:64]

    @classmethod
    def prepare_space(cls, workspace: str) -> str:
        """Prepare space and return space name."""
        space_name = cls.sanitize_space_name(workspace)

        # Get a session without space to create/check space
        with cls.get_session() as session:
            # Check if space exists
            result = session.execute("SHOW SPACES")
            if result.is_succeeded():
                spaces = [row.values()[0].as_string() for row in result]
                if space_name not in spaces:
                    # Create space with fixed string VID type
                    create_space_query = f"""
                    CREATE SPACE IF NOT EXISTS `{space_name}` (
                        partition_num=10, 
                        replica_factor=1, 
                        vid_type=FIXED_STRING(256)
                    )
                    """
                    result = session.execute(create_space_query)
                    if not result.is_succeeded():
                        raise RuntimeError(f"Failed to create space: {result.error_msg()}")

                    logger.info(f"Created NebulaGraph space: {space_name}")

                    # Wait for space to be ready and retry USE command
                    # NebulaGraph space creation is async, need to wait
                    for retry in range(30):  # Try for up to 30 seconds
                        time.sleep(1)
                        test_result = session.execute(f"USE `{space_name}`")
                        if test_result.is_succeeded():
                            logger.info(f"Space {space_name} is ready after {retry + 1} seconds")
                            break
                        if retry == 29:
                            raise RuntimeError(f"Space {space_name} failed to become ready after 30 seconds")

                    # Mark that we need to create schema
                    cls._space_schemas_created.discard(space_name)

            # Create schema if not already created
            if space_name not in cls._space_schemas_created:
                cls._create_schema_for_space(space_name)
                cls._space_schemas_created.add(space_name)

        return space_name

    @classmethod
    def _create_schema_for_space(cls, space_name: str):
        """Create schema (TAGs and EDGE types) for the space."""
        with cls.get_session(space=space_name) as session:
            # Create base TAG for all vertices with all necessary properties
            create_base_tag = """
            CREATE TAG IF NOT EXISTS `base` (
                entity_id string,
                entity_type string,
                entity_name string,
                description string,
                source_id string,
                file_path string,
                created_at int64
            )
            """
            result = session.execute(create_base_tag)
            if not result.is_succeeded():
                logger.warning(f"Failed to create base TAG: {result.error_msg()}")
                # If tag already exists but with different schema, try to add missing columns
                cls._ensure_tag_properties(session, "base", [
                    ("entity_id", "string"),
                    ("entity_type", "string"), 
                    ("entity_name", "string"),
                    ("description", "string"),
                    ("source_id", "string"),
                    ("file_path", "string"),
                    ("created_at", "int64")
                ])

            # Create entity TAG (extends base conceptually) with additional properties
            create_entity_tag = """
            CREATE TAG IF NOT EXISTS `entity` (
                entity_id string,
                entity_type string,
                entity_name string,
                description string,
                source_id string,
                source_chunk_id string,
                occurrence int,
                created_at int64,
                chunk_ids string,
                file_path string
            )
            """
            result = session.execute(create_entity_tag)
            if not result.is_succeeded():
                logger.warning(f"Failed to create entity TAG: {result.error_msg()}")
                # If tag already exists but with different schema, try to add missing columns
                cls._ensure_tag_properties(session, "entity", [
                    ("entity_id", "string"),
                    ("entity_type", "string"),
                    ("entity_name", "string"), 
                    ("description", "string"),
                    ("source_id", "string"),
                    ("source_chunk_id", "string"),
                    ("occurrence", "int"),
                    ("created_at", "int64"),
                    ("chunk_ids", "string"),
                    ("file_path", "string")
                ])

            # Create DIRECTED edge type for relationships
            create_directed_edge = """
            CREATE EDGE IF NOT EXISTS `DIRECTED` (
                weight double,
                source_id string,
                description string,
                keywords string,
                source_chunk_id string,
                file_path string,
                created_at int64
            )
            """
            result = session.execute(create_directed_edge)
            if not result.is_succeeded():
                logger.warning(f"Failed to create DIRECTED edge: {result.error_msg()}")
                # If edge already exists but with different schema, try to add missing columns
                cls._ensure_edge_properties(session, "DIRECTED", [
                    ("weight", "double"),
                    ("source_id", "string"),
                    ("description", "string"),
                    ("keywords", "string"),
                    ("source_chunk_id", "string"),
                    ("file_path", "string"),
                    ("created_at", "int64")
                ])

            # Create indexes for better query performance
            # Index on base.entity_id
            create_base_index = """
            CREATE TAG INDEX IF NOT EXISTS `base_entity_id_idx` 
            ON `base`(entity_id(64))
            """
            result = session.execute(create_base_index)
            if not result.is_succeeded():
                logger.warning(f"Failed to create base index: {result.error_msg()}")

            # Index on entity.entity_name
            create_entity_index = """
            CREATE TAG INDEX IF NOT EXISTS `entity_name_idx` 
            ON `entity`(entity_name(64))
            """
            result = session.execute(create_entity_index)
            if not result.is_succeeded():
                logger.warning(f"Failed to create entity index: {result.error_msg()}")

            # Wait for schemas to be ready - NebulaGraph schema creation is async
            schema_ready = False
            for retry in range(30):  # Try for up to 30 seconds
                time.sleep(1)
                # Verify all schemas exist and are usable
                verify_result = session.execute("SHOW TAGS")
                if verify_result.is_succeeded():
                    tags = [row.values()[0].as_string() for row in verify_result]
                    if "base" in tags and "entity" in tags:
                        # Also verify we can describe the tags (they are fully ready)
                        desc_base = session.execute("DESC TAG `base`")
                        desc_entity = session.execute("DESC TAG `entity`")
                        desc_edge = session.execute("DESC EDGE `DIRECTED`")
                        
                        if desc_base.is_succeeded() and desc_entity.is_succeeded() and desc_edge.is_succeeded():
                            logger.info(f"Schema fully ready for space: {space_name} after {retry + 1} seconds")
                            schema_ready = True
                            break
                
                if retry == 29:
                    logger.error(f"Schema creation timeout for space: {space_name} after 30 seconds")
                    # Still continue, but log the issue

            if not schema_ready:
                logger.warning(f"Schema may not be fully ready for space: {space_name}, but continuing...")

            logger.info(f"Schema creation completed for space: {space_name}")

    @classmethod
    def _ensure_tag_properties(cls, session: Session, tag_name: str, properties: list[tuple[str, str]]):
        """Ensure tag has all required properties, add missing ones."""
        try:
            # Get current tag schema
            desc_result = session.execute(f"DESC TAG `{tag_name}`")
            if not desc_result.is_succeeded():
                return

            existing_props = set()
            for row in desc_result:
                prop_name = row.values()[0].as_string()
                existing_props.add(prop_name)

            # Add missing properties
            for prop_name, prop_type in properties:
                if prop_name not in existing_props:
                    alter_query = f"ALTER TAG `{tag_name}` ADD ({prop_name} {prop_type})"
                    result = session.execute(alter_query)
                    if result.is_succeeded():
                        logger.info(f"Added property {prop_name} to tag {tag_name}")
                    else:
                        logger.warning(f"Failed to add property {prop_name} to tag {tag_name}: {result.error_msg()}")
        except Exception as e:
            logger.warning(f"Failed to ensure tag properties for {tag_name}: {e}")

    @classmethod
    def _ensure_edge_properties(cls, session: Session, edge_name: str, properties: list[tuple[str, str]]):
        """Ensure edge has all required properties, add missing ones."""
        try:
            # Get current edge schema
            desc_result = session.execute(f"DESC EDGE `{edge_name}`")
            if not desc_result.is_succeeded():
                return

            existing_props = set()
            for row in desc_result:
                prop_name = row.values()[0].as_string()
                existing_props.add(prop_name)

            # Add missing properties
            for prop_name, prop_type in properties:
                if prop_name not in existing_props:
                    alter_query = f"ALTER EDGE `{edge_name}` ADD ({prop_name} {prop_type})"
                    result = session.execute(alter_query)
                    if result.is_succeeded():
                        logger.info(f"Added property {prop_name} to edge {edge_name}")
                    else:
                        logger.warning(f"Failed to add property {prop_name} to edge {edge_name}: {result.error_msg()}")
        except Exception as e:
            logger.warning(f"Failed to ensure edge properties for {edge_name}: {e}")

    @classmethod
    def ensure_tag_exists(cls, space_name: str, tag_name: str):
        """Ensure a tag exists in the space, create if it doesn't."""
        with cls.get_session(space=space_name) as session:
            # Check if tag exists
            desc_result = session.execute(f"DESC TAG `{tag_name}`")
            if desc_result.is_succeeded():
                logger.debug(f"Tag {tag_name} already exists in space {space_name}")
                return

            # Create the tag with standard entity properties
            create_tag_query = f"""
            CREATE TAG IF NOT EXISTS `{tag_name}` (
                entity_id string,
                entity_type string,
                entity_name string,
                description string,
                source_id string,
                file_path string,
                created_at int64
            )
            """
            result = session.execute(create_tag_query)
            if result.is_succeeded():
                logger.info(f"Created tag {tag_name} in space {space_name}")
                
                # Create index for this tag
                create_index_query = f"""
                CREATE TAG INDEX IF NOT EXISTS `{tag_name}_entity_id_idx` 
                ON `{tag_name}`(entity_id(64))
                """
                index_result = session.execute(create_index_query)
                if index_result.is_succeeded():
                    logger.debug(f"Created index for tag {tag_name}")
                else:
                    logger.warning(f"Failed to create index for tag {tag_name}: {index_result.error_msg()}")
            else:
                logger.warning(f"Failed to create tag {tag_name}: {result.error_msg()}")

    @classmethod
    def close(cls):
        """Close the connection pool and clean up resources."""
        with cls._lock:
            if cls._connection_pool:
                logger.info(f"Closing NebulaGraph connection pool for worker {os.getpid()}")
                cls._connection_pool.close()
                cls._connection_pool = None
                cls._config = None
                cls._space_schemas_created.clear()


# Celery signal handlers for worker lifecycle
def setup_worker_nebula(**kwargs):
    """Initialize NebulaGraph when worker starts."""
    NebulaSyncConnectionManager.initialize()
    logger.info(f"Worker {os.getpid()}: NebulaGraph sync connection pool initialized")


def cleanup_worker_nebula(**kwargs):
    """Cleanup NebulaGraph when worker shuts down."""
    NebulaSyncConnectionManager.close()
    logger.info(f"Worker {os.getpid()}: NebulaGraph sync connection pool closed")
