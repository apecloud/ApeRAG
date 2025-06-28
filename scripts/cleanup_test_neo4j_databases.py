#!/usr/bin/env python3
"""
Script to delete all Neo4j databases that start with 'test'

This script connects to Neo4j and removes all databases whose names begin with 'test'.
It uses the sync Neo4j driver to avoid event loop issues.

Usage:
    python scripts/cleanup_test_neo4j_databases.py
"""

import logging
import os
import sys
from typing import List

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ClientError, ServiceUnavailable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Neo4jTestDatabaseCleaner:
    """Helper class to clean up test databases in Neo4j"""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the cleaner with connection parameters"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver: Driver = None
    
    def connect(self) -> bool:
        """Connect to Neo4j and verify connectivity"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=5,
                connection_timeout=30.0
            )
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def list_databases(self) -> List[str]:
        """List all databases in Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("SHOW DATABASES")
                databases = [record["name"] for record in result]
                logger.info(f"Found {len(databases)} databases: {databases}")
                return databases
        except ClientError as e:
            if "UnsupportedAdministrationCommand" in str(e):
                logger.warning("This Neo4j instance does not support SHOW DATABASES command (Community Edition)")
                logger.info("Assuming only default database 'neo4j' exists")
                return ["neo4j"]
            else:
                logger.error(f"Error listing databases: {e}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error listing databases: {e}")
            return []
    
    def list_test_databases(self) -> List[str]:
        """List all databases that start with 'test'"""
        all_databases = self.list_databases()
        test_databases = [db for db in all_databases if db.startswith("test")]
        logger.info(f"Found {len(test_databases)} test databases: {test_databases}")
        return test_databases
    
    def drop_database(self, database_name: str) -> bool:
        """Drop a specific database"""
        try:
            with self.driver.session() as session:
                # Use backticks to handle database names with special characters
                query = f"DROP DATABASE `{database_name}` IF EXISTS"
                session.run(query)
                logger.info(f"Successfully dropped database: {database_name}")
                return True
        except ClientError as e:
            if "UnsupportedAdministrationCommand" in str(e):
                logger.warning(f"Cannot drop database {database_name}: Database management not supported (Community Edition)")
                return False
            else:
                logger.error(f"Error dropping database {database_name}: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error dropping database {database_name}: {e}")
            return False
    
    def cleanup_test_databases(self) -> int:
        """Clean up all test databases and return count of databases dropped"""
        test_databases = self.list_test_databases()
        
        if not test_databases:
            logger.info("No test databases found to clean up")
            return 0
        
        dropped_count = 0
        for db_name in test_databases:
            logger.info(f"Attempting to drop test database: {db_name}")
            if self.drop_database(db_name):
                dropped_count += 1
            else:
                logger.warning(f"Failed to drop database: {db_name}")
        
        logger.info(f"Cleanup completed. Dropped {dropped_count} out of {len(test_databases)} test databases")
        return dropped_count


def main():
    """Main function to execute the cleanup"""
    # Get Neo4j connection parameters from environment or use defaults
    neo4j_host = os.environ.get("NEO4J_HOST", "127.0.0.1")
    neo4j_port = os.environ.get("NEO4J_PORT", "7687")
    neo4j_username = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
    
    # Construct URI
    uri = f"bolt://{neo4j_host}:{neo4j_port}"
    
    logger.info("Starting Neo4j test database cleanup")
    logger.info(f"Connecting to Neo4j at {uri} as user {neo4j_username}")
    
    # Create cleaner and connect
    cleaner = Neo4jTestDatabaseCleaner(uri, neo4j_username, neo4j_password)
    
    try:
        if not cleaner.connect():
            logger.error("Failed to connect to Neo4j. Exiting.")
            sys.exit(1)
        
        # Perform cleanup
        dropped_count = cleaner.cleanup_test_databases()
        
        if dropped_count > 0:
            logger.info(f"✅ Cleanup successful! Dropped {dropped_count} test databases")
        else:
            logger.info("✅ No test databases needed to be dropped")
        
    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")
        sys.exit(1)
    finally:
        cleaner.close()


if __name__ == "__main__":
    main() 