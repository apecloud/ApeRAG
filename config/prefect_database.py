#!/usr/bin/env python3
"""
Prefect Database Configuration

Configure Prefect to use ApeRAG's PostgreSQL database with schema separation.
This consolidates all data into a single database while keeping them logically separated.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def configure_prefect_database(
    database_url: Optional[str] = None,
    schema: str = "prefect"
) -> str:
    """
    Configure Prefect to use PostgreSQL database with schema separation.
    
    Args:
        database_url: PostgreSQL connection URL. If None, uses ApeRAG's database URL.
        schema: Schema name for Prefect tables (default: 'prefect')
    
    Returns:
        The configured database URL for Prefect
    """
    
    if database_url is None:
        # Get ApeRAG's database configuration
        try:
            from aperag.config import settings
            database_url = settings.database_url
            logger.info(f"Using ApeRAG database URL: {database_url}")
        except Exception as e:
            logger.error(f"Failed to get ApeRAG database URL: {e}")
            raise
    
    # Convert to Prefect-compatible async PostgreSQL URL if needed
    if database_url.startswith("postgresql://"):
        # Convert to async PostgreSQL URL for Prefect
        prefect_db_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    elif database_url.startswith("postgresql+asyncpg://"):
        prefect_db_url = database_url
    else:
        raise ValueError(f"Unsupported database URL format: {database_url}")
    
    # Add schema parameter if not already present
    if "?schema=" not in prefect_db_url and "&schema=" not in prefect_db_url:
        separator = "&" if "?" in prefect_db_url else "?"
        prefect_db_url = f"{prefect_db_url}{separator}schema={schema}"
    
    # Set Prefect database configuration
    os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = prefect_db_url
    
    logger.info(f"Configured Prefect database: {prefect_db_url}")
    return prefect_db_url


def create_prefect_schema(
    database_url: Optional[str] = None,
    schema: str = "prefect"
) -> None:
    """
    Create Prefect schema in PostgreSQL database if it doesn't exist.
    
    Args:
        database_url: PostgreSQL connection URL
        schema: Schema name to create
    """
    
    if database_url is None:
        try:
            from aperag.config import settings
            database_url = settings.database_url
        except Exception as e:
            logger.error(f"Failed to get database URL: {e}")
            raise
    
    try:
        import asyncpg
        import asyncio
        
        async def create_schema():
            # Parse connection parameters from URL
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            conn = await asyncpg.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:] if parsed.path else 'postgres'
            )
            
            try:
                # Create schema if not exists
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                logger.info(f"Created/verified schema: {schema}")
            finally:
                await conn.close()
        
        # Run the async function
        asyncio.run(create_schema())
        
    except ImportError:
        logger.warning("asyncpg not available, schema creation skipped")
        logger.info("Please manually create schema with: CREATE SCHEMA IF NOT EXISTS prefect;")
    except Exception as e:
        logger.error(f"Failed to create schema {schema}: {e}")
        raise


def initialize_prefect_database(
    database_url: Optional[str] = None,
    schema: str = "prefect",
    create_schema: bool = True
) -> str:
    """
    Initialize Prefect database configuration.
    
    Args:
        database_url: PostgreSQL connection URL
        schema: Schema name for Prefect tables
        create_schema: Whether to create the schema if it doesn't exist
    
    Returns:
        The configured database URL for Prefect
    """
    
    logger.info("Initializing Prefect database configuration...")
    
    # Create schema if requested
    if create_schema:
        try:
            create_prefect_schema(database_url, schema)
        except Exception as e:
            logger.warning(f"Schema creation failed, continuing anyway: {e}")
    
    # Configure Prefect database
    prefect_db_url = configure_prefect_database(database_url, schema)
    
    logger.info("Prefect database initialization completed")
    return prefect_db_url


def get_prefect_database_status() -> dict:
    """
    Get status information about Prefect database configuration.
    
    Returns:
        Dictionary with database status information
    """
    
    from prefect.settings import PREFECT_API_DATABASE_CONNECTION_URL
    
    status = {
        "prefect_db_url": PREFECT_API_DATABASE_CONNECTION_URL.value(),
        "is_postgresql": "postgresql" in PREFECT_API_DATABASE_CONNECTION_URL.value(),
        "is_sqlite": "sqlite" in PREFECT_API_DATABASE_CONNECTION_URL.value(),
    }
    
    # Try to get ApeRAG database URL for comparison
    try:
        from aperag.config import settings
        status["aperag_db_url"] = settings.database_url
        status["same_database"] = (
            status["prefect_db_url"].split("?")[0].replace("postgresql+asyncpg://", "postgresql://") 
            == status["aperag_db_url"]
        )
    except Exception as e:
        status["aperag_db_url"] = f"Error: {e}"
        status["same_database"] = False
    
    return status


if __name__ == "__main__":
    # Test the database configuration
    logging.basicConfig(level=logging.INFO)
    
    print("Current Prefect database status:")
    status = get_prefect_database_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nInitializing Prefect database...")
    try:
        prefect_db_url = initialize_prefect_database()
        print(f"Success! Prefect database URL: {prefect_db_url}")
        
        print("\nUpdated status:")
        status = get_prefect_database_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Failed: {e}") 