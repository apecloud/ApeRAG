#!/usr/bin/env python3
"""
Script to clean up test spaces in Nebula Graph database.
Deletes all spaces that start with 'test'.
"""

import logging
import os
import re
import sys
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _safe_error_msg(result) -> str:
    """Safely extract error message from Nebula result."""
    try:
        error_msg = result.error_msg()
        if isinstance(error_msg, bytes):
            for encoding in ["utf-8", "gbk", "latin-1"]:
                try:
                    return error_msg.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return error_msg.decode("utf-8", errors="replace")
        elif isinstance(error_msg, str):
            return error_msg
        else:
            return str(error_msg)
    except Exception as e:
        logger.warning(f"Failed to get Nebula error message: {e}")
        return f"Nebula operation failed (error code: {result.error_code()})"


def get_all_spaces() -> List[str]:
    """Get list of all spaces in Nebula."""
    spaces = []
    try:
        with NebulaSyncConnectionManager.get_session() as session:
            result = session.execute("SHOW SPACES")
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to show spaces: {_safe_error_msg(result)}")
            
            for row in result:
                space_name = row.values()[0].as_string()
                spaces.append(space_name)
                
        logger.info(f"Found {len(spaces)} spaces: {spaces}")
        return spaces
    except Exception as e:
        logger.error(f"Error getting spaces: {e}")
        return []


def get_test_spaces(spaces: List[str]) -> List[str]:
    """Filter spaces that start with 'test'."""
    test_spaces = [space for space in spaces if space.lower().startswith('test')]
    return test_spaces


def drop_space(space_name: str) -> bool:
    """Drop a single space."""
    try:
        with NebulaSyncConnectionManager.get_session() as session:
            logger.info(f"Dropping space: {space_name}")
            result = session.execute(f"DROP SPACE IF EXISTS {space_name}")
            
            if result.is_succeeded():
                logger.info(f"✅ Successfully dropped space: {space_name}")
                return True
            else:
                logger.error(f"❌ Failed to drop space {space_name}: {_safe_error_msg(result)}")
                return False
    except Exception as e:
        logger.error(f"❌ Exception dropping space {space_name}: {e}")
        return False


def main():
    """Main function to clean up test spaces."""
    logger.info("🚀 Starting Nebula test space cleanup...")
    
    # Check environment variables
    required_vars = ["NEBULA_HOST", "NEBULA_PORT", "NEBULA_USER", "NEBULA_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Using default values...")
    
    # Display connection info
    logger.info(f"Connecting to Nebula at {os.getenv('NEBULA_HOST', '127.0.0.1')}:{os.getenv('NEBULA_PORT', '9669')}")
    
    try:
        # Initialize connection manager
        NebulaSyncConnectionManager.initialize()
        
        # Get all spaces
        all_spaces = get_all_spaces()
        if not all_spaces:
            logger.warning("No spaces found or failed to get spaces list")
            return
        
        # Filter test spaces
        test_spaces = get_test_spaces(all_spaces)
        
        if not test_spaces:
            logger.info("✅ No test spaces found to delete")
            return
        
        logger.info(f"🎯 Found {len(test_spaces)} test spaces to delete: {test_spaces}")
        
        # Confirm deletion
        response = input(f"❓ Do you want to delete {len(test_spaces)} test spaces? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            logger.info("❌ Operation cancelled by user")
            return
        
        # Delete test spaces
        success_count = 0
        failed_count = 0
        
        for space in test_spaces:
            if drop_space(space):
                success_count += 1
            else:
                failed_count += 1
        
        # Summary
        logger.info("="*50)
        logger.info("📊 CLEANUP SUMMARY")
        logger.info(f"✅ Successfully deleted: {success_count} spaces")
        logger.info(f"❌ Failed to delete: {failed_count} spaces")
        logger.info(f"📊 Total processed: {len(test_spaces)} spaces")
        
        if failed_count == 0:
            logger.info("🎉 All test spaces cleaned up successfully!")
        else:
            logger.warning(f"⚠️  {failed_count} spaces failed to delete")
            
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        sys.exit(1)
    finally:
        # Clean up connection
        try:
            NebulaSyncConnectionManager.close()
            logger.info("🔌 Nebula connection closed")
        except Exception:
            pass


if __name__ == "__main__":
    main() 