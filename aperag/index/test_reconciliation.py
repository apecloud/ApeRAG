#!/usr/bin/env python3
"""
Test script for the K8s-inspired reconciliation system

This script demonstrates and tests the basic functionality of the new
document index management system.
"""

import asyncio
import logging
from aperag.index.reconciliation_controller import index_spec_manager, reconciliation_controller
from aperag.services.simple_index_service import simple_index_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_workflow():
    """Test the basic workflow of creating, checking, and deleting indexes"""
    
    test_document_id = "test_doc_123"
    test_user = "test_user@example.com"
    
    logger.info("=== Testing K8s-Inspired Index Management System ===")
    
    try:
        # 1. Create index specifications
        logger.info("1. Creating index specifications...")
        result = await simple_index_service.create_document_indexes(
            document_id=test_document_id,
            user=test_user
        )
        logger.info(f"Create result: {result}")
        
        # 2. Check initial status
        logger.info("2. Checking initial status...")
        status = await simple_index_service.get_document_index_status(test_document_id)
        logger.info(f"Initial status: {status}")
        
        # 3. Run reconciliation manually
        logger.info("3. Running reconciliation...")
        await reconciliation_controller.reconcile_all()
        
        # 4. Check status after reconciliation
        logger.info("4. Checking status after reconciliation...")
        status = await simple_index_service.get_document_index_status(test_document_id)
        logger.info(f"Status after reconciliation: {status}")
        
        # 5. Get system status
        logger.info("5. Getting system status...")
        system_status = await simple_index_service.get_system_index_status()
        logger.info(f"System status: {system_status}")
        
        # 6. Delete index specifications
        logger.info("6. Deleting index specifications...")
        result = await simple_index_service.delete_document_indexes(test_document_id)
        logger.info(f"Delete result: {result}")
        
        # 7. Run reconciliation again
        logger.info("7. Running reconciliation after deletion...")
        await reconciliation_controller.reconcile_all()
        
        # 8. Check final status
        logger.info("8. Checking final status...")
        status = await simple_index_service.get_document_index_status(test_document_id)
        logger.info(f"Final status: {status}")
        
        # 9. Cleanup
        logger.info("9. Cleaning up...")
        result = await simple_index_service.cleanup_document_indexes(test_document_id)
        logger.info(f"Cleanup result: {result}")
        
        logger.info("=== Test completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def test_retry_functionality():
    """Test the retry functionality for failed indexes"""
    
    test_document_id = "test_doc_retry_456"
    test_user = "test_user@example.com"
    
    logger.info("=== Testing Retry Functionality ===")
    
    try:
        # 1. Create index specifications
        logger.info("1. Creating index specifications...")
        await simple_index_service.create_document_indexes(
            document_id=test_document_id,
            user=test_user,
            index_types=["vector"]  # Just test one type
        )
        
        # 2. Simulate failure by manually setting status to failed
        logger.info("2. Simulating index failure...")
        from aperag.db.models import DocumentIndexStatus, DocumentIndexType, IndexActualState
        from aperag.db.ops import get_session
        from sqlalchemy import select, and_
        
        async with get_session() as session:
            stmt = select(DocumentIndexStatus).where(
                and_(
                    DocumentIndexStatus.document_id == test_document_id,
                    DocumentIndexStatus.index_type == DocumentIndexType.VECTOR
                )
            )
            result = await session.execute(stmt)
            status_record = result.scalar_one_or_none()
            
            if status_record:
                status_record.mark_failed("Simulated failure for testing")
                await session.commit()
                logger.info("Marked index as failed")
        
        # 3. Check status
        status = await simple_index_service.get_document_index_status(test_document_id)
        logger.info(f"Status after simulated failure: {status}")
        
        # 4. Retry failed indexes
        logger.info("3. Retrying failed indexes...")
        result = await simple_index_service.retry_failed_indexes(
            document_id=test_document_id,
            user=test_user
        )
        logger.info(f"Retry result: {result}")
        
        # 5. Run reconciliation
        logger.info("4. Running reconciliation...")
        await reconciliation_controller.reconcile_all()
        
        # 6. Check final status
        status = await simple_index_service.get_document_index_status(test_document_id)
        logger.info(f"Status after retry: {status}")
        
        # Cleanup
        await simple_index_service.cleanup_document_indexes(test_document_id)
        
        logger.info("=== Retry test completed! ===")
        
    except Exception as e:
        logger.error(f"Retry test failed: {e}", exc_info=True)


async def main():
    """Main test function"""
    await test_basic_workflow()
    await test_retry_functionality()


if __name__ == "__main__":
    asyncio.run(main()) 