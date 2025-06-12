#!/usr/bin/env python3
"""
Test script for the simplified K8s-inspired document index system

This script demonstrates the two simple chains:
1. Frontend chain: Document operations -> Index specs
2. Backend chain: Reconciliation -> Index status updates

Usage:
    python test_simple_system.py

Requirements:
    - Database must be running and properly configured
    - Run from the aperag project root directory
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_frontend_chain():
    """Test the frontend chain - document operations creating/updating specs"""
    logger.info("=== Testing Frontend Chain ===")
    
    # Import components
    from aperag.document_index_manager import document_index_manager
    from aperag.db.ops import get_session
    from aperag.db.models import DocumentIndexType
    
    test_document_id = "test_doc_12345"
    test_user = "test_user"
    
    async with get_session() as session:
        # Test 1: Create document indexes
        logger.info(f"Creating index specs for document {test_document_id}")
        await document_index_manager.create_document_indexes(
            session, test_document_id, test_user
        )
        await session.commit()
        
        # Test 2: Get status
        status = await document_index_manager.get_document_index_status(session, test_document_id)
        logger.info(f"Document status: {status}")
        
        # Test 3: Update document (increment version)
        logger.info(f"Updating document {test_document_id} (incrementing version)")
        await document_index_manager.update_document_indexes(session, test_document_id)
        await session.commit()
        
        # Test 4: Get updated status
        status = await document_index_manager.get_document_index_status(session, test_document_id)
        logger.info(f"Updated document status: {status}")
        
        # Test 5: Delete document indexes
        logger.info(f"Marking document {test_document_id} for deletion")
        await document_index_manager.delete_document_indexes(session, test_document_id)
        await session.commit()
        
        # Test 6: Get final status
        status = await document_index_manager.get_document_index_status(session, test_document_id)
        logger.info(f"Final document status: {status}")


async def test_backend_chain():
    """Test the backend chain - reconciliation processing specs"""
    logger.info("\n=== Testing Backend Chain ===")
    
    # Import components
    from aperag.index_reconciler import index_reconciler
    
    # Test reconciliation
    logger.info("Running reconciliation...")
    await index_reconciler.reconcile_all()
    logger.info("Reconciliation completed")


async def test_integration():
    """Test integration between frontend and backend chains"""
    logger.info("\n=== Testing Integration ===")
    
    from aperag.document_index_manager import document_index_manager
    from aperag.index_reconciler import index_reconciler
    from aperag.db.ops import get_session
    
    test_document_id = "integration_test_doc"
    test_user = "integration_user"
    
    # Step 1: Create specs (frontend)
    async with get_session() as session:
        logger.info(f"Step 1: Creating specs for {test_document_id}")
        await document_index_manager.create_document_indexes(
            session, test_document_id, test_user
        )
        await session.commit()
    
    # Step 2: Run reconciliation (backend)
    logger.info("Step 2: Running reconciliation")
    await index_reconciler.reconcile_all()
    
    # Step 3: Check results
    async with get_session() as session:
        status = await document_index_manager.get_document_index_status(session, test_document_id)
        logger.info(f"Step 3: Integration test results: {status}")
    
    # Step 4: Update and reconcile again
    async with get_session() as session:
        logger.info("Step 4: Updating document and reconciling")
        await document_index_manager.update_document_indexes(session, test_document_id)
        await session.commit()
    
    await index_reconciler.reconcile_all()
    
    # Step 5: Final check
    async with get_session() as session:
        status = await document_index_manager.get_document_index_status(session, test_document_id)
        logger.info(f"Step 5: Final integration test results: {status}")


async def test_version_tracking():
    """Test version tracking (like k8s generation)"""
    logger.info("\n=== Testing Version Tracking ===")
    
    from aperag.document_index_manager import document_index_manager
    from aperag.db.ops import get_session
    from aperag.db.models import DocumentIndexSpec, DocumentIndexStatus
    from sqlalchemy import select, and_
    
    test_document_id = "version_test_doc"
    test_user = "version_user"
    
    async with get_session() as session:
        # Create initial specs
        await document_index_manager.create_document_indexes(
            session, test_document_id, test_user
        )
        await session.commit()
        
        # Check initial versions
        spec_stmt = select(DocumentIndexSpec).where(DocumentIndexSpec.document_id == test_document_id)
        specs = (await session.execute(spec_stmt)).scalars().all()
        
        for spec in specs:
            logger.info(f"Initial spec {spec.index_type.value}: version {spec.version}")
        
        # Update document (should increment versions)
        await document_index_manager.update_document_indexes(session, test_document_id)
        await session.commit()
        
        # Check updated versions
        specs = (await session.execute(spec_stmt)).scalars().all()
        for spec in specs:
            logger.info(f"Updated spec {spec.index_type.value}: version {spec.version}")
        
        # Simulate reconciliation creating status
        for spec in specs:
            status = DocumentIndexStatus(
                document_id=spec.document_id,
                index_type=spec.index_type,
                observed_version=spec.version - 1  # Simulate being behind
            )
            session.add(status)
        await session.commit()
        
        # Check if reconciliation is needed
        status_stmt = select(DocumentIndexStatus).where(DocumentIndexStatus.document_id == test_document_id)
        statuses = (await session.execute(status_stmt)).scalars().all()
        
        for status in statuses:
            spec = next(s for s in specs if s.index_type == status.index_type)
            needs_reconciliation = status.needs_reconciliation(spec)
            logger.info(f"Status {status.index_type.value}: observed_version={status.observed_version}, "
                       f"spec_version={spec.version}, needs_reconciliation={needs_reconciliation}")


async def cleanup_test_data():
    """Clean up test data"""
    logger.info("\n=== Cleaning up test data ===")
    
    from aperag.db.ops import get_session
    from aperag.db.models import DocumentIndexSpec, DocumentIndexStatus
    from sqlalchemy import delete
    
    test_documents = ["test_doc_12345", "integration_test_doc", "version_test_doc"]
    
    async with get_session() as session:
        for doc_id in test_documents:
            # Delete specs
            spec_stmt = delete(DocumentIndexSpec).where(DocumentIndexSpec.document_id == doc_id)
            await session.execute(spec_stmt)
            
            # Delete statuses  
            status_stmt = delete(DocumentIndexStatus).where(DocumentIndexStatus.document_id == doc_id)
            await session.execute(status_stmt)
        
        await session.commit()
        logger.info("Test data cleaned up")


async def main():
    """Main test function"""
    logger.info("Starting simplified system tests...")
    
    try:
        # Run tests
        await test_frontend_chain()
        await test_backend_chain()
        await test_integration()
        await test_version_tracking()
        
        logger.info("\n=== All Tests Completed Successfully! ===")
        
        # Show system summary
        logger.info("\nSystem Summary:")
        logger.info("- Frontend Chain: Document operations -> Spec updates (✓)")
        logger.info("- Backend Chain: Reconciliation -> Status updates (✓)")
        logger.info("- Version Tracking: K8s-style generation field (✓)")
        logger.info("- Integration: Frontend + Backend working together (✓)")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    finally:
        # Cleanup
        await cleanup_test_data()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 