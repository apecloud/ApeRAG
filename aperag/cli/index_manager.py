#!/usr/bin/env python3
"""
CLI tool for managing the simplified document index system

Usage:
    python -m aperag.cli.index_manager --help
    python -m aperag.cli.index_manager status --document-id doc123
    python -m aperag.cli.index_manager reconcile
    python -m aperag.cli.index_manager create --document-id doc123 --user admin
    python -m aperag.cli.index_manager delete --document-id doc123
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_document_status(document_id: str):
    """Get document index status"""
    from aperag.db.ops import get_session
    from aperag.index.manager import document_index_manager
    
    async with get_session() as session:
        status = await document_index_manager.get_document_index_status(session, document_id)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return status


async def run_reconciliation():
    """Run reconciliation manually"""
    from aperag.index.reconciler import index_reconciler
    
    logger.info("Starting manual reconciliation...")
    await index_reconciler.reconcile_all()
    logger.info("Reconciliation completed")


async def create_document_indexes(document_id: str, user: str, index_types: Optional[list] = None):
    """Create document indexes"""
    from aperag.db.models import DocumentIndexType
    from aperag.db.ops import get_session
    from aperag.index.manager import document_index_manager
    
    if index_types:
        types = [DocumentIndexType(t) for t in index_types]
    else:
        types = None
    
    async with get_session() as session:
        await document_index_manager.create_document_indexes(
            session, document_id, user, types
        )
        await session.commit()
        
        # Show status
        status = await document_index_manager.get_document_index_status(session, document_id)
        print(f"Created indexes for document {document_id}")
        print(json.dumps(status, indent=2, ensure_ascii=False))


async def delete_document_indexes(document_id: str, index_types: Optional[list] = None):
    """Delete document indexes"""
    from aperag.db.models import DocumentIndexType
    from aperag.db.ops import get_session
    from aperag.index.manager import document_index_manager
    
    if index_types:
        types = [DocumentIndexType(t) for t in index_types]
    else:
        types = None
    
    async with get_session() as session:
        await document_index_manager.delete_document_indexes(
            session, document_id, types
        )
        await session.commit()
        
        # Show status
        status = await document_index_manager.get_document_index_status(session, document_id)
        print(f"Marked indexes for deletion for document {document_id}")
        print(json.dumps(status, indent=2, ensure_ascii=False))


async def update_document_indexes(document_id: str):
    """Update document indexes (increment version)"""
    from aperag.db.ops import get_session
    from aperag.index.manager import document_index_manager
    
    async with get_session() as session:
        await document_index_manager.update_document_indexes(session, document_id)
        await session.commit()
        
        # Show status
        status = await document_index_manager.get_document_index_status(session, document_id)
        print(f"Updated indexes for document {document_id}")
        print(json.dumps(status, indent=2, ensure_ascii=False))


async def list_documents_needing_reconciliation():
    """List documents that need reconciliation"""
    from aperag.db.ops import get_session
    from aperag.index.reconciler import index_reconciler
    
    async with get_session() as session:
        pairs = await index_reconciler._get_specs_needing_reconciliation(session)
        
        if not pairs:
            print("No documents need reconciliation")
            return
        
        print(f"Found {len(pairs)} documents needing reconciliation:")
        for spec, status in pairs:
            print(f"- Document {spec.document_id}, Index {spec.index_type.value}")
            print(f"  Spec version: {spec.version}, Status version: {status.observed_version}")
            print(f"  Desired: {spec.desired_state.value}, Actual: {status.actual_state.value}")
            if status.error_message:
                print(f"  Error: {status.error_message}")
            print()


async def show_system_stats():
    """Show system statistics"""
    from sqlalchemy import func, select

    from aperag.db.models import DocumentIndexSpec, DocumentIndexStatus
    from aperag.db.ops import get_session
    
    async with get_session() as session:
        # Count specs by desired state
        spec_stmt = select(
            DocumentIndexSpec.desired_state,
            func.count(DocumentIndexSpec.id)
        ).group_by(DocumentIndexSpec.desired_state)
        spec_results = await session.execute(spec_stmt)
        spec_counts = dict(spec_results.fetchall())
        
        # Count statuses by actual state
        status_stmt = select(
            DocumentIndexStatus.actual_state,
            func.count(DocumentIndexStatus.id)
        ).group_by(DocumentIndexStatus.actual_state)
        status_results = await session.execute(status_stmt)
        status_counts = dict(status_results.fetchall())
        
        # Count version mismatches
        mismatch_stmt = select(func.count(DocumentIndexSpec.id)).select_from(
            DocumentIndexSpec.__table__.join(
                DocumentIndexStatus.__table__,
                (DocumentIndexSpec.document_id == DocumentIndexStatus.document_id) &
                (DocumentIndexSpec.index_type == DocumentIndexStatus.index_type)
            )
        ).where(DocumentIndexSpec.version > DocumentIndexStatus.observed_version)
        mismatch_result = await session.execute(mismatch_stmt)
        version_mismatches = mismatch_result.scalar()
        
        print("=== Index System Statistics ===")
        print("\nSpecs by desired state:")
        for state, count in spec_counts.items():
            print(f"  {state.value}: {count}")
        
        print("\nStatuses by actual state:")
        for state, count in status_counts.items():
            print(f"  {state.value}: {count}")
        
        print(f"\nVersion mismatches: {version_mismatches}")


def main():
    parser = argparse.ArgumentParser(description="Document Index Manager CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get document index status')
    status_parser.add_argument('--document-id', required=True, help='Document ID')
    
    # Reconcile command
    reconcile_parser = subparsers.add_parser('reconcile', help='Run reconciliation manually')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create document indexes')
    create_parser.add_argument('--document-id', required=True, help='Document ID')
    create_parser.add_argument('--user', required=True, help='User creating the indexes')
    create_parser.add_argument('--types', nargs='*', choices=['VECTOR', 'FULLTEXT', 'GRAPH'],
                              help='Index types to create (default: all)')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete document indexes')
    delete_parser.add_argument('--document-id', required=True, help='Document ID')
    delete_parser.add_argument('--types', nargs='*', choices=['VECTOR', 'FULLTEXT', 'GRAPH'],
                              help='Index types to delete (default: all)')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update document indexes')
    update_parser.add_argument('--document-id', required=True, help='Document ID')
    
    # List command
    list_parser = subparsers.add_parser('list-pending', help='List documents needing reconciliation')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'status':
            asyncio.run(get_document_status(args.document_id))
        elif args.command == 'reconcile':
            asyncio.run(run_reconciliation())
        elif args.command == 'create':
            asyncio.run(create_document_indexes(args.document_id, args.user, args.types))
        elif args.command == 'delete':
            asyncio.run(delete_document_indexes(args.document_id, args.types))
        elif args.command == 'update':
            asyncio.run(update_document_indexes(args.document_id))
        elif args.command == 'list-pending':
            asyncio.run(list_documents_needing_reconciliation())
        elif args.command == 'stats':
            asyncio.run(show_system_stats())
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 