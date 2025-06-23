"""empty message

Revision ID: 23c0533b6b63
Revises: 2768dfee8bbc
Create Date: 2025-06-23 11:06:58.841005

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '23c0533b6b63'
down_revision: Union[str, None] = '2768dfee8bbc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename table from searchtesthistory to searchhistory
    op.rename_table('searchtesthistory', 'searchhistory')
    op.rename_table('document_indexes', 'document_index')
    
    # Update any indexes that reference the old table name
    op.drop_index('ix_searchtesthistory_collection_id', table_name='searchhistory')
    op.drop_index('ix_searchtesthistory_gmt_deleted', table_name='searchhistory')  
    op.drop_index('ix_searchtesthistory_user', table_name='searchhistory')
    
    # Create new indexes with updated names
    op.create_index('ix_searchhistory_collection_id', 'searchhistory', ['collection_id'], unique=False)
    op.create_index('ix_searchhistory_gmt_deleted', 'searchhistory', ['gmt_deleted'], unique=False)
    op.create_index('ix_searchhistory_user', 'searchhistory', ['user'], unique=False)


def downgrade() -> None:
    # Reverse the operation
    op.drop_index('ix_searchhistory_user', table_name='searchhistory')
    op.drop_index('ix_searchhistory_gmt_deleted', table_name='searchhistory')
    op.drop_index('ix_searchhistory_collection_id', table_name='searchhistory')
    
    # Recreate old indexes
    op.create_index('ix_searchtesthistory_user', 'searchtesthistory', ['user'], unique=False)
    op.create_index('ix_searchtesthistory_gmt_deleted', 'searchtesthistory', ['gmt_deleted'], unique=False)
    op.create_index('ix_searchtesthistory_collection_id', 'searchtesthistory', ['collection_id'], unique=False)
    
    # Rename table back
    op.rename_table('searchhistory', 'searchtesthistory') 
    op.rename_table('document_index', 'document_indexes')