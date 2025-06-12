"""Add version fields to index tables and remove old status fields

Revision ID: 20250101000001
Revises: 20250101000000
Create Date: 2025-01-01 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20250101000001'
down_revision = '20250101000000'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema"""
    
    # Add version field to DocumentIndexSpec
    op.add_column('document_index_specs', 
                  sa.Column('version', sa.Integer(), nullable=False, server_default='1'))
    
    # Add observed_version field to DocumentIndexStatus 
    op.add_column('document_index_statuses',
                  sa.Column('observed_version', sa.Integer(), nullable=False, server_default='0'))
    
    # Remove old index status fields from Document table
    op.drop_column('documents', 'vector_index_status')
    op.drop_column('documents', 'fulltext_index_status') 
    op.drop_column('documents', 'graph_index_status')


def downgrade():
    """Downgrade database schema"""
    
    # Re-add old index status fields to Document table
    op.add_column('documents',
                  sa.Column('vector_index_status', 
                           sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'SKIPPED', name='documentindexstatusold'),
                           nullable=False, server_default='PENDING'))
    op.add_column('documents',
                  sa.Column('fulltext_index_status',
                           sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'SKIPPED', name='documentindexstatusold'),
                           nullable=False, server_default='PENDING'))
    op.add_column('documents',
                  sa.Column('graph_index_status',
                           sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'SKIPPED', name='documentindexstatusold'),
                           nullable=False, server_default='PENDING'))
    
    # Remove version fields
    op.drop_column('document_index_statuses', 'observed_version')
    op.drop_column('document_index_specs', 'version') 