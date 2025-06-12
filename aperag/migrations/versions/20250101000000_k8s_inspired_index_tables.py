"""K8s-inspired index tables

Revision ID: k8s_inspired_index_tables
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'k8s_inspired_index_tables'
down_revision = None  # This should be set to the latest revision
head = None

def upgrade():
    # Create new enums
    op.execute("CREATE TYPE indexdesiredstate AS ENUM ('present', 'absent')")
    op.execute("CREATE TYPE indexactualstate AS ENUM ('absent', 'creating', 'present', 'deleting', 'failed')")
    
    # Create document_index_specs table
    op.create_table('document_index_specs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.String(length=24), nullable=False),
        sa.Column('index_type', sa.Enum('VECTOR', 'FULLTEXT', 'GRAPH', name='documentindextype'), nullable=False),
        sa.Column('desired_state', sa.Enum('present', 'absent', name='indexdesiredstate'), nullable=False),
        sa.Column('created_by', sa.String(length=256), nullable=False),
        sa.Column('gmt_created', sa.DateTime(timezone=True), nullable=False),
        sa.Column('gmt_updated', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'index_type', name='uq_document_index_spec')
    )
    
    # Create indexes for document_index_specs
    op.create_index('idx_spec_desired_state', 'document_index_specs', ['desired_state'])
    op.create_index(op.f('ix_document_index_specs_document_id'), 'document_index_specs', ['document_id'])
    op.create_index(op.f('ix_document_index_specs_id'), 'document_index_specs', ['id'])
    op.create_index(op.f('ix_document_index_specs_index_type'), 'document_index_specs', ['index_type'])
    
    # Create document_index_statuses table
    op.create_table('document_index_statuses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.String(length=24), nullable=False),
        sa.Column('index_type', sa.Enum('VECTOR', 'FULLTEXT', 'GRAPH', name='documentindextype'), nullable=False),
        sa.Column('actual_state', sa.Enum('absent', 'creating', 'present', 'deleting', 'failed', name='indexactualstate'), nullable=False),
        sa.Column('index_data', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('gmt_created', sa.DateTime(timezone=True), nullable=False),
        sa.Column('gmt_updated', sa.DateTime(timezone=True), nullable=False),
        sa.Column('gmt_last_reconciled', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'index_type', name='uq_document_index_status')
    )
    
    # Create indexes for document_index_statuses
    op.create_index('idx_status_actual_state', 'document_index_statuses', ['actual_state'])
    op.create_index('idx_status_reconcile', 'document_index_statuses', ['gmt_last_reconciled'])
    op.create_index(op.f('ix_document_index_statuses_document_id'), 'document_index_statuses', ['document_id'])
    op.create_index(op.f('ix_document_index_statuses_id'), 'document_index_statuses', ['id'])
    op.create_index(op.f('ix_document_index_statuses_index_type'), 'document_index_statuses', ['index_type'])
    
    # Drop old complex table if it exists
    op.execute("DROP TABLE IF EXISTS document_indexes CASCADE")


def downgrade():
    # Drop new tables
    op.drop_table('document_index_statuses')
    op.drop_table('document_index_specs')
    
    # Drop new enums
    op.execute("DROP TYPE IF EXISTS indexactualstate")
    op.execute("DROP TYPE IF EXISTS indexdesiredstate")
    
    # Note: We don't recreate the old complex table in downgrade
 