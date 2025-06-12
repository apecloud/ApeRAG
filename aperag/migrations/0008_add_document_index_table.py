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

"""
Add DocumentIndex table for declarative index management

Revision ID: 0008
Revises: 0007
Create Date: 2025-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None

# Define enum types
document_index_status_enum = postgresql.ENUM(
    'pending', 'running', 'complete', 'failed', 'deleted',
    name='document_index_status_enum'
)

document_index_type_enum = postgresql.ENUM(
    'vector', 'fulltext', 'graph',
    name='document_index_type_enum'
)


def upgrade():
    """Add DocumentIndex table"""
    
    # Create enum types
    document_index_status_enum.create(op.get_bind())
    document_index_type_enum.create(op.get_bind())
    
    # Create document_indexes table
    op.create_table(
        'document_indexes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.String(24), nullable=False),
        sa.Column('index_type', document_index_type_enum, nullable=False),
        sa.Column('status', document_index_status_enum, nullable=False, default='pending'),
        sa.Column('execution_id', sa.String(64), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, default=1),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('index_data', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('max_retries', sa.Integer(), nullable=False, default=3),
        sa.Column('gmt_created', sa.DateTime(), nullable=False),
        sa.Column('gmt_modified', sa.DateTime(), nullable=False),
        sa.Column('gmt_started', sa.DateTime(), nullable=True),
        sa.Column('gmt_completed', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'index_type', name='uq_document_index_type')
    )
    
    # Create indexes
    op.create_index(
        'idx_document_index_document_id',
        'document_indexes',
        ['document_id']
    )
    op.create_index(
        'idx_document_index_type',
        'document_indexes',
        ['index_type']
    )
    op.create_index(
        'idx_document_index_status',
        'document_indexes',
        ['status']
    )
    op.create_index(
        'idx_document_index_status_type',
        'document_indexes',
        ['status', 'index_type']
    )
    op.create_index(
        'idx_document_index_pending',
        'document_indexes',
        ['status']
    )
    op.create_index(
        'idx_document_index_execution',
        'document_indexes',
        ['execution_id']
    )
    op.create_index(
        'idx_document_index_locked',
        'document_indexes',
        ['locked_until']
    )


def downgrade():
    """Drop DocumentIndex table"""
    
    # Drop indexes
    op.drop_index('idx_document_index_locked', table_name='document_indexes')
    op.drop_index('idx_document_index_execution', table_name='document_indexes')
    op.drop_index('idx_document_index_pending', table_name='document_indexes')
    op.drop_index('idx_document_index_status_type', table_name='document_indexes')
    op.drop_index('idx_document_index_status', table_name='document_indexes')
    op.drop_index('idx_document_index_type', table_name='document_indexes')
    op.drop_index('idx_document_index_document_id', table_name='document_indexes')
    
    # Drop table
    op.drop_table('document_indexes')
    
    # Drop enum types
    document_index_type_enum.drop(op.get_bind())
    document_index_status_enum.drop(op.get_bind()) 