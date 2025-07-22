"""Convert all enum columns to varchar for better flexibility

Revision ID: 20250718000000
Revises: c5cca2fe6e4e
Create Date: 2025-07-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3a2c218442f'
down_revision: Union[str, None] = 'c5cca2fe6e4e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Convert enum columns to varchar."""
    
    # Collection table
    op.alter_column('collection', 'status',
                   existing_type=sa.Enum('INACTIVE', 'ACTIVE', 'DELETED', name='collectionstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    op.alter_column('collection', 'type',
                   existing_type=sa.Enum('document', name='collectiontype'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Collection Summary table
    op.alter_column('collection_summary', 'status',
                   existing_type=sa.Enum('PENDING', 'GENERATING', 'COMPLETE', 'FAILED', name='collectionsummarystatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Document table
    op.alter_column('document', 'status',
                   existing_type=sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'DELETED', name='documentstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Bot table
    op.alter_column('bot', 'type',
                   existing_type=sa.Enum('knowledge', 'common', name='bottype'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    op.alter_column('bot', 'status',
                   existing_type=sa.Enum('ACTIVE', 'DELETED', name='botstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Chat table
    op.alter_column('chat', 'peer_type',
                   existing_type=sa.Enum('system', 'feishu', 'weixin', 'weixin_official', 'web', 'dingtalk', name='chatpeertype'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    op.alter_column('chat', 'status',
                   existing_type=sa.Enum('ACTIVE', 'DELETED', name='chatstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Message Feedback table
    op.alter_column('message_feedback', 'type',
                   existing_type=sa.Enum('good', 'bad', name='messagefeedbacktype'),
                   type_=sa.String(length=70),
                   existing_nullable=True)
    
    op.alter_column('message_feedback', 'tag',
                   existing_type=sa.Enum('Harmful', 'Unsafe', 'Fake', 'Unhelpful', 'Other', name='messagefeedbacktag'),
                   type_=sa.String(length=70),
                   existing_nullable=True)
    
    op.alter_column('message_feedback', 'status',
                   existing_type=sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', name='messagefeedbackstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=True)
    
    # API Key table
    op.alter_column('api_key', 'status',
                   existing_type=sa.Enum('ACTIVE', 'DELETED', name='apikeystatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Model Service Provider table
    op.alter_column('model_service_provider', 'status',
                   existing_type=sa.Enum('ACTIVE', 'INACTIVE', 'DELETED', name='modelserviceproviderstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # LLM Provider Models table
    op.alter_column('llm_provider_models', 'api',
                   existing_type=sa.Enum('completion', 'embedding', 'rerank', name='apitype'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # User table
    op.alter_column('user', 'role',
                   existing_type=sa.Enum('admin', 'rw', 'ro', name='role'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Invitation table
    op.alter_column('invitation', 'role',
                   existing_type=sa.Enum('admin', 'rw', 'ro', name='role'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Document Index table
    op.alter_column('document_index', 'index_type',
                   existing_type=sa.Enum('VECTOR', 'FULLTEXT', 'GRAPH', 'SUMMARY', name='documentindextype'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    op.alter_column('document_index', 'status',
                   existing_type=sa.Enum('PENDING', 'CREATING', 'ACTIVE', 'DELETING', 'DELETION_IN_PROGRESS', 'FAILED', name='documentindexstatus'),
                   type_=sa.String(length=70),
                   existing_nullable=False)
    
    # Audit Log table
    op.alter_column('audit_log', 'resource_type',
                   existing_type=sa.Enum('collection', 'document', 'bot', 'chat', 'message', 'api_key', 'llm_provider', 'llm_provider_model', 'model_service_provider', 'user', 'config', 'invitation', 'auth', 'chat_completion', 'search', 'llm', 'flow', 'system', 'index', name='auditresource'),
                   type_=sa.String(length=70),
                   existing_nullable=True)
    
    # Merge Suggestion table (if exists)
    try:
        op.alter_column('graph_index_merge_suggestions', 'status',
                       existing_type=sa.Enum('PENDING', 'ACCEPTED', 'REJECTED', 'EXPIRED', name='mergesuggestionstatus'),
                       type_=sa.String(length=70),
                       existing_nullable=False)
    except Exception:
        # Table might not exist yet
        pass


def downgrade() -> None:
    """Revert varchar columns back to enums."""
    
    # Note: This downgrade is complex because we need to recreate the enum types
    # and convert the data back. For simplicity, we'll just change the column types
    # back to varchar with the original enum constraints.
    
    # Collection table
    op.alter_column('collection', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('INACTIVE', 'ACTIVE', 'DELETED', name='collectionstatus'),
                   existing_nullable=False)
    
    op.alter_column('collection', 'type',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('document', name='collectiontype'),
                   existing_nullable=False)
    
    # Collection Summary table
    op.alter_column('collection_summary', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('PENDING', 'GENERATING', 'COMPLETE', 'FAILED', name='collectionsummarystatus'),
                   existing_nullable=False)
    
    # Document table
    op.alter_column('document', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'DELETED', name='documentstatus'),
                   existing_nullable=False)
    
    # Bot table
    op.alter_column('bot', 'type',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('knowledge', 'common', name='bottype'),
                   existing_nullable=False)
    
    op.alter_column('bot', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('ACTIVE', 'DELETED', name='botstatus'),
                   existing_nullable=False)
    
    # Chat table
    op.alter_column('chat', 'peer_type',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('system', 'feishu', 'weixin', 'weixin_official', 'web', 'dingtalk', name='chatpeertype'),
                   existing_nullable=False)
    
    op.alter_column('chat', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('ACTIVE', 'DELETED', name='chatstatus'),
                   existing_nullable=False)
    
    # Message Feedback table
    op.alter_column('message_feedback', 'type',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('good', 'bad', name='messagefeedbacktype'),
                   existing_nullable=True)
    
    op.alter_column('message_feedback', 'tag',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('Harmful', 'Unsafe', 'Fake', 'Unhelpful', 'Other', name='messagefeedbacktag'),
                   existing_nullable=True)
    
    op.alter_column('message_feedback', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', name='messagefeedbackstatus'),
                   existing_nullable=True)
    
    # API Key table
    op.alter_column('api_key', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('ACTIVE', 'DELETED', name='apikeystatus'),
                   existing_nullable=False)
    
    # Model Service Provider table
    op.alter_column('model_service_provider', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('ACTIVE', 'INACTIVE', 'DELETED', name='modelserviceproviderstatus'),
                   existing_nullable=False)
    
    # LLM Provider Models table
    op.alter_column('llm_provider_models', 'api',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('completion', 'embedding', 'rerank', name='apitype'),
                   existing_nullable=False)
    
    # User table
    op.alter_column('user', 'role',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('admin', 'rw', 'ro', name='role'),
                   existing_nullable=False)
    
    # Invitation table
    op.alter_column('invitation', 'role',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('admin', 'rw', 'ro', name='role'),
                   existing_nullable=False)
    
    # Document Index table
    op.alter_column('document_index', 'index_type',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('VECTOR', 'FULLTEXT', 'GRAPH', 'SUMMARY', name='documentindextype'),
                   existing_nullable=False)
    
    op.alter_column('document_index', 'status',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('PENDING', 'CREATING', 'ACTIVE', 'DELETING', 'DELETION_IN_PROGRESS', 'FAILED', name='documentindexstatus'),
                   existing_nullable=False)
    
    # Audit Log table
    op.alter_column('audit_log', 'resource_type',
                   existing_type=sa.String(length=70),
                   type_=sa.Enum('collection', 'document', 'bot', 'chat', 'message', 'api_key', 'llm_provider', 'llm_provider_model', 'model_service_provider', 'user', 'config', 'invitation', 'auth', 'chat_completion', 'search', 'llm', 'flow', 'system', 'index', name='auditresource'),
                   existing_nullable=True)
    
    # Merge Suggestion table (if exists)
    try:
        op.alter_column('graph_index_merge_suggestions', 'status',
                       existing_type=sa.String(length=70),
                       type_=sa.Enum('PENDING', 'ACCEPTED', 'REJECTED', 'EXPIRED', name='mergesuggestionstatus'),
                       existing_nullable=False)
    except Exception:
        # Table might not exist yet
        pass
