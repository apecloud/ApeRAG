"""audit_log_table

Revision ID: 20250617113449
Revises: 12ea6d2bf365
Create Date: 2025-06-17 11:34:49.123456

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20250617113449'
down_revision = '12ea6d2bf365'
branch_labels = None
depends_on = None


def upgrade():
    # Create ENUM types
    audit_resource_enum = postgresql.ENUM(
        'COLLECTION', 'DOCUMENT', 'BOT', 'CHAT', 'MESSAGE', 
        'API_KEY', 'LLM_PROVIDER', 'LLM_PROVIDER_MODEL', 
        'MODEL_SERVICE_PROVIDER', 'USER', 'CONFIG',
        name='auditresource',
        create_type=False
    )
    audit_resource_enum.create(op.get_bind(), checkfirst=True)

    # Create audit_log table
    op.create_table(
        'audit_log',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), nullable=True, comment='User ID'),
        sa.Column('username', sa.String(255), nullable=True, comment='Username'),
        sa.Column('resource_type', audit_resource_enum, nullable=True, comment='Resource type'),
        sa.Column('resource_id', sa.String(255), nullable=True, comment='Resource ID (extracted at query time)'),
        sa.Column('api_name', sa.String(255), nullable=False, comment='API operation name'),
        sa.Column('http_method', sa.String(10), nullable=False, comment='HTTP method (POST, PUT, DELETE)'),
        sa.Column('path', sa.String(512), nullable=False, comment='API path'),
        sa.Column('status_code', sa.Integer, nullable=True, comment='HTTP status code'),
        sa.Column('start_time', sa.BigInteger, nullable=False, comment='Request start time (milliseconds since epoch)'),
        sa.Column('end_time', sa.BigInteger, nullable=True, comment='Request end time (milliseconds since epoch)'),
        sa.Column('request_data', sa.Text, nullable=True, comment='Request data (JSON)'),
        sa.Column('response_data', sa.Text, nullable=True, comment='Response data (JSON)'),
        sa.Column('error_message', sa.Text, nullable=True, comment='Error message if failed'),
        sa.Column('ip_address', sa.String(45), nullable=True, comment='Client IP address'),
        sa.Column('user_agent', sa.String(500), nullable=True, comment='User agent string'),
        sa.Column('request_id', sa.String(255), nullable=False, comment='Request ID for tracking'),
        sa.Column('gmt_created', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), comment='Created time'),
    )

    # Create indexes for better query performance
    op.create_index('idx_audit_user_id', 'audit_log', ['user_id'])
    op.create_index('idx_audit_resource_type', 'audit_log', ['resource_type'])
    op.create_index('idx_audit_api_name', 'audit_log', ['api_name'])
    op.create_index('idx_audit_http_method', 'audit_log', ['http_method'])
    op.create_index('idx_audit_status_code', 'audit_log', ['status_code'])
    op.create_index('idx_audit_start_time', 'audit_log', ['start_time'])
    op.create_index('idx_audit_gmt_created', 'audit_log', ['gmt_created'])
    op.create_index('idx_audit_resource_id', 'audit_log', ['resource_id'])
    op.create_index('idx_audit_request_id', 'audit_log', ['request_id'])


def downgrade():
    # Drop indexes
    op.drop_index('idx_audit_request_id', 'audit_log')
    op.drop_index('idx_audit_resource_id', 'audit_log')
    op.drop_index('idx_audit_gmt_created', 'audit_log')
    op.drop_index('idx_audit_start_time', 'audit_log')
    op.drop_index('idx_audit_status_code', 'audit_log')
    op.drop_index('idx_audit_http_method', 'audit_log')
    op.drop_index('idx_audit_api_name', 'audit_log')
    op.drop_index('idx_audit_resource_type', 'audit_log')
    op.drop_index('idx_audit_user_id', 'audit_log')
    
    # Drop table
    op.drop_table('audit_log')
    
    # Drop ENUM types
    postgresql.ENUM(name='auditresource').drop(op.get_bind(), checkfirst=True) 