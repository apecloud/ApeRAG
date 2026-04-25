"""align model_provider.provider_type unique index

The ``model_provider`` table introduced by ``b4f2d91c8e3a`` declared
``provider_type`` with both a separate ``UniqueConstraint`` and a
non-unique index, but the ORM in
``aperag.domains.model_platform.db.models.ModelProvider`` declares the
column as ``unique=True, index=True`` (which SQLAlchemy renders as a
single unique index). ``alembic check`` flagged the drift on CI. This
revision drops the redundant unique constraint and promotes the index
to ``unique=True`` so the autogenerate diff reports clean.

Revision ID: 84fac9e3d8c2
Revises: b4f2d91c8e3a
Create Date: 2026-04-25 21:38:31.188600
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "84fac9e3d8c2"
down_revision: Union[str, None] = "b4f2d91c8e3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(op.f("model_provider_provider_type_key"), "model_provider", type_="unique")
    op.drop_index(op.f("ix_model_provider_provider_type"), table_name="model_provider")
    op.create_index(
        op.f("ix_model_provider_provider_type"),
        "model_provider",
        ["provider_type"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_model_provider_provider_type"), table_name="model_provider")
    op.create_index(
        op.f("ix_model_provider_provider_type"),
        "model_provider",
        ["provider_type"],
        unique=False,
    )
    op.create_unique_constraint(
        op.f("model_provider_provider_type_key"),
        "model_provider",
        ["provider_type"],
    )
