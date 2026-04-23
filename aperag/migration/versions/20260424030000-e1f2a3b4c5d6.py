"""align evaluation_datasets.source_type length with EnumColumn (70 -> 50)

The column was created by revision ``b9c8d7e6f1a2`` with a hand-coded
``sa.String(length=70)``. The model at ``aperag.db.models.EvaluationDataset``
declares it as ``Column(EnumColumn(EvaluationDatasetSourceType), ...)``, and
``EnumColumn`` resolves the length from the enum values as
``max(max(len(v) for v in enum) + 20, 50)`` — which is 50 for this enum
(values ``manual / import / generated``; longest is 9). The 70 in the
original migration was author drift, not intent.

The mismatch was latent until the CI alembic drift check introduced by
``#21`` started running on every push. Shrinking to 50 is safe because
no recorded enum value exceeds 9 characters, so no row can truncate.

Revision ID: e1f2a3b4c5d6
Revises: c1d2e3f4a5b7
Create Date: 2026-04-24 03:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "c1d2e3f4a5b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "evaluation_datasets",
        "source_type",
        existing_type=sa.String(length=70),
        type_=sa.String(length=50),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "evaluation_datasets",
        "source_type",
        existing_type=sa.String(length=50),
        type_=sa.String(length=70),
        existing_nullable=False,
    )
