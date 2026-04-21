"""merge concurrent app startup migration heads

Revision ID: c1d2e3f4a5b6
Revises: f17a6b9c2d4e, b7c3d4e5f6a7
Create Date: 2026-04-22 02:45:00.000000

"""

from typing import Sequence, Union


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = ("f17a6b9c2d4e", "b7c3d4e5f6a7")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
