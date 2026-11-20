import sqlalchemy as sa
from alembic import op

revision = "20261120_03"
down_revision = "20261119_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chats", sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade() -> None:
    op.drop_column("chats", "pinned")
