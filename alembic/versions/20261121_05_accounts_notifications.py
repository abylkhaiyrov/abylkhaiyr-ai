import uuid

import sqlalchemy as sa
from alembic import op

revision = "20261121_05"
down_revision = "20261120_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("uuid", sa.String(36)))
    op.add_column("users", sa.Column("first_name", sa.String(64), nullable=False, server_default=""))
    op.add_column("users", sa.Column("last_name", sa.String(64), nullable=False, server_default=""))
    op.add_column("users", sa.Column("email", sa.String(254)))
    op.add_column("users", sa.Column("role", sa.String(16), nullable=False, server_default="USER"))
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("users", sa.Column("last_login_at", sa.DateTime()))
    op.add_column("users", sa.Column("last_active_at", sa.DateTime()))
    connection = op.get_bind()
    for (identifier,) in connection.execute(sa.text("SELECT id FROM users")).all():
        connection.execute(
            sa.text("UPDATE users SET uuid = :uuid WHERE id = :id"),
            {"uuid": str(uuid.uuid4()), "id": identifier},
        )
    connection.execute(sa.text("UPDATE users SET role = 'ADMIN' WHERE username = 'admin'"))
    op.create_index("uq_users_uuid", "users", ["uuid"], unique=True)
    op.create_index("uq_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"])
    op.add_column("sessions", sa.Column("last_seen_at", sa.DateTime()))
    op.add_column("sessions", sa.Column("user_agent", sa.String(256)))
    op.add_column("chats", sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.create_index("ix_chats_archived", "chats", ["archived"])
    op.add_column("generation_jobs", sa.Column("started_at", sa.DateTime()))
    op.add_column("generation_jobs", sa.Column("completed_at", sa.DateTime()))
    op.create_table(
        "notifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("uuid", sa.String(36), nullable=False, unique=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("message", sa.Text(), nullable=False, server_default=""),
        sa.Column("chat_id", sa.String(36)),
        sa.Column("generation_id", sa.String(64)),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("read_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_notifications_user_read_created", "notifications", ["user_id", "is_read", "created_at"])
    op.create_index(
        "uq_notifications_user_generation",
        "notifications",
        ["user_id", "generation_id"],
        unique=True,
        sqlite_where=sa.text("generation_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("notifications")
    for column in ("completed_at", "started_at"):
        op.drop_column("generation_jobs", column)
    op.drop_column("chats", "archived")
    for column in ("user_agent", "last_seen_at"):
        op.drop_column("sessions", column)
    for column in ("last_active_at", "last_login_at", "is_active", "role", "email", "last_name", "first_name", "uuid"):
        op.drop_column("users", column)
