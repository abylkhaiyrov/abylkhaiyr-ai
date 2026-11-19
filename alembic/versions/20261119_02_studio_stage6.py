import sqlalchemy as sa
from alembic import op

revision = "20261119_02"
down_revision = "20261119_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("model_id", sa.String(256)))
    op.add_column("messages", sa.Column("checkpoint_id", sa.String(1024)))
    op.add_column("messages", sa.Column("generation_params", sa.Text()))
    op.add_column("messages", sa.Column("model_config_json", sa.Text()))
    op.add_column("messages", sa.Column("reasoning_trace", sa.Text()))
    op.add_column(
        "messages",
        sa.Column("status", sa.String(32), nullable=False, server_default="completed"),
    )
    op.create_table(
        "generation_jobs",
        sa.Column("request_id", sa.String(64), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chat_id",
            sa.String(36),
            sa.ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_message_id", sa.String(36)),
        sa.Column("assistant_message_id", sa.String(36)),
        sa.Column("action", sa.String(32), nullable=False, server_default="send"),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("sequence_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("partial_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("reasoning_events", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("model_id", sa.String(256)),
        sa.Column("checkpoint_id", sa.String(1024)),
        sa.Column("queue_reason", sa.String(128)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_detail", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_generation_jobs_user_id", "generation_jobs", ["user_id"])
    op.create_index("ix_generation_jobs_chat_id", "generation_jobs", ["chat_id"])
    op.create_index("ix_generation_jobs_status", "generation_jobs", ["status"])
    op.create_index("ix_generation_jobs_created_at", "generation_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_table("generation_jobs")
    for column in (
        "status",
        "reasoning_trace",
        "model_config_json",
        "generation_params",
        "checkpoint_id",
        "model_id",
    ):
        op.drop_column("messages", column)
