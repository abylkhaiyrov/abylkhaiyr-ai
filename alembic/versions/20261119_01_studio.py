import sqlalchemy as sa
from alembic import op

revision = "20261119_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("users", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("username", sa.String(128), nullable=False, unique=True), sa.Column("password_hash", sa.String(512), nullable=False), sa.Column("requires_password_change", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False))
    op.create_index("ix_users_username", "users", ["username"])
    op.create_table("sessions", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("token_hash", sa.String(128), nullable=False, unique=True), sa.Column("csrf_token", sa.String(128), nullable=False), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("expires_at", sa.DateTime(), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_sessions_token_hash", "sessions", ["token_hash"])
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])
    op.create_table("chats", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("title", sa.String(256), nullable=False), sa.Column("model_name", sa.String(256)), sa.Column("checkpoint", sa.String(1024)), sa.Column("system_prompt", sa.Text(), nullable=False), sa.Column("generation_settings", sa.Text(), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False))
    op.create_index("ix_chats_user_id", "chats", ["user_id"])
    op.create_index("ix_chats_updated_at", "chats", ["updated_at"])
    op.create_table("messages", sa.Column("id", sa.String(36), primary_key=True), sa.Column("chat_id", sa.String(36), sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False), sa.Column("role", sa.String(16), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("input_tokens", sa.Integer()), sa.Column("output_tokens", sa.Integer()), sa.Column("generation_ms", sa.Integer()), sa.Column("tokens_per_second", sa.Float()), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_messages_chat_id", "messages", ["chat_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_index("ix_messages_chat_created", "messages", ["chat_id", "created_at"])
    op.create_table("processes", sa.Column("id", sa.String(36), primary_key=True), sa.Column("pid", sa.Integer(), unique=True), sa.Column("pid_started_at", sa.Float()), sa.Column("kind", sa.String(32), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("command", sa.Text(), nullable=False), sa.Column("config_path", sa.String(1024)), sa.Column("output_dir", sa.String(1024)), sa.Column("started_at", sa.DateTime(), nullable=False), sa.Column("completed_at", sa.DateTime()), sa.Column("exit_code", sa.Integer()), sa.Column("error_reason", sa.Text()))
    op.create_index("ix_processes_kind", "processes", ["kind"])
    op.create_index("ix_processes_status", "processes", ["status"])
    op.create_table("process_events", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("process_id", sa.String(36), sa.ForeignKey("processes.id", ondelete="CASCADE"), nullable=False), sa.Column("level", sa.String(16), nullable=False), sa.Column("message", sa.Text(), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_process_events_process_id", "process_events", ["process_id"])
    op.create_index("ix_process_events_created_at", "process_events", ["created_at"])
    op.create_index("ix_process_events_process_created", "process_events", ["process_id", "created_at"])
    op.create_table("system_metrics", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("recorded_at", sa.DateTime(), nullable=False), sa.Column("payload", sa.Text(), nullable=False))
    op.create_index("ix_system_metrics_recorded_at", "system_metrics", ["recorded_at"])
    op.create_table("audit_logs", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")), sa.Column("category", sa.String(32), nullable=False), sa.Column("action", sa.String(64), nullable=False), sa.Column("details", sa.Text(), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_category", "audit_logs", ["category"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_table("application_settings", sa.Column("key", sa.String(128), primary_key=True), sa.Column("value", sa.Text(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False))


def downgrade() -> None:
    for table in ("application_settings", "audit_logs", "system_metrics", "process_events", "processes", "messages", "chats", "sessions", "users"):
        op.drop_table(table)
