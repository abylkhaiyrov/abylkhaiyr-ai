from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

MESSAGE_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("model_id", "VARCHAR(256)"),
    ("checkpoint_id", "VARCHAR(1024)"),
    ("generation_params", "TEXT"),
    ("model_config_json", "TEXT"),
    ("reasoning_trace", "TEXT"),
    ("variants", "TEXT"),
    ("status", "VARCHAR(32) DEFAULT 'completed' NOT NULL"),
)


CHAT_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pinned", "BOOLEAN DEFAULT 0 NOT NULL"),
    ("archived", "BOOLEAN DEFAULT 0 NOT NULL"),
)
USER_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("uuid", "VARCHAR(36)"),
    ("first_name", "VARCHAR(64) DEFAULT '' NOT NULL"),
    ("last_name", "VARCHAR(64) DEFAULT '' NOT NULL"),
    ("email", "VARCHAR(254)"),
    ("role", "VARCHAR(16) DEFAULT 'USER' NOT NULL"),
    ("is_active", "BOOLEAN DEFAULT 1 NOT NULL"),
    ("last_login_at", "DATETIME"),
    ("last_active_at", "DATETIME"),
)
SESSION_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("last_seen_at", "DATETIME"),
    ("user_agent", "VARCHAR(256)"),
)
JOB_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("started_at", "DATETIME"),
    ("completed_at", "DATETIME"),
)
USER_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_uuid ON users (uuid)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users (email)",
    "CREATE INDEX IF NOT EXISTS ix_users_role ON users (role)",
    "CREATE INDEX IF NOT EXISTS ix_chats_archived ON chats (archived)",
)


class Base(DeclarativeBase):
    pass


def create_sqlite_engine(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path.resolve()}",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record) -> None:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")

    return engine


def _add_missing_columns(
    engine: Engine, table: str, columns: tuple[tuple[str, str], ...]
) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    existing = {column["name"] for column in inspector.get_columns(table)}
    added: set[str] = set()
    with engine.begin() as connection:
        for column_name, column_ddl in columns:
            if column_name not in existing:
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_ddl}")
                )
                added.add(column_name)
    return added


def _backfill_users(engine: Engine, added: set[str]) -> None:
    import uuid

    with engine.begin() as connection:
        rows = connection.execute(text("SELECT id FROM users WHERE uuid IS NULL")).all()
        for (identifier,) in rows:
            connection.execute(
                text("UPDATE users SET uuid = :uuid WHERE id = :id"),
                {"uuid": str(uuid.uuid4()), "id": identifier},
            )
        if "role" in added:
            connection.execute(
                text(
                    "UPDATE users SET role = 'ADMIN' WHERE username = 'admin' "
                    "AND (SELECT COUNT(*) FROM users WHERE role = 'ADMIN') = 0"
                )
            )
        for statement in USER_INDEXES:
            connection.execute(text(statement))


def ensure_studio_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns(engine, "messages", MESSAGE_MIGRATION_COLUMNS)
    _add_missing_columns(engine, "chats", CHAT_MIGRATION_COLUMNS)
    _add_missing_columns(engine, "sessions", SESSION_MIGRATION_COLUMNS)
    _add_missing_columns(engine, "generation_jobs", JOB_MIGRATION_COLUMNS)
    added = _add_missing_columns(engine, "users", USER_MIGRATION_COLUMNS)
    _backfill_users(engine, added)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
