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
    ("status", "VARCHAR(32) DEFAULT 'completed' NOT NULL"),
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


def ensure_studio_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    if "messages" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("messages")}
    with engine.begin() as connection:
        for column_name, column_ddl in MESSAGE_MIGRATION_COLUMNS:
            if column_name not in existing:
                connection.execute(text(f"ALTER TABLE messages ADD COLUMN {column_name} {column_ddl}"))


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
