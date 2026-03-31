"""
Universal SQL connector — PostgreSQL, MySQL, SQLite, MSSQL, Oracle.
Uses SQLAlchemy so any supported dialect works out of the box.
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .base import BaseConnector, DatabaseProfile, TableSchema


class SQLConnector(BaseConnector):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine: Engine | None = None

    def connect(self) -> None:
        self.engine = create_engine(self.db_url, echo=False)
        # Quick connectivity check
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def analyze_schema(self) -> DatabaseProfile:
        insp = inspect(self.engine)
        db_name = self.engine.url.database or "unknown"
        tables: list[TableSchema] = []
        total_rows = 0

        for table_name in insp.get_table_names():
            columns = []
            for col in insp.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "primary_key": False,
                })

            # Mark primary keys
            pk_cols = {c for c in (insp.get_pk_constraint(table_name).get("constrained_columns") or [])}
            for col in columns:
                if col["name"] in pk_cols:
                    col["primary_key"] = True

            # Foreign keys → relationships
            relationships = []
            for fk in insp.get_foreign_keys(table_name):
                relationships.append({
                    "from_col": ", ".join(fk["constrained_columns"]),
                    "to_table": fk["referred_table"],
                    "to_col": ", ".join(fk["referred_columns"]),
                })

            # Row count + sample
            with self.engine.connect() as conn:
                row_count = conn.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')
                ).scalar() or 0
                sample_rows = [
                    dict(r._mapping)
                    for r in conn.execute(
                        text(f'SELECT * FROM "{table_name}" LIMIT 5')
                    )
                ]

            total_rows += row_count
            tables.append(TableSchema(
                name=table_name,
                db_type="sql",
                columns=columns,
                sample_rows=sample_rows,
                row_count=row_count,
                relationships=relationships,
            ))

        # Raw DDL for LLM context
        ddl_parts = []
        for t in tables:
            col_defs = ", ".join(f'{c["name"]} {c["type"]}' for c in t.columns)
            ddl_parts.append(f"CREATE TABLE {t.name} ({col_defs});")

        return DatabaseProfile(
            db_type="sql",
            db_name=db_name,
            tables=tables,
            total_rows=total_rows,
            raw_ddl="\n".join(ddl_parts),
        )

    def extract_documents(self, table_name: str, limit: int = 1000) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f'SELECT * FROM "{table_name}" LIMIT :lim'),
                {"lim": limit},
            )
            return [dict(r._mapping) for r in rows]

    def close(self) -> None:
        if self.engine:
            self.engine.dispose()
