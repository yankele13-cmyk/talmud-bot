"""
MongoDB connector — auto-detects collections, infers schema from documents.
"""

from pymongo import MongoClient

from .base import BaseConnector, DatabaseProfile, TableSchema


class MongoConnector(BaseConnector):
    def __init__(self, connection_string: str, db_name: str):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client: MongoClient | None = None

    def connect(self) -> None:
        self.client = MongoClient(self.connection_string)
        # Connectivity check
        self.client.admin.command("ping")

    def _infer_schema(self, collection_name: str, sample_size: int = 50) -> list[dict]:
        """Infer schema by sampling documents and merging all keys."""
        db = self.client[self.db_name]
        docs = list(db[collection_name].find().limit(sample_size))
        all_keys: dict[str, str] = {}
        for doc in docs:
            for key, value in doc.items():
                if key not in all_keys:
                    all_keys[key] = type(value).__name__
        return [{"name": k, "type": v, "nullable": True} for k, v in all_keys.items()]

    def analyze_schema(self) -> DatabaseProfile:
        db = self.client[self.db_name]
        collection_names = db.list_collection_names()
        tables: list[TableSchema] = []
        total_rows = 0

        for coll_name in collection_names:
            coll = db[coll_name]
            row_count = coll.estimated_document_count()
            columns = self._infer_schema(coll_name)

            # Sample docs (convert ObjectId to str for serialization)
            sample_rows = []
            for doc in coll.find().limit(5):
                clean = {}
                for k, v in doc.items():
                    clean[k] = str(v) if k == "_id" else v
                sample_rows.append(clean)

            total_rows += row_count
            tables.append(TableSchema(
                name=coll_name,
                db_type="mongodb",
                columns=columns,
                sample_rows=sample_rows,
                row_count=row_count,
            ))

        # DDL-equivalent for Mongo (JSON schema description)
        ddl_parts = []
        for t in tables:
            fields = ", ".join(f'{c["name"]}: {c["type"]}' for c in t.columns)
            ddl_parts.append(f"collection {t.name} {{ {fields} }}")

        return DatabaseProfile(
            db_type="mongodb",
            db_name=self.db_name,
            tables=tables,
            total_rows=total_rows,
            raw_ddl="\n".join(ddl_parts),
        )

    def extract_documents(self, table_name: str, limit: int = 1000) -> list[dict]:
        db = self.client[self.db_name]
        docs = []
        for doc in db[table_name].find().limit(limit):
            clean = {}
            for k, v in doc.items():
                clean[k] = str(v) if k == "_id" else v
            docs.append(clean)
        return docs

    def close(self) -> None:
        if self.client:
            self.client.close()
