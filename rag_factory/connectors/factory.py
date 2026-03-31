"""
Connector factory — auto-detects DB type from URL and returns the right connector.
"""

from .base import BaseConnector
from .mongo_connector import MongoConnector
from .sql_connector import SQLConnector


def create_connector(db_url: str, db_name: str | None = None) -> BaseConnector:
    """
    Auto-detect database type from URL and instantiate the right connector.

    Supported URL formats:
      - postgresql://user:pass@host:5432/dbname
      - mysql+pymysql://user:pass@host:3306/dbname
      - sqlite:///path/to/db.sqlite
      - mssql+pyodbc://user:pass@host/dbname?driver=...
      - mongodb://user:pass@host:27017
      - mongodb+srv://user:pass@cluster.mongodb.net
    """
    url_lower = db_url.lower()

    if url_lower.startswith(("mongodb://", "mongodb+srv://")):
        if not db_name:
            raise ValueError("db_name is required for MongoDB connections")
        return MongoConnector(connection_string=db_url, db_name=db_name)

    # Everything else goes through SQLAlchemy
    if any(url_lower.startswith(p) for p in [
        "postgresql", "postgres",
        "mysql",
        "sqlite",
        "mssql",
        "oracle",
        "mariadb",
    ]):
        return SQLConnector(db_url=db_url)

    raise ValueError(
        f"Unsupported database URL: {db_url}\n"
        "Supported: postgresql, mysql, sqlite, mssql, oracle, mongodb"
    )
