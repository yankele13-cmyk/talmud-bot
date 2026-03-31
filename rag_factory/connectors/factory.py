"""
Connector factory — auto-detects DB type from URL and returns the right connector.
"""

from .base import BaseConnector
from .mongo_connector import MongoConnector
from .sefaria_connector import SefariaConnector, SefariaConfig
from .sql_connector import SQLConnector


def create_connector(
    db_url: str,
    db_name: str | None = None,
    sefaria_config: SefariaConfig | None = None,
) -> BaseConnector:
    """
    Auto-detect database type from URL and instantiate the right connector.

    Supported URL formats:
      - postgresql://user:pass@host:5432/dbname
      - mysql+pymysql://user:pass@host:3306/dbname
      - sqlite:///path/to/db.sqlite
      - mssql+pyodbc://user:pass@host/dbname?driver=...
      - mongodb://user:pass@host:27017
      - mongodb+srv://user:pass@cluster.mongodb.net
      - sefaria://talmud          (all Talmud)
      - sefaria://all             (entire library)
      - sefaria://talmud/Berakhot (specific tractate)
    """
    url_lower = db_url.lower()

    # Sefaria special URLs
    if url_lower.startswith("sefaria://"):
        return _create_sefaria_connector(db_url, sefaria_config)

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
        "Supported: postgresql, mysql, sqlite, mssql, oracle, mongodb, sefaria://"
    )


def _create_sefaria_connector(db_url: str, config: SefariaConfig | None) -> SefariaConnector:
    """Parse sefaria:// URL and create connector with appropriate config."""
    # sefaria://talmud
    # sefaria://talmud/Berakhot,Shabbat
    # sefaria://all
    # sefaria://tanakh
    # sefaria://midrash

    path = db_url.split("://", 1)[1] if "://" in db_url else ""
    parts = path.strip("/").split("/")

    category = parts[0] if parts else "talmud"
    tractates_str = parts[1] if len(parts) > 1 else ""

    category_map = {
        "talmud": ["Talmud"],
        "tanakh": ["Tanakh"],
        "midrash": ["Midrash"],
        "mishnah": ["Mishnah"],
        "tosefta": ["Tosefta"],
        "halakhah": ["Halakhah"],
        "kabbalah": ["Kabbalah"],
        "liturgy": ["Liturgy"],
        "all": [],  # empty = everything
    }

    if config:
        return SefariaConnector(config)

    categories = category_map.get(category.lower(), [category])
    tractates = [t.strip() for t in tractates_str.split(",") if t.strip()] if tractates_str else []

    return SefariaConnector(SefariaConfig(
        categories=categories,
        tractates=tractates,
    ))
