"""
Sefaria API Connector — Fetches the entire Sefaria library (Talmud, Tanakh, Midrash, etc.)
via their free public API. No API key needed.

Endpoints used:
  - GET /api/index                → Full table of contents (all texts available)
  - GET /api/shape/Talmud/Bavli   → Exact structure of all Talmud tractates (63 tractates)
  - GET /api/texts/{ref}          → Get text content with commentary (v1, used for commentary)
  - GET /api/v2/index/{title}     → Full metadata for a specific book
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .base import BaseConnector, DatabaseProfile, TableSchema

SEFARIA_BASE = "https://www.sefaria.org"


@dataclass
class SefariaConfig:
    """What to fetch from Sefaria."""
    # Categories to index. Empty = everything.
    categories: list[str] = field(default_factory=lambda: ["Talmud"])
    # Specific tractates (e.g. ["Berakhot", "Shabbat"]). Empty = all in category.
    tractates: list[str] = field(default_factory=list)
    # Include commentary (Rashi, Tosafot, etc.)
    include_commentary: bool = True
    # Language: "he" (Hebrew), "en" (English), "both"
    language: str = "he"
    # Rate limit: seconds between API calls
    delay_between_requests: float = 0.5
    # Max pages per tractate (0 = all)
    max_pages_per_tractate: int = 0


class SefariaConnector(BaseConnector):
    """
    Connector that treats Sefaria's API as a database.
    Auto-discovers all available texts and fetches them.
    """

    def __init__(self, config: SefariaConfig | None = None):
        self.config = config or SefariaConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TalmudBot-RAGFactory/1.0",
            "Accept": "application/json",
        })
        self._index_cache: list[dict] = []
        self._toc_cache: list[dict] = []

    def connect(self) -> None:
        """Verify Sefaria API is reachable and load table of contents."""
        resp = self.session.get(f"{SEFARIA_BASE}/api/index", timeout=30)
        resp.raise_for_status()
        self._toc_cache = resp.json()

        # Also load Shape API for Talmud (gives exact daf counts for all 63 tractates)
        if any("Talmud" in c for c in self.config.categories) or not self.config.categories:
            try:
                shape_resp = self.session.get(
                    f"{SEFARIA_BASE}/api/shape/Talmud/Bavli", timeout=30
                )
                if shape_resp.ok:
                    self._shape_cache = shape_resp.json()
            except Exception:
                self._shape_cache = []

    def _get_talmud_tractates(self) -> list[dict]:
        """Extract all Talmud tractates from the ToC, enriched with Shape data."""
        tractates = []
        self._walk_toc(self._toc_cache, [], tractates)

        # Enrich with Shape data if available (gives exact chapter/daf counts)
        if hasattr(self, "_shape_cache") and self._shape_cache:
            shape_by_title = {}
            for item in self._shape_cache:
                if isinstance(item, dict):
                    shape_by_title[item.get("title", "")] = item

            for t in tractates:
                shape = shape_by_title.get(t["title"])
                if shape:
                    # Shape gives chapters array with lengths
                    t["chapters"] = shape.get("chapters", [])
                    t["length"] = shape.get("length", t.get("length", 0))

        return tractates

    def _walk_toc(self, nodes: list, path: list[str], result: list[dict]):
        """Recursively walk the ToC tree to find texts matching our config."""
        for node in nodes:
            if isinstance(node, dict):
                cat = node.get("category", "")
                current_path = path + [cat] if cat else path

                # Check if this category matches our filter
                if "contents" in node:
                    self._walk_toc(node["contents"], current_path, result)
                elif "title" in node:
                    # This is a leaf (actual text)
                    category_path = "/".join(current_path)
                    matches_category = (
                        not self.config.categories
                        or any(c in category_path for c in self.config.categories)
                    )
                    matches_tractate = (
                        not self.config.tractates
                        or node["title"] in self.config.tractates
                    )

                    if matches_category and matches_tractate:
                        result.append({
                            "title": node["title"],
                            "heTitle": node.get("heTitle", ""),
                            "category": category_path,
                            "addressTypes": node.get("addressTypes", []),
                            "length": node.get("length", 0),
                            "lengths": node.get("lengths", []),
                        })

    def _get_daf_list(self, tractate: dict) -> list[str]:
        """Generate list of all page references for a tractate."""
        title = tractate["title"]
        address_types = tractate.get("addressTypes", [])
        refs = []

        if "Talmud" in address_types:
            # Talmud uses daf system (2a, 2b, 3a, 3b, ...)
            length = tractate.get("length", 0)
            if not length and tractate.get("lengths"):
                length = tractate["lengths"][0] if tractate["lengths"] else 0
            # Sefaria "length" for Talmud = number of amudim (sides)
            # Daf 2a = amud 1, Daf 2b = amud 2, etc.
            num_dafim = (length // 2) + 2  # starts at daf 2
            max_pages = self.config.max_pages_per_tractate
            for daf_num in range(2, num_dafim + 1):
                if max_pages and len(refs) >= max_pages:
                    break
                refs.append(f"{title}.{daf_num}a")
                if max_pages and len(refs) >= max_pages:
                    break
                refs.append(f"{title}.{daf_num}b")
        else:
            # Chapter-based texts (Tanakh, Mishnah, etc.)
            length = tractate.get("length", 0)
            if not length and tractate.get("lengths"):
                length = tractate["lengths"][0] if tractate["lengths"] else 0
            max_pages = self.config.max_pages_per_tractate
            for ch in range(1, length + 1):
                if max_pages and len(refs) >= max_pages:
                    break
                refs.append(f"{title}.{ch}")

        return refs

    def _fetch_text(self, ref: str) -> dict | None:
        """Fetch a single text from Sefaria API."""
        commentary = 1 if self.config.include_commentary else 0
        url = f"{SEFARIA_BASE}/api/texts/{ref}?commentary={commentary}&context=0"

        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 429:
                    # Rate limited — back off
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return None

        return None

    def _clean_html(self, text: str) -> str:
        """Strip HTML tags from Sefaria text."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", "", str(text))
        return clean.strip()

    def _flatten_text(self, text_data) -> str:
        """Flatten nested arrays of text into a single string."""
        if isinstance(text_data, str):
            return self._clean_html(text_data)
        if isinstance(text_data, list):
            parts = []
            for item in text_data:
                flat = self._flatten_text(item)
                if flat:
                    parts.append(flat)
            return "\n".join(parts)
        return ""

    def _extract_commentary(self, data: dict, name: str) -> str:
        """Extract a specific commentary (Rashi, Tosafot, etc.) from API response."""
        commentary = data.get("commentary", [])
        parts = []
        for c in commentary:
            if c.get("collectiveTitle") == name or c.get("indexTitle", "").startswith(name):
                he_text = c.get("he", "")
                if he_text:
                    parts.append(self._flatten_text(he_text))
        return "\n".join(parts)

    def _text_to_document(self, ref: str, data: dict, tractate_info: dict) -> dict:
        """Convert a Sefaria API response into a document dict."""
        # Main text
        if self.config.language in ("he", "both"):
            gemara_text = self._flatten_text(data.get("he", []))
        else:
            gemara_text = self._flatten_text(data.get("text", []))

        if self.config.language == "both":
            en_text = self._flatten_text(data.get("text", []))
            gemara_text = f"{gemara_text}\n\n--- English ---\n{en_text}" if en_text else gemara_text

        sections = [f"[GEMARA — {ref}]\n{gemara_text}"]

        # Commentary
        if self.config.include_commentary:
            rashi = self._extract_commentary(data, "Rashi")
            if rashi:
                sections.append(f"[RASHI]\n{rashi}")

            tosafot = self._extract_commentary(data, "Tosafot")
            if tosafot:
                sections.append(f"[TOSAFOT]\n{tosafot}")

        full_text = "\n\n".join(sections)

        return {
            "text": full_text,
            "ref": ref,
            "title": data.get("indexTitle", tractate_info.get("title", "")),
            "heTitle": data.get("heIndexTitle", tractate_info.get("heTitle", "")),
            "category": tractate_info.get("category", ""),
            "url": f"{SEFARIA_BASE}/{ref.replace(' ', '_')}",
            "language": self.config.language,
            "has_rashi": bool(self._extract_commentary(data, "Rashi")) if self.config.include_commentary else False,
            "has_tosafot": bool(self._extract_commentary(data, "Tosafot")) if self.config.include_commentary else False,
        }

    # ---- BaseConnector interface ----

    def analyze_schema(self) -> DatabaseProfile:
        """Analyze what's available on Sefaria matching our config."""
        tractates = self._get_talmud_tractates()
        tables: list[TableSchema] = []
        total_rows = 0

        for tract in tractates:
            refs = self._get_daf_list(tract)
            row_count = len(refs)
            total_rows += row_count

            tables.append(TableSchema(
                name=tract["title"],
                db_type="sefaria_api",
                columns=[
                    {"name": "ref", "type": "str"},
                    {"name": "gemara_text", "type": "str"},
                    {"name": "rashi", "type": "str"},
                    {"name": "tosafot", "type": "str"},
                    {"name": "heTitle", "type": "str"},
                    {"name": "category", "type": "str"},
                    {"name": "url", "type": "str"},
                ],
                sample_rows=[{"ref": r} for r in refs[:3]],
                row_count=row_count,
            ))

        ddl_parts = []
        for t in tables:
            ddl_parts.append(
                f"text {t.name} ({t.row_count} pages) — "
                f"category: {tractates[tables.index(t)].get('category', '')}"
            )

        return DatabaseProfile(
            db_type="sefaria_api",
            db_name="Sefaria Library",
            tables=tables,
            total_rows=total_rows,
            raw_ddl="\n".join(ddl_parts),
        )

    def extract_documents(self, table_name: str, limit: int = 1000) -> list[dict]:
        """Fetch all pages of a tractate from Sefaria API."""
        tractates = self._get_talmud_tractates()
        tractate_info = None
        for t in tractates:
            if t["title"] == table_name:
                tractate_info = t
                break

        if not tractate_info:
            return []

        refs = self._get_daf_list(tractate_info)
        if limit:
            refs = refs[:limit]

        documents = []
        for ref in refs:
            data = self._fetch_text(ref)
            if data:
                doc = self._text_to_document(ref, data, tractate_info)
                if doc["text"].strip():
                    documents.append(doc)

            time.sleep(self.config.delay_between_requests)

        return documents

    def close(self) -> None:
        self.session.close()
