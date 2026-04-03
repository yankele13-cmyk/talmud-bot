"""
Chunking Strategy Engine — Transforms raw DB rows into optimized LlamaIndex Documents
based on the RAG strategy produced by the Schema Analyzer.
"""

import json

from llama_index.core.schema import Document, TextNode

from rag_factory.connectors.base import BaseConnector, DatabaseProfile


def _render_template(template: str, row: dict) -> str:
    """Render a doc_template string with {column_name} placeholders."""
    text = template
    for key, value in row.items():
        text = text.replace("{" + key + "}", str(value) if value is not None else "")
    return text.strip()


def _row_to_text(row: dict) -> str:
    """Fallback: convert a row to readable key-value text."""
    parts = []
    for k, v in row.items():
        if v is not None and str(v).strip():
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


class Chunker:
    def __init__(self, connector: BaseConnector, profile: DatabaseProfile, strategy: dict):
        self.connector = connector
        self.profile = profile
        self.strategy = strategy

    def generate_documents(self) -> list[Document]:
        """Generate LlamaIndex Documents for all tables in the strategy."""
        all_docs: list[Document] = []

        for table_conf in self.strategy.get("tables_to_index", []):
            table_name = table_conf["table"]
            chunk_strategy = table_conf.get("chunk_strategy", "row_per_doc")
            doc_template = table_conf.get("doc_template")
            text_columns = table_conf.get("text_columns", [])
            metadata_columns = table_conf.get("metadata_columns", [])
            id_column = table_conf.get("id_column")

            # Extract rows from DB
            rows = self.connector.extract_documents(table_name)
            if not rows:
                continue

            if chunk_strategy == "row_per_doc":
                docs = self._row_per_doc(
                    rows, table_name, doc_template, text_columns,
                    metadata_columns, id_column,
                )
            elif chunk_strategy == "concat_text":
                docs = self._concat_text(
                    rows, table_name, doc_template, text_columns,
                    metadata_columns,
                )
            elif chunk_strategy == "hierarchical":
                docs = self._hierarchical(
                    rows, table_name, doc_template, text_columns,
                    metadata_columns, id_column,
                )
            else:
                docs = self._row_per_doc(
                    rows, table_name, doc_template, text_columns,
                    metadata_columns, id_column,
                )

            all_docs.extend(docs)

        return all_docs

    def _build_metadata(self, row: dict, table_name: str, metadata_columns: list[str]) -> dict:
        meta = {"source_table": table_name}
        for col in metadata_columns:
            if col in row:
                meta[col] = str(row[col]) if row[col] is not None else ""
        return meta

    def _row_per_doc(
        self, rows, table_name, doc_template, text_columns, metadata_columns, id_column,
    ) -> list[Document]:
        """One document per row — best for self-contained records."""
        docs = []
        for row in rows:
            if doc_template:
                text = _render_template(doc_template, row)
            elif text_columns:
                text = "\n".join(str(row.get(c, "")) for c in text_columns if row.get(c))
            else:
                text = _row_to_text(row)

            if not text.strip():
                continue

            meta = self._build_metadata(row, table_name, metadata_columns)
            doc_id = str(row.get(id_column, "")) if id_column else None

            docs.append(Document(
                text=text,
                metadata=meta,
                doc_id=doc_id,
            ))
        return docs

    def _concat_text(
        self, rows, table_name, doc_template, text_columns, metadata_columns,
    ) -> list[Document]:
        """Concatenate all rows into larger documents, then chunk by size."""
        chunk_size = self.strategy.get("embedding_config", {}).get("recommended_chunk_size", 1024)
        overlap = self.strategy.get("embedding_config", {}).get("recommended_overlap", 128)

        full_text_parts = []
        for row in rows:
            if doc_template:
                part = _render_template(doc_template, row)
            elif text_columns:
                part = "\n".join(str(row.get(c, "")) for c in text_columns if row.get(c))
            else:
                part = _row_to_text(row)
            if part.strip():
                full_text_parts.append(part)

        full_text = "\n\n---\n\n".join(full_text_parts)

        # Chunk with overlap
        docs = []
        start = 0
        chunk_idx = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]
            if chunk.strip():
                docs.append(Document(
                    text=chunk,
                    metadata={
                        "source_table": table_name,
                        "chunk_index": chunk_idx,
                    },
                ))
                chunk_idx += 1
            start = end - overlap

        return docs

    def _hierarchical(
        self, rows, table_name, doc_template, text_columns, metadata_columns, id_column,
    ) -> list[Document]:
        """
        Hierarchical: group rows by a parent key, embed children in parent context.
        Falls back to row_per_doc if no relationships found.
        """
        joins = self.strategy.get("relationships_to_join", [])
        relevant_join = None
        for j in joins:
            if j.get("child_table") == table_name or j.get("parent_table") == table_name:
                relevant_join = j
                break

        if not relevant_join:
            return self._row_per_doc(
                rows, table_name, doc_template, text_columns, metadata_columns, id_column,
            )

        # Group by join column
        join_col = relevant_join.get("join_column", "")
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = str(row.get(join_col, "unknown"))
            groups.setdefault(key, []).append(row)

        docs = []
        for group_key, group_rows in groups.items():
            parts = []
            for row in group_rows:
                if doc_template:
                    parts.append(_render_template(doc_template, row))
                else:
                    parts.append(_row_to_text(row))

            text = f"[Group: {join_col}={group_key}]\n\n" + "\n\n".join(parts)
            meta = {"source_table": table_name, "group_key": group_key}
            docs.append(Document(text=text, metadata=meta))

        return docs
