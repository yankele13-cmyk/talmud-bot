"""
Schema Analyzer Agent — Uses local LLM to understand database structure
and generate an optimal RAG strategy tailored to the data.
"""

import json

from llama_index.llms.ollama import Ollama

from rag_factory.connectors.base import DatabaseProfile

ANALYSIS_PROMPT = """\
You are an expert database architect and RAG (Retrieval-Augmented Generation) engineer.

Analyze the following database schema and data samples, then produce a JSON RAG strategy.

## DATABASE PROFILE
{db_summary}

## RAW SCHEMA (DDL)
{ddl}

## SAMPLE DATA (first 3 rows per table)
{samples}

## YOUR TASK
Produce a JSON object with this exact structure:
{{
  "tables_to_index": [
    {{
      "table": "<table_name>",
      "priority": "high" | "medium" | "low",
      "reason": "<why this table matters for RAG>",
      "text_columns": ["<columns containing natural language text>"],
      "metadata_columns": ["<columns to use as metadata filters>"],
      "id_column": "<primary key or unique identifier>",
      "chunk_strategy": "row_per_doc" | "concat_text" | "hierarchical",
      "doc_template": "<template string using {{column_name}} placeholders to build the document text>"
    }}
  ],
  "relationships_to_join": [
    {{
      "parent_table": "<table>",
      "child_table": "<table>",
      "join_column": "<column>",
      "strategy": "embed_child_in_parent" | "separate_with_reference"
    }}
  ],
  "embedding_config": {{
    "recommended_chunk_size": <int>,
    "recommended_overlap": <int>,
    "language": "<primary language of the data>"
  }},
  "overall_notes": "<free text advice about this database for RAG>"
}}

RULES:
- Only include tables that have meaningful text content worth indexing.
- Choose chunk_strategy based on data shape: use "row_per_doc" for tables where each row is a self-contained record, "concat_text" when rows should be merged into larger documents, "hierarchical" for parent-child relationships.
- The doc_template should create rich, readable text from multiple columns.
- Be specific about which columns to embed vs. use as metadata.
- Respond ONLY with the JSON, no markdown fences, no explanation.
"""


class SchemaAnalyzer:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def analyze(self, profile: DatabaseProfile) -> dict:
        """Ask the local LLM to produce a tailored RAG strategy from the DB profile."""
        # Build sample preview (max 3 rows per table, truncate long values)
        samples_parts = []
        for table in profile.tables:
            rows_preview = []
            for row in table.sample_rows[:3]:
                truncated = {}
                for k, v in row.items():
                    s = str(v)
                    truncated[k] = s[:200] + "..." if len(s) > 200 else s
                rows_preview.append(truncated)
            samples_parts.append(f"### {table.name}\n{json.dumps(rows_preview, ensure_ascii=False, indent=2)}")

        prompt = ANALYSIS_PROMPT.format(
            db_summary=profile.summary(),
            ddl=profile.raw_ddl,
            samples="\n\n".join(samples_parts),
        )

        response = self.llm.complete(prompt)
        raw = response.text.strip()

        # Extract JSON from response (handle markdown fences if LLM adds them)
        if raw.startswith("```"):
            lines = raw.split("\n")
            json_lines = []
            inside = False
            for line in lines:
                if line.startswith("```") and not inside:
                    inside = True
                    continue
                if line.startswith("```") and inside:
                    break
                if inside:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        return json.loads(raw)
