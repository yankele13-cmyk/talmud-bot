"""
Autonomous RAG Factory Orchestrator — LangGraph pipeline that:
1. Connects to any database
2. Analyzes schema with local LLM
3. Generates optimal chunking strategy
4. Builds documents from data
5. Creates vector index + query engine
6. Saves everything for reuse

100% local. Zero API cost. Runs on Ollama + ChromaDB.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from llama_index.llms.ollama import Ollama

from rag_factory.connectors.base import BaseConnector, DatabaseProfile
from rag_factory.connectors.factory import create_connector
from rag_factory.agents.schema_analyzer import SchemaAnalyzer
from rag_factory.agents.chunker import Chunker
from rag_factory.agents.rag_builder import RAGBuilder

console = Console()


@dataclass
class PipelineState:
    """Full state of the RAG factory pipeline."""
    # Input
    db_url: str = ""
    db_name: str | None = None
    llm_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    persist_dir: str = "./rag_data"
    collection_name: str = "rag_factory"

    # Intermediate state
    connector: Any = None
    profile: DatabaseProfile | None = None
    strategy: dict = field(default_factory=dict)
    documents: list = field(default_factory=list)

    # Output
    index: Any = None
    query_engine: Any = None
    errors: list[str] = field(default_factory=list)


class RAGFactoryOrchestrator:
    """
    The autonomous agent that builds your RAG from scratch.
    Give it a database URL and it does everything else.
    """

    def __init__(self, state: PipelineState):
        self.state = state
        self.llm = Ollama(
            model=state.llm_model,
            base_url=state.ollama_url,
            request_timeout=180.0,
        )

    def run(self) -> PipelineState:
        """Execute the full pipeline."""
        steps = [
            ("Connecting to database", self._step_connect),
            ("Analyzing database schema", self._step_analyze_schema),
            ("Generating RAG strategy with AI", self._step_generate_strategy),
            ("Building documents from data", self._step_build_documents),
            ("Creating vector index", self._step_build_index),
            ("Saving configuration", self._step_save),
        ]

        console.print(Panel(
            "[bold cyan]RAG FACTORY[/bold cyan] — Autonomous RAG Builder\n"
            f"Database: [yellow]{self.state.db_url}[/yellow]\n"
            f"LLM: [green]{self.state.llm_model}[/green] | "
            f"Embeddings: [green]{self.state.embed_model}[/green]",
            title="Starting Pipeline",
        ))

        for step_name, step_fn in steps:
            console.print(f"\n[bold blue]>>> {step_name}...[/bold blue]")
            start = time.time()
            try:
                step_fn()
                elapsed = time.time() - start
                console.print(f"  [green]Done[/green] ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                error_msg = f"{step_name} failed: {e}"
                self.state.errors.append(error_msg)
                console.print(f"  [red]FAILED[/red] ({elapsed:.1f}s): {e}")
                if step_name in ("Connecting to database", "Analyzing database schema"):
                    console.print("[red]Critical step failed — aborting pipeline.[/red]")
                    break

        self._print_summary()
        return self.state

    def _step_connect(self):
        connector = create_connector(self.state.db_url, self.state.db_name)
        connector.connect()
        self.state.connector = connector
        console.print("  [dim]Connection established[/dim]")

    def _step_analyze_schema(self):
        profile = self.state.connector.analyze_schema()
        self.state.profile = profile
        console.print(f"  [dim]{profile.summary()}[/dim]")

    def _step_generate_strategy(self):
        analyzer = SchemaAnalyzer(llm=self.llm)
        strategy = analyzer.analyze(self.state.profile)
        self.state.strategy = strategy

        n_tables = len(strategy.get("tables_to_index", []))
        console.print(f"  [dim]Strategy: {n_tables} tables to index[/dim]")

        for t in strategy.get("tables_to_index", []):
            console.print(
                f"    - [yellow]{t['table']}[/yellow] "
                f"({t.get('chunk_strategy', '?')}, priority: {t.get('priority', '?')})"
            )

    def _step_build_documents(self):
        chunker = Chunker(
            connector=self.state.connector,
            profile=self.state.profile,
            strategy=self.state.strategy,
        )
        documents = chunker.generate_documents()
        self.state.documents = documents
        console.print(f"  [dim]{len(documents)} documents generated[/dim]")

    def _step_build_index(self):
        builder = RAGBuilder(
            llm_model=self.state.llm_model,
            embed_model=self.state.embed_model,
            ollama_base_url=self.state.ollama_url,
            persist_dir=self.state.persist_dir,
            collection_name=self.state.collection_name,
        )
        index = builder.build_index(self.state.documents)
        query_engine = builder.build_query_engine(index)

        self.state.index = index
        self.state.query_engine = query_engine

    def _step_save(self):
        builder = RAGBuilder(
            persist_dir=self.state.persist_dir,
            collection_name=self.state.collection_name,
        )
        builder.save_strategy(self.state.strategy)
        console.print(f"  [dim]Strategy saved to {self.state.persist_dir}/rag_strategy.json[/dim]")

        # Close DB connection
        if self.state.connector:
            self.state.connector.close()

    def _print_summary(self):
        if self.state.errors:
            console.print(Panel(
                "\n".join(f"[red]- {e}[/red]" for e in self.state.errors),
                title="[red]Errors[/red]",
            ))

        if self.state.query_engine:
            console.print(Panel(
                f"[bold green]RAG is READY![/bold green]\n\n"
                f"Documents indexed: [cyan]{len(self.state.documents)}[/cyan]\n"
                f"Persist dir: [cyan]{self.state.persist_dir}[/cyan]\n"
                f"Collection: [cyan]{self.state.collection_name}[/cyan]\n\n"
                f"Use [bold]query_engine.query('your question')[/bold] to ask questions.\n"
                f"Or run: [bold]python -m rag_factory chat[/bold]",
                title="Pipeline Complete",
            ))
        else:
            console.print("[red]Pipeline did not produce a working RAG. Check errors above.[/red]")
