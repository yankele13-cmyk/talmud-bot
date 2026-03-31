"""
CLI entry point: python -m rag_factory

Commands:
  build   — Build a RAG from a database (fully autonomous)
  chat    — Chat with an existing RAG
  info    — Show info about a saved RAG
"""

import argparse
import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def cmd_build(args):
    """Build a new RAG from a database."""
    from rag_factory.orchestrator import RAGFactoryOrchestrator, PipelineState

    state = PipelineState(
        db_url=args.db_url,
        db_name=args.db_name,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    orchestrator = RAGFactoryOrchestrator(state)
    result = orchestrator.run()

    if result.query_engine and args.interactive:
        _chat_loop(result.query_engine)


def cmd_chat(args):
    """Chat with an existing RAG."""
    from rag_factory.agents.rag_builder import RAGBuilder

    builder = RAGBuilder(
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        ollama_base_url=args.ollama_url,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    index = builder.load_existing_index()
    if not index:
        console.print("[red]No existing RAG found. Run 'build' first.[/red]")
        sys.exit(1)

    console.print(f"[green]RAG loaded from {args.persist_dir}[/green]")
    query_engine = builder.build_query_engine(index)
    _chat_loop(query_engine)


def cmd_info(args):
    """Show info about a saved RAG."""
    import json

    strategy_path = os.path.join(args.persist_dir, "rag_strategy.json")
    if not os.path.exists(strategy_path):
        console.print("[red]No RAG strategy found. Run 'build' first.[/red]")
        sys.exit(1)

    with open(strategy_path) as f:
        strategy = json.load(f)

    console.print(Panel(
        json.dumps(strategy, ensure_ascii=False, indent=2),
        title="RAG Strategy",
    ))


def _chat_loop(query_engine):
    """Interactive chat loop."""
    console.print(Panel(
        "[bold cyan]RAG Chat[/bold cyan] — Ask questions about your data.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
        title="Interactive Mode",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold green]Question[/bold green]")
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question.strip():
            continue

        console.print("[dim]Thinking...[/dim]")
        try:
            response = query_engine.query(question)
            console.print(Panel(str(response), title="Answer"))

            # Show sources
            if hasattr(response, "source_nodes") and response.source_nodes:
                console.print("[dim]Sources:[/dim]")
                for i, node in enumerate(response.source_nodes[:5], 1):
                    meta = node.metadata if hasattr(node, "metadata") else {}
                    score = f" (score: {node.score:.3f})" if hasattr(node, "score") and node.score else ""
                    table = meta.get("source_table", "?")
                    console.print(f"  [dim]{i}. {table}{score}[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[dim]Goodbye![/dim]")


def main():
    parser = argparse.ArgumentParser(
        prog="rag_factory",
        description="Autonomous RAG Factory — Build professional RAG systems from any database.",
    )

    # Global options
    parser.add_argument("--llm-model", default="llama3.2", help="Ollama LLM model (default: llama3.2)")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--persist-dir", default="./rag_data", help="Directory to save RAG data")
    parser.add_argument("--collection", default="rag_factory", help="ChromaDB collection name")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # BUILD
    build_parser = subparsers.add_parser("build", help="Build a RAG from a database")
    build_parser.add_argument("db_url", help="Database URL (postgresql://..., mongodb://..., sqlite:///...)")
    build_parser.add_argument("--db-name", help="Database name (required for MongoDB)")
    build_parser.add_argument("--interactive", "-i", action="store_true", help="Start chat after build")

    # CHAT
    subparsers.add_parser("chat", help="Chat with an existing RAG")

    # INFO
    subparsers.add_parser("info", help="Show RAG strategy info")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
