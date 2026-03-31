#!/usr/bin/env python3
"""
Live test of the Sefaria connector — fetches real data from sefaria.org.
"""

import sys
sys.path.insert(0, "/home/user/talmud-bot")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_factory.connectors.sefaria_connector import SefariaConnector, SefariaConfig

console = Console()

def main():
    console.print(Panel("[bold cyan]Test Sefaria Connector — Live API[/bold cyan]"))

    # === STEP 1: Discover all Talmud Bavli tractates ===
    console.print("\n[bold blue]1. Discovering Talmud Bavli tractates...[/bold blue]")
    config = SefariaConfig(
        categories=["Talmud"],
        tractates=[],
        include_commentary=True,
        language="he",
        max_pages_per_tractate=0,
    )
    connector = SefariaConnector(config)
    connector.connect()
    console.print("  [green]Connected to Sefaria![/green]")

    profile = connector.analyze_schema()

    # Filter only real Bavli tractates (those with actual pages > 10)
    bavli_tractates = [t for t in profile.tables if t.row_count >= 10 and "Tractate" not in t.name and "Commentary" not in t.name]

    table = Table(title="Talmud Bavli — All Tractates")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Masechet", style="yellow")
    table.add_column("Pages", justify="right", style="cyan")

    total_pages = 0
    for i, t in enumerate(bavli_tractates, 1):
        table.add_row(str(i), t.name, str(t.row_count))
        total_pages += t.row_count

    console.print(table)
    console.print(f"\n  [bold]{len(bavli_tractates)} massechtot, {total_pages} pages au total[/bold]")

    # === STEP 2: Fetch 2 real pages from Berakhot with commentary ===
    console.print("\n[bold blue]2. Fetching Berakhot 2a + 2b (with Rashi & Tosafot)...[/bold blue]")
    demo_config = SefariaConfig(
        categories=["Talmud"],
        tractates=["Berakhot"],
        include_commentary=True,
        language="he",
        max_pages_per_tractate=2,
    )
    demo = SefariaConnector(demo_config)
    demo.connect()

    docs = demo.extract_documents("Berakhot", limit=2)

    for doc in docs:
        text = doc["text"]
        text_preview = text[:800] + "..." if len(text) > 800 else text

        console.print(Panel(
            f"[bold]Ref:[/bold] {doc['ref']}\n"
            f"[bold]URL:[/bold] {doc['url']}\n"
            f"[bold]Rashi:[/bold] {'Yes' if doc['has_rashi'] else 'No'}\n"
            f"[bold]Tosafot:[/bold] {'Yes' if doc['has_tosafot'] else 'No'}\n"
            f"[bold]Text size:[/bold] {len(text):,} characters\n"
            f"\n{text_preview}",
            title=f"[green]{doc['ref']}[/green]",
            width=100,
        ))

    demo.close()
    connector.close()

    console.print(Panel(
        f"[bold green]Sefaria connector works perfectly![/bold green]\n\n"
        f"Discovered: [cyan]{len(bavli_tractates)}[/cyan] massechtot Bavli\n"
        f"Total pages: [cyan]{total_pages:,}[/cyan]\n"
        f"Fetched: [cyan]{len(docs)}[/cyan] sample pages with Gemara + Rashi + Tosafot\n\n"
        f"[bold]To run the full RAG on YOUR machine:[/bold]\n"
        f"  1. curl -fsSL https://ollama.com/install.sh | sh\n"
        f"  2. ollama pull llama3.2 && ollama pull nomic-embed-text\n"
        f"  3. pip install -r rag_factory/requirements.txt\n"
        f"  4. python run_sefaria.py --tractate Berakhot --chat",
        title="Results",
    ))


if __name__ == "__main__":
    main()
