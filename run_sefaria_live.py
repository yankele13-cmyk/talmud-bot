#!/usr/bin/env python3
"""
Live RAG on Sefaria — 100% gratuit, tourne ici meme.
Utilise sentence-transformers (CPU) + ChromaDB (local).
Pas besoin d'Ollama ni d'API payante.
"""

import sys
import os
import json
import time
import re

sys.path.insert(0, "/home/user/talmud-bot")

import chromadb
import requests
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

console = Console()

# ============================================================
# CONFIG
# ============================================================
SEFARIA_BASE = "https://www.sefaria.org"
PERSIST_DIR = "/home/user/talmud-bot/rag_data"
COLLECTION_NAME = "sefaria_talmud"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast, works on CPU, 384 dims
DELAY_BETWEEN_REQUESTS = 0.3


# ============================================================
# SEFARIA FETCHER
# ============================================================
def clean_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def flatten_text(data):
    if isinstance(data, str):
        return clean_html(data)
    if isinstance(data, list):
        parts = [flatten_text(item) for item in data]
        return "\n".join(p for p in parts if p)
    return ""


def fetch_page(session, ref):
    """Fetch a single daf from Sefaria."""
    url = f"{SEFARIA_BASE}/api/texts/{ref}?commentary=1&context=0"
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return None


def extract_commentary(data, name):
    """Extract Rashi or Tosafot from API response."""
    commentary = data.get("commentary", [])
    parts = []
    for c in commentary:
        if not isinstance(c, dict):
            continue
        if c.get("collectiveTitle") == name or c.get("indexTitle", "").startswith(name):
            he = c.get("he", "")
            if he:
                parts.append(flatten_text(he))
    return "\n".join(parts)


def build_document(ref, data):
    """Build a rich text document from a Sefaria API response."""
    gemara = flatten_text(data.get("he", []))
    sections = [f"[GEMARA — {ref}]\n{gemara}"]

    rashi = extract_commentary(data, "Rashi")
    if rashi:
        sections.append(f"[RASHI]\n{rashi}")

    tosafot = extract_commentary(data, "Tosafot")
    if tosafot:
        sections.append(f"[TOSAFOT]\n{tosafot}")

    return "\n\n".join(sections)


def get_talmud_refs(tractate, max_dafim=0):
    """Generate daf references for a tractate."""
    # Use Shape API to get exact page count
    session = requests.Session()
    resp = session.get(f"{SEFARIA_BASE}/api/shape/Talmud/Bavli", timeout=30)
    resp.raise_for_status()
    shapes = resp.json()

    length = 0
    for item in shapes:
        if isinstance(item, dict) and item.get("title") == tractate:
            length = item.get("length", 0)
            break

    if not length:
        console.print(f"[red]Tractate '{tractate}' not found in Shape API[/red]")
        return []

    refs = []
    num_dafim = (length // 2) + 1
    for daf in range(2, 2 + num_dafim):
        if max_dafim and len(refs) >= max_dafim:
            break
        refs.append(f"{tractate}.{daf}a")
        if max_dafim and len(refs) >= max_dafim:
            break
        refs.append(f"{tractate}.{daf}b")

    return refs


# ============================================================
# INDEXATION
# ============================================================
def index_tractate(tractate="Berakhot", max_dafim=0):
    """Fetch and index a full tractate into ChromaDB."""
    console.print(Panel(
        f"[bold cyan]RAG Factory — Live Indexation[/bold cyan]\n"
        f"Tractate: [yellow]{tractate}[/yellow]\n"
        f"Embeddings: [green]{EMBED_MODEL}[/green] (CPU, local)\n"
        f"Vector DB: [green]ChromaDB[/green] (local)",
        title="Starting",
    ))

    # Step 1: Load embedding model
    console.print("\n[bold blue]1. Loading embedding model...[/bold blue]")
    model = SentenceTransformer(EMBED_MODEL)
    console.print(f"  [green]Model loaded: {EMBED_MODEL} (384 dims)[/green]")

    # Step 2: Setup ChromaDB
    console.print("\n[bold blue]2. Setting up ChromaDB...[/bold blue]")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    # Delete old collection if exists, to start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    console.print(f"  [green]Collection '{COLLECTION_NAME}' created[/green]")

    # Step 3: Get all daf references
    console.print(f"\n[bold blue]3. Getting daf list for {tractate}...[/bold blue]")
    refs = get_talmud_refs(tractate, max_dafim)
    console.print(f"  [green]{len(refs)} pages to fetch[/green]")

    # Step 4: Fetch, embed, and index each page
    console.print(f"\n[bold blue]4. Fetching & indexing pages...[/bold blue]")
    session = requests.Session()
    session.headers.update({"User-Agent": "TalmudBot-RAGFactory/1.0"})

    indexed = 0
    skipped = 0
    batch_ids = []
    batch_docs = []
    batch_embeds = []
    batch_metas = []
    BATCH_SIZE = 20

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Indexing {tractate}...", total=len(refs))

        for ref in refs:
            data = fetch_page(session, ref)
            progress.advance(task)

            if not data:
                skipped += 1
                continue

            doc_text = build_document(ref, data)
            if not doc_text.strip() or len(doc_text) < 30:
                skipped += 1
                continue

            # Generate embedding
            embedding = model.encode(doc_text).tolist()

            batch_ids.append(ref)
            batch_docs.append(doc_text)
            batch_embeds.append(embedding)
            batch_metas.append({
                "ref": ref,
                "tractate": tractate,
                "url": f"{SEFARIA_BASE}/{ref.replace(' ', '_')}",
                "text_length": len(doc_text),
                "has_rashi": "[RASHI]" in doc_text,
                "has_tosafot": "[TOSAFOT]" in doc_text,
            })

            # Flush batch
            if len(batch_ids) >= BATCH_SIZE:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeds,
                    metadatas=batch_metas,
                )
                indexed += len(batch_ids)
                batch_ids, batch_docs, batch_embeds, batch_metas = [], [], [], []

            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Flush remaining
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_metas,
        )
        indexed += len(batch_ids)

    console.print(f"\n  [green]Indexed: {indexed} pages | Skipped: {skipped}[/green]")
    return model, collection, indexed


# ============================================================
# QUERY (search + answer)
# ============================================================
def query_rag(model, collection, question, top_k=5):
    """Search the indexed data and return relevant sources."""
    q_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    sources = []
    for i in range(len(results["ids"][0])):
        sources.append({
            "ref": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })

    return sources


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tractate", default="Berakhot")
    parser.add_argument("--max-dafim", type=int, default=0, help="0=all")
    parser.add_argument("--query", default=None, help="Question to ask")
    args = parser.parse_args()

    model, collection, indexed = index_tractate(args.tractate, args.max_dafim)

    console.print(Panel(
        f"[bold green]RAG READY![/bold green]\n\n"
        f"Indexed [cyan]{indexed}[/cyan] pages from [yellow]{args.tractate}[/yellow]\n"
        f"Stored in: [dim]{PERSIST_DIR}[/dim]",
        title="Indexation Complete",
    ))

    # Interactive query mode
    if args.query:
        questions = [args.query]
    else:
        questions = [
            "מהו הזמן של קריאת שמע בערבית?",
            "What does Rabbi Eliezer say about Shema?",
            "מתי מתחילים לקרוא שמע?",
        ]

    for q in questions:
        console.print(f"\n[bold green]Question:[/bold green] {q}")
        sources = query_rag(model, collection, q, top_k=3)

        for i, src in enumerate(sources, 1):
            preview = src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"]
            console.print(Panel(
                f"[bold]Score:[/bold] {src['score']:.4f}\n"
                f"[bold]URL:[/bold] {src['metadata'].get('url', '')}\n"
                f"[bold]Rashi:[/bold] {src['metadata'].get('has_rashi', False)} | "
                f"[bold]Tosafot:[/bold] {src['metadata'].get('has_tosafot', False)}\n\n"
                f"{preview}",
                title=f"[cyan]#{i} — {src['ref']}[/cyan]",
                width=100,
            ))
