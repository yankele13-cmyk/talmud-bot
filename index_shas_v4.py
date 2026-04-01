#!/usr/bin/env python3
"""
Index the entire Shas with v4 engine:
  - BGE-M3 embeddings (1024d, 8192 tokens)
  - Contextual Retrieval (Claude enriches each chunk)
  - Separate Gemara/Rashi/Tosafot chunks with back-references
  - Resume capability

Usage:
  python index_shas_v4.py                          # Full Shas
  python index_shas_v4.py --resume                 # Resume
  python index_shas_v4.py --tractate Berakhot      # Single tractate
"""

import sys
import os
import re
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

try:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
    CLAUDE_CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None
except Exception:
    CLAUDE_CLIENT = None

from rag_factory.engine_v4 import generate_chunk_context

console = Console()

SEFARIA_BASE = "https://www.sefaria.org"
PERSIST_DIR = "./rag_data_v4"
COLLECTION_NAME = "shas_v4"
EMBED_MODEL = "BAAI/bge-m3"
PROGRESS_FILE = os.path.join(PERSIST_DIR, "progress.json")
DELAY = 0.25
BATCH_SIZE = 32  # Smaller batches for larger embeddings


# ============================================================
# SEFARIA FETCH (reuse proven logic)
# ============================================================
def clean_html(t):
    return re.sub(r"<[^>]+>", "", str(t)).strip() if t else ""

def flatten_text(d):
    if isinstance(d, str): return clean_html(d)
    if isinstance(d, list): return "\n".join(p for p in (flatten_text(i) for i in d) if p)
    return ""

def extract_commentary(data, name):
    parts = []
    for c in data.get("commentary", []):
        if not isinstance(c, dict): continue
        ct = c.get("collectiveTitle", {})
        n = ct.get("en", "") if isinstance(ct, dict) else str(ct)
        if n == name:
            he = c.get("he", "")
            if he: parts.append(flatten_text(he))
    return "\n".join(parts)

def fetch_page(session, ref):
    url = f"{SEFARIA_BASE}/api/texts/{ref}?commentary=1&context=0"
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 429: time.sleep(2 ** (attempt + 1)); continue
            if resp.status_code == 404: return None
            resp.raise_for_status()
            return resp.json()
        except: time.sleep(2 ** attempt) if attempt < 3 else None
    return None

def get_bavli_tractates(only=None):
    s = requests.Session()
    toc = s.get(f"{SEFARIA_BASE}/api/index", timeout=30).json()
    shapes = s.get(f"{SEFARIA_BASE}/api/shape/Talmud/Bavli", timeout=30).json()
    shape_map = {i.get("title", ""): i.get("length", 0) for i in shapes if isinstance(i, dict)}

    tractates = []
    _walk(toc, [], tractates, shape_map)

    skip = ("Tziyyun", "Hagahot", "Pilpula", "Tosafot Ri", "Rosh on", "Rashba on",
            "Ritva on", "Meiri on", "Nimukei", "Shiltei", "Chiddushei")
    real = [t for t in tractates if t["length"] >= 20 and not any(t["title"].startswith(p) for p in skip)]

    if only:
        real = [t for t in real if t["title"] in only]
    return real

def _walk(nodes, path, result, shape_map):
    for n in nodes:
        if not isinstance(n, dict): continue
        cat = n.get("category", "")
        cp = path + [cat] if cat else path
        if "contents" in n: _walk(n["contents"], cp, result, shape_map)
        elif "title" in n and "Talmud" in "/".join(cp) and "Bavli" in "/".join(cp):
            result.append({"title": n["title"], "heTitle": n.get("heTitle", ""),
                          "length": shape_map.get(n["title"], n.get("length", 0))})

def get_daf_refs(title, length):
    refs = []
    for d in range(2, 2 + (length // 2) + 1):
        refs.extend([f"{title}.{d}a", f"{title}.{d}b"])
    return refs


# ============================================================
# CHUNKING — P2: Separate Gemara/Rashi/Tosafot
# ============================================================
def chunk_page(ref, data, max_chunk=2500, overlap=300):
    """
    Create separate chunks for Gemara, Rashi, and Tosafot.
    Each chunk has metadata linking it back to the source.
    """
    gemara = flatten_text(data.get("he", []))
    rashi = extract_commentary(data, "Rashi")
    tosafot = extract_commentary(data, "Tosafot")

    chunks = []

    # Gemara chunks (main text)
    if gemara and len(gemara) > 30:
        for i, chunk_text in enumerate(_split_text(f"[GEMARA — {ref}]\n{gemara}", max_chunk, overlap)):
            chunks.append({
                "text": chunk_text,
                "ref": ref,
                "type": "gemara",
                "chunk_idx": i,
            })

    # Rashi chunks (separate, with back-reference)
    if rashi and len(rashi) > 20:
        for i, chunk_text in enumerate(_split_text(f"[RASHI — {ref}]\n{rashi}", max_chunk, overlap)):
            chunks.append({
                "text": chunk_text,
                "ref": ref,
                "type": "rashi",
                "chunk_idx": i,
            })

    # Tosafot chunks (separate, with back-reference)
    if tosafot and len(tosafot) > 20:
        for i, chunk_text in enumerate(_split_text(f"[TOSAFOT — {ref}]\n{tosafot}", max_chunk, overlap)):
            chunks.append({
                "text": chunk_text,
                "ref": ref,
                "type": "tosafot",
                "chunk_idx": i,
            })

    return chunks, bool(rashi), bool(tosafot)


def _split_text(text, max_chunk, overlap):
    """Split text on Talmudic markers with overlap."""
    markers = [r"מתני׳", r"גמ׳", r"תנו רבנן", r"תניא", r"איתמר",
               r"אמר רב", r"אמר רבי", r"תא שמע", r"מיתיבי", r"שנאמר"]
    pattern = "|".join(f"({m})" for m in markers)

    splits = [0] + [m.start() for m in re.finditer(pattern, text) if m.start() > 0] + [len(text)]

    result = []
    current = ""
    for i in range(len(splits) - 1):
        seg = text[splits[i]:splits[i + 1]]
        if len(current) + len(seg) <= max_chunk:
            current += seg
        else:
            if current.strip():
                result.append(current.strip())
            if len(seg) > max_chunk:
                start = 0
                while start < len(seg):
                    result.append(seg[start:start + max_chunk].strip())
                    start += max_chunk - overlap
                current = ""
            else:
                current = seg
    if current.strip():
        result.append(current.strip())
    return result if result else [text]


# ============================================================
# PROGRESS
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f: return json.load(f)
    return {"completed": [], "stats": {}}

def save_progress(p):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f: json.dump(p, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tractate", action="append", default=[])
    args = parser.parse_args()

    console.print(Panel(
        f"[bold cyan]SHAS INDEXER v4[/bold cyan]\n"
        f"Embeddings: [green]{EMBED_MODEL}[/green] (1024d, 8192 tokens)\n"
        f"Contextual Retrieval: [green]{'Claude Haiku' if CLAUDE_CLIENT else 'Template fallback'}[/green]\n"
        f"Chunks: [green]Separate Gemara/Rashi/Tosafot[/green]",
        title="v4 Indexer",
    ))

    # Load model
    console.print("\n[bold blue]1. Loading BGE-M3...[/bold blue]")
    embedder = SentenceTransformer(EMBED_MODEL)
    dims = embedder.get_sentence_embedding_dimension()
    console.print(f"  [green]Ready: {dims} dimensions[/green]")

    # ChromaDB
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    progress = load_progress() if args.resume else {"completed": [], "stats": {}}

    if not args.resume:
        try: client.delete_collection(COLLECTION_NAME)
        except: pass

    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    console.print(f"  [green]Collection: {collection.count()} existing chunks[/green]")

    # Tractates
    only = args.tractate if args.tractate else None
    tractates = get_bavli_tractates(only)
    to_index = [t for t in tractates if t["title"] not in progress["completed"]]

    table = Table(title="Indexation Plan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Masechet", style="yellow")
    table.add_column("Pages", justify="right", style="cyan")
    table.add_column("Status")
    for i, t in enumerate(tractates, 1):
        st = "[green]DONE[/green]" if t["title"] in progress["completed"] else "pending"
        table.add_row(str(i), t["title"], str((t["length"]//2+1)*2), st)
    console.print(table)
    console.print(f"\n  {len(to_index)} massechtot to index")

    # Index
    session = requests.Session()
    session.headers.update({"User-Agent": "TalmudBot/4.0"})
    total_chunks = 0

    for ti, tract in enumerate(to_index, 1):
        title = tract["title"]
        refs = get_daf_refs(title, tract["length"])
        console.print(f"\n  [bold yellow][{ti}/{len(to_index)}] {title}[/bold yellow] ({len(refs)} pages)")

        all_chunks = []
        rashi_count = tosafot_count = 0

        with Progress(TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()) as prog:
            task = prog.add_task(f"  {title}", total=len(refs))
            for ref in refs:
                data = fetch_page(session, ref)
                prog.advance(task)
                if not data: continue

                chunks, has_r, has_t = chunk_page(ref, data)
                if has_r: rashi_count += 1
                if has_t: tosafot_count += 1

                # Contextual Retrieval: add context prefix
                for c in chunks:
                    ctx = generate_chunk_context(c["ref"], c["text"], CLAUDE_CLIENT)
                    c["enriched_text"] = f"{ctx}\n\n{c['text']}"

                all_chunks.extend(chunks)
                time.sleep(DELAY)

        # Embed and store
        if all_chunks:
            console.print(f"    Embedding {len(all_chunks)} chunks (BGE-M3)...")
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch = all_chunks[i:i + BATCH_SIZE]
                texts = [c["enriched_text"] for c in batch]
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

                ids = [f"{c['ref']}_{c['type']}_chunk{c['chunk_idx']}" for c in batch]
                metas = [{
                    "ref": c["ref"],
                    "tractate": title,
                    "type": c["type"],
                    "chunk_idx": c["chunk_idx"],
                    "url": f"{SEFARIA_BASE}/{c['ref'].replace(' ', '_')}",
                    "has_rashi": c["type"] == "rashi",
                    "has_tosafot": c["type"] == "tosafot",
                } for c in batch]

                # Store enriched text (with context prefix) as document
                collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

            total_chunks += len(all_chunks)

        progress["completed"].append(title)
        progress["stats"][title] = {
            "chunks": len(all_chunks), "pages": len(refs),
            "rashi": rashi_count, "tosafot": tosafot_count,
        }
        save_progress(progress)
        console.print(f"    [green]{len(all_chunks)} chunks (Gemara/Rashi/Tosafot separate)[/green]")

    console.print(Panel(
        f"[bold green]INDEXATION v4 COMPLETE![/bold green]\n\n"
        f"Total chunks: [cyan]{collection.count()}[/cyan]\n"
        f"Model: [cyan]{EMBED_MODEL}[/cyan] (1024d)\n"
        f"Contextual: [cyan]{'Claude' if CLAUDE_CLIENT else 'Template'}[/cyan]\n"
        f"Storage: [dim]{PERSIST_DIR}[/dim]",
        title="Done",
    ))


if __name__ == "__main__":
    main()
