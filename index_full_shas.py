#!/usr/bin/env python3
"""
Index the ENTIRE Talmud Bavli (Shas) into a RAG.
~37 massechtot, ~6000 pages, Gemara + Rashi + Tosafot.

Optimized for:
  - Batch embedding (64 chunks at a time)
  - Progress tracking per masechet
  - Resume capability (skips already-indexed massechtot)
  - Reduced delay (0.25s — respectful but fast)

Usage:
  python index_full_shas.py
  python index_full_shas.py --resume          # skip already indexed
  python index_full_shas.py --query "..."     # query after indexing
"""

import sys
import os
import re
import time
import json

sys.path.insert(0, "/home/user/talmud-bot")

import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

SEFARIA_BASE = "https://www.sefaria.org"
PERSIST_DIR = "/home/user/talmud-bot/rag_data_shas"
COLLECTION_NAME = "shas_complete"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PROGRESS_FILE = os.path.join(PERSIST_DIR, "progress.json")
DELAY = 0.25
BATCH_SIZE = 64

# Talmudic query expansion
TALMUD_SYNONYMS = {
    "שמע": ["קריאת שמע", "שמע בערבין", "בשכבך ובקומך"],
    "תפילה": ["שמונה עשרה", "עמידה", "תפילת שחרית", "מנחה", "ערבית"],
    "שבת": ["שבתא", "יום השבת", "מלאכות שבת"],
    "ברכה": ["ברכות", "ברכת המזון", "ברכת הנהנין"],
    "טומאה": ["טמא", "טהרה", "טבילה", "מקוה"],
    "כהן": ["כהנים", "כהן גדול", "תרומה", "בית המקדש"],
    "קרבן": ["קרבנות", "עולה", "חטאת", "זבח", "מנחה"],
    "גט": ["גיטין", "גירושין", "כריתות"],
    "נזיקין": ["נזק", "בבא קמא", "בבא מציעא", "בבא בתרא"],
    "סנהדרין": ["דיינים", "בית דין", "דיני נפשות"],
    "shema": ["קריאת שמע", "שמע"],
    "prayer": ["תפילה", "tefillah"],
    "shabbat": ["שבת", "sabbath"],
    "sacrifice": ["קרבן", "korban"],
    "purity": ["טהרה", "tahara", "טומאה"],
}


# ============================================================
# SEFARIA API
# ============================================================
def clean_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def flatten_text(data):
    if isinstance(data, str):
        return clean_html(data)
    if isinstance(data, list):
        return "\n".join(p for p in (flatten_text(i) for i in data) if p)
    return ""


def extract_commentary(data, target_name):
    parts = []
    for c in data.get("commentary", []):
        if not isinstance(c, dict):
            continue
        ct = c.get("collectiveTitle", {})
        name_en = ct.get("en", "") if isinstance(ct, dict) else str(ct)
        if name_en == target_name:
            he = c.get("he", "")
            if he:
                parts.append(flatten_text(he))
    return "\n".join(parts)


def fetch_page(session, ref):
    url = f"{SEFARIA_BASE}/api/texts/{ref}?commentary=1&context=0"
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < 3:
                time.sleep(2 ** attempt)
    return None


def get_bavli_tractates():
    """Get all Talmud Bavli tractates with their page counts."""
    session = requests.Session()

    # Get ToC for titles
    toc = session.get(f"{SEFARIA_BASE}/api/index", timeout=30).json()
    # Get Shape for accurate lengths
    shapes = session.get(f"{SEFARIA_BASE}/api/shape/Talmud/Bavli", timeout=30).json()

    shape_map = {}
    for item in shapes:
        if isinstance(item, dict):
            shape_map[item.get("title", "")] = item.get("length", 0)

    # Walk ToC to find Bavli tractates
    tractates = []
    _walk_toc(toc, [], tractates, shape_map)

    # Filter: only real tractates with pages, exclude commentaries
    real = []
    for t in tractates:
        if t["length"] >= 20 and "Commentary" not in t["category"]:
            # Skip commentary compilations
            skip_prefixes = ("Tziyyun", "Hagahot", "Pilpula", "Tosafot Ri",
                           "Rosh on", "Rashba on", "Ritva on", "Meiri on",
                           "Nimukei", "Shiltei", "Chiddushei")
            if not any(t["title"].startswith(p) for p in skip_prefixes):
                real.append(t)

    return real


def _walk_toc(nodes, path, result, shape_map):
    for node in nodes:
        if not isinstance(node, dict):
            continue
        cat = node.get("category", "")
        current_path = path + [cat] if cat else path
        if "contents" in node:
            _walk_toc(node["contents"], current_path, result, shape_map)
        elif "title" in node:
            category_path = "/".join(current_path)
            if "Talmud" in category_path and "Bavli" in category_path:
                title = node["title"]
                length = shape_map.get(title, node.get("length", 0))
                result.append({
                    "title": title,
                    "heTitle": node.get("heTitle", ""),
                    "category": category_path,
                    "length": length,
                })


def get_daf_refs(title, length):
    refs = []
    num_dafim = (length // 2) + 1
    for daf in range(2, 2 + num_dafim):
        refs.append(f"{title}.{daf}a")
        refs.append(f"{title}.{daf}b")
    return refs


# ============================================================
# CHUNKING
# ============================================================
def semantic_chunk(text, ref, max_chunk=1500, overlap=200):
    markers = [
        r"מתני׳", r"גמ׳", r"תנו רבנן", r"תניא", r"איתמר",
        r"בעי רב", r"אמר רב", r"אמר רבי", r"תא שמע", r"מיתיבי",
        r"ורמינהו", r"שנאמר", r"\[RASHI", r"\[TOSAFOT",
    ]
    pattern = "|".join(f"({m})" for m in markers)

    split_points = [0]
    for match in re.finditer(pattern, text):
        if match.start() > 0:
            split_points.append(match.start())
    split_points.append(len(text))

    chunks = []
    current = ""
    idx = 0

    for i in range(len(split_points) - 1):
        segment = text[split_points[i]:split_points[i + 1]]
        if len(current) + len(segment) <= max_chunk:
            current += segment
        else:
            if current.strip():
                chunks.append({"text": current.strip(), "ref": ref, "chunk_idx": idx})
                idx += 1
            if len(segment) > max_chunk:
                start = 0
                while start < len(segment):
                    sub = segment[start:start + max_chunk]
                    if sub.strip():
                        chunks.append({"text": sub.strip(), "ref": ref, "chunk_idx": idx})
                        idx += 1
                    start += max_chunk - overlap
                current = ""
            else:
                current = segment

    if current.strip():
        chunks.append({"text": current.strip(), "ref": ref, "chunk_idx": idx})

    return chunks


# ============================================================
# PROGRESS TRACKING
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "stats": {}}


def save_progress(progress_data):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN INDEXATION
# ============================================================
def index_full_shas(resume=False):
    console.print(Panel(
        "[bold cyan]INDEXATION DU SHAS COMPLET[/bold cyan]\n"
        "Talmud Bavli — Gemara + Rashi + Tosafot\n"
        f"Embeddings: [green]{EMBED_MODEL}[/green]\n"
        f"Storage: [green]{PERSIST_DIR}[/green]",
        title="Shas Indexer",
    ))

    # Load models
    console.print("\n[bold blue]1. Loading models...[/bold blue]")
    embedder = SentenceTransformer(EMBED_MODEL)
    console.print(f"  [green]Embedder ready[/green]")

    # Setup ChromaDB
    console.print("\n[bold blue]2. Setting up ChromaDB...[/bold blue]")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    progress_data = load_progress() if resume else {"completed": [], "stats": {}}

    if not resume:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    console.print(f"  [green]Collection ready (existing: {collection.count()} chunks)[/green]")

    # Get all tractates
    console.print("\n[bold blue]3. Discovering Bavli tractates...[/bold blue]")
    tractates = get_bavli_tractates()

    table = Table(title="Talmud Bavli — Shas")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Masechet", style="yellow")
    table.add_column("Pages", justify="right", style="cyan")
    table.add_column("Status", style="green")

    total_pages = 0
    to_index = []
    for i, t in enumerate(tractates, 1):
        pages = (t["length"] // 2 + 1) * 2
        total_pages += pages
        status = "DONE" if t["title"] in progress_data["completed"] else "pending"
        table.add_row(str(i), t["title"], str(pages), status)
        if t["title"] not in progress_data["completed"]:
            to_index.append(t)

    console.print(table)
    console.print(f"\n  [bold]{len(tractates)} massechtot, ~{total_pages} pages total[/bold]")
    if resume and progress_data["completed"]:
        console.print(f"  [yellow]Resuming: {len(progress_data['completed'])} done, {len(to_index)} remaining[/yellow]")

    # Index each tractate
    console.print(f"\n[bold blue]4. Indexing {len(to_index)} massechtot...[/bold blue]")
    session = requests.Session()
    session.headers.update({"User-Agent": "TalmudBot-RAGFactory/2.0"})

    grand_total_chunks = 0
    grand_total_rashi = 0
    grand_total_tosafot = 0
    start_time = time.time()

    for t_idx, tractate in enumerate(to_index, 1):
        title = tractate["title"]
        refs = get_daf_refs(title, tractate["length"])

        console.print(f"\n  [bold yellow][{t_idx}/{len(to_index)}] {title}[/bold yellow] ({len(refs)} pages)")

        chunks_for_tractate = []
        rashi_count = 0
        tosafot_count = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"  {title}", total=len(refs))

            for ref in refs:
                data = fetch_page(session, ref)
                progress.advance(task)

                if not data:
                    continue

                # Build document
                gemara = flatten_text(data.get("he", []))
                rashi = extract_commentary(data, "Rashi")
                tosafot = extract_commentary(data, "Tosafot")

                sections = [f"[GEMARA — {ref}]\n{gemara}"]
                if rashi:
                    sections.append(f"\n[RASHI — {ref}]\n{rashi}")
                    rashi_count += 1
                if tosafot:
                    sections.append(f"\n[TOSAFOT — {ref}]\n{tosafot}")
                    tosafot_count += 1

                full_text = "\n\n".join(sections)
                if len(full_text) < 30:
                    continue

                # Chunk
                chunks = semantic_chunk(full_text, ref)
                chunks_for_tractate.extend(chunks)

                time.sleep(DELAY)

        # Batch embed and store
        if chunks_for_tractate:
            console.print(f"    Embedding {len(chunks_for_tractate)} chunks...")
            for i in range(0, len(chunks_for_tractate), BATCH_SIZE):
                batch = chunks_for_tractate[i:i + BATCH_SIZE]
                texts = [c["text"] for c in batch]
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

                ids = [f"{c['ref']}_chunk{c['chunk_idx']}" for c in batch]
                metas = [{
                    "ref": c["ref"],
                    "chunk_idx": c["chunk_idx"],
                    "tractate": title,
                    "url": f"{SEFARIA_BASE}/{c['ref'].replace(' ', '_')}",
                    "has_rashi": "[RASHI" in c["text"],
                    "has_tosafot": "[TOSAFOT" in c["text"],
                } for c in batch]

                collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

        grand_total_chunks += len(chunks_for_tractate)
        grand_total_rashi += rashi_count
        grand_total_tosafot += tosafot_count

        # Save progress
        progress_data["completed"].append(title)
        progress_data["stats"][title] = {
            "chunks": len(chunks_for_tractate),
            "rashi": rashi_count,
            "tosafot": tosafot_count,
            "pages": len(refs),
        }
        save_progress(progress_data)

        console.print(f"    [green]{len(chunks_for_tractate)} chunks | Rashi: {rashi_count} | Tosafot: {tosafot_count}[/green]")

    elapsed = time.time() - start_time
    console.print(Panel(
        f"[bold green]SHAS INDEXATION COMPLETE![/bold green]\n\n"
        f"Massechtot: [cyan]{len(tractates)}[/cyan]\n"
        f"Total chunks: [cyan]{collection.count()}[/cyan]\n"
        f"Rashi pages: [cyan]{grand_total_rashi}[/cyan]\n"
        f"Tosafot pages: [cyan]{grand_total_tosafot}[/cyan]\n"
        f"Time: [cyan]{elapsed/60:.1f} minutes[/cyan]\n"
        f"Storage: [dim]{PERSIST_DIR}[/dim]",
        title="Done",
    ))

    return embedder, collection


# ============================================================
# QUERY
# ============================================================
def expand_query(query):
    parts = [query]
    ql = query.lower()
    for key, syns in TALMUD_SYNONYMS.items():
        if key in query or key in ql:
            parts.extend(syns[:2])
    return " ".join(parts)


def query_shas(embedder, reranker, collection, question, top_k=5):
    expanded = expand_query(question)
    q_emb = embedder.encode(expanded).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=15,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return []

    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "bi_score": 1 - results["distances"][0][i],
        })

    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index the entire Talmud Bavli")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--query", default=None, help="Query after indexing")
    args = parser.parse_args()

    embedder, collection = index_full_shas(resume=args.resume)

    if args.query:
        console.print(f"\n[bold blue]Loading reranker...[/bold blue]")
        reranker = CrossEncoder(RERANKER_MODEL)

        console.print(f"\n[bold green]Question:[/bold green] {args.query}")
        results = query_shas(embedder, reranker, collection, args.query)

        for i, r in enumerate(results, 1):
            text = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
            console.print(Panel(
                f"[bold]Score:[/bold] {r['rerank_score']:.3f} (bi: {r['bi_score']:.3f})\n"
                f"[bold]Masechet:[/bold] {r['metadata'].get('tractate', '?')}\n"
                f"[bold]Ref:[/bold] {r['metadata'].get('ref', '?')}\n"
                f"[bold]URL:[/bold] {r['metadata'].get('url', '')}\n\n{text}",
                title=f"#{i}",
                width=100,
            ))
