#!/usr/bin/env python3
"""
RAG Factory v2 — Production-grade Talmud RAG
=============================================
Improvements over v1:
  1. Fixed Rashi/Tosafot parsing (collectiveTitle is a dict, not string)
  2. Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
  3. Semantic chunking (split by sugya markers, not by daf)
  4. Cross-encoder reranker after retrieval
  5. Query expansion with Talmudic terminology
  6. LLM-free response synthesis from sources

100% gratuit. Zero API. Tourne sur CPU.
"""

import sys
import os
import re
import time
import json
from typing import Optional

sys.path.insert(0, "/home/user/talmud-bot")

import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

console = Console()

# ============================================================
# CONFIG
# ============================================================
SEFARIA_BASE = "https://www.sefaria.org"
PERSIST_DIR = "/home/user/talmud-bot/rag_data_v2"
COLLECTION_NAME = "talmud_v2"
# Multilingual model — handles Hebrew/Aramaic properly (120M params, 384 dims)
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# Cross-encoder for reranking (much more accurate than bi-encoder alone)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DELAY = 0.3

# Talmudic query expansion dictionary
TALMUD_SYNONYMS = {
    "שמע": ["קריאת שמע", "שמע בערבין", "שמע בשחרית", "בשכבך ובקומך"],
    "תפילה": ["שמונה עשרה", "עמידה", "תפילת שחרית", "תפילת מנחה", "תפילת ערבית"],
    "שבת": ["שבתא", "יום השבת", "הלכות שבת", "מלאכות שבת"],
    "ברכה": ["ברכות", "מברך", "ברכת המזון", "ברכת הנהנין"],
    "טומאה": ["טמא", "טהרה", "טבילה", "מקוה"],
    "כהן": ["כהנים", "כהן גדול", "תרומה", "בית המקדש"],
    "shema": ["קריאת שמע", "שמע", "shema yisrael", "krias shema"],
    "prayer": ["תפילה", "tefillah", "shemoneh esrei", "amidah"],
    "shabbat": ["שבת", "sabbath", "shabbos"],
    "blessing": ["ברכה", "bracha", "beracha", "ברכות"],
}


# ============================================================
# 1. SEFARIA FETCHER (fixed commentary parsing)
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


def extract_commentary(data: dict, target_name: str) -> str:
    """
    Extract commentary — FIXED: collectiveTitle is {"en": "Rashi", "he": "רש\"י"},
    not a plain string.
    """
    commentary = data.get("commentary", [])
    parts = []
    for c in commentary:
        if not isinstance(c, dict):
            continue

        ct = c.get("collectiveTitle", {})
        # collectiveTitle can be a dict or a string
        if isinstance(ct, dict):
            name_en = ct.get("en", "")
            name_he = ct.get("he", "")
        else:
            name_en = str(ct)
            name_he = ""

        if name_en == target_name or name_he == target_name:
            he_text = c.get("he", "")
            if he_text:
                parts.append(flatten_text(he_text))

    return "\n".join(parts)


def fetch_page(session, ref):
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


def build_full_document(ref: str, data: dict) -> dict:
    """Build a rich document with Gemara + Rashi + Tosafot."""
    gemara = flatten_text(data.get("he", []))
    rashi = extract_commentary(data, "Rashi")
    tosafot = extract_commentary(data, "Tosafot")

    sections = [f"[GEMARA — {ref}]\n{gemara}"]
    if rashi:
        sections.append(f"\n[RASHI — {ref}]\n{rashi}")
    if tosafot:
        sections.append(f"\n[TOSAFOT — {ref}]\n{tosafot}")

    return {
        "full_text": "\n\n".join(sections),
        "gemara": gemara,
        "rashi": rashi,
        "tosafot": tosafot,
        "ref": ref,
        "url": f"{SEFARIA_BASE}/{ref.replace(' ', '_')}",
    }


def get_daf_refs(tractate: str, max_dafim: int = 0) -> list[str]:
    session = requests.Session()
    resp = session.get(f"{SEFARIA_BASE}/api/shape/Talmud/Bavli", timeout=30)
    resp.raise_for_status()
    length = 0
    for item in resp.json():
        if isinstance(item, dict) and item.get("title") == tractate:
            length = item.get("length", 0)
            break
    if not length:
        return []
    refs = []
    for daf in range(2, 2 + (length // 2) + 1):
        if max_dafim and len(refs) >= max_dafim:
            break
        refs.append(f"{tractate}.{daf}a")
        if max_dafim and len(refs) >= max_dafim:
            break
        refs.append(f"{tractate}.{daf}b")
    return refs


# ============================================================
# 2. SEMANTIC CHUNKING (by sugya, not by daf)
# ============================================================
def semantic_chunk(text: str, ref: str, max_chunk: int = 1500, overlap: int = 200) -> list[dict]:
    """
    Split text into semantic chunks based on Talmudic markers.
    Markers: Mishna headers, Gemara transitions, major topic shifts.
    """
    # Talmudic section markers
    markers = [
        r"מתני׳",           # Mishna
        r"גמ׳",             # Gemara start
        r"תנו רבנן",        # Baraita
        r"תניא",            # Baraita
        r"איתמר",           # Amoraic statement
        r"בעי רב",          # Question
        r"בעי רבי",         # Question
        r"אמר רב",          # Statement
        r"אמר רבי",         # Statement
        r"תא שמע",          # "Come and hear" - proof
        r"מיתיבי",          # Objection
        r"ורמינהו",         # Contradiction
        r"לימא מסייע",      # Support
        r"שנאמר",           # Scriptural proof
        r"\[RASHI",         # Rashi section
        r"\[TOSAFOT",       # Tosafot section
    ]

    pattern = "|".join(f"({m})" for m in markers)

    # Find all split points
    split_points = [0]
    for match in re.finditer(pattern, text):
        pos = match.start()
        if pos > 0:
            split_points.append(pos)
    split_points.append(len(text))

    # Merge small segments, split large ones
    chunks = []
    current = ""
    chunk_idx = 0

    for i in range(len(split_points) - 1):
        segment = text[split_points[i]:split_points[i + 1]]

        if len(current) + len(segment) <= max_chunk:
            current += segment
        else:
            if current.strip():
                chunks.append({
                    "text": current.strip(),
                    "ref": ref,
                    "chunk_idx": chunk_idx,
                })
                chunk_idx += 1
            # If segment itself is too large, split it with overlap
            if len(segment) > max_chunk:
                start = 0
                while start < len(segment):
                    end = start + max_chunk
                    sub = segment[start:end]
                    if sub.strip():
                        chunks.append({
                            "text": sub.strip(),
                            "ref": ref,
                            "chunk_idx": chunk_idx,
                        })
                        chunk_idx += 1
                    start = end - overlap
                current = ""
            else:
                current = segment

    if current.strip():
        chunks.append({
            "text": current.strip(),
            "ref": ref,
            "chunk_idx": chunk_idx,
        })

    return chunks


# ============================================================
# 3. QUERY EXPANSION
# ============================================================
def expand_query(query: str) -> str:
    """Expand query with Talmudic synonyms and terminology."""
    expanded_parts = [query]

    query_lower = query.lower()
    for key, synonyms in TALMUD_SYNONYMS.items():
        if key in query or key in query_lower:
            expanded_parts.extend(synonyms[:2])  # Add top 2 synonyms

    return " ".join(expanded_parts)


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(tractate: str = "Berakhot", max_dafim: int = 0):
    console.print(Panel(
        f"[bold cyan]RAG Factory v2 — Production Pipeline[/bold cyan]\n"
        f"Tractate: [yellow]{tractate}[/yellow]\n"
        f"Embeddings: [green]{EMBED_MODEL}[/green] (multilingual)\n"
        f"Reranker: [green]{RERANKER_MODEL}[/green] (cross-encoder)\n"
        f"Chunking: [green]Semantic (by sugya)[/green]\n"
        f"Vector DB: [green]ChromaDB[/green]",
        title="RAG v2",
    ))

    # --- Load models ---
    console.print("\n[bold blue]1. Loading models...[/bold blue]")
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    console.print(f"  [green]Embedder: {EMBED_MODEL} (multilingual, 384d)[/green]")
    console.print(f"  [green]Reranker: {RERANKER_MODEL} (cross-encoder)[/green]")

    # --- Setup ChromaDB ---
    console.print("\n[bold blue]2. Setting up ChromaDB...[/bold blue]")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Fetch and index ---
    console.print(f"\n[bold blue]3. Fetching {tractate} from Sefaria...[/bold blue]")
    refs = get_daf_refs(tractate, max_dafim)
    console.print(f"  [dim]{len(refs)} pages to fetch[/dim]")

    session = requests.Session()
    session.headers.update({"User-Agent": "TalmudBot-RAGFactory/2.0"})

    all_chunks = []
    rashi_count = 0
    tosafot_count = 0

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Fetching & chunking {tractate}...", total=len(refs))
        for ref in refs:
            data = fetch_page(session, ref)
            progress.advance(task)
            if not data:
                continue

            doc = build_full_document(ref, data)
            if not doc["full_text"].strip() or len(doc["full_text"]) < 30:
                continue

            if doc["rashi"]:
                rashi_count += 1
            if doc["tosafot"]:
                tosafot_count += 1

            # Semantic chunking
            chunks = semantic_chunk(doc["full_text"], ref)
            all_chunks.extend(chunks)

            time.sleep(DELAY)

    console.print(f"  [green]Pages with Rashi: {rashi_count}[/green]")
    console.print(f"  [green]Pages with Tosafot: {tosafot_count}[/green]")
    console.print(f"  [green]Total chunks: {len(all_chunks)}[/green]")

    # --- Embed and store ---
    console.print(f"\n[bold blue]4. Embedding {len(all_chunks)} chunks...[/bold blue]")
    BATCH = 64
    indexed = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Embedding...", total=len(all_chunks))
        for i in range(0, len(all_chunks), BATCH):
            batch = all_chunks[i:i + BATCH]
            texts = [c["text"] for c in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

            ids = [f"{c['ref']}_chunk{c['chunk_idx']}" for c in batch]
            metas = [{
                "ref": c["ref"],
                "chunk_idx": c["chunk_idx"],
                "tractate": tractate,
                "url": f"{SEFARIA_BASE}/{c['ref'].replace(' ', '_')}",
                "has_rashi": "[RASHI" in c["text"],
                "has_tosafot": "[TOSAFOT" in c["text"],
                "text_length": len(c["text"]),
            } for c in batch]

            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metas,
            )
            indexed += len(batch)
            progress.advance(task, len(batch))

    console.print(f"  [green]{indexed} chunks indexed[/green]")

    console.print(Panel(
        f"[bold green]Indexation Complete![/bold green]\n\n"
        f"Chunks: [cyan]{indexed}[/cyan]\n"
        f"Rashi: [cyan]{rashi_count}[/cyan] pages | "
        f"Tosafot: [cyan]{tosafot_count}[/cyan] pages\n"
        f"Dir: [dim]{PERSIST_DIR}[/dim]",
        title="Done",
    ))

    return embedder, reranker, collection


# ============================================================
# QUERY ENGINE (with expansion + reranking + synthesis)
# ============================================================
def query_rag(
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    collection,
    question: str,
    top_k_retrieve: int = 15,
    top_k_rerank: int = 5,
):
    """
    Full retrieval pipeline:
    1. Expand query with Talmudic terms
    2. Embed expanded query
    3. Retrieve top-K candidates (broad)
    4. Rerank with cross-encoder (precise)
    5. Synthesize answer from top sources
    """

    # Step 1: Query expansion
    expanded = expand_query(question)
    if expanded != question:
        console.print(f"  [dim]Expanded: {expanded[:100]}...[/dim]")

    # Step 2: Embed
    q_embedding = embedder.encode(expanded).tolist()

    # Step 3: Broad retrieval
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k_retrieve,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return [], "No results found."

    # Step 4: Cross-encoder reranking
    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "bi_score": 1 - results["distances"][0][i],
        })

    # Score each (query, passage) pair with cross-encoder
    pairs = [(question, c["text"]) for c in candidates]
    cross_scores = reranker.predict(pairs)

    for c, score in zip(candidates, cross_scores):
        c["rerank_score"] = float(score)

    # Sort by rerank score
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_results = candidates[:top_k_rerank]

    # Step 5: Synthesize answer from sources
    answer = synthesize_answer(question, top_results)

    return top_results, answer


def synthesize_answer(question: str, sources: list[dict]) -> str:
    """
    Build a structured answer from the retrieved sources.
    No LLM needed — uses extractive synthesis.
    """
    if not sources:
        return "No relevant sources found."

    parts = []
    parts.append(f"**Question:** {question}\n")
    parts.append(f"**{len(sources)} sources found (ranked by relevance):**\n")

    for i, src in enumerate(sources, 1):
        ref = src["metadata"].get("ref", src["id"])
        url = src["metadata"].get("url", "")
        score = src["rerank_score"]
        bi_score = src["bi_score"]

        text = src["text"]
        # Extract the most relevant 500 chars
        if len(text) > 500:
            # Try to find the section most relevant to the question
            text = text[:500] + "..."

        parts.append(f"### Source {i}: {ref} (score: {score:.3f})")
        parts.append(f"[{url}]({url})")

        # Label sections
        if "[RASHI" in text:
            parts.append("*Includes Rashi commentary*")
        if "[TOSAFOT" in text:
            parts.append("*Includes Tosafot commentary*")

        parts.append(f"\n{text}\n")

    # Summary line
    refs = [s["metadata"].get("ref", "") for s in sources]
    parts.append(f"\n---\n**References:** {', '.join(refs)}")

    return "\n".join(parts)


# ============================================================
# INTERACTIVE CHAT
# ============================================================
def chat_loop(embedder, reranker, collection):
    console.print(Panel(
        "[bold cyan]Talmud RAG v2 — Chat[/bold cyan]\n"
        "Features: multilingual embeddings, cross-encoder reranking, query expansion\n"
        "Type [bold]quit[/bold] to exit.",
        title="Chat",
    ))

    while True:
        try:
            question = input("\n\033[1;32mQuestion:\033[0m ")
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question.strip():
            continue

        console.print("[dim]Searching...[/dim]")
        sources, answer = query_rag(embedder, reranker, collection, question)

        console.print(Panel(answer, title="Answer", width=100))

        # Show rerank comparison
        if sources:
            table = Table(title="Reranking Details")
            table.add_column("Ref", style="yellow")
            table.add_column("Bi-encoder", justify="right", style="dim")
            table.add_column("Cross-encoder", justify="right", style="cyan")
            table.add_column("Rashi", justify="center")
            table.add_column("Tosafot", justify="center")

            for s in sources:
                table.add_row(
                    s["metadata"].get("ref", "?"),
                    f"{s['bi_score']:.4f}",
                    f"{s['rerank_score']:.4f}",
                    "Yes" if s["metadata"].get("has_rashi") else "-",
                    "Yes" if s["metadata"].get("has_tosafot") else "-",
                )
            console.print(table)

    console.print("[dim]Lehitraot![/dim]")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Factory v2 — Production Talmud RAG")
    parser.add_argument("--tractate", default="Berakhot")
    parser.add_argument("--max-dafim", type=int, default=0)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    embedder, reranker, collection = run_pipeline(args.tractate, args.max_dafim)

    # Demo queries
    demo_questions = [
        "מהו הזמן של קריאת שמע בערבית?",
        "What does Rabbi Eliezer say about Shema?",
        "מה אומר רש״י על הכהנים שנכנסים לאכול בתרומתן?",
    ]

    if args.query:
        demo_questions = [args.query]

    for q in demo_questions:
        console.print(f"\n{'='*80}")
        console.print(f"[bold green]Question:[/bold green] {q}")
        sources, answer = query_rag(embedder, reranker, collection, q)
        console.print(Panel(answer, title="Answer", width=100))

        if sources:
            table = Table(title="Reranking")
            table.add_column("Ref", style="yellow")
            table.add_column("Bi-encoder", justify="right", style="dim")
            table.add_column("Rerank", justify="right", style="cyan")
            table.add_column("Rashi")
            table.add_column("Tosafot")
            for s in sources:
                table.add_row(
                    s["metadata"].get("ref", "?"),
                    f"{s['bi_score']:.4f}",
                    f"{s['rerank_score']:.4f}",
                    "Yes" if s["metadata"].get("has_rashi") else "-",
                    "Yes" if s["metadata"].get("has_tosafot") else "-",
                )
            console.print(table)

    if args.chat:
        chat_loop(embedder, reranker, collection)
