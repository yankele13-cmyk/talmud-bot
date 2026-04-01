#!/usr/bin/env python3
"""
Shas Indexer v5 — Uses Sefaria-Export (offline bulk) + Qdrant + BGE-M3.

Two modes:
  1. OFFLINE (recommended): Clone Sefaria-Export, read JSON files locally
     git clone https://github.com/Sefaria/Sefaria-Export.git sefaria-export
     python index_shas_v5.py --export-dir ./sefaria-export

  2. ONLINE (fallback): Fetch from Sefaria API (slower, 0.25s/page)
     python index_shas_v5.py --online

Usage:
  python index_shas_v5.py --export-dir ./sefaria-export              # Full Shas from export
  python index_shas_v5.py --export-dir ./sefaria-export --tractate Berakhot  # Single
  python index_shas_v5.py --online --tractate Berakhot               # API fallback
  python index_shas_v5.py --resume                                   # Resume from checkpoint
"""

import sys
import os
import re
import json
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

try:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    CLAUDE_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")) if os.environ.get("ANTHROPIC_API_KEY") else None
except Exception:
    CLAUDE_CLIENT = None

from rag_factory.engine_v5 import (
    TalmudRAGEngine, ensure_collection, text_to_sparse_vector,
    generate_chunk_context, DEFAULT_COLLECTION, DENSE_DIM, MASECHET_TO_SEDER,
    models,
)

console = Console()
SEFARIA_BASE = "https://www.sefaria.org"
PROGRESS_FILE = "./rag_progress_v5.json"
BATCH_SIZE = 32


# ============================================================
# DATA LOADING — Sefaria Export (offline) or API (online)
# ============================================================
def load_tractate_from_export(export_dir: str, tractate: str) -> list[dict]:
    """Load a tractate from Sefaria-Export JSON files."""
    # Path: sefaria-export/json/Talmud/Bavli/Seder X/Tractate/Hebrew/merged.json
    pattern = os.path.join(export_dir, "json", "Talmud", "Bavli", "**", tractate, "Hebrew", "merged.json")
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        # Try English merged too
        pattern = os.path.join(export_dir, "json", "Talmud", "Bavli", "**", tractate, "Hebrew", "*.json")
        matches = glob.glob(pattern, recursive=True)

    if not matches:
        console.print(f"  [red]Not found in export: {tractate}[/red]")
        return []

    pages = []
    for filepath in matches:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Sefaria export format: {"text": [[...], [...], ...], "versions": [...]}
        # or {"he": [[...]], "text": [[...]]} depending on version
        he_text = data.get("he", data.get("text", []))

        if not isinstance(he_text, list):
            continue

        # For Talmud, text is organized by daf (amud)
        for daf_idx, daf_content in enumerate(he_text):
            if not daf_content:
                continue

            # Calculate daf reference
            daf_num = (daf_idx // 2) + 2  # Starts at daf 2
            amud = "a" if daf_idx % 2 == 0 else "b"
            ref = f"{tractate}.{daf_num}{amud}"

            # Flatten text content
            if isinstance(daf_content, list):
                text_parts = []
                for item in daf_content:
                    if isinstance(item, str):
                        text_parts.append(clean_html(item))
                    elif isinstance(item, list):
                        text_parts.extend(clean_html(str(x)) for x in item if x)
                gemara_text = "\n".join(p for p in text_parts if p)
            else:
                gemara_text = clean_html(str(daf_content))

            if len(gemara_text) > 30:
                pages.append({
                    "ref": ref,
                    "gemara": gemara_text,
                    "rashi": "",  # Will load separately
                    "tosafot": "",
                })

    # Try to load Rashi and Tosafot from export
    for commentary, field in [("Rashi", "rashi"), ("Tosafot", "tosafot")]:
        pattern = os.path.join(export_dir, "json", "Talmud", "Bavli", "**",
                             f"{commentary} on {tractate}", "Hebrew", "merged.json")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            try:
                with open(matches[0], "r", encoding="utf-8") as f:
                    comm_data = json.load(f)
                he = comm_data.get("he", comm_data.get("text", []))
                if isinstance(he, list):
                    for daf_idx, content in enumerate(he):
                        if daf_idx < len(pages) and content:
                            flat = _flatten_nested(content)
                            if flat:
                                pages[daf_idx][field] = flat
            except Exception:
                pass

    return pages


def _flatten_nested(data):
    if isinstance(data, str):
        return clean_html(data)
    if isinstance(data, list):
        return "\n".join(p for p in (_flatten_nested(i) for i in data) if p)
    return ""


def clean_html(t):
    return re.sub(r"<[^>]+>", "", str(t)).strip() if t else ""


# API fallback (reuses existing logic)
def load_tractate_from_api(tractate: str, session=None) -> list[dict]:
    """Fetch from Sefaria API (slow but works without export)."""
    import requests
    if not session:
        session = requests.Session()
        session.headers.update({"User-Agent": "TalmudBot/5.0"})

    # Get length from Shape API
    shapes = session.get(f"{SEFARIA_BASE}/api/shape/Talmud/Bavli", timeout=30).json()
    length = 0
    for item in shapes:
        if isinstance(item, dict) and item.get("title") == tractate:
            length = item.get("length", 0)
            break

    if not length:
        return []

    pages = []
    refs = []
    for daf in range(2, 2 + (length // 2) + 1):
        refs.extend([f"{tractate}.{daf}a", f"{tractate}.{daf}b"])

    for ref in refs:
        url = f"{SEFARIA_BASE}/api/texts/{ref}?commentary=1&context=0"
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()

            gemara = _flatten_nested(data.get("he", []))
            rashi = _extract_commentary(data, "Rashi")
            tosafot = _extract_commentary(data, "Tosafot")

            if len(gemara) > 30:
                pages.append({"ref": ref, "gemara": gemara, "rashi": rashi, "tosafot": tosafot})
        except Exception:
            pass
        time.sleep(0.25)

    return pages


def _extract_commentary(data, name):
    parts = []
    for c in data.get("commentary", []):
        if not isinstance(c, dict):
            continue
        ct = c.get("collectiveTitle", {})
        n = ct.get("en", "") if isinstance(ct, dict) else str(ct)
        if n == name:
            he = c.get("he", "")
            if he:
                parts.append(_flatten_nested(he))
    return "\n".join(parts)


# ============================================================
# CHUNKING — Separate Gemara/Rashi/Tosafot
# ============================================================
def chunk_page(ref, page, max_chunk=2500, overlap=300):
    chunks = []
    for text_type, text in [("gemara", page["gemara"]), ("rashi", page.get("rashi", "")), ("tosafot", page.get("tosafot", ""))]:
        if not text or len(text) < 20:
            continue
        header = f"[{text_type.upper()} — {ref}]\n"
        for idx, chunk_text in enumerate(_split_text(header + text, max_chunk, overlap)):
            ctx = generate_chunk_context(ref, chunk_text, CLAUDE_CLIENT)
            chunks.append({
                "text": f"{ctx}\n\n{chunk_text}",
                "ref": ref,
                "type": text_type,
                "chunk_idx": idx,
                "tractate": ref.split(".")[0],
                "url": f"{SEFARIA_BASE}/{ref.replace(' ', '_')}",
            })
    return chunks


def _split_text(text, max_chunk, overlap):
    markers = [r"מתני׳", r"גמ׳", r"תנו רבנן", r"תניא", r"איתמר",
               r"אמר רב", r"אמר רבי", r"תא שמע", r"מיתיבי"]
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
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "stats": {}}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


# ============================================================
# BAVLI TRACTATE LIST
# ============================================================
BAVLI_TRACTATES = [
    "Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah",
    "Yoma", "Sukkah", "Beitzah", "Taanit", "Megillah", "Moed Katan",
    "Chagigah", "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah",
    "Gittin", "Kiddushin", "Bava Kamma", "Bava Metzia", "Bava Batra",
    "Sanhedrin", "Makkot", "Shevuot", "Avodah Zarah", "Horayot",
    "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin",
    "Temurah", "Keritot", "Meilah", "Tamid", "Niddah",
]


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Index Shas v5 (Qdrant + BGE-M3)")
    parser.add_argument("--export-dir", help="Path to cloned Sefaria-Export repo")
    parser.add_argument("--online", action="store_true", help="Fetch from API instead of export")
    parser.add_argument("--tractate", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qdrant-url", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    args = parser.parse_args()

    mode = "export" if args.export_dir else "api" if args.online else None
    if not mode:
        console.print("[red]Specify --export-dir or --online[/red]")
        console.print("  Recommended: git clone https://github.com/Sefaria/Sefaria-Export.git sefaria-export")
        console.print("  Then: python index_shas_v5.py --export-dir ./sefaria-export")
        return

    console.print(Panel(
        f"[bold cyan]SHAS INDEXER v5[/bold cyan]\n"
        f"Source: [green]{mode.upper()}[/green] {'(' + args.export_dir + ')' if args.export_dir else ''}\n"
        f"Embeddings: [green]BGE-M3[/green] (1024d)\n"
        f"Vector DB: [green]Qdrant[/green] ({args.qdrant_url}:{args.qdrant_port})\n"
        f"Contextual: [green]{'Claude Haiku' if CLAUDE_CLIENT else 'Template'}[/green]",
        title="v5 Indexer",
    ))

    # Load embedder
    console.print("\n[bold blue]1. Loading BGE-M3...[/bold blue]")
    embedder = SentenceTransformer("BAAI/bge-m3")
    console.print(f"  [green]{embedder.get_sentence_embedding_dimension()}d ready[/green]")

    # Connect Qdrant
    console.print(f"\n[bold blue]2. Connecting to Qdrant...[/bold blue]")
    qdrant = QdrantClient(host=args.qdrant_url, port=args.qdrant_port)
    ensure_collection(qdrant, DEFAULT_COLLECTION)
    console.print(f"  [green]Collection '{DEFAULT_COLLECTION}': {qdrant.count(DEFAULT_COLLECTION).count} points[/green]")

    # Tractate list
    tractates = args.tractate if args.tractate else BAVLI_TRACTATES
    progress = load_progress() if args.resume else {"completed": [], "stats": {}}
    to_index = [t for t in tractates if t not in progress["completed"]]

    console.print(f"\n[bold blue]3. {len(to_index)} massechtot to index[/bold blue]")

    total_chunks = 0
    for ti, tractate in enumerate(to_index, 1):
        console.print(f"\n  [bold yellow][{ti}/{len(to_index)}] {tractate}[/bold yellow]")

        # Load data
        if mode == "export":
            pages = load_tractate_from_export(args.export_dir, tractate)
        else:
            pages = load_tractate_from_api(tractate)

        if not pages:
            console.print(f"    [red]No data found, skipping[/red]")
            continue

        console.print(f"    {len(pages)} pages loaded")

        # Chunk
        all_chunks = []
        rashi_count = sum(1 for p in pages if p.get("rashi"))
        tosafot_count = sum(1 for p in pages if p.get("tosafot"))

        for page in pages:
            chunks = chunk_page(page["ref"], page)
            all_chunks.extend(chunks)

        console.print(f"    {len(all_chunks)} chunks (Rashi: {rashi_count}, Tosafot: {tosafot_count})")

        # Embed and index into Qdrant
        console.print(f"    Embedding & indexing...")
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            dense_vecs = embedder.encode(texts, show_progress_bar=False).tolist()

            points = []
            for j, chunk in enumerate(batch):
                chunk_id = f"{chunk['ref']}_{chunk['type']}_chunk{chunk['chunk_idx']}"
                point_id = abs(hash(chunk_id)) % (2**63)
                sparse = text_to_sparse_vector(chunk["text"])

                points.append(models.PointStruct(
                    id=point_id,
                    vector={"dense": dense_vecs[j], "sparse": sparse},
                    payload={
                        "chunk_id": chunk_id,
                        "text": chunk["text"],
                        "ref": chunk["ref"],
                        "tractate": chunk["tractate"],
                        "type": chunk["type"],
                        "chunk_idx": chunk["chunk_idx"],
                        "url": chunk["url"],
                    },
                ))

            qdrant.upsert(collection_name=DEFAULT_COLLECTION, points=points)

        total_chunks += len(all_chunks)

        progress["completed"].append(tractate)
        progress["stats"][tractate] = {
            "chunks": len(all_chunks), "pages": len(pages),
            "rashi": rashi_count, "tosafot": tosafot_count,
        }
        save_progress(progress)
        console.print(f"    [green]Done ({len(all_chunks)} chunks indexed)[/green]")

    console.print(Panel(
        f"[bold green]v5 INDEXATION COMPLETE![/bold green]\n\n"
        f"Total: [cyan]{qdrant.count(DEFAULT_COLLECTION).count}[/cyan] chunks\n"
        f"Massechtot: [cyan]{len(progress['completed'])}[/cyan]",
        title="Done",
    ))


if __name__ == "__main__":
    main()
