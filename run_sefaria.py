#!/usr/bin/env python3
"""
Quick launcher — Build a RAG on the entire Sefaria Talmud library.

Usage:
  # Tout le Talmud Bavli (Gemara + Rashi + Tosafot)
  python run_sefaria.py

  # Un seul traite
  python run_sefaria.py --tractate Berakhot

  # Plusieurs traites
  python run_sefaria.py --tractate Berakhot --tractate Shabbat --tractate Eruvin

  # Tout le Tanakh
  python run_sefaria.py --category tanakh

  # Toute la bibliotheque Sefaria
  python run_sefaria.py --category all

  # Avec chat interactif apres la construction
  python run_sefaria.py --tractate Berakhot --chat
"""

import argparse
import sys

from rag_factory.connectors.sefaria_connector import SefariaConfig
from rag_factory.orchestrator import RAGFactoryOrchestrator, PipelineState


def main():
    parser = argparse.ArgumentParser(
        description="Build a RAG from Sefaria texts — 100% local, zero cost",
    )
    parser.add_argument(
        "--category", default="talmud",
        help="Category: talmud, tanakh, mishnah, midrash, all (default: talmud)",
    )
    parser.add_argument(
        "--tractate", action="append", default=[],
        help="Specific tractate(s) to index (can repeat). Empty = all in category.",
    )
    parser.add_argument("--language", default="he", choices=["he", "en", "both"])
    parser.add_argument("--no-commentary", action="store_true", help="Skip Rashi/Tosafot")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages per tractate (0=all)")
    parser.add_argument("--llm-model", default="llama3.2")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--persist-dir", default="./rag_data")
    parser.add_argument("--collection", default="sefaria_talmud")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat after build")
    args = parser.parse_args()

    # Build the sefaria:// URL
    if args.tractate:
        tractates = ",".join(args.tractate)
        db_url = f"sefaria://{args.category}/{tractates}"
    else:
        db_url = f"sefaria://{args.category}"

    state = PipelineState(
        db_url=db_url,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    orchestrator = RAGFactoryOrchestrator(state)
    result = orchestrator.run()

    if result.query_engine and args.chat:
        from rag_factory.__main__ import _chat_loop
        _chat_loop(result.query_engine)

    sys.exit(0 if not result.errors else 1)


if __name__ == "__main__":
    main()
