"""
Talmud RAG API v3 — FastAPI backend with full optimization suite.
Uses engine_v3: HyDE, Hybrid Search, Small-to-Big, Cross-Encoder Reranking.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from rag_factory.engine_v4 import TalmudRAGEngine, SEDER_MAP

# ============================================================
# CONFIG
# ============================================================
PERSIST_CANDIDATES = [
    ("rag_data_shas", "shas_complete"),
    ("rag_data_v2", "talmud_v2"),
    ("rag_data", "sefaria_talmud"),
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# APP
# ============================================================
app = FastAPI(title="Talmud RAG API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: TalmudRAGEngine | None = None
db_info = {"status": "loading"}


def init_engine():
    global engine, db_info

    for dir_name, coll_name in PERSIST_CANDIDATES:
        full_path = os.path.join(BASE_DIR, dir_name)
        if os.path.exists(full_path):
            try:
                e = TalmudRAGEngine(
                    persist_dir=full_path,
                    collection_name=coll_name,
                )
                e.load()
                stats = e.get_stats()
                if stats["chunks"] > 0:
                    engine = e
                    db_info = {
                        "status": "ready",
                        "version": "v3",
                        "features": [
                            "HyDE (Hypothetical Document Embedding)",
                            "Hybrid Search (Vector + BM25)",
                            "Cross-Encoder Reranking",
                            "Small-to-Big Retrieval",
                            "Query Expansion (Talmudic Dictionary)",
                            "Metadata Filtering (Masechet, Seder)",
                        ],
                        **stats,
                    }
                    print(f"Engine v3 ready: {stats['chunks']} chunks, BM25: {stats['bm25_index_size']}")
                    return
            except Exception as e:
                print(f"Could not load {coll_name}: {e}")

    db_info = {"status": "no_data", "message": "No indexed data found. Run indexation first."}


# ============================================================
# ROUTES
# ============================================================
@app.on_event("startup")
async def startup():
    init_engine()


@app.get("/api/status")
async def status():
    return db_info


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Question to search"),
    top_k: int = Query(5, ge=1, le=20),
    tractate: str = Query(None, description="Filter by masechet"),
    seder: str = Query(None, description="Filter by seder (Moed, Nashim, etc.)"),
    hyde: bool = Query(True, description="Use HyDE"),
    bm25: bool = Query(True, description="Use BM25 hybrid search"),
    context: bool = Query(True, description="Include parent context"),
):
    if not engine:
        return {"error": "Engine not loaded", "results": []}

    result = engine.search(
        question=q,
        top_k=top_k,
        tractate=tractate,
        seder=seder,
        use_hyde=hyde,
        use_bm25=bm25,
        include_context=context,
    )

    return result


@app.get("/api/tractates")
async def list_tractates():
    if not engine or not engine.collection:
        return {"tractates": [], "sedarim": list(SEDER_MAP.keys())}

    try:
        sample = engine.collection.get(limit=10000, include=["metadatas"])
        tractates = set()
        for m in sample["metadatas"]:
            if m and "tractate" in m:
                tractates.add(m["tractate"])
        return {
            "tractates": sorted(tractates),
            "sedarim": list(SEDER_MAP.keys()),
        }
    except Exception:
        return {"tractates": [], "sedarim": list(SEDER_MAP.keys())}


@app.get("/api/sedarim")
async def list_sedarim():
    return SEDER_MAP


# Serve frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
