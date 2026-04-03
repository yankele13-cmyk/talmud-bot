"""
Talmud RAG API v5 — FastAPI backend.
Qdrant + BGE-M3 + bge-reranker-v2-m3 + Claude Sonnet synthesis.
SSE streaming for real-time Rav responses.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from rag_factory.engine_v5 import TalmudRAGEngine, SEDER_MAP

# ============================================================
# APP
# ============================================================
app = FastAPI(title="Talmud RAG API", version="5.0")

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
    try:
        e = TalmudRAGEngine(
            qdrant_url=os.environ.get("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", 6333)),
        )
        e.load()
        stats = e.get_stats()
        if stats["chunks"] > 0:
            engine = e
            db_info = {
                "status": "ready",
                "version": "v5",
                "features": [
                    "BGE-M3 embeddings (1024d, 8192 tokens)",
                    "BGE-Reranker-v2-M3 (568M, multilingual)",
                    "Qdrant hybrid search (dense + sparse + RRF)",
                    "HyDE (3-doc average, Claude Haiku)",
                    "Contextual Retrieval",
                    "Rav synthesis (Claude Sonnet, CoT + citations)",
                    "SSE streaming",
                ],
                **stats,
            }
        else:
            engine = e  # Keep engine for indexing even if empty
            db_info = {"status": "empty", "message": "Qdrant connected but no data. Run indexation.", **stats}
    except Exception as ex:
        db_info = {"status": "error", "message": str(ex)}
        print(f"Engine init failed: {ex}")


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
    q: str = Query(..., description="Question"),
    top_k: int = Query(5, ge=1, le=20),
    tractate: str = Query(None),
    seder: str = Query(None),
    hyde: bool = Query(True),
    context: bool = Query(True),
):
    if not engine:
        return {"error": "Engine not loaded", "results": []}

    return engine.search(
        question=q, top_k=top_k, tractate=tractate,
        seder=seder, use_hyde=hyde, include_context=context,
    )


@app.get("/api/search/stream")
async def api_search_stream(
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=20),
    tractate: str = Query(None),
    seder: str = Query(None),
):
    """SSE endpoint — streams sources then Claude tokens."""
    if not engine:
        return {"error": "Engine not loaded"}

    async def generate():
        for event in engine.search_stream(
            question=q, top_k=top_k, tractate=tractate, seder=seder,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/tractates")
async def list_tractates():
    if not engine or not engine.qdrant:
        return {"tractates": [], "sedarim": list(SEDER_MAP.keys())}
    try:
        # Scroll a sample to find unique tractates
        points, _ = engine.qdrant.scroll(
            collection_name=engine.collection_name,
            limit=100,
            with_payload=["tractate"],
        )
        tractates = set()
        for p in points:
            if p.payload and "tractate" in p.payload:
                tractates.add(p.payload["tractate"])
        return {"tractates": sorted(tractates), "sedarim": list(SEDER_MAP.keys())}
    except Exception:
        return {"tractates": [], "sedarim": list(SEDER_MAP.keys())}


@app.get("/api/sedarim")
async def list_sedarim():
    return SEDER_MAP


# Frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
