"""
Talmud RAG API — FastAPI backend.
Serves the RAG query engine + static frontend.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# CONFIG
# ============================================================
# Try shas first, fall back to v2, then v1
PERSIST_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_data_shas"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_data_v2"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_data"),
]
COLLECTIONS = ["shas_complete", "talmud_v2", "sefaria_talmud"]
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TALMUD_SYNONYMS = {
    "שמע": ["קריאת שמע", "שמע בערבין", "בשכבך ובקומך"],
    "תפילה": ["שמונה עשרה", "עמידה", "תפילת שחרית"],
    "שבת": ["שבתא", "יום השבת", "מלאכות שבת"],
    "ברכה": ["ברכות", "ברכת המזון", "ברכת הנהנין"],
    "טומאה": ["טמא", "טהרה", "טבילה"],
    "כהן": ["כהנים", "כהן גדול", "תרומה"],
    "קרבן": ["קרבנות", "עולה", "חטאת", "זבח"],
    "shema": ["קריאת שמע", "שמע"],
    "prayer": ["תפילה", "tefillah"],
    "shabbat": ["שבת", "sabbath"],
}

# ============================================================
# INIT
# ============================================================
app = FastAPI(title="Talmud RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
embedder = None
reranker = None
collection = None
db_info = {"status": "loading"}


def init_models():
    global embedder, reranker, collection, db_info

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    # Find available collection
    for persist_dir, coll_name in zip(PERSIST_DIRS, COLLECTIONS):
        if os.path.exists(persist_dir):
            try:
                client = chromadb.PersistentClient(path=persist_dir)
                collection = client.get_collection(coll_name)
                count = collection.count()
                if count > 0:
                    db_info = {
                        "status": "ready",
                        "collection": coll_name,
                        "chunks": count,
                        "persist_dir": persist_dir,
                    }
                    print(f"Loaded: {coll_name} ({count} chunks)")
                    return
            except Exception as e:
                print(f"Could not load {coll_name}: {e}")

    db_info = {"status": "no_data", "message": "No indexed data found. Run indexation first."}
    print("WARNING: No indexed data found!")


# ============================================================
# QUERY
# ============================================================
def expand_query(query: str) -> str:
    parts = [query]
    ql = query.lower()
    for key, syns in TALMUD_SYNONYMS.items():
        if key in query or key in ql:
            parts.extend(syns[:2])
    return " ".join(parts)


def search(question: str, top_k: int = 5, tractate_filter: str = None):
    if not collection:
        return []

    expanded = expand_query(question)
    q_emb = embedder.encode(expanded).tolist()

    # Build query params
    query_params = {
        "query_embeddings": [q_emb],
        "n_results": min(top_k * 3, 20),
        "include": ["documents", "metadatas", "distances"],
    }

    if tractate_filter:
        query_params["where"] = {"tractate": tractate_filter}

    results = collection.query(**query_params)

    if not results["ids"][0]:
        return []

    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "bi_score": round(1 - results["distances"][0][i], 4),
        })

    # Rerank
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# ROUTES
# ============================================================
@app.on_event("startup")
async def startup():
    init_models()


@app.get("/api/status")
async def status():
    return db_info


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Question to search"),
    top_k: int = Query(5, ge=1, le=20),
    tractate: str = Query(None, description="Filter by tractate name"),
):
    start = time.time()
    results = search(q, top_k=top_k, tractate_filter=tractate)
    elapsed = time.time() - start

    return {
        "query": q,
        "expanded": expand_query(q),
        "results": results,
        "count": len(results),
        "time_ms": round(elapsed * 1000),
    }


@app.get("/api/tractates")
async def list_tractates():
    """List all indexed tractates."""
    if not collection:
        return {"tractates": []}

    # Sample metadata to find unique tractates
    try:
        sample = collection.get(limit=10000, include=["metadatas"])
        tractates = set()
        for m in sample["metadatas"]:
            if m and "tractate" in m:
                tractates.add(m["tractate"])
        return {"tractates": sorted(tractates)}
    except Exception:
        return {"tractates": []}


# Serve frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
