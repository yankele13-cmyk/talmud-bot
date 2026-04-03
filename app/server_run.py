"""Ultra-light server — loads ChromaDB, no BM25 (saves RAM)."""
import os, sys, time, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI(title="Talmud RAG Live")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embedder = reranker = collection = None
db_info = {"status": "loading"}

DICT = {
    "שמע": ["קריאת שמע", "שמע בערבין"], "תפילה": ["שמונה עשרה", "עמידה"],
    "גאולה": ["גאולה לתפילה", "סמיכת גאולה"], "שבת": ["שבתא", "מלאכות שבת"],
    "shema": ["קריאת שמע"], "prayer": ["תפילה"], "shabbat": ["שבת"],
}
SEDER = {"Zeraim": ["Berakhot"], "Moed": ["Shabbat", "Eruvin"]}
M2S = {}
for s, ms in SEDER.items():
    for m in ms: M2S[m] = s

def expand(q):
    parts = [q]
    for k, syns in DICT.items():
        if k in q or k in q.lower(): parts.extend(syns)
    seen = set()
    return " ".join(x for x in parts if x not in seen and not seen.add(x))

def init():
    global embedder, reranker, collection, db_info
    print("Loading embedder...", flush=True)
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Loading reranker...", flush=True)
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    print("Loading ChromaDB...", flush=True)
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_data_shas")
    client = chromadb.PersistentClient(path=p)
    collection = client.get_collection("shas_complete")
    count = collection.count()
    db_info = {"status": "ready", "version": "live", "chunks": count,
               "massechtot": "Berakhot, Shabbat, Eruvin",
               "features": ["Multilingual embeddings", "Cross-encoder reranking", "Query expansion"]}
    print(f"READY! {count} chunks", flush=True)

@app.on_event("startup")
async def startup(): init()

@app.get("/api/status")
async def status(): return db_info

@app.get("/api/search")
async def search(q: str = Query(...), top_k: int = Query(5, ge=1, le=10), tractate: str = Query(None)):
    start = time.time()
    expanded = expand(q)
    qe = embedder.encode(expanded).tolist()

    params = {"query_embeddings": [qe], "n_results": 15, "include": ["documents", "metadatas", "distances"]}
    if tractate: params["where"] = {"tractate": tractate}
    vr = collection.query(**params)

    candidates = []
    for i in range(len(vr["ids"][0])):
        candidates.append({
            "id": vr["ids"][0][i], "text": vr["documents"][0][i],
            "metadata": vr["metadatas"][0][i],
            "bi_score": round(1 - vr["distances"][0][i], 4),
        })

    if candidates:
        pairs = [(q, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores): c["rerank_score"] = round(float(s), 4)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    top = candidates[:top_k]
    qw = set(re.findall(r'[\w\u0590-\u05FF]{3,}', q))
    sources = []
    for i, src in enumerate(top):
        t = src["text"]; m = src["metadata"]
        sents = [s.strip() for s in re.split(r'\n', t) if len(s.strip()) >= 10]
        scored = [(len(qw & set(re.findall(r'[\w\u0590-\u05FF]+', s))), s) for s in sents]
        scored.sort(reverse=True)
        sources.append({
            "rank": i+1, "ref": m.get("ref", src["id"]), "tractate": m.get("tractate", ""),
            "seder": M2S.get(m.get("tractate", ""), ""), "url": m.get("url", ""),
            "rerank_score": src.get("rerank_score", 0), "bi_score": src.get("bi_score", 0),
            "has_gemara": "[GEMARA" in t, "has_rashi": m.get("has_rashi", False) or "[RASHI" in t,
            "has_tosafot": m.get("has_tosafot", False) or "[TOSAFOT" in t,
            "full_text": t, "key_passages": [s for _, s in scored[:3]],
        })

    s0 = sources[0] if sources else {}
    summary = f"**{s0.get('ref','')}** ({s0.get('tractate','')}) | Score: {s0.get('rerank_score',0):.2f}" if sources else ""
    return {
        "query": q, "expanded": expanded, "hyde_used": False,
        "answer": {"summary": summary, "llm_answer": None, "sources": sources, "total": len(sources)},
        "time_ms": round((time.time()-start)*1000), "candidates_evaluated": len(candidates),
    }

@app.get("/api/tractates")
async def tractates():
    try:
        s = collection.get(limit=5000, include=["metadatas"])
        t = set(m["tractate"] for m in s["metadatas"] if m and "tractate" in m)
        return {"tractates": sorted(t), "sedarim": list(SEDER.keys())}
    except: return {"tractates": [], "sedarim": []}

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
@app.get("/")
async def index(): return FileResponse(os.path.join(STATIC, "index.html"))

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
