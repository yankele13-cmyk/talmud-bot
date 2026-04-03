"""Minimal server using the v2 data (151 chunks, 3.3MB, loads fast)."""
import os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI(title="Talmud RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

E = R = C = None
INFO = {"status": "loading"}

def init():
    global E, R, C, INFO
    print("Loading models...", flush=True)
    E = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    R = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    print("Loading data...", flush=True)
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_data_v2")
    C = chromadb.PersistentClient(path=p).get_collection("talmud_v2")
    n = C.count()
    INFO = {"status": "ready", "version": "live", "chunks": n, "data": "Berakhot (20 dafim)"}
    print(f"READY — {n} chunks", flush=True)

@app.on_event("startup")
async def startup(): init()

@app.get("/api/status")
async def status(): return INFO

@app.get("/api/search")
async def search(q: str = Query(...), top_k: int = Query(5, ge=1, le=10)):
    t0 = time.time()
    qe = E.encode(q).tolist()
    vr = C.query(query_embeddings=[qe], n_results=15, include=["documents", "metadatas", "distances"])
    cands = [{"id": vr["ids"][0][i], "text": vr["documents"][0][i],
              "metadata": vr["metadatas"][0][i], "bi_score": round(1-vr["distances"][0][i], 4)}
             for i in range(len(vr["ids"][0]))]
    if cands:
        scores = R.predict([(q, c["text"]) for c in cands])
        for c, s in zip(cands, scores): c["rerank_score"] = round(float(s), 4)
        cands.sort(key=lambda x: x["rerank_score"], reverse=True)

    qw = set(re.findall(r'[\w\u0590-\u05FF]{3,}', q))
    sources = []
    for i, c in enumerate(cands[:top_k]):
        t, m = c["text"], c["metadata"]
        sents = [s.strip() for s in re.split(r'\n', t) if len(s.strip()) >= 10]
        scored = sorted([(len(qw & set(re.findall(r'[\w\u0590-\u05FF]+', s))), s) for s in sents], reverse=True)
        sources.append({
            "rank": i+1, "ref": m.get("ref", c["id"]), "tractate": m.get("tractate", "Berakhot"),
            "seder": "Zeraim", "url": m.get("url", ""), "rerank_score": c.get("rerank_score", 0),
            "bi_score": c["bi_score"], "has_gemara": "[GEMARA" in t,
            "has_rashi": "[RASHI" in t, "has_tosafot": "[TOSAFOT" in t,
            "full_text": t, "key_passages": [s for _, s in scored[:3]],
        })
    s0 = sources[0] if sources else {}
    return {
        "query": q, "expanded": q, "hyde_used": False,
        "answer": {"summary": f"**{s0.get('ref','')}** | Score: {s0.get('rerank_score',0):.2f}" if sources else "",
                   "llm_answer": None, "sources": sources, "total": len(sources)},
        "time_ms": round((time.time()-t0)*1000), "candidates_evaluated": len(cands),
    }

@app.get("/api/tractates")
async def tractates(): return {"tractates": ["Berakhot"], "sedarim": ["Zeraim"]}

ST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=ST), name="static")
@app.get("/")
async def idx(): return FileResponse(os.path.join(ST, "index.html"))

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
