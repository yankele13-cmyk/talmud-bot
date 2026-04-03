"""
Lightweight server — runs with ChromaDB data available in this environment.
No Qdrant needed. Uses engine_v3 with the shas_complete collection.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict
import numpy as np

app = FastAPI(title="Talmud RAG", version="live")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Globals
embedder = None
reranker = None
collection = None
bm25_ids = []
bm25_index = None
db_info = {"status": "loading"}

TALMUD_DICT = {
    "שמע": ["קריאת שמע", "שמע בערבין", "בשכבך ובקומך"],
    "תפילה": ["שמונה עשרה", "עמידה", "תפילת שחרית", "מנחה", "ערבית"],
    "גאולה": ["גאולה לתפילה", "סמיכת גאולה"],
    "שבת": ["שבתא", "מלאכות שבת", "ל״ט מלאכות"],
    "ברכה": ["ברכות", "ברכת המזון"],
    "shema": ["קריאת שמע", "שמע"],
    "prayer": ["תפילה", "tefillah"],
    "shabbat": ["שבת", "sabbath"],
    "prière": ["תפילה"],
}

SEDER_MAP = {
    "Zeraim": ["Berakhot"],
    "Moed": ["Shabbat", "Eruvin", "Pesachim", "Yoma", "Sukkah", "Beitzah", "Taanit", "Megillah", "Moed Katan", "Chagigah"],
    "Nashim": ["Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin"],
    "Nezikin": ["Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot", "Avodah Zarah"],
    "Kodashim": ["Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah", "Tamid"],
    "Tahorot": ["Niddah"],
}
MASECHET_TO_SEDER = {}
for s, ms in SEDER_MAP.items():
    for m in ms: MASECHET_TO_SEDER[m] = s


def expand_query(q):
    parts = [q]
    ql = q.lower()
    for k, syns in TALMUD_DICT.items():
        if k in q or k in ql: parts.extend(syns[:3])
    seen = set()
    return " ".join(x for x in parts if x not in seen and not seen.add(x))


def init():
    global embedder, reranker, collection, bm25_ids, bm25_index, db_info

    print("Loading embedder...")
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Loading reranker...")
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    print("Loading ChromaDB...")
    persist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_data_shas")
    client = chromadb.PersistentClient(path=persist)
    collection = client.get_collection("shas_complete")
    count = collection.count()
    print(f"  {count} chunks loaded")

    # Build BM25
    print("Building BM25...")
    all_data = {"ids": [], "docs": [], "metas": []}
    offset = 0
    while True:
        batch = collection.get(limit=5000, offset=offset, include=["documents", "metadatas"])
        if not batch["ids"]: break
        all_data["ids"].extend(batch["ids"])
        all_data["docs"].extend(batch["documents"])
        all_data["metas"].extend(batch["metadatas"])
        offset += 5000

    bm25_ids = all_data["ids"]
    corpus = [re.findall(r'[\w\u0590-\u05FF]+', d.lower()) for d in all_data["docs"]]
    bm25_index = BM25Okapi(corpus)
    print(f"  BM25: {len(bm25_ids)} docs")

    db_info = {
        "status": "ready",
        "version": "live",
        "chunks": count,
        "features": ["Multilingual embeddings", "Cross-encoder reranking", "Hybrid search (Vector+BM25)", "Query expansion"],
        "massechtot": "Berakhot, Shabbat, Eruvin",
    }
    print("Server ready!")


@app.on_event("startup")
async def startup():
    init()


@app.get("/api/status")
async def status():
    return db_info


@app.get("/api/search")
async def search(
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=20),
    tractate: str = Query(None),
    seder: str = Query(None),
):
    start = time.time()
    expanded = expand_query(q)
    q_emb = embedder.encode(expanded).tolist()

    # Vector search
    vparams = {"query_embeddings": [q_emb], "n_results": 20, "include": ["documents", "metadatas", "distances"]}
    if tractate: vparams["where"] = {"tractate": tractate}
    vr = collection.query(**vparams)

    candidates = {}
    vec_ranked = []
    for i in range(len(vr["ids"][0])):
        cid = vr["ids"][0][i]
        candidates[cid] = {
            "id": cid, "text": vr["documents"][0][i],
            "metadata": vr["metadatas"][0][i],
            "bi_score": round(1 - vr["distances"][0][i], 4),
        }
        vec_ranked.append(cid)

    # BM25
    tokens = re.findall(r'[\w\u0590-\u05FF]+', expanded.lower())
    bm25_scores = bm25_index.get_scores(tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_ranked = [bm25_ids[i] for i in bm25_top if bm25_scores[i] > 0]

    # RRF
    rrf = defaultdict(float)
    for ranked in [vec_ranked, bm25_ranked]:
        for rank, pid in enumerate(ranked):
            rrf[pid] += 1.0 / (rank + 60)

    for cid in bm25_ranked:
        if cid not in candidates:
            idx = bm25_ids.index(cid)
            batch = collection.get(ids=[cid], include=["documents", "metadatas"])
            if batch["documents"]:
                meta = batch["metadatas"][0] if batch["metadatas"] else {}
                if tractate and meta.get("tractate") != tractate: continue
                candidates[cid] = {"id": cid, "text": batch["documents"][0], "metadata": meta, "bi_score": 0}

    for cid in candidates:
        candidates[cid]["rrf_score"] = round(rrf.get(cid, 0), 6)

    sorted_c = sorted(candidates.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)[:15]

    # Rerank
    if sorted_c:
        pairs = [(q, c["text"]) for c in sorted_c]
        scores = reranker.predict(pairs)
        for c, s in zip(sorted_c, scores):
            c["rerank_score"] = round(float(s), 4)
        sorted_c.sort(key=lambda x: x["rerank_score"], reverse=True)

    top = sorted_c[:top_k]

    # Format sources
    question_words = set(re.findall(r'[\w\u0590-\u05FF]{3,}', q))
    sources = []
    for i, src in enumerate(top):
        text = src.get("text", "")
        meta = src.get("metadata", {})
        sentences = re.split(r'[.!?]\s|[.!?]$|\n', text)
        scored = [(len(question_words & set(re.findall(r'[\w\u0590-\u05FF]+', s))), s.strip())
                  for s in sentences if len(s.strip()) >= 10]
        scored.sort(reverse=True)

        sources.append({
            "rank": i + 1,
            "ref": meta.get("ref", src["id"]),
            "tractate": meta.get("tractate", ""),
            "seder": MASECHET_TO_SEDER.get(meta.get("tractate", ""), ""),
            "url": meta.get("url", ""),
            "rerank_score": src.get("rerank_score", 0),
            "bi_score": src.get("bi_score", 0),
            "has_gemara": "[GEMARA" in text,
            "has_rashi": "[RASHI" in text or meta.get("has_rashi", False),
            "has_tosafot": "[TOSAFOT" in text or meta.get("has_tosafot", False),
            "full_text": text,
            "key_passages": [s for _, s in scored[:3]],
        })

    elapsed = time.time() - start
    top_s = sources[0] if sources else {}
    summary = f"**{top_s.get('ref', '')}** ({top_s.get('tractate', '')}, {top_s.get('seder', '')}) | Score: {top_s.get('rerank_score', 0):.2f}" if sources else ""

    return {
        "query": q, "expanded": expanded,
        "answer": {"summary": summary, "llm_answer": None, "sources": sources, "total": len(sources)},
        "time_ms": round(elapsed * 1000),
        "candidates_evaluated": len(candidates),
    }


@app.get("/api/tractates")
async def tractates():
    if not collection: return {"tractates": [], "sedarim": list(SEDER_MAP.keys())}
    try:
        sample = collection.get(limit=10000, include=["metadatas"])
        t = set()
        for m in sample["metadatas"]:
            if m and "tractate" in m: t.add(m["tractate"])
        return {"tractates": sorted(t), "sedarim": list(SEDER_MAP.keys())}
    except: return {"tractates": [], "sedarim": list(SEDER_MAP.keys())}


STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
