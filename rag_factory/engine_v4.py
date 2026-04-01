"""
RAG Engine v4 — Professional-grade Talmud search engine.

Based on industry audit (April 2026):

P0 (Critical fixes):
  1. BGE-M3 embeddings (1024d, 8192 tokens, Hebrew-stable)
  2. Multilingual reranker (mmarco-mMiniLMv2-L12)
  3. Reciprocal Rank Fusion (replaces arbitrary weights)

P1 (Major improvements):
  4. Contextual Retrieval — Claude enriches chunks at indexing time
  5. Multi-HyDE — 3 hypothetical docs averaged

P2 (Quality improvements):
  6. Separate Gemara/Rashi/Tosafot chunks with back-references
  7. Structured Rav response via Claude

Claude API optional: graceful fallback without it.
"""

import os
import re
import time
import json
import numpy as np
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


# ============================================================
# DEFAULTS — upgraded models
# ============================================================
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"                                  # 1024d, 8192 tokens, multilingual
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual reranker
RRF_K = 60  # Reciprocal Rank Fusion constant


# ============================================================
# TALMUDIC KNOWLEDGE BASE
# ============================================================
TALMUD_DICTIONARY = {
    "שמע": ["קריאת שמע", "שמע בערבין", "שמע בשחרית", "בשכבך ובקומך"],
    "תפילה": ["שמונה עשרה", "עמידה", "תפילת שחרית", "מנחה", "ערבית"],
    "גאולה": ["גאולה לתפילה", "סמיכת גאולה", "גאל ישראל"],
    "שבת": ["שבתא", "יום השבת", "מלאכות שבת", "ל״ט מלאכות"],
    "מלאכה": ["מלאכות", "אבות מלאכות", "תולדות"],
    "ברכה": ["ברכות", "ברכת המזון", "ברכת הנהנין"],
    "טומאה": ["טמא", "טהרה", "טבילה", "מקוה", "נדה"],
    "כהן": ["כהנים", "כהן גדול", "תרומה", "בית המקדש"],
    "קרבן": ["קרבנות", "עולה", "חטאת", "אשם", "שלמים", "זבח"],
    "גט": ["גיטין", "גירושין", "כריתות"],
    "קידושין": ["אירוסין", "נישואין", "כתובה"],
    "נזק": ["נזיקין", "שור", "בור", "אש", "מבעה"],
    "סנהדרין": ["בית דין", "דיינים", "דיני נפשות"],
    "מחלוקת": ["פלוגתא", "dispute", "machloket"],
    "הלכה": ["פסק", "halacha", "ruling"],
    "דאורייתא": ["דרבנן", "מדאורייתא", "מדרבנן"],
    "shema": ["קריאת שמע", "שמע", "krias shema"],
    "prayer": ["תפילה", "tefillah", "amidah"],
    "shabbat": ["שבת", "sabbath", "shabbos"],
    "sacrifice": ["קרבן", "korban"],
    "purity": ["טהרה", "tahara", "טומאה"],
    "prière": ["תפילה", "tefillah"],
    "bénédiction": ["ברכה", "bracha"],
    "mariage": ["קידושין", "kiddushin"],
}

SEDER_MAP = {
    "Zeraim": ["Berakhot"],
    "Moed": ["Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah", "Yoma", "Sukkah",
              "Beitzah", "Taanit", "Megillah", "Moed Katan", "Chagigah"],
    "Nashim": ["Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin"],
    "Nezikin": ["Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot",
                "Shevuot", "Avodah Zarah", "Horayot"],
    "Kodashim": ["Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin",
                 "Temurah", "Keritot", "Meilah", "Tamid"],
    "Tahorot": ["Niddah"],
}

MASECHET_TO_SEDER = {}
for seder, masechtot in SEDER_MAP.items():
    for m in masechtot:
        MASECHET_TO_SEDER[m] = seder


# ============================================================
# 1. QUERY EXPANSION
# ============================================================
def expand_query(query: str) -> str:
    expanded = [query]
    ql = query.lower()
    query_words = set(re.findall(r'[\w\u0590-\u05FF]+', query))

    for key, syns in TALMUD_DICTIONARY.items():
        if key in query or key in ql:
            expanded.extend(syns[:3])

    for word in query_words:
        for key, syns in TALMUD_DICTIONARY.items():
            if len(word) >= 3 and word in key:
                expanded.extend(syns[:2])
                break

    seen = set()
    return " ".join(x for x in expanded if x not in seen and not seen.add(x))


# ============================================================
# 2. MULTI-HyDE (3 hypothetical docs averaged)
# ============================================================
HYDE_SYSTEM = """\
You are a Talmudic scholar. Given a question, write a short hypothetical \
Talmud passage (Hebrew/Aramaic) that would contain the answer. Include \
Gemara-style language. Keep it under 150 words. Write ONLY the passage."""

HYDE_TEMPLATES = [
    "גמרא: תנו רבנן {q} אמר רבי יוחנן {q} ורש\"י פירש {q}",
    "מתני׳ {q} גמ׳ מנא הני מילי אמר רב {q} תוספות {q}",
    "איתמר {q} רב אמר {q} ושמואל אמר {q} והלכתא {q}",
]


def generate_hyde_documents(query: str, claude_client=None, n: int = 3) -> list[str]:
    """Generate multiple hypothetical documents for averaged embedding."""
    docs = []

    if claude_client:
        try:
            for i in range(n):
                response = claude_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=300,
                    system=HYDE_SYSTEM,
                    messages=[{"role": "user", "content": f"[Version {i+1}] {query}"}],
                )
                docs.append(response.content[0].text)
            return docs
        except Exception:
            pass

    # Template fallback
    for tmpl in HYDE_TEMPLATES[:n]:
        docs.append(tmpl.format(q=query))
    return docs


# ============================================================
# 3. RECIPROCAL RANK FUSION
# ============================================================
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Merge multiple ranked lists using RRF.
    Each list is a list of IDs ordered by relevance.
    Returns {id: rrf_score}.
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (rank + k)
    return dict(scores)


# ============================================================
# 4. BM25 INDEX
# ============================================================
class BM25Index:
    def __init__(self):
        self.corpus = []
        self.ids = []
        self.metadatas = []
        self.bm25 = None

    def build(self, ids, documents, metadatas):
        self.ids = ids
        self.metadatas = metadatas
        self.corpus = [re.findall(r'[\w\u0590-\u05FF]+', doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 20) -> list[str]:
        """Return ranked list of IDs (for RRF)."""
        if not self.bm25:
            return []
        tokens = re.findall(r'[\w\u0590-\u05FF]+', query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.ids[i] for i in top_idx if scores[i] > 0]

    def get_score(self, query: str, doc_id: str) -> float:
        if doc_id not in self.ids:
            return 0.0
        idx = self.ids.index(doc_id)
        tokens = re.findall(r'[\w\u0590-\u05FF]+', query.lower())
        scores = self.bm25.get_scores(tokens)
        return float(scores[idx])


# ============================================================
# 5. SMALL-TO-BIG RETRIEVAL
# ============================================================
def get_parent_context(collection, chunk_id: str, window: int = 1) -> str:
    parts = chunk_id.rsplit("_chunk", 1)
    if len(parts) != 2:
        return ""
    ref = parts[0]
    chunk_idx = int(parts[1])
    neighbor_ids = [f"{ref}_chunk{chunk_idx + offset}" for offset in range(-window, window + 1) if offset != 0]
    try:
        result = collection.get(ids=neighbor_ids, include=["documents"])
        return "\n\n---\n\n".join(doc for doc in result["documents"] if doc)
    except Exception:
        return ""


# ============================================================
# 6. CONTEXTUAL RETRIEVAL — Enrich chunks at indexing time
# ============================================================
CONTEXT_PROMPT = """\
You are indexing Talmud Bavli for a search engine. Given a chunk of Talmudic text \
and its reference, write a concise 1-2 sentence context prefix in Hebrew that describes:
- Which masechet and daf this is from
- The main topic/sugya being discussed
- Key halakhic or aggadic concepts mentioned

Write ONLY the prefix, no explanation. Example:
"מסכת ברכות דף ב עמוד א. סוגיית זמן קריאת שמע של ערבית, מחלוקת תנא קמא ורבן גמליאל."
"""


def generate_chunk_context(ref: str, text: str, claude_client=None) -> str:
    """Generate a context prefix for a chunk (Anthropic's Contextual Retrieval)."""
    if not claude_client:
        # Fallback: simple metadata prefix
        parts = ref.replace("_", " ").split(".")
        tractate = parts[0] if parts else ref
        daf = parts[1] if len(parts) > 1 else ""
        return f"מסכת {tractate} דף {daf}."

    try:
        response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system=CONTEXT_PROMPT,
            messages=[{"role": "user", "content": f"Reference: {ref}\n\nText:\n{text[:1500]}"}],
        )
        return response.content[0].text.strip()
    except Exception:
        parts = ref.replace("_", " ").split(".")
        tractate = parts[0] if parts else ref
        daf = parts[1] if len(parts) > 1 else ""
        return f"מסכת {tractate} דף {daf}."


# ============================================================
# 7. ANSWER SYNTHESIS (Claude Rav)
# ============================================================
RAV_SYSTEM = """\
You are a Talmudic scholar (Rav) who answers questions based ONLY on the \
provided sources from the Talmud Bavli. Style: clear Yeshiva teacher.

Structure:
1. **Pshat (פשט)** — Straightforward answer, 2-3 sentences.
2. **Iyun (עיון)** — Deeper analysis, machloket if present.
3. **Mekorot (מקורות)** — Exact references [Masechet Daf].

Rules:
- ONLY use provided sources. Say so if not found.
- Include Rashi/Tosafot commentary when in sources.
- Respond in the SAME LANGUAGE as the question.
- Under 300 words."""


def synthesize_answer(question: str, sources: list[dict], claude_client=None) -> dict:
    if not sources:
        return {"summary": "No relevant sources found.", "llm_answer": None, "sources": [], "total": 0}

    question_words = set(re.findall(r'[\w\u0590-\u05FF]{3,}', question))
    formatted = []

    for i, src in enumerate(sources):
        text = src.get("text", "")
        meta = src.get("metadata", {})

        sentences = re.split(r'[.!?]\s|[.!?]$|\n', text)
        scored = [(len(question_words & set(re.findall(r'[\w\u0590-\u05FF]+', s))), s.strip())
                  for s in sentences if len(s.strip()) >= 10]
        scored.sort(reverse=True)

        formatted.append({
            "rank": i + 1,
            "ref": meta.get("ref", src.get("id", "")),
            "tractate": meta.get("tractate", ""),
            "seder": MASECHET_TO_SEDER.get(meta.get("tractate", ""), ""),
            "url": meta.get("url", ""),
            "rerank_score": src.get("rerank_score", 0),
            "bi_score": src.get("bi_score", 0),
            "bm25_score": src.get("bm25_score", 0),
            "rrf_score": src.get("rrf_score", 0),
            "has_gemara": "[GEMARA" in text,
            "has_rashi": "[RASHI" in text,
            "has_tosafot": "[TOSAFOT" in text,
            "full_text": text,
            "key_passages": [s for _, s in scored[:3]],
            "parent_context": src.get("parent_context", ""),
        })

    # Claude LLM synthesis
    llm_answer = None
    if claude_client:
        try:
            ctx = "\n\n---\n\n".join(
                f"[{s['ref']} — {s['tractate']}]\n{s['full_text'][:2000]}" for s in formatted[:5]
            )
            resp = claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                system=RAV_SYSTEM,
                messages=[{"role": "user", "content": f"SOURCES:\n{ctx}\n\nQUESTION: {question}"}],
            )
            llm_answer = resp.content[0].text
        except Exception as e:
            print(f"Claude synthesis failed: {e}")

    top = formatted[0]
    summary = f"Top: **{top['ref']}** ({top['tractate']}, {top['seder']}) | Score: {top['rerank_score']:.2f}"
    if top["key_passages"]:
        summary += f" | {top['key_passages'][0][:150]}..."

    return {"summary": summary, "llm_answer": llm_answer, "sources": formatted, "total": len(formatted)}


# ============================================================
# 8. FULL RAG ENGINE v4
# ============================================================
class TalmudRAGEngine:
    def __init__(
        self,
        persist_dir: str = "./rag_data_shas",
        collection_name: str = "shas_complete",
        embed_model: str = DEFAULT_EMBED_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        anthropic_api_key: str | None = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.embedder = None
        self.reranker = None
        self.collection = None
        self.bm25_index = None

        self.claude_client = None
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key and CLAUDE_AVAILABLE:
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            print("Claude API enabled (Haiku for HyDE + Contextual Retrieval + Rav synthesis)")

    def load(self):
        print(f"Loading embedding model: {self.embed_model}...")
        self.embedder = SentenceTransformer(self.embed_model)
        print(f"  Dimensions: {self.embedder.get_sentence_embedding_dimension()}")

        print(f"Loading reranker: {self.reranker_model}...")
        self.reranker = CrossEncoder(self.reranker_model)

        print("Loading ChromaDB...")
        client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = client.get_collection(self.collection_name)
        print(f"  Collection: {self.collection.count()} chunks")

        print("Building BM25 index...")
        self._build_bm25()
        print("Engine v4 ready!")
        return self

    def _build_bm25(self):
        self.bm25_index = BM25Index()
        all_ids, all_docs, all_metas = [], [], []
        offset = 0
        while True:
            batch = self.collection.get(limit=5000, offset=offset, include=["documents", "metadatas"])
            if not batch["ids"]:
                break
            all_ids.extend(batch["ids"])
            all_docs.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += 5000
        self.bm25_index.build(all_ids, all_docs, all_metas)
        print(f"  BM25: {len(all_ids)} docs indexed")

    def search(
        self,
        question: str,
        top_k: int = 5,
        tractate: str = None,
        seder: str = None,
        use_hyde: bool = True,
        use_bm25: bool = True,
        include_context: bool = True,
    ) -> dict:
        start = time.time()

        # 1. Query expansion
        expanded = expand_query(question)

        # 2. Multi-HyDE embedding
        if use_hyde:
            hyde_docs = generate_hyde_documents(question, self.claude_client, n=3)
            embed_texts = [expanded] + hyde_docs
            embeddings = self.embedder.encode(embed_texts)
            # Average all embeddings (original + 3 hypothetical)
            q_embedding = np.mean(embeddings, axis=0).tolist()
        else:
            q_embedding = self.embedder.encode(expanded).tolist()

        # 3. Vector search
        vector_params = {
            "query_embeddings": [q_embedding],
            "n_results": 20,
            "include": ["documents", "metadatas", "distances"],
        }
        where = {}
        if tractate:
            where["tractate"] = tractate
        elif seder:
            masech = SEDER_MAP.get(seder, [])
            if masech:
                where["tractate"] = {"$in": masech}
        if where:
            vector_params["where"] = where

        vr = self.collection.query(**vector_params)

        # Build doc cache
        doc_cache = {}
        vector_ranked = []
        for i in range(len(vr["ids"][0])):
            cid = vr["ids"][0][i]
            doc_cache[cid] = {
                "id": cid,
                "text": vr["documents"][0][i],
                "metadata": vr["metadatas"][0][i],
                "bi_score": round(1 - vr["distances"][0][i], 4),
            }
            vector_ranked.append(cid)

        # 4. BM25 search
        bm25_ranked = []
        if use_bm25 and self.bm25_index:
            bm25_ranked = self.bm25_index.search(expanded, top_k=20)
            # Fetch docs not in cache
            for cid in bm25_ranked:
                if cid not in doc_cache:
                    try:
                        doc = self.collection.get(ids=[cid], include=["documents", "metadatas"])
                        if doc["documents"]:
                            meta = doc["metadatas"][0] if doc["metadatas"] else {}
                            if tractate and meta.get("tractate") != tractate:
                                continue
                            doc_cache[cid] = {
                                "id": cid,
                                "text": doc["documents"][0],
                                "metadata": meta,
                                "bi_score": 0.0,
                            }
                    except Exception:
                        pass

        # 5. RECIPROCAL RANK FUSION
        rrf_scores = reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=RRF_K)

        for cid, score in rrf_scores.items():
            if cid in doc_cache:
                doc_cache[cid]["rrf_score"] = round(score, 6)
                doc_cache[cid]["bm25_score"] = round(
                    self.bm25_index.get_score(expanded, cid) if self.bm25_index else 0, 2
                )

        # Sort by RRF, take top for reranking
        candidates = sorted(
            [c for c in doc_cache.values() if "rrf_score" in c],
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:15]

        # 6. Cross-encoder reranking (multilingual)
        if candidates:
            pairs = [(question, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = round(float(s), 4)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        top = candidates[:top_k]

        # 7. Small-to-big context
        if include_context:
            for r in top:
                r["parent_context"] = get_parent_context(self.collection, r["id"])

        # 8. Synthesis
        answer = synthesize_answer(question, top, self.claude_client)

        return {
            "query": question,
            "expanded": expanded,
            "hyde_used": use_hyde,
            "bm25_used": use_bm25,
            "rrf_k": RRF_K,
            "answer": answer,
            "time_ms": round((time.time() - start) * 1000),
            "candidates_evaluated": len(doc_cache),
        }

    def get_stats(self) -> dict:
        count = self.collection.count() if self.collection else 0
        bm25_size = len(self.bm25_index.ids) if self.bm25_index else 0
        return {
            "collection": self.collection_name,
            "chunks": count,
            "bm25_index_size": bm25_size,
            "persist_dir": self.persist_dir,
            "claude_enabled": self.claude_client is not None,
            "embed_model": self.embed_model,
            "reranker_model": self.reranker_model,
        }
