"""
RAG Engine v3 — Production-grade Talmud search engine.
All optimizations implemented:

1. HyDE (Hypothetical Document Embedding) — with Claude API
2. Hybrid Search (Vector + BM25)
3. Small-to-Big Retrieval (search chunks, return parent context)
4. Cross-encoder Reranking
5. Advanced Query Expansion (Talmudic dictionary + fuzzy)
6. Metadata Filtering (masechet, seder, text type)
7. LLM Answer Synthesis — Claude generates structured Talmudic responses

Claude API optional: works without it (falls back to extractive mode).
"""

import os
import re
import time
import json
import math
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Claude API (optional — graceful fallback if not available)
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


# ============================================================
# TALMUDIC KNOWLEDGE BASE — for query expansion & HyDE
# ============================================================

# Comprehensive bilingual/trilingual synonym dictionary
TALMUD_DICTIONARY = {
    # Kriat Shema
    "שמע": ["קריאת שמע", "שמע בערבין", "שמע בשחרית", "בשכבך ובקומך", "פרשת שמע", "קרית שמע"],
    "shema": ["קריאת שמע", "שמע", "krias shema", "recitation of shema"],
    # Tefila
    "תפילה": ["שמונה עשרה", "עמידה", "תפילת שחרית", "מנחה", "ערבית", "מוסף", "נעילה"],
    "prayer": ["תפילה", "tefillah", "amidah", "shemoneh esrei"],
    "גאולה": ["גאולה לתפילה", "סמיכת גאולה", "גאל ישראל", "redemption"],
    # Shabbat
    "שבת": ["שבתא", "יום השבת", "מלאכות שבת", "שמירת שבת", "הלכות שבת"],
    "shabbat": ["שבת", "sabbath", "shabbos", "hilchot shabbat"],
    "מלאכה": ["מלאכות", "ל״ט מלאכות", "אבות מלאכות", "תולדות"],
    # Berakhot
    "ברכה": ["ברכות", "ברכת המזון", "ברכת הנהנין", "ברכה ראשונה", "ברכה אחרונה"],
    "blessing": ["ברכה", "bracha", "beracha"],
    # Tumah/Tahara
    "טומאה": ["טמא", "טהרה", "טבילה", "מקוה", "נדה"],
    "purity": ["טהרה", "tahara", "tumah", "mikveh"],
    # Kodashim
    "קרבן": ["קרבנות", "עולה", "חטאת", "אשם", "שלמים", "זבח", "מנחה"],
    "sacrifice": ["קרבן", "korban", "offering"],
    # Nezikin
    "נזק": ["נזיקין", "בבא קמא", "שור", "בור", "אש", "מבעה"],
    "damage": ["נזק", "nezek", "nezikin"],
    # Nashim
    "גט": ["גיטין", "גירושין", "כריתות", "ספר כריתות"],
    "divorce": ["גט", "get", "gittin"],
    "קידושין": ["קידושין", "אירוסין", "נישואין", "כתובה"],
    "marriage": ["קידושין", "kiddushin", "ketubah"],
    # Sanhedrin
    "סנהדרין": ["בית דין", "דיינים", "דיני נפשות", "דיני ממונות"],
    "court": ["סנהדרין", "beit din", "sanhedrin"],
    # Key Rabbis — maps names to alternative spellings
    "רבי אליעזר": ["ר' אליעזר", "ר״א", "rabbi eliezer"],
    "רבי יהושע": ["ר' יהושע", "ר״י", "rabbi yehoshua"],
    "רבי עקיבא": ["ר' עקיבא", "ר״ע", "rabbi akiva"],
    "רבן גמליאל": ["ר״ג", "rabban gamliel"],
    "רבי יוחנן": ["ר' יוחנן", "rabbi yochanan"],
    "ריש לקיש": ["ר״ל", "שמעון בן לקיש", "reish lakish"],
    "אביי": ["אביי ורבא", "abaye"],
    "רבא": ["אביי ורבא", "rava"],
    "רב": ["רב ושמואל", "rav"],
    "שמואל": ["רב ושמואל", "shmuel"],
    # Halachic concepts
    "לכתחילה": ["בדיעבד", "lechatchila", "bedieved"],
    "דאורייתא": ["דרבנן", "מדאורייתא", "מדרבנן"],
    "מחלוקת": ["פלוגתא", "dispute", "machloket"],
    "הלכה": ["פסק", "halacha", "ruling"],
    # French (for the user)
    "prière": ["תפילה", "tefillah", "prayer"],
    "bénédiction": ["ברכה", "blessing", "bracha"],
    "pureté": ["טהרה", "tahara", "purity"],
    "tribunal": ["בית דין", "sanhedrin", "court"],
    "mariage": ["קידושין", "kiddushin", "marriage"],
}

# Seder → Massechtot mapping for metadata enrichment
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

# Reverse: masechet → seder
MASECHET_TO_SEDER = {}
for seder, masechtot in SEDER_MAP.items():
    for m in masechtot:
        MASECHET_TO_SEDER[m] = seder


# ============================================================
# 1. ADVANCED QUERY EXPANSION
# ============================================================
def expand_query(query: str) -> str:
    """
    Multi-level query expansion:
    1. Exact dictionary match
    2. Substring/fuzzy match
    3. Add related Aramaic/Hebrew terms
    """
    expanded_parts = [query]
    query_lower = query.lower()
    query_words = set(re.findall(r'[\w\u0590-\u05FF]+', query))

    # Level 1: Exact matches
    for key, synonyms in TALMUD_DICTIONARY.items():
        if key in query or key in query_lower:
            expanded_parts.extend(synonyms[:3])

    # Level 2: Word-level fuzzy match
    for word in query_words:
        for key, synonyms in TALMUD_DICTIONARY.items():
            # Check if word appears in any synonym
            if len(word) >= 3 and word in key:
                expanded_parts.extend(synonyms[:2])
                break

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in expanded_parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return " ".join(unique)


# ============================================================
# 2. HyDE (Hypothetical Document Embedding)
# ============================================================
HYDE_SYSTEM_PROMPT = """\
You are a Talmudic scholar. Given a question, write a short hypothetical \
Talmud passage (in Hebrew/Aramaic) that would contain the answer. \
Include Gemara-style language (תנו רבנן, אמר רבי, תא שמע, etc.) and \
if relevant, add a brief Rashi-style comment. Keep it under 200 words. \
Write ONLY the passage, no explanation."""


def generate_hyde_document(query: str, claude_client=None) -> str:
    """
    Generate a hypothetical Talmudic passage that would answer the question.
    Uses Claude API if available, falls back to template-based generation.
    """
    # Try Claude API first
    if claude_client:
        try:
            response = claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=HYDE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            return response.content[0].text
        except Exception:
            pass  # Fall back to templates

    # Template fallback
    has_hebrew = bool(re.search(r'[\u0590-\u05FF]', query))

    if has_hebrew:
        hyde = (
            f"גמרא: {query} "
            f"תנו רבנן: {query} "
            f"אמר רבי יוחנן: {query} "
            f"רש\"י: {query} "
            f"תוספות: {query}"
        )
    else:
        hyde = (
            f"The Gemara discusses: {query}. "
            f"Rabbi Yochanan says regarding {query}. "
            f"Rashi explains: {query}. "
            f"The Tosafot comment on this: {query}. "
            f"גמרא: תנו רבנן אמר רבי"
        )

    return hyde


# ============================================================
# 3. BM25 INDEX for hybrid search
# ============================================================
class BM25Index:
    """Keyword-based search using BM25 — complements vector search."""

    def __init__(self):
        self.corpus = []
        self.ids = []
        self.metadatas = []
        self.bm25 = None

    def build(self, ids: list, documents: list, metadatas: list):
        """Build BM25 index from documents."""
        self.ids = ids
        self.metadatas = metadatas
        # Tokenize: split on whitespace and punctuation, keep Hebrew
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'[\w\u0590-\u05FF]+', text.lower())

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        if not self.bm25:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": self.ids[idx],
                    "score": float(scores[idx]),
                    "metadata": self.metadatas[idx],
                })

        return results


# ============================================================
# 4. SMALL-TO-BIG RETRIEVAL
# ============================================================
def get_parent_context(collection, chunk_id: str, window: int = 1) -> str:
    """
    Given a chunk, retrieve its neighboring chunks to provide broader context.
    chunk_id format: "Berakhot.2a_chunk3"
    """
    parts = chunk_id.rsplit("_chunk", 1)
    if len(parts) != 2:
        return ""

    ref = parts[0]
    chunk_idx = int(parts[1])

    # Fetch neighboring chunks
    neighbor_ids = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        neighbor_ids.append(f"{ref}_chunk{chunk_idx + offset}")

    if not neighbor_ids:
        return ""

    try:
        result = collection.get(ids=neighbor_ids, include=["documents"])
        texts = [doc for doc in result["documents"] if doc]
        return "\n\n---\n\n".join(texts)
    except Exception:
        return ""


# ============================================================
# 5. ANSWER SYNTHESIS (Claude LLM + extractive fallback)
# ============================================================

RAV_SYSTEM_PROMPT = """\
You are a Talmudic scholar (Rav) who answers questions based ONLY on the \
provided sources from the Talmud Bavli. Your style is that of a Yeshiva \
teacher — clear, precise, and pedagogical.

RULES:
- Answer ONLY based on the sources provided. If the sources don't contain \
  the answer, say so clearly.
- Structure your answer as:
  1. **Pshat (פשט)** — The straightforward answer in 2-3 sentences.
  2. **Iyun (עיון)** — Deeper analysis if the sources allow it. \
     Mention any machloket (dispute) between Tannaim/Amoraim.
  3. **Mekorot (מקורות)** — Cite the exact references [Masechet Daf].
- Use Hebrew/Aramaic terms naturally but explain them.
- If Rashi or Tosafot are in the sources, incorporate their commentary.
- Keep it concise (under 300 words).
- Respond in the SAME LANGUAGE as the question (Hebrew, English, or French)."""


def synthesize_answer(question: str, sources: list[dict], claude_client=None) -> dict:
    """
    Build a structured answer. Uses Claude for LLM synthesis if available,
    otherwise falls back to extractive synthesis.
    """
    if not sources:
        return {"summary": "No relevant sources found.", "sources": [], "llm_answer": None}

    # Extract key terms for highlighting
    question_words = set(re.findall(r'[\w\u0590-\u05FF]{3,}', question))

    formatted_sources = []
    for i, src in enumerate(sources):
        text = src.get("text", "")
        meta = src.get("metadata", {})

        # Find most relevant sentences
        sentences = re.split(r'[.!?]\s|[.!?]$|\n', text)
        scored_sentences = []
        for sent in sentences:
            if len(sent.strip()) < 10:
                continue
            sent_words = set(re.findall(r'[\w\u0590-\u05FF]+', sent))
            overlap = len(question_words & sent_words)
            scored_sentences.append((overlap, sent.strip()))

        scored_sentences.sort(reverse=True)
        best_sentences = [s for _, s in scored_sentences[:3]]

        has_gemara = "[GEMARA" in text
        has_rashi = "[RASHI" in text
        has_tosafot = "[TOSAFOT" in text

        formatted_sources.append({
            "rank": i + 1,
            "ref": meta.get("ref", src.get("id", "")),
            "tractate": meta.get("tractate", ""),
            "seder": MASECHET_TO_SEDER.get(meta.get("tractate", ""), ""),
            "url": meta.get("url", f"https://www.sefaria.org/{meta.get('ref', '')}"),
            "rerank_score": src.get("rerank_score", 0),
            "bi_score": src.get("bi_score", 0),
            "bm25_score": src.get("bm25_score", 0),
            "has_gemara": has_gemara,
            "has_rashi": has_rashi,
            "has_tosafot": has_tosafot,
            "full_text": text,
            "key_passages": best_sentences,
            "parent_context": src.get("parent_context", ""),
        })

    # Try Claude LLM synthesis
    llm_answer = None
    if claude_client:
        llm_answer = _claude_synthesize(question, formatted_sources, claude_client)

    # Extractive summary fallback
    top = formatted_sources[0]
    summary_parts = [
        f"Top result: **{top['ref']}** ({top['tractate']}, Seder {top['seder']})",
        f"Score: {top['rerank_score']:.2f}",
    ]
    if top["key_passages"]:
        summary_parts.append(f"Key passage: {top['key_passages'][0][:200]}...")

    return {
        "summary": " | ".join(summary_parts),
        "llm_answer": llm_answer,
        "sources": formatted_sources,
        "total": len(formatted_sources),
    }


def _claude_synthesize(question: str, sources: list[dict], client) -> str | None:
    """Generate a Rav-style answer using Claude."""
    try:
        # Build context from sources
        context_parts = []
        for s in sources[:5]:
            text = s["full_text"][:2000]
            context_parts.append(f"[{s['ref']} — {s['tractate']}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=RAV_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"SOURCES:\n{context}\n\nQUESTION: {question}",
            }],
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude synthesis failed: {e}")
        return None


# ============================================================
# 6. FULL RAG ENGINE
# ============================================================
class TalmudRAGEngine:
    """
    Complete RAG engine with all optimizations.
    """

    def __init__(
        self,
        persist_dir: str = "./rag_data_shas",
        collection_name: str = "shas_complete",
        embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        anthropic_api_key: str | None = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = None
        self.reranker = None
        self.collection = None
        self.bm25_index = None
        self.embed_model = embed_model
        self.reranker_model = reranker_model

        # Claude API client (optional)
        self.claude_client = None
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key and CLAUDE_AVAILABLE:
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            print("Claude API enabled (Haiku for HyDE + synthesis)")

    def load(self):
        """Load models and indexes."""
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(self.embed_model)

        print("Loading reranker...")
        self.reranker = CrossEncoder(self.reranker_model)

        print("Loading ChromaDB collection...")
        client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = client.get_collection(self.collection_name)
        count = self.collection.count()
        print(f"Collection loaded: {count} chunks")

        # Build BM25 index
        print("Building BM25 index...")
        self._build_bm25()
        print("Engine ready!")

        return self

    def _build_bm25(self):
        """Load all documents and build BM25 index for keyword search."""
        self.bm25_index = BM25Index()

        # Load in batches
        all_ids = []
        all_docs = []
        all_metas = []
        batch_size = 5000
        offset = 0

        while True:
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if not batch["ids"]:
                break
            all_ids.extend(batch["ids"])
            all_docs.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += batch_size

        self.bm25_index.build(all_ids, all_docs, all_metas)
        print(f"BM25 index built: {len(all_ids)} documents")

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
        """
        Full search pipeline:
        1. Query expansion
        2. HyDE embedding
        3. Vector search (broad)
        4. BM25 keyword search
        5. Merge & deduplicate
        6. Cross-encoder reranking
        7. Small-to-big context expansion
        8. Answer synthesis
        """
        start_time = time.time()

        # Step 1: Query expansion
        expanded = expand_query(question)

        # Step 2: Embed (with optional HyDE — uses Claude if available)
        if use_hyde:
            hyde_doc = generate_hyde_document(question, claude_client=self.claude_client)
            # Combine original + expanded + hyde for richer embedding
            embed_text = f"{expanded} {hyde_doc}"
        else:
            embed_text = expanded

        q_embedding = self.embedder.encode(embed_text).tolist()

        # Step 3: Vector search
        vector_params = {
            "query_embeddings": [q_embedding],
            "n_results": 20,
            "include": ["documents", "metadatas", "distances"],
        }

        # Metadata filtering
        where_filters = {}
        if tractate:
            where_filters["tractate"] = tractate
        if seder:
            seder_masechtot = SEDER_MAP.get(seder, [])
            if seder_masechtot:
                where_filters["tractate"] = {"$in": seder_masechtot}

        if where_filters:
            vector_params["where"] = where_filters

        vector_results = self.collection.query(**vector_params)

        # Build candidate pool from vector search
        candidates = {}
        for i in range(len(vector_results["ids"][0])):
            cid = vector_results["ids"][0][i]
            candidates[cid] = {
                "id": cid,
                "text": vector_results["documents"][0][i],
                "metadata": vector_results["metadatas"][0][i],
                "bi_score": round(1 - vector_results["distances"][0][i], 4),
                "bm25_score": 0.0,
            }

        # Step 4: BM25 search (keyword-based)
        if use_bm25 and self.bm25_index:
            bm25_results = self.bm25_index.search(expanded, top_k=20)
            for r in bm25_results:
                cid = r["id"]
                if cid in candidates:
                    candidates[cid]["bm25_score"] = r["score"]
                else:
                    # Need to fetch the document text
                    try:
                        doc = self.collection.get(ids=[cid], include=["documents", "metadatas"])
                        if doc["documents"]:
                            # Apply metadata filter manually
                            meta = doc["metadatas"][0] if doc["metadatas"] else {}
                            if tractate and meta.get("tractate") != tractate:
                                continue
                            if seder and meta.get("tractate") not in SEDER_MAP.get(seder, []):
                                continue

                            candidates[cid] = {
                                "id": cid,
                                "text": doc["documents"][0],
                                "metadata": meta,
                                "bi_score": 0.0,
                                "bm25_score": r["score"],
                            }
                    except Exception:
                        pass

        # Step 5: Compute combined score
        if candidates:
            max_bi = max(c["bi_score"] for c in candidates.values()) or 1
            max_bm25 = max(c["bm25_score"] for c in candidates.values()) or 1

            for c in candidates.values():
                norm_bi = c["bi_score"] / max_bi
                norm_bm25 = c["bm25_score"] / max_bm25
                # Weighted combination: 60% vector + 40% BM25
                c["combined_score"] = 0.6 * norm_bi + 0.4 * norm_bm25

        # Sort by combined score, take top candidates for reranking
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )[:15]

        # Step 6: Cross-encoder reranking
        if sorted_candidates:
            pairs = [(question, c["text"]) for c in sorted_candidates]
            rerank_scores = self.reranker.predict(pairs)
            for c, score in zip(sorted_candidates, rerank_scores):
                c["rerank_score"] = round(float(score), 4)
            sorted_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        top_results = sorted_candidates[:top_k]

        # Step 7: Small-to-big context expansion
        if include_context:
            for r in top_results:
                parent = get_parent_context(self.collection, r["id"], window=1)
                r["parent_context"] = parent

        # Step 8: Answer synthesis (Claude LLM if available)
        answer = synthesize_answer(question, top_results, claude_client=self.claude_client)

        elapsed = time.time() - start_time

        return {
            "query": question,
            "expanded": expanded,
            "hyde_used": use_hyde,
            "bm25_used": use_bm25,
            "answer": answer,
            "time_ms": round(elapsed * 1000),
            "candidates_evaluated": len(candidates),
        }

    def get_stats(self) -> dict:
        """Return engine statistics."""
        count = self.collection.count() if self.collection else 0
        bm25_size = len(self.bm25_index.ids) if self.bm25_index else 0
        return {
            "collection": self.collection_name,
            "chunks": count,
            "bm25_index_size": bm25_size,
            "persist_dir": self.persist_dir,
            "claude_enabled": self.claude_client is not None,
        }
