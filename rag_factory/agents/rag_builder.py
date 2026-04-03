"""
RAG Builder — Takes Documents and builds a fully operational RAG query engine.
Uses ChromaDB (local, no server) + Ollama embeddings + Ollama LLM.
All 100% local, zero API cost.
"""

import os
import json

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


class RAGBuilder:
    def __init__(
        self,
        llm_model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        persist_dir: str = "./rag_data",
        collection_name: str = "rag_factory",
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Init local LLM
        self.llm = Ollama(
            model=llm_model,
            base_url=ollama_base_url,
            request_timeout=180.0,
        )

        # Init local embeddings
        self.embed = OllamaEmbedding(
            model_name=embed_model,
            base_url=ollama_base_url,
        )

        # Set as global defaults for LlamaIndex
        Settings.llm = self.llm
        Settings.embed_model = self.embed

    def build_index(self, documents: list[Document]) -> VectorStoreIndex:
        """Build a vector index from documents using ChromaDB for persistence."""
        os.makedirs(self.persist_dir, exist_ok=True)

        # ChromaDB local persistent client
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

        return index

    def build_query_engine(self, index: VectorStoreIndex, top_k: int = 10):
        """Build a query engine with similarity filtering."""
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.3),
            ],
            response_mode="tree_summarize",
        )
        return query_engine

    def load_existing_index(self) -> VectorStoreIndex | None:
        """Load a previously built index from disk."""
        if not os.path.exists(self.persist_dir):
            return None

        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            chroma_collection = chroma_client.get_collection(self.collection_name)
        except Exception:
            return None

        if chroma_collection.count() == 0:
            return None

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return VectorStoreIndex.from_vector_store(vector_store)

    def save_strategy(self, strategy: dict, path: str | None = None):
        """Save the RAG strategy to disk for reuse."""
        path = path or os.path.join(self.persist_dir, "rag_strategy.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(strategy, f, ensure_ascii=False, indent=2)

    def load_strategy(self, path: str | None = None) -> dict | None:
        """Load a previously saved strategy."""
        path = path or os.path.join(self.persist_dir, "rag_strategy.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
