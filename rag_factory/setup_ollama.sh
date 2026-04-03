#!/bin/bash
# ==============================================
# RAG Factory — Ollama Setup Script
# Run this once to install models locally.
# ==============================================

set -e

echo "=== RAG Factory — Setup ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "[!] Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "[+] Ollama installed."
else
    echo "[+] Ollama already installed."
fi

# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "[*] Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Pull required models
echo ""
echo "[*] Pulling LLM model (llama3.2)..."
ollama pull llama3.2

echo ""
echo "[*] Pulling embedding model (nomic-embed-text)..."
ollama pull nomic-embed-text

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  pip install -r rag_factory/requirements.txt"
echo "  python -m rag_factory build 'sqlite:///my_database.db'"
echo "  python -m rag_factory build 'postgresql://user:pass@localhost:5432/mydb'"
echo "  python -m rag_factory build 'mongodb://localhost:27017' --db-name mydb"
echo "  python -m rag_factory chat"
echo ""
