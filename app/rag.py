"""
rag.py — RAG Pipeline: Document Ingestion + Retrieval.

Handles PDF/TXT file ingestion and URL scraping via Firecrawl,
chunking with 512-token windows and 50-token overlap,
embedding with sentence-transformers all-MiniLM-L6-v2,
and storage/retrieval from ChromaDB.
"""

import os
import logging
import hashlib
from typing import Optional

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

from app.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "capstone_docs"
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 50     # tokens
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.65  # minimum cosine similarity threshold


# ═══════════════════════════════════════════════════════════════════════════
#  Lazy-loaded singletons (avoid re-loading on every call)
# ═══════════════════════════════════════════════════════════════════════════

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.HttpClient] = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazily load the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_collection():
    """Lazily connect to ChromaDB and get/create the collection."""
    global _chroma_client, _collection
    if _collection is None:
        logger.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ═══════════════════════════════════════════════════════════════════════════
#  Text Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _extract_text_from_txt(file_path: str) -> str:
    """Read plain text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_text_from_url(url: str) -> str:
    """Scrape URL content using Firecrawl, returning clean Markdown text."""
    from firecrawl import Firecrawl

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable is required for URL ingestion")

    app = Firecrawl(api_key=api_key)
    result = app.scrape(url, formats=["markdown"])

    # Firecrawl returns a Document object (or dict) with 'markdown' key/attribute
    if hasattr(result, "markdown") and result.markdown is not None:
        return result.markdown
    elif isinstance(result, dict) and "markdown" in result and result["markdown"] is not None:
        return result["markdown"]
    else:
        raise ValueError(f"Firecrawl returned unexpected result format or empty markdown for URL: {url}")



# ═══════════════════════════════════════════════════════════════════════════
#  Chunking (token-aware with overlap)
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks of approximately `chunk_size` tokens
    with `overlap` token overlap between consecutive chunks.

    Uses simple whitespace tokenization (word-level) as an approximation.
    For production, consider tiktoken or the model's tokenizer.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) words
        start += chunk_size - overlap

    return chunks


def _generate_chunk_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic, unique ID for a chunk."""
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
#  Ingestion
# ═══════════════════════════════════════════════════════════════════════════

def ingest_document(file_path: str, file_type: str = "auto") -> list[RetrievedChunk]:
    """
    Ingest a PDF or TXT document into the ChromaDB vector store.

    Args:
        file_path: Path to the document file.
        file_type: "pdf", "txt", or "auto" (detect from extension).

    Returns:
        List of RetrievedChunk with metadata for each stored chunk.
    """
    # Detect file type
    if file_type == "auto":
        ext = os.path.splitext(file_path)[1].lower()
        file_type = "pdf" if ext == ".pdf" else "txt"

    # Extract text
    if file_type == "pdf":
        text = _extract_text_from_pdf(file_path)
    else:
        text = _extract_text_from_txt(file_path)

    source_name = os.path.basename(file_path)
    return _ingest_text(text, source_name)


def ingest_url(url: str) -> list[RetrievedChunk]:
    """
    Ingest a web page into the ChromaDB vector store via Firecrawl.

    Args:
        url: The URL to scrape and ingest.

    Returns:
        List of RetrievedChunk with metadata for each stored chunk.
    """
    text = _extract_text_from_url(url)
    return _ingest_text(text, source=url)


def _ingest_text(text: str, source: str) -> list[RetrievedChunk]:
    """
    Core ingestion: chunk text → embed → store in ChromaDB → return chunks.
    """
    if not text.strip():
        logger.warning(f"No text extracted from source: {source}")
        return []

    # Chunk
    chunks = _chunk_text(text)
    logger.info(f"Split '{source}' into {len(chunks)} chunks")

    # Embed
    model = _get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Prepare for ChromaDB
    ids = [_generate_chunk_id(source, i) for i in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk_index": i, "total_chunks": len(chunks)}
        for i in range(len(chunks))
    ]

    # Upsert into ChromaDB
    collection = _get_collection()
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Stored {len(chunks)} chunks from '{source}' in ChromaDB")

    # Return chunk metadata
    return [
        RetrievedChunk(
            text=chunk,
            source=source,
            chunk_index=i,
            similarity_score=1.0,  # self-similarity at ingestion
            metadata=metadatas[i],
        )
        for i, chunk in enumerate(chunks)
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks from ChromaDB for a given query.

    Args:
        query: The search query string.
        top_k: Maximum number of results to return.

    Returns:
        List of RetrievedChunk sorted by similarity score (descending),
        filtered to score >= 0.65.
    """
    model = _get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()[0]

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            # ChromaDB returns distances (lower = more similar for cosine)
            # Convert cosine distance to similarity: similarity = 1 - distance
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            if similarity >= MIN_SIMILARITY:
                chunks.append(
                    RetrievedChunk(
                        text=doc,
                        source=metadata.get("source", "unknown"),
                        chunk_index=metadata.get("chunk_index", 0),
                        similarity_score=round(similarity, 4),
                        metadata=metadata,
                    )
                )

    # Sort by score descending
    chunks.sort(key=lambda c: c.similarity_score, reverse=True)
    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
    return chunks
