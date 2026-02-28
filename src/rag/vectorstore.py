"""FAISS-backed vector store for Retrieval-Augmented Generation.

RAG improves LLM grounding by retrieving only the most relevant chunks
of collected research instead of dumping the full text into the prompt.

Pipeline:
    1. **Chunk** — split raw text into ~1 000-char pieces with overlap.
    2. **Embed** — convert each chunk to a dense vector via HuggingFace.
    3. **Store** — index vectors in FAISS for fast similarity search.
    4. **Retrieve** — given a query, return the *k* nearest chunks.

The embedding model (default ``all-MiniLM-L6-v2``, ~80 MB) is
downloaded on first use and cached locally by sentence-transformers.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


class RAGStore:
    """Manages the full ingest → retrieve lifecycle for RAG.

    Usage::

        store = RAGStore()
        store.ingest(["Some long article text …"])
        chunks = store.retrieve("relevant question")
    """

    def __init__(self) -> None:
        logger.info(
            "rag_init",
            model=settings.embedding_model,
            msg="Downloading embedding model if not cached (~80 MB first time)",
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
        )
        self._vectorstore: FAISS | None = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        logger.info("rag_ready", model=settings.embedding_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> int:
        """Chunk, embed, and store texts in the FAISS index.

        Can be called multiple times — subsequent calls add to the
        existing index rather than replacing it.

        Args:
            texts: Raw text strings (search results, page content, etc.).
            metadatas: Optional per-text metadata (e.g. ``{"source": url}``).

        Returns:
            Total number of chunks stored in this call.
        """
        docs: list[Document] = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            chunks = self._splitter.split_text(text)
            docs.extend(Document(page_content=c, metadata=meta) for c in chunks)

        if not docs:
            return 0

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(docs, self._embeddings)
        else:
            self._vectorstore.add_documents(docs)

        logger.info("rag_ingested", chunk_count=len(docs))
        return len(docs)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the *k* chunks most relevant to *query*.

        Args:
            query: Natural-language question or search phrase.
            k: Number of chunks to return (default 3).

        Returns:
            List of chunk texts, most relevant first.
            Empty list if nothing has been ingested yet.
        """
        if self._vectorstore is None:
            return []

        results = self._vectorstore.similarity_search(query, k=k)
        logger.info("rag_retrieved", query=query, result_count=len(results))
        return [doc.page_content for doc in results]

    def clear(self) -> None:
        """Reset the vector store (useful between independent queries)."""
        self._vectorstore = None
        logger.info("rag_cleared")
