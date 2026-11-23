"""Pipeline RAG complet : ingestion → chunking → embeddings → indexation → recherche."""

from pathlib import Path
from typing import List, Optional
from loguru import logger

from .ingestion import ingester_dossier, ingester_document
from .chunking import chunker_documents
from .embeddings import EmbeddingGenerator
from .search import HybridSearcher
from .models import Document, Chunk, SearchQuery, SearchResult
from .config import settings


class RAGPipeline:
    """Pipeline RAG de bout en bout."""

    def __init__(self):
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.searcher: Optional[HybridSearcher] = None
        self._initialise()

    def _initialise(self):
        self.embedding_generator = EmbeddingGenerator()
        self.searcher = HybridSearcher(embedding_generator=self.embedding_generator)

    def ingester_et_indexer(self, dossier: Path = None, force_rebuild: bool = False) -> int:
        """Pipeline complet : dossier → documents → chunks → embeddings → index."""
        logger.info("=== DÉBUT PIPELINE RAG ===")

        # 1. Ingestion
        self.documents = ingester_dossier(dossier)
        if not self.documents:
            logger.warning("Aucun document ingéré")
            return 0

        # 2. Chunking
        self.chunks = chunker_documents(self.documents)
        if not self.chunks:
            logger.warning("Aucun chunk généré")
            return 0

        # 3. Indexation (embeddings + vector store + BM25)
        self.searcher.indexer_chunks(self.chunks)

        logger.info(f"=== PIPELINE TERMINÉ : {len(self.documents)} docs, {len(self.chunks)} chunks ===")
        return len(self.chunks)

    def ajouter_document(self, path: Path) -> int:
        """Ajoute un document incrémentalement."""
        doc = ingester_document(path)
        if not doc:
            return 0

        # Vérifie si déjà présent (même hash)
        if any(d.id == doc.id for d in self.documents):
            logger.info(f"Document déjà indexé: {path.name}")
            return 0

        self.documents.append(doc)
        nouveaux_chunks = chunker_documents([doc])
        self.chunks.extend(nouveaux_chunks)
        self.searcher.indexer_chunks(nouveaux_chunks)
        logger.info(f"Document ajouté: {path.name} ({len(nouveaux_chunks)} chunks)")
        return len(nouveaux_chunks)

    def rechercher(self, query: SearchQuery) -> List[SearchResult]:
        """Recherche hybride."""
        if not self.searcher:
            raise RuntimeError("Pipeline non initialisé. Appelez ingester_et_indexer() d'abord.")
        return self.searcher.rechercher(query)

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vector_store_count": self.searcher.vector_store.compter() if self.searcher else 0,
            "embedding_model": settings.embedding_model,
            "backend": settings.vector_db_backend,
        }

    def fermer(self):
        if self.searcher and self.searcher.vector_store:
            self.searcher.vector_store.fermer()