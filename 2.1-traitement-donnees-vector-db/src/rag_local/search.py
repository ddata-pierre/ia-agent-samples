"""Recherche hybride : vectorielle (dense) + BM25 (sparse) + reranking cross-encoder."""

from typing import List, Dict, Any, Optional
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

from .models import Chunk, SearchResult, SearchQuery
from .embeddings import EmbeddingGenerator
from .vector_db.duckdb_vss import DuckDBVectorStore
from .vector_db.qdrant_embedded import QdrantVectorStore
from .config import settings


class HybridSearcher:
    """Moteur de recherche hybride dense + sparse + rerank."""

    def __init__(self, vector_store=None, embedding_generator=None):
        self.vector_store = vector_store or self._creer_vector_store()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[Chunk] = []
        self.reranker: Optional[CrossEncoder] = None
        if settings.use_reranker:
            self._charger_reranker()

    def _creer_vector_store(self):
        if settings.vector_db_backend == "qdrant":
            return QdrantVectorStore()
        return DuckDBVectorStore()

    def _charger_reranker(self):
        logger.info(f"Chargement reranker: {settings.reranker_model}")
        self.reranker = CrossEncoder(settings.reranker_model, device=settings.device, max_length=512)

    def indexer_chunks(self, chunks: List[Chunk]):
        """Indexe chunks pour BM25 (sparse) et vector store (dense)."""
        self.corpus_chunks = chunks
        # Corpus BM25 : tokenization simple français
        corpus_tokens = [c.contenu.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"BM25 indexé sur {len(chunks)} chunks")

        # Indexe dans vector store
        chunks_avec_emb = self.embedding_generator.generer_pour_chunks(chunks)
        self.vector_store.ajouter_chunks(chunks_avec_emb)

    def rechercher(self, query: SearchQuery) -> List[SearchResult]:
        """Pipeline complet : dense + sparse → fusion → rerank → top-k final."""
        # 1. Embedding requête
        query_emb = self.embedding_generator.generer([query.texte])[0]

        # 2. Recherche dense (vectorielle)
        resultats_dense = self.vector_store.recherche_dense(
            query_emb.tolist(),
            top_k=query.top_k * 2,  # Large pour fusion
            filtres=query.filtres,
        )

        # 3. Recherche sparse (BM25)
        resultats_sparse = self._rechercher_bm25(query.texte, query.top_k * 2)

        # 4. Fusion hybride (RRF - Reciprocal Rank Fusion)
        resultats_fusion = self._fusion_rrf(resultats_dense, resultats_sparse, query.top_k * 2)

        # 5. Reranking cross-encoder (optionnel)
        if query.utiliser_reranker and self.reranker and resultats_fusion:
            resultats_fusion = self._reranker(query.texte, resultats_fusion)

        # 6. Top-k final
        resultats_fusion = resultats_fusion[:query.top_k]
        for i, r in enumerate(resultats_fusion):
            r.rang_final = i + 1

        return resultats_fusion

    def _rechercher_bm25(self, query: str, top_k: int) -> List[SearchResult]:
        if not self.bm25 or not self.corpus_chunks:
            return []

        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_score = scores[top_indices[0]] if len(top_indices) > 0 else 1.0
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            chunk = self.corpus_chunks[idx]
            results.append(SearchResult(
                chunk=chunk,
                score_dense=0.0,
                score_sparse=float(scores[idx] / max_score),  # Normalisé [0,1]
                score_hybride=0.0,
                rang_final=0,
            ))
        return results

    def _fusion_rrf(self, dense: List[SearchResult], sparse: List[SearchResult],
                    top_k: int, k: int = 60) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion : score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        k=60 standard (empirique).
        """
        # Map chunk_id → (rank_dense, rank_sparse)
        ranks = {}

        for rank, r in enumerate(dense):
            ranks.setdefault(r.chunk.id, {"dense": rank+1, "sparse": None, "chunk": r.chunk})
            ranks[r.chunk.id]["dense"] = rank + 1

        for rank, r in enumerate(sparse):
            ranks.setdefault(r.chunk.id, {"dense": None, "sparse": rank+1, "chunk": r.chunk})
            ranks[r.chunk.id]["sparse"] = rank + 1

        # Calcule score RRF
        fused = []
        for chunk_id, info in ranks.items():
            score_dense = 1.0 / (k + info["dense"]) if info["dense"] else 0.0
            score_sparse = 1.0 / (k + info["sparse"]) if info["sparse"] else 0.0
            score_rrf = score_dense + score_sparse

            fused.append(SearchResult(
                chunk=info["chunk"],
                score_dense=score_dense,
                score_sparse=score_sparse,
                score_hybride=score_rrf,
                rang_final=0,
            ))

        # Tri par score RRF descendant
        fused.sort(key=lambda x: x.score_hybride, reverse=True)
        return fused[:top_k]

    def _reranker(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Cross-encoder reranking sur les top candidats."""
        if not results:
            return results

        pairs = [(query, r.chunk.contenu) for r in results]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        for r, score in zip(results, scores):
            r.score_rerank = float(score)

        # Tri par score rerank descendant
        results.sort(key=lambda x: x.score_rerank or 0.0, reverse=True)
        return results