"""Tests recherche hybride (mock vector store)."""

import pytest
from unittest.mock import Mock, MagicMock
from rag_local.models import Chunk, ChunkMetadata, SearchQuery, SearchResult
from rag_local.search import HybridSearcher
from rag_local.config import settings


def test_fusion_rrf():
    searcher = HybridSearcher(vector_store=Mock(), embedding_generator=Mock())
    searcher.bm25 = Mock()
    searcher.corpus_chunks = []

    # Crée résultats mock
    chunk1 = Chunk(id="1", contenu="chunk 1", metadata=ChunkMetadata(document_id="d1", index_chunk=0))
    chunk2 = Chunk(id="2", contenu="chunk 2", metadata=ChunkMetadata(document_id="d1", index_chunk=1))
    chunk3 = Chunk(id="3", contenu="chunk 3", metadata=ChunkMetadata(document_id="d1", index_chunk=2))

    dense = [
        SearchResult(chunk=chunk1, score_dense=0.9, score_sparse=0, score_hybride=0, rang_final=0),
        SearchResult(chunk=chunk2, score_dense=0.8, score_sparse=0, score_hybride=0, rang_final=0),
    ]
    sparse = [
        SearchResult(chunk=chunk2, score_dense=0, score_sparse=0.9, score_hybride=0, rang_final=0),
        SearchResult(chunk=chunk3, score_dense=0, score_sparse=0.8, score_hybride=0, rang_final=0),
    ]

    fused = searcher._fusion_rrf(dense, sparse, top_k=3)
    assert len(fused) == 3
    # chunk2 présent dans les deux → score RRF le plus haut
    assert fused[0].chunk.id == "2"