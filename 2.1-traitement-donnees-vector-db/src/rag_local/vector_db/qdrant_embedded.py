"""Vector DB avec Qdrant en mode embedded (binaire local).
Plus de fonctionnalités : filtres payload riches, HNSW natif, quantification.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)
import numpy as np

from ..models import Chunk, SearchResult, SearchQuery
from ..config import settings


class QdrantVectorStore:
    """Stockage vectoriel Qdrant Embedded."""

    def __init__(self, path: Path = None):
        self.path = path or settings.qdrant_path
        self.path.mkdir(parents=True, exist_ok=True)

        # Client embedded (pas de serveur réseau)
        self.client = QdrantClient(path=str(self.path))
        self.collection_name = "rag_chunks"
        self._init_collection()

    def _init_collection(self):
        """Crée collection avec HNSW + quantification scalaire."""

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                    ),
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                ),
            )

            logger.info(
                f"Collection Qdrant '{self.collection_name}' "
                "créée avec HNSW + quantification int8"
            )
        else:
            logger.info(
                f"Collection Qdrant '{self.collection_name}' existante"
            )
    def _normaliser_id(self, chunk_id: str) -> str:
        """Convertit un ID de chunk en UUID déterministe."""
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            str(chunk_id)
        ))

    def ajouter_chunks(self, chunks: List[Chunk]) -> int:
        """Insère chunks par batch."""
        if not chunks:
            return 0

        points = []
        for c in chunks:
            if c.embedding is None:
                continue
            payload = c.metadata.model_dump()
            payload["contenu"] = c.contenu  # Dupliqué pour récupération facile
            points.append(PointStruct(
                id=self._normaliser_id(c.id),
                vector=c.embedding,
                payload=payload,
            ))

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        logger.info(f"{len(points)} chunks insérés dans Qdrant")
        return len(points)

    def recherche_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filtres: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Recherche vectorielle avec filtres payload."""

        query_filter = None

        if filtres:
            conditions = []

            for key, value in filtres.items():
                if isinstance(value, list):
                    for v in value:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=v)
                            )
                        )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

            if conditions:
                query_filter = Filter(should=conditions)

        # API Qdrant 1.20.0
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(
                hnsw_ef=128,
                exact=False,
            ),
        )

        search_results = []

        for r in response.points:
            payload = r.payload or {}

            from ..models import ChunkMetadata

            meta = ChunkMetadata(
                document_id=payload.get("document_id", ""),
                document_titre=payload.get("document_titre"),
                section=payload.get("section"),
                niveau_section=payload.get("niveau_section"),
                index_chunk=payload.get("index_chunk", 0),
                mots=payload.get("mots", 0),
                tokens_estimes=payload.get("tokens_estimes", 0),
            )

            chunk = Chunk(
                id=payload.get("chunk_id", str(r.id)),
                contenu=payload.get("contenu", ""),
                metadata=meta,
            )

            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score_dense=r.score,
                    score_sparse=0.0,
                    score_hybride=r.score,
                    rang_final=0,
                )
            )

        return search_results

    def compter(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def vider(self):
        self.client.delete_collection(self.collection_name)
        self._init_collection()
        logger.info("Collection Qdrant vidée")

    def fermer(self):
        self.client.close()