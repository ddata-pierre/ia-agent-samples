"""Vector DB avec DuckDB + extension VSS (Vector Similarity Search).
Zéro dépendance externe, fichier unique, SQL natif, filtres métadonnées.
"""

import duckdb
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger
import numpy as np

from ..models import Chunk, SearchResult, SearchQuery
from ..config import settings


class DuckDBVectorStore:
    """Stockage vectoriel DuckDB + VSS."""
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or settings.duckdb_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._load_vss()
        self._init_schema()

    def _load_vss(self):
        """Charge extension VSS (Vector Similarity Search)."""
        try:
            self.conn.execute("INSTALL AND LOAD vss;")
            self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
            logger.info("Extension DuckDB VSS chargée")
        except Exception as e:
            logger.warning(f"Extension VSS non disponible: {e}. Utilisation fallback cosinus manuel.")
            self._vss_available = False
        else:
            self._vss_available = True

    def _init_schema(self):
        """Crée tables et index."""
        # Table chunks avec métadonnées JSON
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                contenu TEXT NOT NULL,
                document_id VARCHAR NOT NULL,
                document_titre VARCHAR,
                section VARCHAR,
                niveau_section INTEGER,
                index_chunk INTEGER,
                mots INTEGER,
                tokens_estimes INTEGER,
                metadata_json JSON,  -- Métadonnées complètes pour filtrage
                embedding FLOAT[1024]  -- Dimension bge-m3 = 1024
            )
        """)

        # Index HNSW via VSS (si dispo)
        if self._vss_available:
            try:
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                    ON chunks USING HNSW(embedding)
                    WITH (metric = 'cosine')
                """)
                logger.info("Index HNSW créé sur embeddings")
            except Exception as e:
                logger.warning(f"Index HNSW non créé: {e}")

        # Index métadonnées pour filtrage rapide
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section)")

    def ajouter_chunks(self, chunks: List[Chunk]) -> int:
        """Insère ou met à jour des chunks (upsert)."""
        if not chunks:
            return 0

        # Prépare données
        data = []
        for c in chunks:
            if c.embedding is None:
                logger.warning(f"Chunk {c.id} sans embedding, ignoré")
                continue
            meta = c.metadata.model_dump()
            data.append((
                c.id,
                c.contenu,
                c.metadata.document_id,
                c.metadata.document_titre,
                c.metadata.section,
                c.metadata.niveau_section,
                c.metadata.index_chunk,
                c.metadata.mots,
                c.metadata.tokens_estimes,
                json.dumps(meta),
                c.embedding,
            ))

        # Upsert via INSERT OR REPLACE
        self.conn.executemany("""
            INSERT OR REPLACE INTO chunks
            (id, contenu, document_id, document_titre, section, niveau_section,
             index_chunk, mots, tokens_estimes, metadata_json, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        logger.info(f"{len(data)} chunks insérés/mis à jour dans DuckDB")
        return len(data)

    def recherche_dense(self, query_embedding: List[float], top_k: int = 10,
                        filtres: Dict[str, Any] = None) -> List[SearchResult]:
        """Recherche vectorielle pure (cosinus)."""
        if self._vss_available:
            # Utilise fonction VSS native
            sql = """
                SELECT id, contenu, document_id, document_titre, section, niveau_section,
                       index_chunk, mots, tokens_estimes, metadata_json,
                       array_cosine_similarity(embedding, ?::FLOAT[1024]) as score
                FROM chunks
            """
            params = [query_embedding]
        else:
            # Fallback : calcul manuel (plus lent)
            sql = """
                SELECT id, contenu, document_id, document_titre, section, niveau_section,
                       index_chunk, mots, tokens_estimes, metadata_json, embedding
                FROM chunks
            """
            params = []

        # Filtres métadonnées (JSON)
        if filtres:
            conditions = []
            for key, value in filtres.items():
                if isinstance(value, list):
                    placeholders = ",".join(["?"] * len(value))
                    conditions.append(f"json_extract(metadata_json, '$.{key}') IN ({placeholders})")
                    params.extend(value)
                else:
                    conditions.append(f"json_extract(metadata_json, '$.{key}') = ?")
                    params.append(value)
            sql += " WHERE " + " AND ".join(conditions)

        if self._vss_available:
            # Recherche SQL native
            sql += " ORDER BY score DESC LIMIT ?"
            params.append(top_k)

            rows = self.conn.execute(sql, params).fetchall()

            results = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                score = row[-1]

                results.append(SearchResult(
                    chunk=chunk,
                    score_dense=float(score),
                    score_sparse=0.0,
                    score_hybride=float(score),
                    rang_final=0,
                ))

        else:
            # Fallback cosinus manuel
            rows = self.conn.execute(sql, params).fetchall()

            results = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                score = self._cosinus_manuel(query_embedding, row[-1])

                results.append(SearchResult(
                    chunk=chunk,
                    score_dense=float(score),
                    score_sparse=0.0,
                    score_hybride=float(score),
                    rang_final=0,
                ))

            results.sort(key=lambda x: x.score_dense, reverse=True)
            results = results[:top_k]

        return results

    def _cosinus_manuel(self, q: List[float], emb_blob) -> float:
        """Cosinus manuel si VSS pas dispo."""
        emb = np.array(emb_blob, dtype=np.float32)
        q_arr = np.array(q, dtype=np.float32)
        return float(np.dot(q_arr, emb) / (np.linalg.norm(q_arr) * np.linalg.norm(emb) + 1e-8))

    def _row_to_chunk(self, row) -> Chunk:
        """Convertit ligne DuckDB en objet Chunk."""
        from ..models import ChunkMetadata
        meta = ChunkMetadata(**json.loads(row[9]))
        return Chunk(
            id=row[0],
            contenu=row[1],
            metadata=meta,
            embedding=None,  # Pas chargé pour économiser mémoire
        )

    def compter(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def vider(self):
        self.conn.execute("DELETE FROM chunks")
        logger.info("Base DuckDB vidée")

    def fermer(self):
        self.conn.close()