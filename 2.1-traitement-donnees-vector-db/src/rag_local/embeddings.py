"""Génération d'embeddings locale avec sentence-transformers, batch, cache disque."""

import pickle
import hashlib
from pathlib import Path
from typing import List, Optional
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .models import Chunk
from .config import settings


class EmbeddingGenerator:
    """Générateur d'embeddings avec cache disque et batch processing."""

    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_length = settings.embedding_max_length

        # Cache disque : {hash_texte: embedding}
        self.cache_dir = settings.vector_db_dir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"cache_{self.model_name.replace('/', '_')}.pkl"
        self._cache = self._charger_cache()

        # Chargement modèle (lazy)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Chargement modèle embeddings: {self.model_name} sur {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_length
        return self._model

    def _charger_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                logger.info(f"Cache embeddings chargé: {len(cache)} entrées")
                return cache
            except Exception as e:
                logger.warning(f"Cache corrompu, nouveau cache: {e}")
        return {}

    def _sauver_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.error(f"Échec sauvegarde cache: {e}")

    def _hash_texte(self, texte: str) -> str:
        return hashlib.sha256(texte.encode()).hexdigest()[:32]

    def generer(self, textes: List[str], show_progress: bool = True) -> np.ndarray:
        """Génère embeddings pour une liste de textes (avec cache)."""
        # Vérifie cache
        embeddings = []
        a_calculer = []
        indices_a_calculer = []

        for i, txt in enumerate(textes):
            h = self._hash_texte(txt)
            if h in self._cache:
                embeddings.append(self._cache[h])
            else:
                embeddings.append(None)
                a_calculer.append(txt)
                indices_a_calculer.append(i)

        # Calcule les manquants par batch
        if a_calculer:
            logger.info(f"Génération embeddings pour {len(a_calculer)} nouveaux textes (batch={self.batch_size})")
            nouveaux = self.model.encode(
                a_calculer,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Important pour cosinus
            )
            # Met à jour cache et résultat
            for idx, emb in zip(indices_a_calculer, nouveaux):
                h = self._hash_texte(textes[idx])
                self._cache[h] = emb
                embeddings[idx] = emb
            self._sauver_cache()

        return np.array(embeddings, dtype=np.float32)

    def generer_pour_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remplit chunk.embedding pour une liste de chunks."""
        textes = [c.contenu for c in chunks]
        embeddings = self.generer(textes)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        return chunks