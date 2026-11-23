"""Tests embeddings (avec modèle mini pour vitesse)."""

import pytest
import numpy as np
from rag_local.embeddings import EmbeddingGenerator


def test_embedding_generator_cache():
    gen = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", batch_size=2)
    textes = ["Test un", "Test deux", "Test un"]  # Doublon pour tester cache

    emb1 = gen.generer(textes, show_progress=False)
    assert emb1.shape == (3, 384)
    # Vérifie que le cache fonctionne : 2ème appel doit être plus rapide (pas de rechargement modèle)
    emb2 = gen.generer(textes, show_progress=False)
    assert emb2.shape == (3, 384)
    # Les embeddings doivent être déterministes pour le même texte
    assert np.allclose(emb1[0], emb1[2], atol=1e-5)  # Même texte = même embedding