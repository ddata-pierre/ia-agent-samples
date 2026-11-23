"""Configuration centralisée — un seul point de modification."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Chemins
    data_raw_dir: Path = Field(default=Path("data/raw"), description="Documents sources")
    data_processed_dir: Path = Field(default=Path("data/processed"), description="Chunks JSONL")
    vector_db_dir: Path = Field(default=Path("data/vector_db"), description="Base vectorielle")

    # Modèles
    embedding_model: str = Field(default="BAAI/bge-m3", description="Modèle embeddings (HF Hub)")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder reranking")
    embedding_batch_size: int = Field(default=32, description="Batch size embeddings (RAM CPU)")
    embedding_max_length: int = Field(default=8192, description="Longueur max tokens entrée (bge-m3 supporte 8192)")

    # Chunking
    chunk_target_size: int = Field(default=512, description="Taille cible tokens par chunk")
    chunk_overlap: int = Field(default=64, description="Overlap tokens entre chunks")
    chunk_min_size: int = Field(default=100, description="Taille min pour garder un chunk")

    # Vector DB
    vector_db_backend: str = Field(default="duckdb", description="Backend: 'duckdb' ou 'qdrant'")
    duckdb_path: Path = Field(default=Path("data/vector_db/rag.duckdb"), description="Fichier DuckDB")
    qdrant_path: Path = Field(default=Path("data/vector_db/qdrant"), description="Dossier Qdrant embedded")

    # Recherche
    search_top_k: int = Field(default=10, description="Candidats recherche dense")
    search_bm25_k: int = Field(default=10, description="Candidats recherche sparse")
    search_final_k: int = Field(default=5, description="Résultats finaux après rerank")
    use_reranker: bool = Field(default=True, description="Activer cross-encoder reranking")

    # Device
    device: str = Field(default="cpu", description="Device torch: 'cpu' (N100) ou 'cuda'")

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"


settings = Settings()

# Création dossiers
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
settings.vector_db_dir.mkdir(parents=True, exist_ok=True)