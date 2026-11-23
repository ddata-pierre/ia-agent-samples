"""Modèles de données du pipeline RAG."""

from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    TXT = "txt"
    HTML = "html"


class DocumentMetadata(BaseModel):
    """Métadonnées document source."""
    source_path: Path
    source_type: SourceType
    titre: Optional[str] = None
    auteur: Optional[str] = None
    date_creation: Optional[datetime] = None
    date_modification: Optional[datetime] = None
    pages: int = 0
    mots_approx: int = 0
    langue: str = "fr"
    tags: List[str] = Field(default_factory=list)  # ex: ["conges", "politique", "2024"]


class Document(BaseModel):
    """Document brut après extraction."""
    id: str  # hash chemin + mtime
    metadata: DocumentMetadata
    contenu_brut: str  # Texte complet extrait


class ChunkMetadata(BaseModel):
    """Métadonnées chunk — essentielles pour citation et filtrage."""
    document_id: str
    document_titre: Optional[str] = None
    page_debut: Optional[int] = None
    page_fin: Optional[int] = None
    section: Optional[str] = None  # Titre section détectée
    niveau_section: Optional[int] = None  # 1=H1, 2=H2...
    index_chunk: int  # Ordre dans le document
    mots: int = 0
    tokens_estimes: int = 0


class Chunk(BaseModel):
    """Chunk sémantique avec embedding."""
    id: str  # hash(document_id + index_chunk + contenu)
    contenu: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None  # Rempli après génération


class SearchResult(BaseModel):
    """Résultat de recherche unifié."""
    chunk: Chunk
    score_dense: float  # Similarité cosinus [0,1]
    score_sparse: float  # Score BM25 normalisé [0,1]
    score_hybride: float  # Combinaison pondérée
    score_rerank: Optional[float] = None  # Score cross-encoder si activé
    rang_final: int


class SearchQuery(BaseModel):
    """Requête de recherche structurée."""
    texte: str
    top_k: int = 5
    filtres: Dict[str, Any] = Field(default_factory=dict)  # ex: {"tags": ["conges"], "annee": 2024}
    utiliser_reranker: bool = True