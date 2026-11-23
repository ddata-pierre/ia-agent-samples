# 03-traitement-donnees-vector-db - RAG local : ingestion données, chunking sémantique, DuckDB/Qdrant embarqué

> **Objectif** : construire un pipeline RAG complet local pour documents — ingestion PDF/Word, nettoyage, chunking sémantique, embeddings CPU, vector DB embarquée, recherche hybride.
> **Durée** : ~25 min.
> **Prérequis** : Tutos #1-2 fonctionnels, Python 3.11+, `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`.

---

## 1. Architecture du pipeline RAG local

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Documents  │───▶│  Nettoyage   │───▶│ Chunking sémantique│───▶│  Embeddings     │───▶│  Vector DB      │
│  (PDF/DOCX) │    │  + Métadonnées│    │  (sections, §)   │    │  (bge-small-fr) │    │  (DuckDB/Qdrant)│
└─────────────┘    └──────────────┘    └────────────────┘    └──────────────────┘    └─────────────────┘
                                                                                          │
                                                                                          ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Réponse    │◀───│  Génération  │◀───│  Reranking     │◀───│  Recherche       │◀───│  Requête user   │
│  finale     │    │  (LLM local) │    │  (cross-encoder) │    │  Hybride         │    │  (embedding)    │
└─────────────┘    └──────────────┘    └────────────────┘    └──────────────────┘    └─────────────────┘
                                                          ▲
                              ┌───────────────────────────┘
                              │ Dense (vectoriel) + Sparse (BM25) = Hybride
```

**Choix techniques** :

| Composant | Option retenue | Pourquoi |
|-----------|----------------|----------|
| Embeddings | `sentence-transformers` + `BAAI/bge-m3` | 568M params, 2.2 Go, multilingue, SOTA français, CPU rapide, supporte 8192 tokens |
| Chunking | Sémantique (paragraphes + sections) + overlap | Respecte structure doc, évite couper au milieu d'une phrase |
| Vector DB | **DuckDB + VSS** (par défaut) / **Qdrant Embedded** | DuckDB = 0 déps, SQL natif, fichier unique. Qdrant = filtres payload, HNSW, plus complet |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (optionnel) | Améliore précision top-k, ~50 ms CPU |
| Recherche | Hybride : vectoriel (cosinus) + BM25 (mots-clés) | Capture sens + termes exacts (acronymes, refs légales) |

---

## 2. Installation et structure

```bash
# Avec uv (plus rapide pour tout : venv + install)
uv venv .venv
source .venv/bin/activate

# Installation dépendances + editable
uv pip install -e ".[dev]"
```

**Structure** :
```
03-traitement-donnees-vector-db/
├── data/
│   ├── raw/           # Documents sources (PDF, DOCX) — à vous de les placer
│   └── processed/     # Chunks JSONL intermédiaire
├── src/
│   ├── __init__.py
│   ├── config.py          # Config centralisée (paths, modèles, params)
│   ├── models.py          # Modèles Pydantic (Document, Chunk, SearchResult)
│   ├── ingestion.py       # Extraction PDF/DOCX + nettoyage
│   ├── chunking.py        # Chunking sémantique
│   ├── embeddings.py      # Génération embeddings (batch, cache)
│   ├── vector_db/
│   │   ├── __init__.py
│   │   ├── duckdb_vss.py  # Implémentation DuckDB + VSS
│   │   └── qdrant_embedded.py  # Implémentation Qdrant Embedded
│   ├── search.py          # Recherche hybride + reranking
│   ├── pipeline.py        # Orchestration complète
│   └── cli.py             # CLI demo
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   └── test_search.py
├── pyproject.toml
└── docs/tutoriel.md
```

---

## 3. Configuration centralisée

**Fichier** : `src/config.py`

```python
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
```

---

## 4. Modèles Pydantic — Documents, Chunks, Résultats

**Fichier** : `src/models.py`

```python
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
```

---

## 5. Ingestion — Extraction PDF/DOCX + Nettoyage

**Fichier** : `src/ingestion.py`

```python
"""Extraction texte depuis PDF/DOCX + nettoyage basique."""

import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from loguru import logger
from pypdf import PdfReader
from docx import Document as DocxDocument

from .models import Document, DocumentMetadata, SourceType
from .config import settings


def calculer_hash_fichier(path: Path) -> str:
    """Hash SHA256 du fichier + mtime pour détecter changements."""
    stat = path.stat()
    content = f"{path}:{stat.st_size}:{stat.st_mtime}".encode()
    return hashlib.sha256(content).hexdigest()[:16]


def extraire_pdf(path: Path) -> tuple[str, dict]:
    """Extrait texte + métadonnées PDF."""
    reader = PdfReader(str(path))
    textes = []
    meta = {"pages": len(reader.pages), "titre": None, "auteur": None}

    if reader.metadata:
        meta["titre"] = reader.metadata.get("/Title")
        meta["auteur"] = reader.metadata.get("/Author")
        if "/CreationDate" in reader.metadata:
            # Format PDF: D:YYYYMMDDHHmmSS
            try:
                d = reader.metadata["/CreationDate"][2:16]
                meta["date_creation"] = datetime.strptime(d, "%Y%m%d%H%M%S")
            except Exception:
                pass

    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text()
            if txt:
                textes.append(f"[PAGE {i+1}]\n{txt}")
        except Exception as e:
            logger.warning(f"Page {i+1} PDF {path.name} illisible: {e}")

    return "\n\n".join(textes), meta


def extraire_docx(path: Path) -> tuple[str, dict]:
    """Extrait texte + métadonnées DOCX (garde structure paragraphes)."""
    doc = DocxDocument(str(path))
    textes = []
    meta = {"pages": 0, "titre": None, "auteur": None}

    # Propriétés noyau
    if doc.core_properties.title:
        meta["titre"] = doc.core_properties.title
    if doc.core_properties.author:
        meta["auteur"] = doc.core_properties.author
    if doc.core_properties.created:
        meta["date_creation"] = doc.core_properties.created
    if doc.core_properties.modified:
        meta["date_modification"] = doc.core_properties.modified

    for para in doc.paragraphs:
        if para.text.strip():
            # Détecte styles titre (Heading 1, 2, 3...)
            style = para.style.name.lower()
            prefix = ""
            if "heading 1" in style or "titre 1" in style:
                prefix = "# "
            elif "heading 2" in style or "titre 2" in style:
                prefix = "## "
            elif "heading 3" in style or "titre 3" in style:
                prefix = "### "
            textes.append(f"{prefix}{para.text}")

    # Tableaux → texte simple
    for table in doc.tables:
        for row in table.rows:
            cellules = [cell.text.strip() for cell in row.cells]
            if any(cellules):
                textes.append(" | ".join(cellules))

    return "\n\n".join(textes), meta


def nettoyer_texte(texte: str) -> str:
    """Nettoyage basique : espaces, en-têtes/pieds de page répétés, caractères de contrôle."""
    # Supprime caractères de contrôle sauf \n \t
    texte = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", texte)

    # Normalise espaces multiples (garde sauts de ligne)
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)

    # Supprime lignes d'en-tête/pied de page répétitives (heuristique simple)
    lignes = texte.split("\n")
    freq = {}
    for lig in lignes:
        lig_propre = lig.strip()
        if 3 <= len(lig_propre) <= 80:
            freq[lig_propre] = freq.get(lig_propre, 0) + 1

    lignes_filtrees = []
    for lig in lignes:
        lig_propre = lig.strip()
        if freq.get(lig_propre, 0) > 3 and len(lig_propre) < 60:
            # Ligne courte répétée > 3 fois = probable header/footer
            continue
        lignes_filtrees.append(lig)

    return "\n".join(lignes_filtrees).strip()


def ingester_document(path: Path) -> Optional[Document]:
    """Pipeline complet : extraction → nettoyage → Document."""
    if not path.exists():
        logger.error(f"Fichier introuvable: {path}")
        return None

    doc_id = calculer_hash_fichier(path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            texte_brut, meta = extraire_pdf(path)
            source_type = SourceType.PDF
        elif suffix in (".docx", ".doc"):
            texte_brut, meta = extraire_docx(path)
            source_type = SourceType.DOCX
        elif suffix in (".md", ".markdown"):
            texte_brut = path.read_text(encoding="utf-8")
            meta = {"pages": 1}
            source_type = SourceType.MD
        elif suffix == ".txt":
            texte_brut = path.read_text(encoding="utf-8")
            meta = {"pages": 1}
            source_type = SourceType.TXT
        else:
            logger.warning(f"Format non supporté: {suffix} ({path})")
            return None

        texte_nettoye = nettoyer_texte(texte_brut)
        mots = len(texte_nettoye.split())

        metadata = DocumentMetadata(
            source_path=path,
            source_type=source_type,
            titre=meta.get("titre") or path.stem,
            auteur=meta.get("auteur"),
            date_creation=meta.get("date_creation"),
            date_modification=meta.get("date_modification"),
            pages=meta.get("pages", 1),
            mots_approx=mots,
        )

        return Document(id=doc_id, metadata=metadata, contenu_brut=texte_nettoye)

    except Exception as e:
        logger.exception(f"Erreur ingestion {path}: {e}")
        return None


def ingester_dossier(dossier: Path = None) -> List[Document]:
    """Ingeste tous les documents supportés d'un dossier."""
    dossier = dossier or settings.data_raw_dir
    extensions = {".pdf", ".docx", ".doc", ".md", ".txt"}
    documents = []

    for path in dossier.rglob("*"):
        if path.suffix.lower() in extensions and path.is_file():
            logger.info(f"Ingestion: {path.relative_to(dossier)}")
            doc = ingester_document(path)
            if doc:
                documents.append(doc)

    logger.info(f"Total documents ingérés: {len(documents)}")
    return documents
```

---

## 6. Chunking Sémantique — Respecter la structure

**Fichier** : `src/chunking.py`

```python
"""Chunking sémantique : découpe par sections/paragraphes, pas au hasard."""

import re
import hashlib
from typing import List, Optional
from loguru import logger

from .models import Document, Chunk, ChunkMetadata
from .config import settings


# Patterns pour détecter structure (Markdown + heuristiques)
PATTERN_TITRE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
PATTERN_PAGE = re.compile(r"^\[PAGE (\d+)\]$", re.MULTILINE)


def estimer_tokens(texte: str) -> int:
    """Estimation grossière tokens (français ~ 1.3 tokens/mot)."""
    return int(len(texte.split()) * 1.3)


def detecter_sections(texte: str) -> List[tuple[int, int, str, int]]:
    """
    Retourne liste (debut, fin, titre_section, niveau) pour chaque section détectée.
    Niveau = nombre de # (1 à 6).
    """
    sections = []
    matches = list(PATTERN_TITRE.finditer(texte))

    for i, m in enumerate(matches):
        debut = m.end()
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(texte)
        niveau = len(m.group(1))
        titre = m.group(2).strip()
        sections.append((debut, fin, titre, niveau))

    # Si pas de titres markdown, essaie de détecter pages PDF
    if not sections:
        pages = list(PATTERN_PAGE.finditer(texte))
        for i, m in enumerate(pages):
            debut = m.end()
            fin = pages[i + 1].start() if i + 1 < len(pages) else len(texte)
            sections.append((debut, fin, f"Page {m.group(1)}", 1))

    # Fallback : document entier comme une section
    if not sections:
        sections.append((0, len(texte), "Document", 1))

    return sections


def decouper_paragraphe(texte: str, taille_cible: int, overlap: int) -> List[str]:
    """Découpe un long paragraphe en chunks avec overlap (phrase par phrase)."""
    phrases = re.split(r"(?<=[.!?])\s+", texte)
    chunks = []
    courant = ""
    tokens_courant = 0

    for phrase in phrases:
        tokens_phrase = estimer_tokens(phrase)
        if tokens_courant + tokens_phrase > taille_cible and courant:
            chunks.append(courant.strip())
            # Overlap : garde les dernières phrases
            mots_overlap = courant.split()[-overlap//2:]
            courant = " ".join(mots_overlap) + " " + phrase
            tokens_courant = estimer_tokens(courant)
        else:
            courant += " " + phrase
            tokens_courant += tokens_phrase

    if courant.strip():
        chunks.append(courant.strip())

    return chunks


def chunker_document(doc: Document) -> List[Chunk]:
    """Pipeline chunking sémantique complet pour un document."""
    sections = detecter_sections(doc.contenu_brut)
    chunks = []

    for idx_section, (debut, fin, titre_section, niveau) in enumerate(sections):
        texte_section = doc.contenu_brut[debut:fin].strip()
        if not texte_section:
            continue

        # Si section courte → un seul chunk
        if estimer_tokens(texte_section) <= settings.chunk_target_size:
            chunk_id = hashlib.sha256(f"{doc.id}:{idx_section}:{texte_section[:50]}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                id=chunk_id,
                contenu=texte_section,
                metadata=ChunkMetadata(
                    document_id=doc.id,
                    document_titre=doc.metadata.titre,
                    section=titre_section,
                    niveau_section=niveau,
                    index_chunk=len(chunks),
                    mots=len(texte_section.split()),
                    tokens_estimes=estimer_tokens(texte_section),
                )
            ))
        else:
            # Section longue → découpe paragraphes avec overlap
            sous_chunks = decouper_paragraphe(
                texte_section,
                settings.chunk_target_size,
                settings.chunk_overlap
            )
            for sc in sous_chunks:
                if estimer_tokens(sc) >= settings.chunk_min_size:
                    chunk_id = hashlib.sha256(f"{doc.id}:{idx_section}:{sc[:50]}".encode()).hexdigest()[:16]
                    chunks.append(Chunk(
                        id=chunk_id,
                        contenu=sc,
                        metadata=ChunkMetadata(
                            document_id=doc.id,
                            document_titre=doc.metadata.titre,
                            section=titre_section,
                            niveau_section=niveau,
                            index_chunk=len(chunks),
                            mots=len(sc.split()),
                            tokens_estimes=estimer_tokens(sc),
                        )
                    ))

    logger.debug(f"Document {doc.metadata.titre}: {len(chunks)} chunks créés")
    return chunks


def chunker_documents(documents: List[Document]) -> List[Chunk]:
    """Chunke une liste de documents."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker_document(doc))
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks
```

---

## 7. Embeddings — Génération batch avec cache

**Fichier** : `src/embeddings.py`

```python
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
```

---

## 8. Vector DB — DuckDB + VSS (Embedded, zéro déps)

**Fichier** : `src/vector_db/duckdb_vss.py`

```python
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
        self._init_schema()
        self._load_vss()

    def _load_vss(self):
        """Charge extension VSS (Vector Similarity Search)."""
        try:
            self.conn.execute("LOAD vss")
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

        sql += " ORDER BY score DESC LIMIT ?"
        params.append(top_k)

        rows = self.conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            score = row[-1] if self._vss_available else self._cosinus_manuel(query_embedding, row[-1])
            results.append(SearchResult(
                chunk=chunk,
                score_dense=float(score),
                score_sparse=0.0,
                score_hybride=float(score),
                rang_final=0,
            ))
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
```

---

## 9. Vector DB — Qdrant Embedded (Alternative plus complète)

**Fichier** : `src/vector_db/qdrant_embedded.py`

```python
"""Vector DB avec Qdrant en mode embedded (binaire local).
Plus de fonctionnalités : filtres payload riches, HNSW natif, quantification.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
    SearchParams, HnswConfigDiff, QuantizationConfig, ScalarQuantization
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
        """Crée collection avec HNSW + quantification scalaire (économie RAM)."""
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
                    quantization_config=QuantizationConfig(
                        scalar=ScalarQuantization(
                            type="int8",
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                ),
            )
            logger.info(f"Collection Qdrant '{self.collection_name}' créée avec HNSW + quantification int8")
        else:
            logger.info(f"Collection Qdrant '{self.collection_name}' existante")

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
                id=c.id,
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

    def recherche_dense(self, query_embedding: List[float], top_k: int = 10,
                        filtres: Dict[str, Any] = None) -> List[SearchResult]:
        """Recherche vectorielle avec filtres payload."""
        query_filter = None
        if filtres:
            conditions = []
            for key, value in filtres.items():
                if isinstance(value, list):
                    # Qdrant ne supporte pas IN direct, on fait OR
                    for v in value:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(should=conditions)  # OR

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(hnsw_ef=128, exact=False),
        )

        search_results = []
        for r in results:
            payload = r.payload
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
                id=str(r.id),
                contenu=payload.get("contenu", ""),
                metadata=meta,
            )
            search_results.append(SearchResult(
                chunk=chunk,
                score_dense=r.score,
                score_sparse=0.0,
                score_hybride=r.score,
                rang_final=0,
            ))
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
```

---

## 10. Recherche Hybride — Dense + Sparse (BM25) + Reranking

**Fichier** : `src/search.py`

```python
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
```

---

## 11. Pipeline Complet — Orchestration

**Fichier** : `src/pipeline.py`

```python
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
```

---

## 12. CLI Demo

**Fichier** : `src/cli.py`

```python
"""CLI pour tester le pipeline RAG."""

import asyncio
import sys
from pathlib import Path
from loguru import logger

from .pipeline import RAGPipeline
from .models import SearchQuery
from .config import settings


def demo_recherche():
    """Demo interactive recherche."""
    pipeline = RAGPipeline()

    # Indexe si pas déjà fait
    if pipeline.stats()["chunks"] == 0:
        logger.info("Indexation en cours...")
        pipeline.ingester_et_indexer()
        logger.info(f"Stats: {pipeline.stats()}")

    print(f"\n=== RAG Local ({settings.vector_db_backend}) ===")
    print(f"Documents: {pipeline.stats()['documents']} | Chunks: {pipeline.stats()['chunks']}")
    print("Tapez votre question (ou 'quit')\n")

    while True:
        try:
            question = input("❓ Question : ").strip()
            if question.lower() in ("quit", "q", "exit"):
                break
            if not question:
                continue

            query = SearchQuery(texte=question, top_k=5, utiliser_reranker=True)
            resultats = pipeline.rechercher(query)

            print(f"\n🔍 {len(resultats)} résultats :\n")
            for r in resultats:
                c = r.chunk
                print(f"  #{r.rang_final} | Score hybride: {r.score_hybride:.3f} | Dense: {r.score_dense:.3f} | Sparse: {r.score_sparse:.3f}", end="")
                if r.score_rerank is not None:
                    print(f" | Rerank: {r.score_rerank:.3f}", end="")
                print()
                print(f"      📄 {c.metadata.document_titre} | § {c.metadata.section or 'N/A'}")
                print(f"      💬 {c.contenu[:200]}...")
                print()

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    pipeline.fermer()
    print("\nAu revoir !")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    demo_recherche()
```

---

## 13. Tests Unitaires

**Fichier** : `tests/test_chunking.py`

```python
"""Tests chunking sémantique."""

import pytest
from src.models import Document, DocumentMetadata, SourceType
from src.chunking import chunker_document, detecter_sections, estimer_tokens
from pathlib import Path
from datetime import datetime


def test_estimer_tokens():
    assert estimer_tokens("Bonjour le monde") == 5  # 4 mots * 1.3 ≈ 5
    assert estimer_tokens("") == 0


def test_detecter_sections_markdown():
    texte = """# Titre principal
Contenu section 1.
## Sous-section
Contenu section 2.
# Autre titre
Fin."""
    sections = detecter_sections(texte)
    assert len(sections) == 3
    assert sections[0][2] == "Titre principal"
    assert sections[0][3] == 1
    assert sections[1][3] == 2
    assert sections[2][3] == 1


def test_chunker_document_court():
    doc = Document(
        id="test1",
        metadata=DocumentMetadata(source_path=Path("test.pdf"), source_type=SourceType.PDF, titre="Test"),
        contenu_brut="# Intro\nCeci est un court document.\n## Section\nContenu section."
    )
    chunks = chunker_document(doc)
    assert len(chunks) >= 1
    assert all(c.metadata.document_id == "test1" for c in chunks)


def test_chunker_document_long():
    """Document avec section longue → découpée."""
    long_text = "Phrase. " * 200  # ~200 phrases = ~2600 tokens
    doc = Document(
        id="test2",
        metadata=DocumentMetadata(source_path=Path("test.pdf"), source_type=SourceType.PDF, titre="Long"),
        contenu_brut=f"# Longue\n{long_text}"
    )
    chunks = chunker_document(doc)
    assert len(chunks) > 1  # Doit être découpé
    assert all(c.metadata.tokens_estimes <= 600 for c in chunks)  # Respecte target + overlap
```

**Fichier** : `tests/test_embeddings.py`

```python
"""Tests embeddings (avec modèle mini pour vitesse)."""

import pytest
import numpy as np
from src.embeddings import EmbeddingGenerator


def test_embedding_generator_cache():
    gen = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", batch_size=2)
    textes = ["Test un", "Test deux", "Test un"]  # Doublon pour tester cache

    emb1 = gen.generer(textes, show_progress=False)
    assert emb1.shape == (3, 384)
    assert np.allclose(emb1[0], emb1[2])  # Cache : même texte = même embedding

    # Deuxième appel = tout du cache
    emb2 = gen.generer(textes, show_progress=False)
    assert np.allclose(emb1, emb2)
```

**Fichier** : `tests/test_search.py`

```python
"""Tests recherche hybride (mock vector store)."""

import pytest
from unittest.mock import Mock, MagicMock
from src.models import Chunk, ChunkMetadata, SearchQuery, SearchResult
from src.search import HybridSearcher
from src.config import settings


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
```

---

## 14. Configuration Projet — pyproject.toml

**Fichier** : `pyproject.toml`

```toml
[project]
name = "rag-local"
version = "0.1.0"
description = "Pipeline RAG local pour documents : ingestion, chunking, embeddings, vector DB embarquée"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers>=3.0",
    "pypdf>=4.0",
    "python-docx>=1.1",
    "duckdb>=1.0",
    "qdrant-client>=1.10",
    "rank-bm25>=0.2",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "loguru>=0.7",
    "tqdm>=4.66",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
```

---

## 15. Exécution

```bash
# 1. Placez vos documents dans data/raw/
#    Ex: data/raw/politique_conges.pdf, charte_teletravail.docx, FAQ_RH.md

# 2. Installez dépendances
uv pip install -e ".[dev]"

# 3. Tests unitaires
pytest tests/ -v

# 4. Lancez le pipeline (indexation + recherche interactive)
python -m src.cli
```

**Exemple session** :
```
=== RAG Local (duckdb) ===
Documents: 3 | Chunks: 47

❓ Question : Combien de jours de congés par an ?
🔍 5 résultats :

  #1 | Score hybride: 0.042 | Dense: 0.87 | Sparse: 0.12 | Rerank: 8.45
      📄 Politique congés 2024 | § Congés annuels
      💬 Les salariés bénéficient de 25 jours ouvrés de congés payés par an...

  #2 | Score hybride: 0.038 | Dense: 0.82 | Sparse: 0.15 | Rerank: 7.91
      📄 Charte télétravail | § Congés et télétravail
      💬 Les jours de télétravail ne comptent pas dans le décompte des congés...
```

5 Exemples de Données à Chercher (Factuelles et Absentes des LLM à partir des pdf factice fournit)

Ces requêtes contiennent des faits arbitraires créés sur mesure. Un modèle généraliste sans RAG ne pourra **jamais** deviner ces réponses ; si ton RAG fonctionne, il ressortira exactement la valeur exacte.

| N° | Dom. | Question de Test pour le RAG | Réponse Attendue (dans le PDF) | Document Source |
| --- | --- | --- | --- | --- |
| **1** | **RH** | *Quel est le montant exact de la prime forfaitaire d'équipement ergonomique attribuée pour le télétravail ?* | **345,50 €** | `rh_politique_teletravail_2026.pdf` |
| **2** | **RH** | *Combien de jours de congé sont accordés pour l'adoption d'un animal en refuge SPA et quel est le code du congé ?* | **1 jour unique** (Code congé : **`#RH-SPA-01`**) | `rh_guide_conges_cooptation_2026.pdf` |
| **3** | **RH / IT** | *Quel est le montant de la prime de cooptation brute versée pour l'embauche d'un ingénieur data ?* | **2 450 € brut** (versé en deux fois) | `rh_guide_conges_cooptation_2026.pdf` |
| **4** | **IT** | *Tous les combien de jours les clés d'API inter-services doivent-elles subir une rotation automatique ?* | Tous les **42 jours** | `it_politique_securite_passwords.pdf` |
| **5** | **IT** | *Quelle est l'adresse IP fixe et le nom réseau du serveur d'impression réservé à la direction générale ?* | IP : **`10.200.4.55`** / Nom : **`PRN-MARS-EXEC04.ddata.local`** | `it_procedure_incidents_escalade.pdf` |

---

## 16. Pièges et limites — Soyons honnêtes

| Piège | Réalité |
|-------|---------|
| « Chunking sémantique = parfait » | Heuristiques regex. Les PDF mal structurés (scans, colonnes) donnent du bruit. Prévoir nettoyage manuel ou `pdfplumber`/`marker` pour PDF complexes. |
| « Embeddings locaux = qualité OpenAI » | `bge-m3` est excellent pour le français, mais < `text-embedding-3-large` sur tâches complexes. Pour production critique : évaluer `bge-base` (137M params, ~2x plus lent). |
| « DuckDB VSS = production ready » | Extension VSS encore jeune. Pas de quantization native, RAM = taille index. Pour >50k chunks : Qdrant embedded (quantification int8) ou migration vers Qdrant serveur. |
| « BM25 + Dense = tout capture » | BM25 rate les synonymes. Dense rate les acronymes exacts (ex: "RGPD", "CPF"). Hybride aide, mais **reranker cross-encoder est crucial** pour précision. |
| « Ça tient sur N100 » | 47 chunks ≈ 50 Mo RAM (embeddings + index). 10k chunks ≈ 500 Mo + index HNSW. Sur 16 Go : OK jusqu'à ~50k chunks. Au-delà : swappe → latence x10. |

---

## 17. Aller plus loin

- [sentence-transformers docs](https://www.sbert.net/) — modèles, fine-tuning, évaluation
- [DuckDB VSS extension](https://duckdb.org/docs/2.0/core_extensions/vss) — HNSW, paramètres
- [Qdrant Embedded](https://qdrant.tech/documentation/embeddings/) — quantification, filtres, payload
- [BM25 Okapi](https://github.com/dorianbrown/rank_bm25) — paramètres k1, b
- [Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html) — reranking
- Prochaine étape : **04-architecture-memoires-court-long-terme** — mémoire court terme (fenêtre conversation) vs long terme (profil utilisateur, préférences, historique persistant).