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