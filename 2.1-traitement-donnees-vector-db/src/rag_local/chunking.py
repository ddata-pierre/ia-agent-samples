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