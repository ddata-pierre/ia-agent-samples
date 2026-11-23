"""Tests chunking sémantique."""

import pytest
from rag_local.models import Document, DocumentMetadata, SourceType
from rag_local.chunking import chunker_document, detecter_sections, estimer_tokens
from pathlib import Path
from datetime import datetime


def test_estimer_tokens():
    assert estimer_tokens("Bonjour le monde") == 3  # 3 mots * 1.3 = 3.9 -> int = 3
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
    long_text = "Phrase. " * 1000  # ~1000 mots = ~1300 tokens > 512 target
    doc = Document(
        id="test2",
        metadata=DocumentMetadata(source_path=Path("test.pdf"), source_type=SourceType.PDF, titre="Long"),
        contenu_brut=f"# Longue\n{long_text}"
    )
    chunks = chunker_document(doc)
    assert len(chunks) > 1  # Doit être découpé
    # Overlap peut faire dépasser légèrement la target, on vérifie juste qu'on a plusieurs chunks
    assert len(chunks) >= 2