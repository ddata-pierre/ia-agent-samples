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