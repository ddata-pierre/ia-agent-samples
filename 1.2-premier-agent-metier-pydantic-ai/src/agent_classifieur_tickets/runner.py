"""Exécution de l'agent avec logging structuré, métriques, fallback."""

import json
import time
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from pydantic_ai import AgentRunResult
from pydantic_ai.exceptions import UnexpectedModelBehavior, ModelRetry

from .agent import agent, construire_prompt_utilisateur
from .models import TicketEntree, TicketSortie, ErreurClassification


@dataclass
class ResultatExecution:
    """Résultat enrichi pour monitoring/observabilité."""
    succes: bool
    sortie: Optional[TicketSortie] = None
    erreur: Optional[ErreurClassification] = None
    duree_ms: float = 0.0
    tokens_entree: int = 0
    tokens_sortie: int = 0
    nb_retries: int = 0


class AgentRunner:
    """Wrapper autour de l'agent Pydantic AI pour production."""

    def __init__(self, agent_instance: Agent = None):
        self.agent = agent_instance or agent
        logger.add("logs/agent_classifieur.log", rotation="10 MB", retention="7 days", level="INFO")

    async def classifier(self, ticket: TicketEntree) -> ResultatExecution:
        """Classifie un ticket, gère erreurs, log métriques."""
        start = time.perf_counter()
        prompt = construire_prompt_utilisateur(ticket)

        try:
                    # Exécution avec historique (pour debug/contexte)
                    result: AgentRunResult = await self.agent.run(prompt)

                    # Métriques d'usage (tokens)
                    usage = result.usage
                    duree = (time.perf_counter() - start) * 1000

                    logger.info(
                        "Classification OK | titre={} | categorie={} | urgence={} | confidence={:.2f} | "
                        "tokens_in={} tokens_out={} | duree_ms={:.0f}",
                        ticket.titre, result.output.categorie, result.output.urgence,
                        result.output.confidence, usage.input_tokens, usage.output_tokens, duree
                    )

                    return ResultatExecution(
                        succes=True,
                        sortie=result.output,
                        duree_ms=duree,
                        tokens_entree=usage.input_tokens,
                        tokens_sortie=usage.output_tokens,
                        nb_retries=getattr(result, '_retries', 0),
                    )

        except ModelRetry as e:
            # Échec après tous les retries (validation Pydantic ou validateur métier)
            duree = (time.perf_counter() - start) * 1000
            err = ErreurClassification(
                code="RETRY_EXHAUSTED",
                message=f"Échec après retries : {e}",
                entree_brute=ticket,
                details=str(e.__cause__) if e.__cause__ else None,
            )
            logger.error("Classification RETRY_EXHAUSTED | titre={} | erreur={}", ticket.titre, err.message)
            return ResultatExecution(succes=False, erreur=err, duree_ms=duree)

        except UnexpectedModelBehavior as e:
            # Modèle a retourné quelque chose d'inattendu (pas JSON, format cassé)
            duree = (time.perf_counter() - start) * 1000
            err = ErreurClassification(
                code="MODEL_BEHAVIOR",
                message=f"Comportement modèle inattendu : {e}",
                entree_brute=ticket,
            )
            logger.error("Classification MODEL_BEHAVIOR | titre={} | erreur={}", ticket.titre, err.message)
            return ResultatExecution(succes=False, erreur=err, duree_ms=duree)

        except Exception as e:
            # Erreur inattendue (réseau, timeout, bug code)
            duree = (time.perf_counter() - start) * 1000
            err = ErreurClassification(
                code="INTERNAL_ERROR",
                message=f"Erreur interne : {type(e).__name__}: {e}",
                entree_brute=ticket,
            )
            logger.exception("Classification INTERNAL_ERROR | titre={}", ticket.titre)
            return ResultatExecution(succes=False, erreur=err, duree_ms=duree)


# Instance globale pour usage simple
runner = AgentRunner()