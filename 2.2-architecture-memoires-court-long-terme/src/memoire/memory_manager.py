"""Orchestrateur mémoire : combine hot + cold, injection contextuelle sélective."""

from dataclasses import dataclass
from typing import Optional, List
from loguru import logger

from .hot_memory import HotMemoryManager, HotMemoryState
from .cold_memory import ColdMemoryManager
from .models import Message, RoleMessage, InteractionHistorique
from .config import settings


@dataclass
class ContexteInjection:
    """Bloc contexte final injecté dans le prompt de l'agent."""
    hot_context: str      # Mémoire court terme (toujours)
    cold_context: str     # Mémoire long terme (à la demande)
    user_id: str
    session_id: str


class MemoryManager:
    """Point d'entrée unique pour la mémoire de l'agent."""

    def __init__(self, hot: HotMemoryManager = None, cold: ColdMemoryManager = None):
        self.hot = hot or HotMemoryManager()
        self.cold = cold or ColdMemoryManager()

    async def demarrer_session(self, session_id: str, user_id: str) -> HotMemoryState:
        """Initialise ou reprend une session."""
        state = await self.hot.get_or_create(session_id, user_id)
        logger.info(f"Session {session_id} démarrée pour user {user_id}")
        return state

    async def ajouter_echange(self, state: HotMemoryState, user_msg: str, assistant_msg: str,
                              metadata: dict = None):
        """Enregistre un tour complet (user + assistant)."""
        self.hot.ajouter_message(state, RoleMessage.USER, user_msg, **(metadata or {}))
        self.hot.ajouter_message(state, RoleMessage.ASSISTANT, assistant_msg, **(metadata or {}))
        await self.hot.sauvegarder(state)

    async def construire_contexte_complet(self, state: HotMemoryState,
                                    inclure_long_terme: bool = True,
                                    cles_long_terme: List[str] = None) -> ContexteInjection:
        """Construit le contexte complet pour injection dans le prompt agent."""
        hot_ctx = self.hot.construire_contexte_prompt(state)

        cold_ctx = ""
        if inclure_long_terme:
            try:
                cold_ctx = await self.cold.get_contexte_injection(
                    state.user_id,
                    cles_long_terme
                )
            except Exception as e:
                logger.error(f"Erreur récupération mémoire long terme : {e}")
                cold_ctx = ""

        return ContexteInjection(
            hot_context=hot_ctx,
            cold_context=cold_ctx,
            user_id=state.user_id,
            session_id=state.session_id,
        )

    def formater_prompt_systeme(self, prompt_base: str, contexte: ContexteInjection) -> str:
        """Insère le contexte dans le prompt système de l'agent."""
        blocs = [prompt_base]

        if contexte.hot_context:
            blocs.append(f"\n=== MÉMOIRE COURT TERME ===\n{contexte.hot_context}")

        if contexte.cold_context:
            blocs.append(f"\n=== MÉMOIRE LONG TERME ===\n{contexte.cold_context}")

        blocs.append("\n=== INSTRUCTIONS ===")
        blocs.append("Utilise la mémoire court terme pour la continuité de conversation.")
        blocs.append("Utilise la mémoire long terme pour personnaliser (rôle, préférences, faits).")
        blocs.append("Ne RÉVÈLE PAS explicitement que tu as accès à ces mémoires sauf si l'utilisateur demande.")

        return "\n".join(blocs)

    async def extraire_faits_post_echange(self, state: HotMemoryState, agent_type: str):
        """Extraction faits long terme après un échange (à appeler en background)."""
        # Construit texte conversation récente
        recent_msgs = state.messages[-6:]  # 3 derniers tours
        conversation = "\n".join(m.to_prompt() for m in recent_msgs)

        await self.cold.extraire_et_stocker_faits(state.user_id, conversation, agent_type)

    async def logger_interaction(self, state: HotMemoryState, agent_type: str,
                                 entree: str, sortie: str, succes: bool,
                                 duree_ms: int, tokens_in: int, tokens_out: int):
        """Enregistre interaction dans historique long terme."""
        interaction = InteractionHistorique(
            user_id=state.user_id,
            session_id=state.session_id,
            agent_type=agent_type,
            entree=entree[:500],  # Tronque pour stockage
            sortie_resume=sortie[:200],
            succes=succes,
            duree_ms=duree_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        await self.cold.logger_interaction(interaction)

    async def get_stats_utilisateur(self, user_id: str) -> dict:
        return await self.cold.get_stats_utilisateur(user_id)