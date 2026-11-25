"""Mémoire court terme : fenêtre glissante + résumé récursif, persistée en SQLite."""

import aiosqlite
import json
from datetime import datetime
from typing import List, Optional
from loguru import logger

from .models import HotMemoryState, Message, SessionSummary, RoleMessage
from .config import settings


class HotMemoryStore:
    """Gestionnaire mémoire court terme (session)."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.sessions_db)
        self._init_db()

    def _init_db(self):
        """Crée tables si absent."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    maj_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

    async def charger(self, session_id: str) -> Optional[HotMemoryState]:
        """Charge état session depuis DB."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return HotMemoryState.model_validate_json(row[0])
        return None

    async def sauvegarder(self, state: HotMemoryState):
        """Sauvegarde état session (upsert)."""
        state.maj_le = datetime.now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO sessions (session_id, user_id, state_json, maj_le)
                   VALUES (?, ?, ?, ?)""",
                (state.session_id, state.user_id, state.model_dump_json(), state.maj_le),
            )
            await db.commit()

    async def supprimer(self, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()


class HotMemoryManager:
    """Logique métier mémoire court terme : fenêtre + résumé récursif."""

    def __init__(self, store: HotMemoryStore = None):
        self.store = store or HotMemoryStore()

    async def get_or_create(self, session_id: str, user_id: str) -> HotMemoryState:
        state = await self.store.charger(session_id)
        if state is None:
            state = HotMemoryState(session_id=session_id, user_id=user_id)
            await self.store.sauvegarder(state)
        return state

    def ajouter_message(self, state: HotMemoryState, role: RoleMessage, contenu: str, **metadata):
        """Ajoute message + gère fenêtre glissante + déclenche résumé si besoin."""
        msg = Message(role=role, contenu=contenu, metadata=metadata)
        state.messages.append(msg)

        # Fenêtre glissante : garde seulement les N derniers messages bruts
        if len(state.messages) > settings.hot_window_size:
            # Les messages qui sortent de la fenêtre sont "absorbés" dans le résumé
            messages_a_resumer = state.messages[:-settings.hot_window_size]
            state.messages = state.messages[-settings.hot_window_size:]
            self._mettre_a_jour_resume(state, messages_a_resumer)

        # Déclencheur résumé périodique (tous les N messages totaux)
        total_messages = (state.resume_actuel.messages_couverts if state.resume_actuel else 0) + len(state.messages)
        if total_messages >= settings.hot_summary_trigger and total_messages % settings.hot_summary_trigger == 0:
            # Résumé complet de la session
            self._mettre_a_jour_resume(state, state.messages[:-settings.hot_window_size])

    def _mettre_a_jour_resume(self, state: HotMemoryState, nouveaux_messages: List[Message]):
        """Met à jour le résumé récursif (appel LLM asynchrone recommandé).
        Ici version synchrone simplifiée pour demo — en prod : tâche de fond."""
        if not nouveaux_messages:
            return

        # Construit texte à résumer
        if state.resume_actuel:
            texte_base = f"Résumé précédent : {state.resume_actuel.resume}\n\nNouveaux messages :\n"
        else:
            texte_base = "Début de conversation :\n"

        texte_base += "\n".join(m.to_prompt() for m in nouveaux_messages)

        # TODO: Appel LLM réel pour résumé (voir _generer_resume_llm ci-dessous)
        # Pour demo : résumé heuristique simple
        resume_texte = self._resume_heuristique(nouveaux_messages, state.resume_actuel)

        state.resume_actuel = SessionSummary(
            session_id=state.session_id,
            resume=resume_texte,
            messages_couverts=(state.resume_actuel.messages_couverts if state.resume_actuel else 0) + len(nouveaux_messages),
            debut=nouveaux_messages[0].timestamp,
            fin=nouveaux_messages[-1].timestamp,
            tokens_estimes=int(len(resume_texte.split()) * 1.3),
        )
        logger.debug(f"Résumé mis à jour session {state.session_id}: {len(resume_texte)} chars")

    def _resume_heuristique(self, messages: List[Message], resume_precedent: Optional[SessionSummary]) -> str:
        """Résumé heuristique sans LLM (fallback). En prod → appel LLM."""
        sujets = set()
        for m in messages:
            if m.role == RoleMessage.USER:
                # Extraction mots-clés simple
                mots = [w for w in m.contenu.lower().split() if len(w) > 4]
                sujets.update(mots[:5])

        base = resume_precedent.resume if resume_precedent else "Conversation initiée."
        return f"{base} Sujets abordés : {', '.join(list(sujets)[:10])}. {len(messages)} échanges."

    async def _generer_resume_llm(self, texte: str) -> str:
        """Appel réel au LLM local pour résumé (à implémenter avec client Ollama/OpenAI)."""
        # Exemple avec client OpenAI-compatible :
        # from openai import AsyncOpenAI
        # client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
        # resp = await client.chat.completions.create(
        #     model=settings.summarizer_model,
        #     messages=[{"role": "system", "content": "Résume en 3 phrases max, français."},
        #               {"role": "user", "content": texte}],
        #     max_tokens=200, temperature=0.3
        # )
        # return resp.choices[0].message.content
        return self._resume_heuristique([], None)  # Placeholder

    def construire_contexte_prompt(self, state: HotMemoryState) -> str:
        """Construit le bloc contexte à injecter dans le prompt système."""
        parties = []

        if state.resume_actuel:
            parties.append(f"[RÉSUMÉ SESSION] {state.resume_actuel.resume}")

        if state.messages:
            parties.append("[DERNIERS ÉCHANGES]")
            parties.extend(m.to_prompt() for m in state.messages)

        return "\n".join(parties) if parties else "[Nouvelle session]"

    async def sauvegarder(self, state: HotMemoryState):
        await self.store.sauvegarder(state)

    async def nettoyer_sessions_anciennes(self, jours: int = 30):
        """Maintenance : supprime sessions > N jours."""
        import sqlite3
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=jours)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE maj_le < ?", (cutoff,))
            conn.commit()