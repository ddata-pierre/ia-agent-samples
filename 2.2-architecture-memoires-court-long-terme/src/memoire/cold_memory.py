"""Mémoire long terme : profil, faits, historique — persistée en SQLite par user_id."""

import aiosqlite
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger

from .models import (
    ProfilUtilisateur, FaitUtilisateur, InteractionHistorique,
    TypeFait
)
from .config import settings


class ColdMemoryStore:
    """Couche d'accès données SQLite pour mémoire long terme."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.memory_db)
        self._init_db()

    def _init_db(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            # Profil utilisateur
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profils (
                    user_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    maj_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Faits extraits
            conn.execute("""
                CREATE TABLE IF NOT EXISTS faits (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    cle TEXT NOT NULL,
                    valeur TEXT NOT NULL,
                    confiance REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'extraction_auto',
                    actif INTEGER DEFAULT 1,
                    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    maj_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faits_user ON faits(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faits_cle ON faits(user_id, cle)")

            # Historique interactions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historique (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    agent_type TEXT,
                    entree TEXT,
                    sortie_resume TEXT,
                    succes INTEGER,
                    duree_ms INTEGER,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_histo_user ON historique(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_histo_session ON historique(session_id)")

    # --- Profil ---
    async def get_profil(self, user_id: str) -> Optional[ProfilUtilisateur]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data_json FROM profils WHERE user_id = ?", (user_id,)
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return ProfilUtilisateur.model_validate_json(row[0])
        return None

    async def upsert_profil(self, profil: ProfilUtilisateur):
        profil.maj_le = datetime.now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO profils (user_id, data_json, maj_le) VALUES (?, ?, ?)",
                (profil.user_id, profil.model_dump_json(), profil.maj_le),
            )
            await db.commit()

    # --- Faits ---
    async def get_faits(self, user_id: str, actifs_seulement: bool = True) -> List[FaitUtilisateur]:
        query = "SELECT * FROM faits WHERE user_id = ?"
        params = [user_id]
        if actifs_seulement:
            query += " AND actif = 1"
        query += " ORDER BY maj_le DESC LIMIT ?"
        params.append(settings.cold_max_facts_per_user)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [self._row_to_fait(r) for r in rows]

    async def upsert_fait(self, fait: FaitUtilisateur):
            fait.maj_le = datetime.now()
            async with aiosqlite.connect(self.db_path) as db:
                # Vérifie si un fait avec même user_id + cle existe déjà
                async with db.execute(
                    "SELECT id FROM faits WHERE user_id = ? AND cle = ?",
                    (fait.user_id, fait.cle)
                ) as cur:
                    row = await cur.fetchone()
            
                if row:
                    # Met à jour le fait existant (garde son id)
                    fait.id = row[0]
                    await db.execute("""UPDATE faits
                        SET type = ?, valeur = ?, confiance = ?, source = ?, actif = ?, maj_le = ?
                        WHERE id = ?""",
                        (fait.type.value, fait.valeur, fait.confiance, fait.source, int(fait.actif), fait.maj_le, fait.id))
                else:
                    # Insère nouveau fait
                    await db.execute("""INSERT INTO faits
                        (id, user_id, type, cle, valeur, confiance, source, actif, maj_le)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (fait.id, fait.user_id, fait.type.value, fait.cle, fait.valeur,
                          fait.confiance, fait.source, int(fait.actif), fait.maj_le))
                await db.commit()

    async def supprimer_fait(self, user_id: str, cle: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE faits SET actif = 0 WHERE user_id = ? AND cle = ?", (user_id, cle))
            await db.commit()

    def _row_to_fait(self, row) -> FaitUtilisateur:
        return FaitUtilisateur(
            id=row[0], user_id=row[1], type=TypeFait(row[2]), cle=row[3],
            valeur=row[4], confiance=row[5], source=row[6], actif=bool(row[7]),
            cree_le=row[8], maj_le=row[9],
        )

    # --- Historique ---
    async def ajouter_interaction(self, interaction: InteractionHistorique):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO historique
                (id, user_id, session_id, agent_type, entree, sortie_resume,
                 succes, duree_ms, tokens_in, tokens_out, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (interaction.id, interaction.user_id, interaction.session_id,
                  interaction.agent_type, interaction.entree, interaction.sortie_resume,
                  int(interaction.succes), interaction.duree_ms,
                  interaction.tokens_in, interaction.tokens_out, interaction.timestamp))
            # Purge ancien
            await db.execute("""
                DELETE FROM historique
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM historique WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
                )
            """, (interaction.user_id, interaction.user_id, settings.cold_max_history_per_user))
            await db.commit()

    async def get_historique_recent(self, user_id: str, limite: int = 10) -> List[InteractionHistorique]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM historique WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limite)
            ) as cur:
                rows = await cur.fetchall()
                return [self._row_to_historique(r) for r in rows]

    def _row_to_historique(self, row) -> InteractionHistorique:
        return InteractionHistorique(
            id=row[0], user_id=row[1], session_id=row[2], agent_type=row[3],
            entree=row[4], sortie_resume=row[5], succes=bool(row[6]),
            duree_ms=row[7], tokens_in=row[8], tokens_out=row[9], timestamp=row[10],
        )


class ColdMemoryManager:
    """Logique métier mémoire long terme : extraction faits, consultation profil."""

    def __init__(self, store: ColdMemoryStore = None):
        self.store = store or ColdMemoryStore()

    # --- Profil ---
    async def get_profil(self, user_id: str) -> ProfilUtilisateur:
        profil = await self.store.get_profil(user_id)
        if profil is None:
            profil = ProfilUtilisateur(user_id=user_id)
            await self.store.upsert_profil(profil)
        return profil

    async def maj_profil(self, user_id: str, **champs) -> ProfilUtilisateur:
        profil = await self.get_profil(user_id)
        for k, v in champs.items():
            if hasattr(profil, k):
                setattr(profil, k, v)
        await self.store.upsert_profil(profil)
        return profil

    # --- Faits ---
    async def get_faits_contextuels(self, user_id: str, cles: List[str] = None) -> List[FaitUtilisateur]:
        """Récupère faits pertinents (tous ou filtrés par clés)."""
        faits = await self.store.get_faits(user_id)
        if cles:
            faits = [f for f in faits if f.cle in cles]
        return faits

    async def extraire_et_stocker_faits(self, user_id: str, conversation: str, agent_type: str) -> List[FaitUtilisateur]:
            """
            Extrait des faits depuis une conversation (appel LLM recommandé).
            Version heuristique ici — en prod : prompt d'extraction structurée vers LLM.
            """
            faits_nouveaux = []

            # Heuristiques simples (remplacer par LLM en production)
            patterns = {
                "style.reponse": ["court", "courtement", "bref", "détail", "détaillé", "long"],
                "langue.preference": ["anglais", "english", "français", "french"],
                "client.principal": ["client", "société", "entreprise"],
            }

            conv_lower = conversation.lower()
            for cle, mots_cles in patterns.items():
                for mk in mots_cles:
                    if mk in conv_lower:
                        # Extraction valeur : phrase utilisateur contenant le mot-clé
                        idx = conv_lower.index(mk)
                        # Trouve le début de la ligne utilisateur
                        debut_ligne = conversation.rfind('\n', 0, idx) + 1
                        if debut_ligne <= 0:
                            debut_ligne = 0
                        # Trouve la fin de la ligne
                        fin_ligne = conversation.find('\n', idx)
                        if fin_ligne == -1:
                            fin_ligne = len(conversation)
                        ligne = conversation[debut_ligne:fin_ligne].strip()
                        # Nettoie le préfixe "Utilisateur: "
                        if ligne.startswith("Utilisateur: "):
                            ligne = ligne[13:]
                    
                        if not ligne:
                            ligne = conversation[max(0, idx-50):idx+100].strip()
                    
                        fait = FaitUtilisateur(
                            user_id=user_id,
                            type=TypeFait.PREFERENCE if "style" in cle or "langue" in cle else TypeFait.ENTITE,
                            cle=cle,
                            valeur=ligne,
                            confiance=0.7,
                            source=f"extraction_auto_{agent_type}",
                        )
                        await self.store.upsert_fait(fait)
                        faits_nouveaux.append(fait)
                        break  # Un seul fait par clé par extraction

            return faits_nouveaux

    async def get_contexte_injection(self, user_id: str, cles_demandees: List[str] = None) -> str:
        """Construit bloc contexte long terme pour injection dans prompt."""
        profil = await self.get_profil(user_id)
        faits = await self.get_faits_contextuels(user_id, cles_demandees)

        parties = ["[CONTEXTE UTILISATEUR (MÉMOIRE LONG TERME)]"]

        # Profil
        if profil.nom or profil.role or profil.service:
            p = []
            if profil.nom: p.append(f"Nom: {profil.nom}")
            if profil.role: p.append(f"Rôle: {profil.role}")
            if profil.service: p.append(f"Service: {profil.service}")
            if profil.style_reponse != "standard": p.append(f"Style préféré: {profil.style_reponse}")
            parties.append("Profil : " + " | ".join(p))

        # Faits
        if faits:
            parties.append("Faits connus :")
            for f in faits:
                if f.actif:
                    parties.append(f"  - {f.cle} = {f.valeur} (confiance: {f.confiance:.1f})")

        return "\n".join(parties) if len(parties) > 1 else ""

    # --- Historique ---
    async def logger_interaction(self, interaction: InteractionHistorique):
        await self.store.ajouter_interaction(interaction)

    async def get_stats_utilisateur(self, user_id: str) -> Dict[str, Any]:
        profil = await self.get_profil(user_id)
        faits = await self.store.get_faits(user_id)
        histo = await self.store.get_historique_recent(user_id, 5)

        return {
            "user_id": user_id,
            "profil_complet": bool(profil.nom and profil.role),
            "nb_faits": len(faits),
            "derniere_activite": histo[0].timestamp.isoformat() if histo else None,
            "taux_succes": sum(1 for h in histo if h.succes) / len(histo) if histo else 0,
        }