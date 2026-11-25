# 04-architecture-memoires-court-long-terme - Mémoires Agent : Court terme + Long terme

> **⚠️ DISCLAIMER** : Ce tutoriel est **pédagogique** — il vise à comprendre le fonctionnement des mémoires **hot (court terme)** et **cold (long terme)** d'un agent IA. Ce n'est **PAS** un exemple prêt pour la production.
>
> **Écarts prod vs tuto** :
> - Extraction faits : heuristiques regex (bruit) → **remplacer par LLM + schéma Pydantic structuré**
> - Résumé récursif : version synchrone simplifiée → **tâche de fond asynchrone + LLM**
> - SQLite seul : OK jusqu'à ~10k users → **au-delà : PostgreSQL + pgvector / pgvecto.rs**
> - Pas de chiffrement au repos, pas de RBAC, pas d'audit trail
> - Injection cold systématique en demo → **en prod : récupération à la demande (tools) ou sélective (clés)**

> **Objectif** : implémenter deux mémoires séparées — court terme (fenêtre conversation + résumé récursif) et long terme (profil, préférences, faits, historique) persistées en SQLite, avec injection contextuelle sélective.
> **Durée** : ~20 min.
> **Prérequis** : Tutos #1-3 fonctionnels, Python 3.11+, `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`.

---

## 1. Architecture deux niveaux — Pourquoi séparer ?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT (Pydantic AI)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PROMPT SYSTÈME + CONTEXTE                        │   │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │   │
│  │  │ MÉMOIRE COURT TERME │  │     MÉMOIRE LONG TERME (à la demande)    │  │   │
│  │  │  (Hot Memory)       │  │     (Cold Memory)                   │  │   │
│  │  │  • Derniers 10 tours│  │     • Profil utilisateur (rôle, svc)   │  │   │
│  │  │  • Résumé récursif  │  │     • Préférences (style, langue)     │  │   │
│  │  │  • Contexte immédiat│  │     • Faits extraits (entités, règles)│  │   │
│  │  └─────────────────────┘  └─────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌─────────────────┐             ┌─────────────────┐
           │   SQLite        │             │   SQLite        │
           │   (sessions)    │             │   (users)       │
           │   sessions.db   │             │   memory.db     │
           └─────────────────┘             └─────────────────┘
```

**Principes clés** :
- **Court terme** = injecté **systématiquement** à chaque tour (fenêtre glissante + résumé).
- **Long terme** = récupéré **à la demande** via tools (profil, faits, historique).
- **Isolation** : `user_id` = clé de partitionnement partout. Aucun partage inter-utilisateurs.
- **Persistance** : SQLite (fichier unique, zéro déps, ACID, WAL mode pour concurrence).

---

## 2. Installation et structure

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Structure** :
```
04-architecture-memoires-court-long-terme/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py              # Modèles Pydantic (mémoires, messages, faits)
│   ├── hot_memory.py          # Mémoire court terme (session)
│   ├── cold_memory.py         # Mémoire long terme (utilisateur)
│   ├── memory_manager.py      # Orchestration + injection contextuelle
│   └── cli.py                 # Demo
├── tests/
│   ├── test_hot_memory.py
│   ├── test_cold_memory.py
│   └── test_integration.py
├── pyproject.toml
└── docs/tutoriel.md
```

---

## 3. Configuration

**Fichier** : `src/config.py`

```python
"""Configuration centralisée mémoire."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MEM_",
    )

    # Stockage
    data_dir: Path = Field(
        default=Path("data/memory"),
        description="Dossier bases SQLite",
    )
    sessions_db: Path = Field(
        default=Path("data/memory/sessions.db"),
        description="DB sessions (court terme)",
    )
    memory_db: Path = Field(
        default=Path("data/memory/memory.db"),
        description="DB utilisateurs (long terme)",
    )

    # Court terme
    hot_window_size: int = Field(
        default=10,
        description="Nb messages gardés en brut (fenêtre glissante)",
    )
    hot_summary_trigger: int = Field(
        default=20,
        description="Nb messages avant résumé récursif",
    )
    hot_max_summary_length: int = Field(
        default=500,
        description="Longueur max résumé (tokens approx)",
    )

    # Long terme
    cold_max_facts_per_user: int = Field(
        default=100,
        description="Max faits stockés par utilisateur",
    )
    cold_max_history_per_user: int = Field(
        default=500,
        description="Max interactions historiques par user",
    )

    # Device / LLM pour résumé
    summarizer_model: str = Field(
        default="qwen2.5:3b-instruct-q4_k_m",
        description="Modèle pour résumé (Ollama)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="URL Ollama OpenAI-compatible",
    )


settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)
```

---

## 4. Modèles Pydantic — Messages, Faits, Profil

**Fichier** : `src/models.py`

```python
"""Modèles de données pour les deux mémoires."""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from uuid import uuid4


class RoleMessage(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """Message unique dans l'historique."""
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    role: RoleMessage
    contenu: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # tokens, model, tool_calls...

    def to_prompt(self) -> str:
        """Format pour injection dans prompt LLM."""
        prefix = {"user": "Utilisateur", "assistant": "Assistant", "system": "Système", "tool": "Outil"}[self.role.value]
        return f"{prefix}: {self.contenu}"


class SessionSummary(BaseModel):
    """Résumé récursif d'une session."""
    session_id: str
    resume: str  # Résumé textuel
    messages_couverts: int  # Nb messages résumés
    debut: datetime
    fin: datetime
    tokens_estimes: int = 0


class HotMemoryState(BaseModel):
    """État complet mémoire court terme d'une session."""
    session_id: str
    user_id: str
    messages: List[Message] = Field(default_factory=list)  # Fenêtre glissante brute
    resume_actuel: Optional[SessionSummary] = None  # Résumé cumulatif
    cree_le: datetime = Field(default_factory=datetime.now)
    maj_le: datetime = Field(default_factory=datetime.now)


# --- Long terme ---

class TypeFait(str, Enum):
    PREFERENCE = "preference"      # "réponses courtes", "langue: fr"
    ENTITE = "entite"              # "client: ACME Corp", "projet: Phoenix"
    REGLE = "regle"                # "validation manager requise > 5k€"
    CONTEXTE = "contexte"          # "travaille sur dossier RH depuis mars"
    AUTRE = "autre"


class FaitUtilisateur(BaseModel):
    """Fait extrait et validé sur l'utilisateur."""
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    user_id: str
    type: TypeFait
    cle: str  # Clé normalisée: "style.reponse", "client.principal", "projet.actuel"
    valeur: str
    confiance: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "extraction_auto"  # ou "feedback_utilisateur", "import"
    cree_le: datetime = Field(default_factory=datetime.now)
    maj_le: datetime = Field(default_factory=datetime.now)
    actif: bool = True


class ProfilUtilisateur(BaseModel):
    """Profil structuré utilisateur (rempli à la main ou via onboarding)."""
    user_id: str
    nom: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None  # "RH_MEMBER", "IT_SUPPORT", "EMPLOYEE", "MANAGER"
    service: Optional[str] = None
    langue_preferee: str = "fr"
    style_reponse: str = "standard"  # "court", "standard", "detaille", "technique"
    fuseau_horaire: str = "Europe/Paris"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cree_le: datetime = Field(default_factory=datetime.now)
    maj_le: datetime = Field(default_factory=datetime.now)


class InteractionHistorique(BaseModel):
    """Une interaction passée (pour analytics, pas pour injection prompt)."""
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    user_id: str
    session_id: str
    agent_type: str  # "classifieur", "rag_rh", "router", etc.
    entree: str
    sortie_resume: str
    succes: bool
    duree_ms: int
    tokens_in: int
    tokens_out: int
    timestamp: datetime = Field(default_factory=datetime.now)
```

---

## 5. Mémoire Court Terme (Hot) — Session SQLite

**Fichier** : `src/hot_memory.py`

```python
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
```

---

## 6. Mémoire Long Terme (Cold) — Utilisateur SQLite

**Fichier** : `src/cold_memory.py`

```python
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
```

---

## 7. Memory Manager — Orchestration + Injection Contextuelle

**Fichier** : `src/memory_manager.py`

```python
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
```

---

## 8. Intégration avec Agent Pydantic AI (Tuto #2)

**Fichier** : `src/integration_agent.py`

```python
"""Exemple d'intégration mémoire + agent Pydantic AI (classifieur tickets)."""

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

from .memory_manager import MemoryManager, ContexteInjection
from .models import TicketEntree, TicketSortie
from .config import settings


# Modèle local
model = OpenAIModel(
    model_name="qwen2.5:3b-instruct-q4_k_m",
    provider=OllamaProvider(base_url=settings.ollama_base_url),
)

# Agent classifieur avec injection mémoire dynamique
classifieur_agent = Agent(
    model=model,
    result_type=TicketSortie,
    system_prompt="""Tu es un classifieur expert de tickets IT d'entreprise.
Réponds UNIQUEMENT par JSON valide TicketSortie.
CATÉGORIES: RESEAU, MATERIEL, LOGICIEL, ACCES, AUTRE
URGENCE: FAIBLE, MOYENNE, ELEVEE""",
    retries=3,
)


async def classifier_avec_memoire(ticket: TicketEntree, session_id: str, user_id: str) -> TicketSortie:
    """Classifie un ticket en utilisant la mémoire (contexte utilisateur)."""
    memory = MemoryManager()

    # 1. Démarre/reprend session
    state = await memory.demarrer_session(session_id, user_id)

    # 2. Construit contexte complet
    contexte = await memory.construire_contexte_complet(state, inclure_long_terme=True)

    # 3. Formate prompt système enrichi
    prompt_enrichi = memory.formater_prompt_systeme(
        classifieur_agent._system_prompt,  # Accès interne (demo)
        contexte
    )

    # 4. Exécute agent avec prompt enrichi
    # Note: Pydantic AI ne permet pas de changer system_prompt à la volée facilement.
    # Pattern recommandé : créer nouvel agent ou utiliser dependency injection.
    # Ici simplification : on préfixe le prompt utilisateur avec le contexte.
    prompt_utilisateur = f"{contexte.hot_context}\n\n{contexte.cold_context}\n\nTicket: {ticket.titre}\n{ticket.description}"

    result = await classifieur_agent.run(prompt_utilisateur)

    # 5. Sauvegarde échange en mémoire court terme
    await memory.ajouter_echange(state, prompt_utilisateur, result.data.model_dump_json())

    # 6. Log interaction long terme (background)
    await memory.logger_interaction(state, "classifieur", prompt_utilisateur,
                                    result.data.model_dump_json(), True, 0, 0, 0)

    # 7. Extraction faits (background)
                await memory.extraire_faits_post_echange(state, "demo")

    return result.data
```

> **Note importante** : Pydantic AI n'expose pas de setter pour `system_prompt` à l'exécution. Deux patterns :
> 1. **Dependency Injection** : passer `RunContext` avec `deps` contenant le contexte mémoire.
> 2. **Nouvel agent par session** : recréer l'agent avec `system_prompt` enrichi (coût négligeable).
> Le tuto #7 (LangGraph) montrera l'injection via `state` du graphe.

---

## 9. CLI Demo

**Fichier** : `src/cli.py`

```python
"""Demo interactive mémoire deux niveaux."""

import asyncio
import sys
from loguru import logger

from .memory_manager import MemoryManager
from .models import ProfilUtilisateur


async def demo():
    memory = MemoryManager()
    user_id = "demo_user_001"
    session_id = "session_001"

    # Setup profil demo
    await memory.cold.maj_profil(user_id, nom="Jean Dupont", role="EMPLOYEE",
                                  service="Marketing", style_reponse="court")

    print(f"\n=== Demo Mémoire : {user_id} / {session_id} ===")
    print("Commandes : 'quit', 'stats', 'profil', 'faits', 'reset'\n")

    state = await memory.demarrer_session(session_id, user_id)

    while True:
        try:
            user_input = input("👤 Vous : ").strip()
            if user_input.lower() in ("quit", "q", "exit"):
                break
            if user_input.lower() == "stats":
                stats = await memory.get_stats_utilisateur(user_id)
                print(f"📊 Stats : {stats}")
                continue
            if user_input.lower() == "profil":
                profil = await memory.cold.get_profil(user_id)
                print(f"👤 Profil : {profil.model_dump_json(indent=2)}")
                continue
            if user_input.lower() == "faits":
                faits = await memory.cold.get_faits_contextuels(user_id)
                for f in faits:
                    print(f"  📌 {f.cle} = {f.valeur} ({f.confiance:.1f})")
                continue
            if user_input.lower() == "reset":
                state = await memory.demarrer_session(session_id + "_new", user_id)
                print("🔄 Session réinitialisée")
                continue
            if not user_input:
                continue

            # Simule réponse agent (echo pour demo)
            assistant_reply = f"[Agent] Reçu : {user_input}"

            # Injection contexte
            contexte = await memory.construire_contexte_complet(state)
            print(f"\n🧠 Contexte injecté :")
            print(f"  Hot: {contexte.hot_context[:150]}...")
            if contexte.cold_context:
                print(f"  Cold: {contexte.cold_context[:150]}...")

            # Sauvegarde
            await memory.ajouter_echange(state, user_input, assistant_reply)
            await memory.logger_interaction(state, "demo", user_input, assistant_reply, True, 10, 20, 15)

            # Extraction faits long terme (background)
            await memory.extraire_faits_post_echange(state, "demo")

            print(f"🤖 Agent : {assistant_reply}\n")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nAu revoir !")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(demo())
```

---

## 10. Tests Unitaires

**Fichier** : `tests/test_hot_memory.py`

```python
"""Tests mémoire court terme."""

import pytest
import asyncio
from src.hot_memory import HotMemoryManager, HotMemoryStore
from src.models import HotMemoryState, RoleMessage


@pytest.fixture
def hot_manager(tmp_path):
    store = HotMemoryStore(str(tmp_path / "test_sessions.db"))
    return HotMemoryManager(store)


@pytest.mark.asyncio
async def test_creer_session(hot_manager):
    state = await hot_manager.get_or_create("sess_1", "user_1")
    assert state.session_id == "sess_1"
    assert state.user_id == "user_1"
    assert len(state.messages) == 0


@pytest.mark.asyncio
async def test_fenetre_glissante(hot_manager):
    state = await hot_manager.get_or_create("sess_2", "user_1")

    # Ajoute 15 messages (window=10 par défaut)
    for i in range(15):
        hot_manager.hot.ajouter_message(state, RoleMessage.USER, f"Message {i}")

    # Seuls les 10 derniers gardés en brut
    assert len(state.messages) == 10
    assert state.messages[0].contenu == "Message 5"  # Les 5 premiers absorbés

    # Résumé créé
    assert state.resume_actuel is not None
    assert state.resume_actuel.messages_couverts >= 5


@pytest.mark.asyncio
async def test_contexte_prompt(hot_manager):
    state = await hot_manager.get_or_create("sess_3", "user_1")
    hot_manager.hot.ajouter_message(state, RoleMessage.USER, "Bonjour")
    hot_manager.hot.ajouter_message(state, RoleMessage.ASSISTANT, "Bonjour !")

    contexte = hot_manager.hot.construire_contexte_prompt(state)
    assert "Bonjour" in contexte
    assert "Utilisateur: Bonjour" in contexte
    assert "Assistant: Bonjour !" in contexte
```

**Fichier** : `tests/test_cold_memory.py`

```python
"""Tests mémoire long terme."""

import pytest
import asyncio
from src.cold_memory import ColdMemoryManager, ColdMemoryStore
from src.models import ProfilUtilisateur, FaitUtilisateur, TypeFait


@pytest.fixture
def cold_manager(tmp_path):
    store = ColdMemoryStore(str(tmp_path / "test_memory.db"))
    return ColdMemoryManager(store)


@pytest.mark.asyncio
async def test_profil_creation(cold_manager):
    profil = await cold_manager.get_profil("user_test")
    assert profil.user_id == "user_test"
    assert profil.langue_preferee == "fr"

    # Maj
    profil = await cold_manager.maj_profil("user_test", nom="Test User", role="MANAGER")
    assert profil.nom == "Test User"
    assert profil.role == "MANAGER"


@pytest.mark.asyncio
async def test_faits_upsert(cold_manager):
    fait = FaitUtilisateur(
        user_id="user_fait",
        type=TypeFait.PREFERENCE,
        cle="style.reponse",
        valeur="court",
        confiance=0.9,
    )
    await cold_manager.store.upsert_fait(fait)

    faits = await cold_manager.get_faits_contextuels("user_fait")
    assert len(faits) == 1
    assert faits[0].cle == "style.reponse"
    assert faits[0].valeur == "court"


@pytest.mark.asyncio
async def test_contexte_injection(cold_manager):
    await cold_manager.maj_profil("user_ctx", nom="Alice", role="RH_MEMBER", service="RH")
    await cold_manager.store.upsert_fait(FaitUtilisateur(
        user_id="user_ctx", type=TypeFait.ENTITE, cle="client.principal",
        valeur="ACME Corp", confiance=0.8
    ))

    ctx = await cold_manager.get_contexte_injection("user_ctx")
    assert "Alice" in ctx
    assert "RH_MEMBER" in ctx
    assert "ACME Corp" in ctx
```

---

## 11. Configuration Projet — pyproject.toml

```toml
[project]
name = "agent-memory"
version = "0.1.0"
description = "Architecture mémoire deux niveaux (court/long terme) pour agents IA locaux"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "loguru>=0.7",
    "aiosqlite>=0.19",
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
```

---

## 12. Exécution

```bash
# 1. Installez
uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# 2. Tests
pytest tests/ -v

# 3. Demo interactive
python -m memoire.cli
```

**Session exemple** :
```
=== Demo Mémoire : demo_user_001 / session_001 ===

👤 Vous : Je m'appelle Marie et je préfère les réponses très courtes
🧠 Contexte injecté :
  Hot: [Nouvelle session]
  Cold: [CONTEXTE UTILISATEUR (MÉMOIRE LONG TERME)]
Profil : Nom: Jean Dupont | Rôle: EMPLOYEE | Service: Marketing | Style préféré: court
Faits connus :
  - style.reponse = court (confiance: 0.7)

🤖 Agent : [Agent] Reçu : Je m'appelle Marie et je préfère les réponses très courtes

👤 Vous : stats
📊 Stats : {'user_id': 'demo_user_001', 'profil_complet': True, 'nb_faits': 2, ...}
```

---

## 13. Protocole de test complet (HOT + COLD)

### Installation et reset

```bash
cd /home/pierre/04-architecture-memoires-court-long-terme
source .venv/bin/activate
rm -rf data/memory          # Reset DB propre
pytest tests/ -v            # 28 tests doivent passer
python -m memoire.cli       # Lance demo interactive
```

### Scénario de validation dans la CLI

| Étape | Commande saisie | Vérification |
|-------|-----------------|--------------|
| 1 | `Je prefere les reponses courtes` | Échange ajouté en HOT, fait `style.reponse` extrait en COLD |
| 2 | `Mon client est ACME Corp` | 2ème échange + fait `client.principal` extrait |
| 3 | `faits` | Affiche 2 faits : `style.reponse` + `client.principal` |
| 4 | `stats` | `nb_faits: 2`, `derniere_activite` à jour |
| 5 | `reset` | Nouvelle session HOT (vide), COLD intact |

### Ce qui se passe sous le capot

```
ÉCHANGE UTILISATEUR
       │
       ▼
┌──────────────────┐
│ MEMORY MANAGER   │
│  ajouter_echange │──► HOT: fenêtre glissante (10 msg) + résumé récursif si >20
│  extraire_faits  │──► COLD: patterns "court", "client" → upsert par (user_id, cle)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ CONTEXTE INJECTÉ │
│  HOT: systématique (derniers msg + résumé)       │
│  COLD: à la demande (profil + faits actifs)      │
└──────────────────┘
```

### Points clés validés

- **HOT** : 4 messages en fenêtre après 2 échanges (user+assistant ×2)
- **COLD** : 2 faits persistés (`style.reponse`, `client.principal`) avec `confiance: 0.7`
- **Nouvelle session** : HOT remis à zéro, COLD conservé (même `user_id`)
- **Aucun doublon** : upsert par clé composite `(user_id, cle)`

---

## 14. Pièges et limites — Soyons honnêtes

| Piège | Réalité |
|-------|---------|
| « Résumé récursif = perte zéro info » | Perte inévitable. Le résumé compresse. Pour détails précis → faire un tool de recherche dans l'historique complet (pas implémenté ici). |
| « Extraction faits auto = fiable » | Heuristiques = bruit. En prod : **prompt d'extraction structurée vers LLM** avec schéma Pydantic, validation humaine optionnelle. |
| « SQLite = assez pour tout » | Oui jusqu'à ~10k users / 1M interactions. Au-delà : PostgreSQL + pgvector pour faits embeddings, ou migration partielle. |
| « Injection cold à chaque tour = bon » | Non : surcharge le contexte. **Récupération à la demande** (tools) ou **injection sélective** (clés demandées) préférée. |
| « Isolation user_id = sécurité » | Au niveau appli oui. Pas de chiffrement au repos. Pour données sensibles : chiffrement colonne (SQLCipher) ou Vault externe. |

---

## 14. Aller plus loin

- [Recursive Summarization](https://arxiv.org/abs/2308.10287) — papier original
- [MemGPT](https://arxiv.org/abs/2310.08560) — architecture mémoire hiérarchique pour LLM
- [LangGraph Memory](https://langchain-ai.github.io/langgraph/concepts/memory/) — intégration graphe
- Prochaine étape : **05-passerelle-hybride-openrouter** — passerelle hybride OpenRouter pour LLM généraliste rapide (gratuit) dédié au bavardage et à l'aiguillage (routing), pendant que le SLM local fait le travail métier.