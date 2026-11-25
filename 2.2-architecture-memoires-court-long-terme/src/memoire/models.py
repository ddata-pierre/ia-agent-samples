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