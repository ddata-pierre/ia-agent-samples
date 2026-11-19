"""Modèles Pydantic = contrat strict de l'agent classifieur tickets IT."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class CategorieTicket(str, Enum):
    """Catégories métier normalisées — pas de texte libre."""
    RESEAU = "RESEAU"
    MATERIEL = "MATERIEL"
    LOGICIEL = "LOGICIEL"
    ACCES = "ACCES"
    AUTRE = "AUTRE"


class NiveauUrgence(str, Enum):
    """Urgence normalisée — évite 'urgent', 'critique', 'P1' mélangés."""
    FAIBLE = "FAIBLE"
    MOYENNE = "MOYENNE"
    ELEVEE = "ELEVEE"


class TicketEntree(BaseModel):
    """Entrée brute : ce que l'utilisateur ou le système nous donne."""
    titre: str = Field(..., min_length=3, max_length=200, description="Titre du ticket")
    description: str = Field(..., min_length=10, max_length=5000, description="Description détaillée")
    utilisateur_id: Optional[str] = Field(None, description="Identifiant utilisateur (optionnel)")
    systeme_source: Optional[str] = Field(None, description="Système d'origine : email, portal, monitoring...")

    @field_validator("titre", "description")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class TicketSortie(BaseModel):
    """Sortie structurée : ce que l'agent DOIT produire, validé automatiquement."""
    categorie: CategorieTicket = Field(..., description="Catégorie métier normalisée")
    urgence: NiveauUrgence = Field(..., description="Niveau d'urgence normalisé")
    resume: str = Field(..., min_length=5, max_length=100, description="Résumé exécutif ≤ 100 caractères")
    mots_cles: list[str] = Field(default_factory=list, max_length=10, description="Mots-clés techniques extraits")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance du modèle 0.0-1.0")
    necessite_escalade: bool = Field(default=False, description="True si urgence ELEVEE ou catégorie ACCES")

    @model_validator(mode="after")
    def compute_escalade(self) -> "TicketSortie":
        """Calcule automatiquement necessite_escalade selon les règles métier."""
        self.necessite_escalade = (
            self.urgence == NiveauUrgence.ELEVEE or 
            self.categorie == CategorieTicket.ACCES
        )
        return self

    @field_validator("resume")
    @classmethod
    def resume_pas_trop_long(cls, v: str) -> str:
        if len(v) > 100:
            raise ValueError("Résumé dépasse 100 caractères")
        return v


class ErreurClassification(BaseModel):
    """Structure d'erreur normalisée pour logging/monitoring."""
    code: str
    message: str
    entree_brute: Optional[TicketEntree] = None
    details: Optional[str] = None