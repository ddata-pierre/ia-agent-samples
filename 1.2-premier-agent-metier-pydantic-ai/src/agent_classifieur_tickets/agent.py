"""Agent Pydantic AI : définition, outils, retry, validation."""

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel

from .models import TicketEntree, TicketSortie, CategorieTicket, NiveauUrgence
from .prompts import PROMPT_SYSTEME_CLASSIFIEUR, PROMPT_UTILISATEUR_TEMPLATE


# Configuration modèle local via Ollama (compatible OpenAI API)
# Ollama expose /v1 compatible OpenAI sur le port 11434
model = OpenAIChatModel(
    model_name="qwen2.5:3b-instruct-q4_k_m",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

# Agent typé : entrée = TicketEntree, sortie = TicketSortie
# Le système de validation Pydantic s'applique AUTOMATIQUEMENT à la sortie
agent = Agent(
    model=model,
    output_type=TicketSortie,           # Validation sortie automatique
    system_prompt=PROMPT_SYSTEME_CLASSIFIEUR,
    retries=3,                          # Retry global (peut être affiné par outil)
)


# --- Outils (optionnels) : ici on n'en a pas besoin, le prompt suffit ---
# Mais Pydantic AI permet d'ajouter des tools avec @agent.tool
# Exemple pour plus tard (RAG, API externe) :
#
# @agent.tool
# async def chercher_base_connaissance(ctx: RunContext, query: str) -> str:
#     \"\"\"Recherche dans la base de connaissance interne.\"\"\"
#     # Implémentation RAG ici
#     return "Résultat de recherche..."


# --- Retry personnalisé par type d'erreur ---
@agent.output_validator
async def valider_sortie_metier(ctx: RunContext, result: TicketSortie) -> TicketSortie:
    """
    Validateur métier : règles qui vont au-delà du typage Pydantic.
    Relance l'agent (retry) si règles violées.
    """
    from pydantic_ai.exceptions import ModelRetry
    # Règle : confiance trop faible → retry
    if result.confidence < 0.5:
        raise ModelRetry(f"Confiance trop faible ({result.confidence:.2f}) — ticket ambigu")

    # Règle : cohérence urgence/catégorie
    if result.categorie == CategorieTicket.ACCES and result.urgence == NiveauUrgence.FAIBLE:
        raise ModelRetry("Catégorie ACCES incompatible avec urgence FAIBLE")

    # Règle : escalade cohérente
    if result.necessite_escalade != (result.urgence == NiveauUrgence.ELEVEE or result.categorie == CategorieTicket.ACCES):
        raise ModelRetry("Champ necessite_escalade incohérent avec urgence/catégorie")

    return result


# --- Helper pour construire le prompt utilisateur ---
def construire_prompt_utilisateur(ticket: TicketEntree) -> str:
    return PROMPT_UTILISATEUR_TEMPLATE.format(
        titre=ticket.titre,
        description=ticket.description,
        utilisateur_id=ticket.utilisateur_id or "inconnu",
        systeme_source=ticket.systeme_source or "inconnu",
    )