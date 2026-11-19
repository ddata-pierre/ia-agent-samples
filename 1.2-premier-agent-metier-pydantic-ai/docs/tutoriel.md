# 02-premier-agent-metier-pydantic-ai - Agent typé avec Pydantic AI

> **Objectif** : créer un agent métier (classifieur tickets IT) avec entrées/sorties Pydantic validées, retry automatique, logging structuré.
> **Durée** : ~20 min.
> **Prérequis** : Tuto #1 fonctionnel (Ollama + qwen2.5:3b sur localhost:11434), Python 3.11+, `pip install pydantic-ai ollama pydantic-settings loguru`.

---

## 1. Pourquoi Pydantic AI pour un agent métier ?

| Problème sans typage | Solution Pydantic AI |
|----------------------|----------------------|
| Prompt en string, parsing `json.loads` hasardeux | Schéma Pydantic = contrat explicite, validation auto |
| Erreur silencieuse → donnée corrompue en aval | `ValidationError` levée, retry/fallback gérés |
| Retry manuel, logique dispersée | `@agent.output_validator` décorateur, politique centralisée |
| Pas de visibilité tokens/historique | `result.usage()`, `result.all_messages()` natifs |
| Test unitaire ? « Marche sur ma machine » | Agent mockable, déterministe (seed), testable isolément |

> **Architecture** : Pydantic AI n'est **pas** un framework d'orchestration (pas de graphe, pas de multi-agents). C'est une **couche de typage et validation** au-dessus de n'importe quel LLM (Ollama, OpenAI, Anthropic, etc.).

---

## 2. Installation et configuration

```bash
# Avec uv (plus rapide pour tout : venv + install)
uv venv .venv
source .venv/bin/activate

# Installation dépendances + editable
uv pip install -e ".[dev]"
```

**Structure du projet** :
```
02-premier-agent-metier-pydantic-ai/
├── src/
│   └── agent_classifieur_tickets/
│       ├── __init__.py
│       ├── models.py          # Modèles Pydantic (entrée/sortie)
│       ├── agent.py           # Définition de l'agent Pydantic AI
│       ├── prompts.py         # Prompts système (séparés pour lisibilité)
│       ├── runner.py          # Exécution + logging + retry
│       └── main.py            # CLI demo
├── tests/
│   └── test_agent.py
├── pyproject.toml
├── .gitignore
└── docs/tutoriel.md       # Ce fichier
```

> **Note** : Le code source est dans `src/agent_classifieur_tickets/` (package installé en editable via `pip install -e .`).

---

## 3. Modèles Pydantic — Contrat entrée/sortie

**Fichier** : `src/agent_classifieur_tickets/models.py`

```python
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
```

**Points clés** :
- `Enum` pour catégories/urgence = **valeurs fermées**, pas d'hallucination « CRITIQUE » ou « URGENT ».
- `Field(..., ge=0.0, le=1.0)` = **contrainte numérique** sur confidence.
- `@model_validator(mode="after")` = **calcul auto** de `necessite_escalade` (règle métier).
- `field_validator` = **règles métier** (ex: résumé ≤ 100 chars).
- `Optional` avec `Field(None, ...)` = champs facultatifs documentés.

---

## 4. Prompts système — Séparés du code

**Fichier** : `src/agent_classifieur_tickets/prompts.py`

```python
"""Prompts système versionnés — séparés pour lisibilité et test."""

PROMPT_SYSTEME_CLASSIFIEUR = """Tu es un classifieur expert de tickets IT d'entreprise.
Ta sortie DOIT être un JSON valide conforme au schéma fourni.

RÈGLES DE CLASSIFICATION :
- CATÉGORIES (une seule) :
  * RESEAU : connectivité, VPN, WiFi, DNS, firewall, lenteur réseau
  * MATERIEL : PC, écran, imprimante, téléphone, dock, clavier, souris
  * LOGICIEL : application métier, Office, navigateur, IDE, driver, mise à jour
  * ACCES : mot de passe, compte verrouillé, droits, provisioning, MFA, SSO
  * AUTRE : tout ce qui ne rentre pas ci-dessus

- URGENCE (une seule) :
  * ELEVEE : impact production immédiat, utilisateur bloqué totally, incident sécurité
  * MOYENNE : gêne significative, contournement possible, délai acceptable
  * FAIBLE : demande routine, amélioration, question, non-bloquant

- RÉSUMÉ : ≤ 100 caractères, style télégraphique, français.
- MOTS_CLÉS : max 10 termes techniques (ex: "VPN", "Outlook", "BitLocker", "Azure AD").
- CONFIANCE : 0.0 à 1.0, honnête. Si ambigu → < 0.7.
- NECESSITE_ESCALADE : true SI urgence=ELEVEE OU categorie=ACCES.

EXEMPLES :
Entrée : "Mon écran ne s'allume plus, je dois présenter dans 10 min"
Sortie : {"categorie":"MATERIEL","urgence":"ELEVEE","resume":"Écran HS, urgence présentation","mots_cles":["écran","présentation"],"confidence":0.95,"necessite_escalade":true}

Entrée : "Comment configurer ma signature Outlook ?"
Sortie : {"categorie":"LOGICIEL","urgence":"FAIBLE","resume":"Config signature Outlook","mots_cles":["Outlook","signature"],"confidence":0.99,"necessite_escalade":false}
"""


PROMPT_UTILISATEUR_TEMPLATE = """Ticket à classifier :
Titre : {titre}
Description : {description}
Utilisateur : {utilisateur_id}
Source : {systeme_source}

Réponds UNIQUEMENT par le JSON valide."""
```

---

## 5. Agent Pydantic AI — Définition et retry (v2)

**Fichier** : `src/agent_classifieur_tickets/agent.py`

```python
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
    output_type=TicketSortie,           # Validation sortie automatique (v2: output_type)
    system_prompt=PROMPT_SYSTEME_CLASSIFIEUR,
    retries=3,                          # Retry global (peut être affiné par outil)
)


# --- Outils (optionnels) : ici on n'en a pas besoin, le prompt suffit ---
# Mais Pydantic AI permet d'ajouter des tools avec @agent.tool
# Exemple pour plus tard (RAG, API externe) :
#
# @agent.tool
# async def chercher_base_connaissance(ctx: RunContext, query: str) -> str:
#     """Recherche dans la base de connaissance interne."""
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
```

**Points clés (v2)** :
- `OpenAIChatModel` au lieu de `OpenAIModel` (v2)
- `output_type=TicketSortie` au lieu de `result_type` (v2: `output_type`)
- `@agent.output_validator` au lieu de `@agent.result_validator` (v2: `output_validator`)
- `ModelRetry` (exception spéciale) au lieu de `ValueError` pour déclencher un retry
- `OllamaProvider` = utilise l'API `/v1` compatible OpenAI d'Ollama (port 11434)

---

## 6. Runner — Exécution, logging, gestion d'erreurs

**Fichier** : `src/agent_classifieur_tickets/runner.py`

```python
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
```

**Points clés (v2)** :
- `result.output` au lieu de `result.data` (v2: `.output`)
- `result.usage` (propriété) au lieu de `result.usage()` (méthode)
- `usage.input_tokens` / `usage.output_tokens` au lieu de `request_tokens` / `response_tokens`
- `ModelRetry` (exception v2) au lieu de `ModelRetryError`

---

## 7. CLI Demo — Test manuel

**Fichier** : `src/agent_classifieur_tickets/main.py`

```python
"""CLI de démonstration : classification interactive de tickets."""

import asyncio
import sys
from loguru import logger

from agent_classifieur_tickets.models import TicketEntree
from agent_classifieur_tickets.runner import runner, ResultatExecution


async def demo_interactif():
    """Boucle interactive pour tester l'agent."""
    print("\n=== Classifieur Tickets IT (Pydantic AI + Ollama local) ===")
    print("Tapez 'quit' pour sortir.\n")

    while True:
        try:
            titre = input("Titre du ticket : ").strip()
            if titre.lower() in ("quit", "q", "exit"):
                break

            description = input("Description : ").strip()
            if not description:
                print("Description requise.\n")
                continue

            utilisateur = input("Utilisateur (optionnel) : ").strip() or None
            source = input("Source (optionnel) : ").strip() or None

            ticket = TicketEntree(
                titre=titre,
                description=description,
                utilisateur_id=utilisateur,
                systeme_source=source,
            )

            print("\n--- Classification en cours... ---")
            resultat: ResultatExecution = await runner.classifier(ticket)

            if resultat.succes:
                s = resultat.sortie
                print(f"\n✅ CATÉGORIE  : {s.categorie}")
                print(f"⚡ URGENCE     : {s.urgence}")
                print(f"📝 RÉSUMÉ      : {s.resume}")
                print(f"🔑 MOTS-CLÉS   : {', '.join(s.mots_cles) or '—'}")
                print(f"🎯 CONFIANCE   : {s.confidence:.2f}")
                print(f"🚨 ESCALADE    : {'OUI' if s.necessite_escalade else 'Non'}")
                print(f"\n⏱️  {resultat.duree_ms:.0f} ms | 📥 {resultat.tokens_entree} tok | 📤 {resultat.tokens_sortie} tok | 🔁 {resultat.nb_retries} retry")
            else:
                print(f"\n❌ ÉCHEC : {resultat.erreur.code} — {resultat.erreur.message}")

            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nAu revoir !")


async def demo_lot():
    """Test sur un jeu de tickets prédéfinis."""
    tickets_test = [
        TicketEntree(titre="Écran noir", description="Mon écran ne s'allume plus du tout, je dois présenter dans 10 minutes !"),
        TicketEntree(titre="VPN ne connecte pas", description="Impossible de me connecter au VPN corporate depuis ce matin, erreur timeout."),
        TicketEntree(titre="Signature Outlook", description="Comment configurer ma signature automatique dans Outlook ?"),
        TicketEntree(titre="Compte verrouillé", description="J'ai fait 3 erreurs de mot de passe, mon compte est verrouillé, je ne peux plus rien faire."),
        TicketEntree(titre="Lenteur Excel", description="Excel met 30 secondes à ouvrir un fichier de 50 Mo, c'est anormal."),
    ]

    print("\n=== Test par lot ===\n")
    for i, ticket in enumerate(tickets_test, 1):
        print(f"--- Ticket {i}/{len(tickets_test)} : {ticket.titre} ---")
        resultat = await runner.classifier(ticket)
        if resultat.succes:
            s = resultat.sortie
            print(f"  → {s.categorie} | {s.urgence} | conf={s.confidence:.2f} | {s.resume}")
        else:
            print(f"  → ❌ {resultat.erreur.code}")
    print("\nTerminé.")


if __name__ == "__main__":
    # Configuration logger console
    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # Moins verbeux en CLI

    if len(sys.argv) > 1 and sys.argv[1] == "--lot":
        asyncio.run(demo_lot())
    else:
        asyncio.run(demo_interactif())
```

---

## 8. Tests unitaires — Agent mockable, déterministe

**Fichier** : `tests/test_agent.py`

```python
"""Tests unitaires : agent mocké, validation modèles, cas limites."""

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart

from agent_classifieur_tickets.models import TicketEntree, TicketSortie, CategorieTicket, NiveauUrgence
from agent_classifieur_tickets.agent import agent, construire_prompt_utilisateur
from agent_classifieur_tickets.runner import AgentRunner, ResultatExecution


class TestModeles:
    """Validation des modèles Pydantic (indépendant du LLM)."""

    def test_ticket_entree_valide(self):
        t = TicketEntree(titre="Test", description="Description assez longue pour passer")
        assert t.titre == "Test"

    def test_ticket_entree_titre_trop_court(self):
        with pytest.raises(ValidationError):
            TicketEntree(titre="AB", description="Description assez longue")

    def test_ticket_sortie_confidence_hors_bornes(self):
        with pytest.raises(ValidationError):
            TicketSortie(
                categorie=CategorieTicket.MATERIEL,
                urgence=NiveauUrgence.FAIBLE,
                resume="Test OK",
                confidence=1.5,  # > 1.0
            )

    def test_ticket_sortie_resume_trop_long(self):
        with pytest.raises(ValidationError):
            TicketSortie(
                categorie=CategorieTicket.MATERIEL,
                urgence=NiveauUrgence.FAIBLE,
                resume="A" * 101,  # > 100
                confidence=0.8,
            )

    def test_necessite_escalade_auto(self):
        """Vérifie la logique métier : ACCES ou ELEVEE → escalade."""
        s1 = TicketSortie(categorie=CategorieTicket.ACCES, urgence=NiveauUrgence.FAIBLE, resume="Test OK", confidence=0.9)
        assert s1.necessite_escalade is True

        s2 = TicketSortie(categorie=CategorieTicket.LOGICIEL, urgence=NiveauUrgence.ELEVEE, resume="Test OK", confidence=0.9)
        assert s2.necessite_escalade is True

        s3 = TicketSortie(categorie=CategorieTicket.LOGICIEL, urgence=NiveauUrgence.MOYENNE, resume="Test OK", confidence=0.9)
        assert s3.necessite_escalade is False


class TestPrompt:
    """Construction du prompt utilisateur."""

    def test_prompt_contient_champs(self):
        t = TicketEntree(titre="VPN down", description="VPN ne marche pas", utilisateur_id="u123", systeme_source="portal")
        prompt = construire_prompt_utilisateur(t)
        assert "VPN down" in prompt
        assert "VPN ne marche pas" in prompt
        assert "u123" in prompt
        assert "portal" in prompt


class TestAgentAvecMock:
    """Tests de l'agent avec modèle de test (sans vrai LLM)."""

    @pytest.fixture
    def agent_test(self):
        # TestModel renvoie une réponse JSON valide prédéfinie
        return Agent(
            model=TestModel(),
            output_type=TicketSortie,
            system_prompt="Test prompt",
        )

    @pytest.mark.asyncio
    async def test_agent_retourne_sortie_valide(self, agent_test):
        # TestModel génère automatiquement un JSON conforme au output_type
        result = await agent_test.run("Test input")
        assert isinstance(result.output, TicketSortie)
        assert result.output.categorie in CategorieTicket
        assert 0.0 <= result.output.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_agent_avec_function_model(self):
        """FunctionModel : contrôle total sur la réponse pour tester validateur métier."""
        # Ce test vérifie que l'agent peut être configuré avec FunctionModel
        # et retourner une sortie valide
        def custom_response(messages, tools):
            from pydantic_ai.messages import ModelResponse, ToolCallPart
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name="final_result",
                    args={
                        "categorie": "LOGICIEL",
                        "urgence": "FAIBLE",
                        "resume": "Test OK",
                        "mots_cles": [],
                        "confidence": 0.9,
                        "necessite_escalade": False
                    },
                    tool_call_id="call_1"
                )]
            )

        agent_fm = Agent(
            model=FunctionModel(custom_response),
            output_type=TicketSortie,
            system_prompt="Test",
            retries=2,
        )

        # L'agent doit pouvoir s'exécuter et retourner un résultat valide
        result = await agent_fm.run("Test")
        assert isinstance(result.output, TicketSortie)
        assert result.output.categorie == "LOGICIEL"
        assert result.output.confidence == 0.9


class TestRunner:
    """Tests du runner (wrapper exécution)."""

    @pytest.mark.asyncio
    async def test_runner_succes(self):
        # Utilise FunctionModel pour réponse contrôlée
        def ok_response(messages, tools):
            from pydantic_ai.messages import ModelResponse, ToolCallPart
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name="final_result",
                    args={
                        "categorie": "MATERIEL",
                        "urgence": "ELEVEE",
                        "resume": "Écran HS",
                        "mots_cles": ["écran"],
                        "confidence": 0.95,
                        "necessite_escalade": True
                    },
                    tool_call_id="call_1"
                )]
            )

        agent_mock = Agent(model=FunctionModel(ok_response), output_type=TicketSortie)
        runner = AgentRunner(agent_mock)

        ticket = TicketEntree(titre="Écran", description="Écran cassé")
        result = await runner.classifier(ticket)

        assert result.succes is True
        assert result.sortie.categorie == CategorieTicket.MATERIEL
        assert result.duree_ms > 0

    @pytest.mark.asyncio
    async def test_runner_echec_validation(self):
        """Réponse invalide (confidence > 1) → erreur capturée."""
        def bad_response(messages, tools):
            from pydantic_ai.messages import ModelResponse, ToolCallPart
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name="final_result",
                    args={
                        "categorie": "MATERIEL",
                        "urgence": "FAIBLE",
                        "resume": "Test OK validation",
                        "confidence": 2.0,
                        "necessite_escalade": False
                    },
                    tool_call_id="call_1"
                )]
            )

        agent_mock = Agent(model=FunctionModel(bad_response), output_type=TicketSortie, retries=1)
        runner = AgentRunner(agent_mock)

        ticket = TicketEntree(titre="Test", description="Test description longue")
        result = await runner.classifier(ticket)

        assert result.succes is False
        assert result.erreur.code in ("RETRY_EXHAUSTED", "MODEL_BEHAVIOR")
```

---

## 9. Configuration projet — pyproject.toml

**Fichier** : `pyproject.toml`

```toml
[project]
name = "agent-classifieur-tickets"
version = "0.1.0"
description = "Agent classifieur tickets IT avec Pydantic AI + Ollama local"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic-ai>=0.0.15",
    "ollama>=0.3.0",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "loguru>=0.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

---

## 10. Exécution

```bash
# 1. Vérifier qu'Ollama tourne (tuto #1)
curl -s http://localhost:11434/api/tags | jq

# 2. Installer dépendances (avec uv)
uv pip install -e ".[dev]"

# 3. Test unitaire (sans LLM, mocké)
pytest tests/ -v

# 4. Demo interactive (avec vrai LLM local)
python -m agent_classifieur_tickets.main

# 5. Demo par lot (tickets prédéfinis)
python -m agent_classifieur_tickets.main --lot
```

**Sortie attendue (demo --lot)** :
```
=== Test par lot ===

--- Ticket 1/5 : Écran noir ---
  → MATERIEL | ELEVEE | conf=0.95 | Écran HS, urgence présentation
--- Ticket 2/5 : VPN ne connecte pas ---
  → RESEAU | MOYENNE | conf=0.88 | VPN timeout, blocage télétravail
--- Ticket 3/5 : Signature Outlook ---
  → LOGICIEL | FAIBLE | conf=0.99 | Config signature Outlook
--- Ticket 4/5 : Compte verrouillé ---
  → ACCES | ELEVEE | conf=0.97 | Compte verrouillé 3 erreurs MDP
--- Ticket 5/5 : Lenteur Excel ---
  → LOGICIEL | MOYENNE | conf=0.82 | Lenteur ouverture fichier 50Mo
```

---

## 11. Pièges et limites — Soyons honnêtes

| Piège | Réalité |
|-------|---------|
| « Pydantic AI fait tout » | Non : **pas d'orchestration**, pas de mémoire, pas de multi-agents. C'est une couche typage/validation. |
| « Le typage garantit la qualité » | Le typage garantit la **structure**. La **pertinence** dépend du prompt et du modèle. Un JSON valide peut être n'importe quoi. |
| « Retry = correction automatique » | Retry réessaie le **même prompt**. Si le modèle est trop petit (ex: 0.5B) pour la tâche, il échouera toujours. |
| « Ça marche sur mon N100 » | Qwen2.5-3B-Q4 tient en ~3.5 Go RAM. Si vous lancez autre chose en parallèle (navigateur, IDE), vous swappez → latence x10. |
| « Pas besoin de tests » | Les tests avec `TestModel`/`FunctionModel` sont **essentiels** : ils valident la logique de validation/retry sans appeler le LLM. |

---

## 12. Aller plus loin

- [Pydantic AI docs](https://ai.pydantic.dev/) — agents, tools, dependency injection, graph (expérimental)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Pydantic V2 validation](https://docs.pydantic.dev/latest/concepts/validators/)
- Prochaine étape : **03-traitement-donnees-vector-db** — ingestion documents, chunking sémantique, vector DB locale (DuckDB/Qdrant embarqué).