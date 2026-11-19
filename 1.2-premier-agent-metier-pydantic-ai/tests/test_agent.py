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