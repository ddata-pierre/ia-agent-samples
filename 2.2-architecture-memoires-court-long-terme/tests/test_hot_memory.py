"""Tests mémoire court terme."""

import pytest
import asyncio
from datetime import datetime
from memoire.models import Message, RoleMessage, HotMemoryState, SessionSummary
from memoire.hot_memory import HotMemoryStore, HotMemoryManager
from memoire.config import settings


@pytest.fixture
def temp_db(tmp_path):
    """DB temporaire pour tests."""
    db_path = tmp_path / "test_sessions.db"
    return str(db_path)


@pytest.fixture
def store(temp_db):
    return HotMemoryStore(db_path=temp_db)


@pytest.fixture
def manager(store):
    return HotMemoryManager(store=store)


@pytest.fixture
def state():
    return HotMemoryState(session_id="test_session", user_id="test_user")


class TestHotMemoryStore:
    """Tests persistance SQLite."""

    @pytest.mark.asyncio
    async def test_sauvegarder_charger(self, store, state):
        await store.sauvegarder(state)
        loaded = await store.charger("test_session")
        assert loaded is not None
        assert loaded.session_id == "test_session"
        assert loaded.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_supprimer(self, store, state):
        await store.sauvegarder(state)
        await store.supprimer("test_session")
        loaded = await store.charger("test_session")
        assert loaded is None


class TestHotMemoryManager:
    """Tests logique métier mémoire court terme."""

    @pytest.mark.asyncio
    async def test_get_or_create_nouveau(self, manager):
        state = await manager.get_or_create("new_session", "user1")
        assert state.session_id == "new_session"
        assert state.user_id == "user1"
        assert len(state.messages) == 0

    @pytest.mark.asyncio
    async def test_get_or_create_existant(self, manager, state):
        await manager.store.sauvegarder(state)
        loaded = await manager.get_or_create("test_session", "test_user")
        assert loaded.session_id == "test_session"

    def test_ajouter_message_simple(self, manager, state):
        manager.ajouter_message(state, RoleMessage.USER, "Bonjour")
        assert len(state.messages) == 1
        assert state.messages[0].role == RoleMessage.USER
        assert state.messages[0].contenu == "Bonjour"

    def test_fenetre_glissante(self, manager, state):
        """Au-delà de hot_window_size, les anciens messages sortent."""
        for i in range(settings.hot_window_size + 5):
            manager.ajouter_message(state, RoleMessage.USER, f"Message {i}")

        # Seuls les derniers hot_window_size restent en brut
        assert len(state.messages) == settings.hot_window_size
        assert state.messages[0].contenu == f"Message {5}"  # Les 5 premiers sont partis

    def test_resume_cree_quand_depassement(self, manager, state):
        """Quand messages > fenêtre, résumé est créé."""
        for i in range(settings.hot_window_size + 3):
            manager.ajouter_message(state, RoleMessage.USER, f"Msg {i}")

        assert state.resume_actuel is not None
        assert state.resume_actuel.messages_couverts >= 3

    def test_resume_recursif_accumule(self, manager, state):
        """Résumé accumule les messages couverts."""
        # Premier dépassement
        for i in range(settings.hot_window_size + 3):
            manager.ajouter_message(state, RoleMessage.USER, f"Batch 1 - {i}")
        premier_resume = state.resume_actuel.messages_couverts

        # Deuxième dépassement
        for i in range(settings.hot_window_size + 3):
            manager.ajouter_message(state, RoleMessage.USER, f"Batch 2 - {i}")

        assert state.resume_actuel.messages_couverts > premier_resume

    def test_construire_contexte_prompt(self, manager, state):
        """Contexte inclut résumé + derniers messages."""
        manager.ajouter_message(state, RoleMessage.USER, "Question 1")
        manager.ajouter_message(state, RoleMessage.ASSISTANT, "Réponse 1")
        manager.ajouter_message(state, RoleMessage.USER, "Question 2")

        # Force résumé en ajoutant assez de messages pour dépasser la fenêtre
        # Les 3 premiers messages (Question 1, Réponse 1, Question 2) seront résumés
        for i in range(settings.hot_window_size + 2):
            manager.ajouter_message(state, RoleMessage.USER, f"Old {i}")

        ctx = manager.construire_contexte_prompt(state)
        assert "RÉSUMÉ SESSION" in ctx
        assert "DERNIERS ÉCHANGES" in ctx
        # Question 2 est dans le résumé, pas dans les derniers messages
        # Vérifie que le résumé mentionne le sujet
        assert "question" in ctx.lower()


class TestMessage:
    def test_to_prompt_format(self):
        msg = Message(role=RoleMessage.USER, contenu="Test")
        assert msg.to_prompt() == "Utilisateur: Test"

        msg2 = Message(role=RoleMessage.ASSISTANT, contenu="Réponse")
        assert msg2.to_prompt() == "Assistant: Réponse"