"""Tests intégration MemoryManager (hot + cold)."""

import pytest
import tempfile
from memoire.memory_manager import MemoryManager
from memoire.models import RoleMessage
from memoire.hot_memory import HotMemoryStore
from memoire.cold_memory import ColdMemoryStore


@pytest.fixture
def temp_dirs(tmp_path):
    """Crée dossiers temporaires isolés pour chaque test."""
    sessions_db = tmp_path / "sessions.db"
    memory_db = tmp_path / "memory.db"
    return str(sessions_db), str(memory_db)


@pytest.fixture
def manager(temp_dirs):
    sessions_db, memory_db = temp_dirs
    hot_store = HotMemoryStore(db_path=sessions_db)
    cold_store = ColdMemoryStore(db_path=memory_db)
    from memoire.hot_memory import HotMemoryManager
    from memoire.cold_memory import ColdMemoryManager
    return MemoryManager(
        hot=HotMemoryManager(store=hot_store),
        cold=ColdMemoryManager(store=cold_store),
    )


@pytest.mark.asyncio
async def test_demarrer_session_cree_et_charge(manager):
    state = await manager.demarrer_session("sess1", "user1")
    assert state.session_id == "sess1"
    assert state.user_id == "user1"

    # Recharger même session
    state2 = await manager.demarrer_session("sess1", "user1")
    assert state2.session_id == "sess1"


@pytest.mark.asyncio
async def test_ajouter_echange_met_a_jour_hot(manager):
    state = await manager.demarrer_session("sess2", "user2")
    await manager.ajouter_echange(state, "Bonjour", "Bonjour ! Comment puis-je aider ?")

    assert len(state.messages) == 2
    assert state.messages[0].role == RoleMessage.USER
    assert state.messages[1].role == RoleMessage.ASSISTANT


@pytest.mark.asyncio
async def test_construire_contexte_complet_hot_seulement(manager):
    state = await manager.demarrer_session("sess3", "user3")
    await manager.ajouter_echange(state, "Question", "Réponse")

    ctx = await manager.construire_contexte_complet(state, inclure_long_terme=False)

    assert ctx.user_id == "user3"
    assert ctx.session_id == "sess3"
    assert "DERNIERS ÉCHANGES" in ctx.hot_context
    assert "Question" in ctx.hot_context


@pytest.mark.asyncio
async def test_formater_prompt_systeme_inclut_contexte(manager):
    state = await manager.demarrer_session("sess4", "user4")
    await manager.ajouter_echange(state, "Test", "OK")

    ctx = await manager.construire_contexte_complet(state, inclure_long_terme=False)
    prompt = manager.formater_prompt_systeme("Tu es un assistant.", ctx)

    assert "Tu es un assistant." in prompt
    assert "MÉMOIRE COURT TERME" in prompt
    assert "INSTRUCTIONS" in prompt


@pytest.mark.asyncio
async def test_extraire_faits_post_echange(manager):
    state = await manager.demarrer_session("sess5", "user5")
    await manager.ajouter_echange(state, "Je préfère les réponses courtes", "Noté.")

    # Ne doit pas lever d'erreur
    await manager.extraire_faits_post_echange(state, "test_agent")

    # Vérifie qu'un fait a été potentiellement créé
    faits = await manager.cold.get_faits_contextuels("user5")
    # L'extraction heuristique peut ne rien trouver, on vérifie juste que ça ne crash pas


@pytest.mark.asyncio
async def test_logger_interaction_ajoute_historique(manager):
    state = await manager.demarrer_session("sess6", "user6")

    await manager.logger_interaction(
        state, "rh_agent", "Combien de congés ?", "25 jours",
        True, 150, 20, 10
    )

    stats = await manager.get_stats_utilisateur("user6")
    assert stats["user_id"] == "user6"
    assert stats["derniere_activite"] is not None


@pytest.mark.asyncio
async def test_session_isolation(manager):
    """Sessions différentes = mémoires courtes isolées."""
    state1 = await manager.demarrer_session("sess_a", "user1")
    state2 = await manager.demarrer_session("sess_b", "user1")  # Même user, session différente

    await manager.ajouter_echange(state1, "Session A", "Réponse A")
    await manager.ajouter_echange(state2, "Session B", "Réponse B")

    assert len(state1.messages) == 2
    assert len(state2.messages) == 2
    assert "Session A" in state1.messages[0].contenu
    assert "Session B" in state2.messages[0].contenu