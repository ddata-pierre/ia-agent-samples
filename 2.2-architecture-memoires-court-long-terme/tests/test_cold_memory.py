"""Tests mémoire long terme."""

import pytest
import asyncio
from memoire.models import (
    ProfilUtilisateur, FaitUtilisateur, TypeFait, InteractionHistorique
)
from memoire.cold_memory import ColdMemoryStore, ColdMemoryManager


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_memory.db"
    return str(db_path)


@pytest.fixture
def store(temp_db):
    return ColdMemoryStore(db_path=temp_db)


@pytest.fixture
def manager(store):
    return ColdMemoryManager(store=store)


class TestColdMemoryStore:
    """Tests persistance SQLite long terme."""

    @pytest.mark.asyncio
    async def test_profil_upsert_get(self, store):
        profil = ProfilUtilisateur(user_id="user1", nom="Test", role="EMPLOYEE")
        await store.upsert_profil(profil)

        loaded = await store.get_profil("user1")
        assert loaded is not None
        assert loaded.nom == "Test"
        assert loaded.role == "EMPLOYEE"

    @pytest.mark.asyncio
    async def test_profil_inexistant_retourne_none(self, store):
        loaded = await store.get_profil("inconnu")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_faits_upsert_get(self, store):
        fait = FaitUtilisateur(
            user_id="user1", type=TypeFait.PREFERENCE,
            cle="style.reponse", valeur="court", confiance=0.9
        )
        await store.upsert_fait(fait)

        faits = await store.get_faits("user1")
        assert len(faits) == 1
        assert faits[0].cle == "style.reponse"
        assert faits[0].valeur == "court"

    @pytest.mark.asyncio
    async def test_faits_inactifs_exclus_par_defaut(self, store):
        fait1 = FaitUtilisateur(user_id="user1", type=TypeFait.PREFERENCE, cle="a", valeur="1", actif=True)
        fait2 = FaitUtilisateur(user_id="user1", type=TypeFait.PREFERENCE, cle="b", valeur="2", actif=False)
        await store.upsert_fait(fait1)
        await store.upsert_fait(fait2)

        faits = await store.get_faits("user1")  # actifs_seulement=True par défaut
        assert len(faits) == 1
        assert faits[0].cle == "a"

        tous = await store.get_faits("user1", actifs_seulement=False)
        assert len(tous) == 2

    @pytest.mark.asyncio
    async def test_historique_ajout_et_purge(self, store):
        for i in range(5):
            inter = InteractionHistorique(
                user_id="user1", session_id=f"s{i}", agent_type="test",
                entree=f"in{i}", sortie_resume=f"out{i}", succes=True,
                duree_ms=100, tokens_in=10, tokens_out=5
            )
            await store.ajouter_interaction(inter)

        histo = await store.get_historique_recent("user1", limite=10)
        assert len(histo) == 5
        assert histo[0].entree == "in4"  # Plus récent en premier


class TestColdMemoryManager:
    """Tests logique métier mémoire long terme."""

    @pytest.mark.asyncio
    async def test_get_profil_cree_si_absent(self, manager):
        profil = await manager.get_profil("nouveau_user")
        assert profil.user_id == "nouveau_user"
        assert profil.langue_preferee == "fr"

    @pytest.mark.asyncio
    async def test_maj_profil(self, manager):
        await manager.maj_profil("user1", nom="Alice", service="IT")
        profil = await manager.get_profil("user1")
        assert profil.nom == "Alice"
        assert profil.service == "IT"

    @pytest.mark.asyncio
    async def test_get_faits_contextuels(self, manager):
        await manager.store.upsert_fait(FaitUtilisateur(
            user_id="user1", type=TypeFait.PREFERENCE, cle="style.reponse", valeur="court"
        ))
        await manager.store.upsert_fait(FaitUtilisateur(
            user_id="user1", type=TypeFait.ENTITE, cle="client.principal", valeur="ACME"
        ))

        tous = await manager.get_faits_contextuels("user1")
        assert len(tous) == 2

        filtres = await manager.get_faits_contextuels("user1", cles=["style.reponse"])
        assert len(filtres) == 1
        assert filtres[0].cle == "style.reponse"

    @pytest.mark.asyncio
    async def test_contexte_injection_vide(self, manager):
        ctx = await manager.get_contexte_injection("user_inconnu")
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_contexte_injection_avec_profil_et_faits(self, manager):
        await manager.maj_profil("user1", nom="Bob", role="MANAGER", service="Finance")
        await manager.store.upsert_fait(FaitUtilisateur(
            user_id="user1", type=TypeFait.PREFERENCE, cle="style.reponse", valeur="detaille", confiance=0.8
        ))

        ctx = await manager.get_contexte_injection("user1")
        assert "CONTEXTE UTILISATEUR" in ctx
        assert "Bob" in ctx
        assert "MANAGER" in ctx
        assert "Finance" in ctx
        assert "style.reponse" in ctx
        assert "detaille" in ctx

    @pytest.mark.asyncio
    async def test_logger_interaction(self, manager):
        inter = InteractionHistorique(
            user_id="user1", session_id="s1", agent_type="rh",
            entree="Combien de congés ?", sortie_resume="25 jours", succes=True,
            duree_ms=200, tokens_in=30, tokens_out=15
        )
        await manager.logger_interaction(inter)

        stats = await manager.get_stats_utilisateur("user1")
        assert stats["nb_faits"] >= 0
        assert stats["derniere_activite"] is not None