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