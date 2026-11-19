"""CLI de démonstration : classification interactive de tickets."""

import asyncio
import sys
from loguru import logger

from .models import TicketEntree
from .runner import runner, ResultatExecution


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