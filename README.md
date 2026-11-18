# Plan de la Série de Tutoriels : Système d'Agents IA d'Entreprise (Architecture Hybride & Local CPU)

## Module 1 : Premier Agent Métier Local (Pydantic AI & SLM)
- **Article 1.1 : Exécution locale d'un SLM sur CPU**
  - Installation et configuration d'Ollama / llama.cpp sur CPU.
  - Sélection de petits modèles quantifiés (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5) et benchmarks locaux.
- **Article 1.2 : Mon premier agent métier typé avec Pydantic AI**
  - Prise en main de Pydantic AI, typage strict des entrées/sorties, validation et gestion robuste des erreurs.

## Module 2 : Connaissance Métier & Gestion des Mémoires
- **Article 2.1 : Traitement des données RH et Vector DB locale**
  - Ingestion, nettoyage et chunking sémantique de documents RH (politiques internes, chartes).
  - Stockage vectoriel local et léger avec DuckDB / Qdrant embarqué.
- **Article 2.2 : Architecture des Mémoires (Court Terme vs Long Terme)**
  - Implémentation de la mémoire à court terme (fenêtre de conversation / état de la session).
  - Implémentation de la mémoire à long terme (profil utilisateur, préférences et historique d'interactions persistant).

## Module 3 : Orchestration Hybride & Système Multi-Agents (LangGraph)
- **Article 3.1 : Passerelle Hybride avec OpenRouter**
  - Configuration de l'accès OpenRouter pour exploiter un LLM généraliste rapide (gratuit) dédié au bavardage et à l'aiguillage.
- **Article 3.2 : Routage et Orchestration Multi-Agents avec LangGraph**
  - Création du graphe d'états : Agent Routeur (OpenRouter) -> Agent RH (Local RAG) vs Agent IT (Local Ticket).
- **Article 3.3 : Implémentation de l'Agent IT (Collecte & Ticketing)**
  - Conception de l'agent IT sous forme de machine à états pour interroger l'utilisateur et valider la création d'un ticket.

## Module 4 : Sécurité d'Entreprise, SSO & Control d'Accès (Keycloak)
- **Article 4.1 : Déploiement et Configuration de Keycloak**
  - Déploiement local de Keycloak (Podman / Docker) et création d'un Realm entreprise avec gestion des rôles (`RH_MEMBER`, `IT_SUPPORT`, `EMPLOYEE`).
- **Article 4.2 : Sécurisation du RAG et propagation de l'identité**
  - Propagation du jeton JWT Keycloak à travers le graphe LangGraph.
  - Filtrage RBAC au niveau de la Vector DB : limiter l'accès aux documents RH confidentiels selon les droits de l'utilisateur connecté.

## Module 5 : Industrialisation, MCP & Interface Utilisateur
- **Article 5.1 : Standardisation des outils IT via le protocole MCP (Model Context Protocol)**
  - Interconnexion de l'agent IT avec un serveur MCP local simulant une API de ticketing (ServiceNow/Jira/SQLite).
- **Article 5.2 : Observabilité, Audit et Guardrails**
  - Traçabilité complète des exécutions d'agents et suivi de la latence avec Langfuse (ou alternative open source légère).
  - Validation des garde-fous (Guardrails) avant restitution.
- **Article 5.3 : Évaluation Automatisée de la Qualité (RAG & Agents avec RAGAS)**
  - Jeux de données de test synthétiques, métriques d'évaluation du RAG (faithfulness, answer relevance, context precision/recall) et tests automatisés d'agents sur CPU.