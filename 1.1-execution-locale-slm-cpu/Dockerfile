# Image de base officielle Ollama (basée sur Debian slim)
FROM docker.io/ollama/ollama:latest

# Auteur / métadonnées
LABEL maintainer="ddata.fr"
LABEL description="Ollama avec SLM quantifiés Q4_K_M pour N100 (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5)"

# Variable d'environnement : modèles à pull au build (cache layers)
# Modèles choisis : tous < 2.5 Go en Q4_K_M,performants sur CPU
ENV OLLAMA_MODELS="qwen2.5:3b-instruct-q4_k_m llama3.2:3b-instruct-q4_k_m phi3.5:3.8b-mini-instruct-q4_k_m"

# Script d'initialisation : pull les modèles pendant le build
# Utilise un script shell explicite pour éviter les problèmes de continuation de ligne
RUN set -eux; \
    for m in $OLLAMA_MODELS; do \
        ollama serve & \
        sleep 5 && \
        ollama pull "$m" && \
        pkill ollama; \
    done

# Expose le port API REST
EXPOSE 11434

# Ajout outils réseau pour démonstration hors réseau
RUN apt-get update \
    && apt-get install -y --no-install-recommends iputils-ping curl jq \
    && rm -rf /var/lib/apt/lists/*

# Point d'entrée : lance le serveur Ollama (ENTRYPOINT du base = /bin/ollama)
CMD ["serve"]