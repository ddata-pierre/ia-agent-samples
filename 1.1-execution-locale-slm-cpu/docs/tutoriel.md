# 01-execution-locale-slm-cpu - SLM en conteneur (Podman + Ollama)

> **Objectif** : faire tourner des SLM (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5) quantifiés 4-bit en conteneur, comprendre l'API REST, benchmarker sur CPU pur.
> **Durée** : ~15 min.
> **Prérequis** : Podman installé, 16 Go RAM, WSL2 Ubuntu, modèles ~2–2.5 Go chacun en Q4_K_M.

---

## 1. Pourquoi Podman + Ollama + SLM quantifiés ?

| Avantages | Inconvénients |
|-----------|---------------|
| Isolation complète, pas de pollution hôte | Légère surcouche (Dockerfile) |
| Persistance via volumes nommés | Port mapping requis pour API |
| Reproductible (image versionnée, modèles figés) | Nécessite Podman/Docker |
| Modèles préchargés = zéro téléchargement runtime | Image ~7 Go (base + 3 modèles) |
| Quantification Q4_K_M = meilleur rapport qualité/poids CPU | Moins précis que FP16 (acceptable pour classif/extraction) |

> Alternative locale sans Ollama : tutoriel 02 (llama.cpp brut via `llama-cpp-python`).

---

## 2. Dockerfile - Image Ollama avec modèles préchargés

**Fichier** : `Dockerfile` (à la racine de `01-execution-locale-slm-cpu/`)

```dockerfile
# Image de base officielle Ollama (basée sur Debian slim)
FROM docker.io/ollama/ollama:latest

# Auteur / métadonnées
LABEL maintainer="ddata.fr"
LABEL description="Ollama avec SLM quantifiés Q4_K_M pour N100 (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5)"

# Variable d'environnement : modèles à pull au build (cache layers)
# Modèles choisis : tous < 2.5 Go en Q4_K_M, performants sur CPU
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
```

> **Build** (une seule fois) :
> ```bash
> podman build -t local-ia/slm-cpu:q4 .
> ```
> Temps ~5–8 min (téléchargement modèles ~6–7 Go). Les layers sont cachés : rebuild instantané si Dockerfile inchangé.

**Explications des modèles choisis** :

| Modèle | Taille Q4_K_M | Contexte | Force | Usage recommandé |
|--------|---------------|----------|-------|------------------|
| `qwen2.5:3b-instruct-q4_k_m` | ~1.9 Go | 32k | Raisonnement, code, multilangue | Agent généraliste, extraction |
| `llama3.2:3b-instruct-q4_k_m` | ~2.0 Go | 128k | Instruction following, résumé | RAG long contexte, synthèse |
| `phi3.5:3.8b-mini-instruct-q4_k_m` | ~2.3 Go | 128k | Compact, fort en logique | Classification, routing, JSON strict |

---

## 3. Lancer le conteneur (avec réseau pour API REST)

```bash
# Crée un volume nommé pour persister les modèles (+ configs)
podman volume create ollama-slm-data

# Lance le conteneur en détaché (-d), mappe port 11434, monte le volume
podman run -d \
  --name ollama-slm \
  -p 11434:11434 \
  -v ollama-slm-data:/root/.ollama \
  local-ia/slm-cpu:q4
```

**Explications** :

| Option | Rôle |
|--------|------|
| `-d` | Détaché (arrière-plan) |
| `--name ollama-slm` | Nom explicite pour `podman stop ollama-slm` |
| `-p 11434:11434` | Port hôte:conteneur (API accessible sur localhost:11434) |
| `-v ollama-slm-data:/root/.ollama` | Persistance modèles dans volume nommé |
| `local-ia/slm-cpu:q4` | Image construite à l'étape 2 |

**Vérifier** :
```bash
podman logs ollama-slm           # logs du serveur
curl -s http://localhost:11434/api/tags | jq    # liste modèles (JSON)
```

---

## 3bis. Démonstration : fonctionnement 100% local (sans réseau)

Pour prouver qu'aucune connexion externe n'est nécessaire **après le build**, on lance le conteneur **sans réseau** :

```bash
# Arrête le conteneur précédent si besoin
podman stop ollama-slm && podman rm ollama-slm

# Lance SANS réseau (--network none)
# Note: --network none ne supporte PAS le port mapping (-p) dans Podman
podman run -d \
  --name ollama-slm-local \
  --network none \
  -v ollama-slm-data:/root/.ollama \
  local-ia/slm-cpu:q4
```

**Vérifier l'isolation** :
```bash
# Depuis le conteneur : modèle tourne en local (pas de pull, pas de réseau)
podman exec ollama-slm-local /bin/ollama run qwen2.5:3b-instruct-q4_k_m "test"
# → génère une réponse, modèles déjà dans l'image

# Aucun accès réseau externe
podman exec ollama-slm-local ping -c 1 8.8.8.8 2>&1 || echo "ping échoué (attendu : pas de réseau)"
```

**Explications** :

| Option | Rôle |
|--------|------|
| `--network none` | Pas d'interface réseau dans le conteneur (isolé total) |
| Modèles dans l'image | Aucun `pull` au runtime, inférence 100% locale |
| Pas de `-p` | Port mapping impossible avec `--network none` (limite Podman) |

> **Preuve** : le conteneur démarre, le modèle répond, mais **aucune connexion sortante n'est possible**. L'inférence est 100% locale.

> **Pour l'API REST** : utilisez le conteneur avec réseau (étape 3). Les modèles sont déjà dans l'image, donc **zéro téléchargement au démarrage**.

---

## 4. Utilisation CLI - `podman exec`

Si vous avez fait le test sans réseau, pensez à relancer le conteneur *ollama-slm* (étape 3).

```bash
# REPL interactive dans le conteneur (équivalent `ollama run`)
podman exec -it ollama-slm ollama run qwen2.5:3b-instruct-q4_k_m
```

**Explications** :
- `-it` : interactif + TTY (nécessaire pour REPL)
- `ollama run qwen2.5:3b-instruct-q4_k_m` : commande *dans* le conteneur
- `Ctrl+D` ou `/bye` pour quitter

---

## 5. API REST - Exemples pas à pas (depuis l'hôte)

### 5.1 Chat non-streaming (conversation multi-tours)

**Commande** :
```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b-instruct-q4_k_m",
    "messages": [
      {"role": "system", "content": "Tu es un assistant concis."},
      {"role": "user", "content": "C'\''est quoi un embedding ?"}
    ],
    "stream": false
  }' | jq -r '.message.content'
```

**Explications** :
- `curl -s` : silencieux (pas de barre de progression)
- `-H "Content-Type: application/json"` : header JSON
- `-d '{...}'` : payload JSON (modèle, messages, stream=false)
- `| jq -r '.message.content'` : extrait le champ texte de la réponse

**Flux** : client → POST /api/chat → Ollama → modèle → JSON response → `jq` filtre.

### 5.2 Generate (complétion simple, sans historique)

**Différence clé** : `/api/generate` = complétion unique (prompt brut, pas de rôles), `/api/chat` = conversation structurée.

#### 5.2.1 Réponse pour humain (texte libre)

```bash
curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b-instruct-q4_k_m",
    "prompt": "La capitale de la France est",
    "stream": false
  }' | jq -r '.response'
```

**Sortie** : `Paris. C'est une ville...` (phrase complète, langage naturel)

#### 5.2.2 Réponse pour machine (JSON strict, exploitable)

Pour obtenir une réponse parsable par un autre process, on force le format via le prompt système + options :

```bash
curl -s http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b-instruct-q4_k_m",
    "prompt": "Réponds UNIQUEMENT par un JSON valide : {\"capital\": \"...\"}. Question : Quelle est la capitale de la France ?",
    "stream": false,
    "options": {
      "temperature": 0,
      "num_predict": 16
    }
  }' | jq '.response | fromjson'
```

**Sortie** : `{"capital": "Paris"}` (JSON valide, `jq` le parse avec `fromjson`)

**Paramètres clés** :
- `temperature: 0` : déterministe (toujours même réponse)
- `num_predict: 16` : limite tokens (économique, évite verbiage)
- Prompt explicite : exige format JSON exact

> **Usage machine** : `capital=$(curl ... | jq -r '.response | fromjson | .capital')` → variable shell = `Paris`

### 5.3 Streaming (temps réel, token par token)

**Commande** :
```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Écris un haïku sur le code."}],
    "stream": true
  }' | while read line; do
      echo "$line" | jq -r '.message.content // empty' | tr -d '\n'
  done
echo
```

**Explications** :
- `stream: true` : renvoie du NDJSON (une ligne JSON par token)
- `while read line` : lit chaque ligne au fur et à mesure (flux continu)
- `jq -r '.message.content // empty'` : extrait le token, ignore lignes sans contenu
- `tr -d '\n'` : supprime newlines pour affichage fluide
- `echo` final : ajoute le newline de fin

**Pourquoi cette boucle** : `curl` sort le flux brut NDJSON ; la boucle traite chaque ligne JSON au fil de l'eau.

---

## 6. Personnaliser un modèle (Modelfile)

Crée un fichier `Modelfile` (sur l'hôte) :

```dockerfile
# Modelfile : définit un modèle dérivé
FROM qwen2.5:3b-instruct-q4_k_m                              # modèle de base (déjà dans l'image)
SYSTEM "Tu es un expert data engineering. Réponds en français, max 3 phrases."  # prompt système
PARAMETER temperature 0.3                      # créativité (0 = déterministe, 1 = créatif)
PARAMETER top_p 0.9                            # noyau d'échantillonnage
PARAMETER num_ctx 8192                         # fenêtre de contexte (tokens)
```

Puis crée le modèle dans le conteneur :

```bash
# Copie le Modelfile dans le conteneur et crée le modèle
podman cp Modelfile ollama-slm:/tmp/Modelfile
podman exec ollama-slm ollama create mon-expert-data -f /tmp/Modelfile
podman exec -it ollama-slm ollama run mon-expert-data   # test
```

---

## 7. Benchmark local des 3 modèles (CPU pur)

Script de benchmark simple à lancer depuis l'hôte :

```bash
#!/usr/bin/env bash
# benchmark.sh - à placer dans 01-execution-locale-slm-cpu/

MODELS=(
  "qwen2.5:3b-instruct-q4_k_m"
  "llama3.2:3b-instruct-q4_k_m"
  "phi3.5:3.8b-mini-instruct-q4_k_m"
)

PROMPT="Explique la différence entre HTTP et HTTPS en exactement 2 phrases."

echo "Modèle, Temps_1er_token(s), Tokens/sec, RAM_pic(Go)"
for m in "${MODELS[@]}"; do
  # Warm-up
  curl -s http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$m\",\"prompt\":\"test\",\"stream\":false}" > /dev/null

  # Mesure
  start=$(date +%s.%N)
  curl -s http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$m\",\"prompt\":\"$PROMPT\",\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":64}}" \
    | jq -r '.response' > /dev/null
  end=$(date +%s.%N)

  duration=$(echo "$end - $start" | bc -l)
  # Approximation tokens/sec (num_predict / durée)
  tps=$(echo "64 / $duration" | bc -l)

  # RAM pic via podman stats (échantillon)
  ram=$(podman stats --no-stream --format "{{.MemPerc}}" ollama-slm 2>/dev/null | sed 's/%//' || echo "N/A")

  printf "%s, %.2f, %.1f, %s\n" "$m" "$duration" "$tps" "$ram"
done
```

** Les résultats sont variables selon charge système. **

> **Observation** : Tester les modèles en fonction de vos besoins réel : Qwen2.5-3B est le meilleur compromis vitesse/qualité. Phi-3.5 est plus lent mais excelle sur le JSON strict et la logique. Llama-3.2 brille sur les contextes longs (128k).

---

## 8. Client Python (bibliothèque `ollama`)

```bash
# Dans un venv (recommandé)
python3 -m venv .venv && source .venv/bin/activate
pip install ollama
```

```python
# client.py
import ollama
import time

# Client pointe vers localhost:11434 (défaut)
client = ollama.Client(host="http://localhost:11434")

# Chat non-streaming
print("=== Chat non-streaming ===")
start = time.time()
resp = client.chat(
    model="qwen2.5:3b-instruct-q4_k_m",
    messages=[
        {"role": "system", "content": "Tu es un assistant concis."},
        {"role": "user", "content": "C'est quoi un embedding ?"}
    ],
    stream=False
)
elapsed = time.time() - start
print(f"Réponse ({elapsed:.2f}s) : {resp['message']['content']}")

# Chat streaming
print("\n=== Chat streaming ===")
stream = client.chat(
    model="qwen2.5:3b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "Compte de 1 à 5"}],
    stream=True
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()

# Generate avec JSON forcé
print("\n=== Generate JSON strict ===")
resp = client.generate(
    model="qwen2.5:3b-instruct-q4_k_m",
    prompt='Réponds UNIQUEMENT par un JSON valide : {"capital": "..."}. Question : Quelle est la capitale de la France ?',
    stream=False,
    options={"temperature": 0, "num_predict": 16}
)
import json
try:
    data = json.loads(resp["response"])
    print(f"Capital : {data['capital']}")
except json.JSONDecodeError:
    print("Échec parsing JSON :", resp["response"])
```

```bash
python3 client.py
```

---

## 9. Gestion des données (volume)

```bash
# Liste volumes
podman volume ls

# Inspecte le volume (point de montage hôte)
podman volume inspect ollama-slm-data

# Sauvegarde (export tar)
podman run --rm -v ollama-slm-data:/data -v $(pwd):/backup alpine tar czf /backup/ollama-slm-backup.tar.gz -C /data .

# Restauration
podman run --rm -v ollama-slm-data:/data -v $(pwd):/backup alpine tar xzf /backup/ollama-slm-backup.tar.gz -C /data
```

---

## 10. Exercices + Corrections

### Exercice 1 - Modèle « Classifieur de tickets IT »

**Consigne** : Crée un Modelfile qui classe un ticket IT en JSON : `{"categorie": "RESEAU|MATERIEL|LOGICIEL|ACCES|AUTRE", "urgence": "FAIBLE|MOYENNE|ELEVEE", "resume": "..."}`.

**Correction** :
```dockerfile
# Modelfile.classifieur
FROM qwen2.5:3b-instruct-q4_k_m
SYSTEM "Tu es un classifieur de tickets IT. Réponds UNIQUEMENT par un JSON valide avec les clés : categorie (RESEAU|MATERIEL|LOGICIEL|ACCES|AUTRE), urgence (FAIBLE|MOYENNE|ELEVEE), resume (max 15 mots). Pas de texte hors JSON."
PARAMETER temperature 0
PARAMETER top_p 0.1
PARAMETER num_predict 64
```
```bash
podman cp Modelfile.classifieur ollama-slm:/tmp/Modelfile.classifieur
podman exec ollama-slm ollama create classifieur-ticket -f /tmp/Modelfile.classifieur
podman exec -it ollama-slm ollama run classifieur-ticket
# Test : "Mon écran ne s'allume plus, je dois présenter dans 10 min"
# → {"categorie":"MATERIEL","urgence":"ELEVEE","resume":"Écran ne s'allume pas, urgence présentation"}
```

### Exercice 2 - Comparer modèles sur tâche identique

**Consigne** : Pose la même question aux 3 modèles, note la différence de style, précision, vitesse.

**Correction** :
```bash
Q="Résume en 3 points clés la RGPD pour un développeur."

for m in qwen2.5:3b-instruct-q4_k_m llama3.2:3b-instruct-q4_k_m phi3.5:3.8b-mini-instruct-q4_k_m; do
  echo "=== $m ==="
  time podman exec ollama-slm ollama run "$m" "$Q"
  echo
done
```

**Observation attendue** : Qwen = structuré, complet. Llama = fluide, parfois verbeux. Phi = concis, va à l'essentiel.

### Exercice 3 - Streaming en Python avec callback

**Consigne** : Réécris le streaming avec une fonction callback pour afficher token par token dans une UI future.

**Correction** :
```python
# streaming_callback.py
import ollama

def stream_with_callback(model, messages, on_token):
    """Streaming avec callback par token."""
    stream = ollama.Client().chat(model=model, messages=messages, stream=True)
    full = ""
    for chunk in stream:
        token = chunk["message"]["content"]
        full += token
        on_token(token)
    return full

def print_token(token):
    print(token, end="", flush=True)

if __name__ == "__main__":
    stream_with_callback(
        "qwen2.5:3b-instruct-q4_k_m",
        [{"role": "user", "content": "Écris un haïku sur le debug."}],
        print_token
    )
    print()
```

---

## 11. Nettoyage

```bash
# Arrête et supprime le conteneur
podman stop ollama-slm && podman rm ollama-slm

# Supprime l'image (libère ~7 Go)
podman rmi local-ia/slm-cpu:q4

# Supprime le volume (supprime modèles + configs)
podman volume rm ollama-slm-data
```

---

## 12. Aller plus loin

- [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Modelfile reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx)
- [Podman docs](https://docs.podman.io/)
- [GGUF quantification guide](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- Prochaine étape : **02-premier-agent-metier-pydantic-ai** - Agent typé avec Pydantic AI.