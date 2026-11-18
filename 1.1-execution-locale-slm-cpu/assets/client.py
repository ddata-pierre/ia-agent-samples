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
