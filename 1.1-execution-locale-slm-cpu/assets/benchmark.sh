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
