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
