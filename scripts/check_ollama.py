import ollama


def main():
    """Send a small prompt to verify the local Ollama chat model is available."""
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Explain what a database is in simple terms."}],
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    main()