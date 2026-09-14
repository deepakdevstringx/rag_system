import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Explain what a database is in simple terms."
        }
    ]
)

print(response["message"]["content"])