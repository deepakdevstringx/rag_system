import ollama

from .config import CHAT_MODEL
from .vector_store import ingest_pending_pdfs, retrieve_context

def optimize_search_query(user_query):
    """Rewrite a question for semantic retrieval without changing its intent."""
    prompt = f"""
    Rewrite the user's question for semantic search against an enterprise PDF knowledge base.
    Rules:
    1. Fix spelling mistakes and grammar.
    2. Preserve the user's original intent.
    3. Do not answer the question or invent facts.
    4. Add relevant synonyms when useful and keep it concise.
    5. Return ONLY the optimized search query.

    User question:
    {user_query}
    """
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You optimize user queries for semantic search."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0, "num_predict": 100}
    )
    return response["message"]["content"].strip()

def run_chat_session():
    """Run the interactive question-answering loop and maintain recent history."""
    chat_history = []
    print("\nSmart RAG Assistant Ready! Type 'exit' to quit.\n")

    while True:
        user_query = input("\n Ask a question: ").strip()

        if user_query.lower() == 'exit':
            break
        if not user_query:
            continue

        optimized_query = optimize_search_query(user_query)
        print(f"\n🔎 Optimized query: {optimized_query}")

        context, citations = retrieve_context(optimized_query, k=5)

        system_instruction = f"""You are a precise enterprise research assistant.
            Answer the user's question using ONLY the provided document context below. 

            INSTRUCTIONS:
            1. Base your answer strictly on the provided context. Do not invent information.
            2. If the context does not contain enough info, state clearly that you cannot find it.

            DOCUMENT CONTEXT:
            {context}
            """

        messages = [{"role": "system", "content": system_instruction}] + chat_history[-4:] + [{"role": "user", "content": user_query}]

        print("\n🤔 Searching and generating response...\n")
        
        response_stream = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            keep_alive="24h",
            options={
                "num_predict": 512,
                "temperature": 0.1
            }
        )

        full_response = ""
        print("🤖 ANSWER:")
        for chunk in response_stream:
            content = chunk["message"]["content"]
            full_response += content
            print(content, end="", flush=True)

        print("\n\n📄 Sources consulted:")
        for citation in citations:
            print(f" . {citation}")
        print("-" * 50)
        
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": full_response})

def main():
    """Index pending PDFs before starting the interactive chat session."""
    ingest_pending_pdfs()
    run_chat_session()

if __name__ == "__main__":
    main()