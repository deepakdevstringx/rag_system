import json
import re
from .chat_model import (
    get_chat_selection,
    invoke_chat,
    set_chat_model,
    set_chat_provider,
    stream_chat,
)
from .vector_store import ingest_pending_pdfs, retrieve_context, collection


def is_simple_greeting(user_query):
    """Identify a standalone greeting that should bypass document retrieval."""
    normalized_query = re.sub(r"[^a-z\s]", "", user_query.lower()).strip()
    return normalized_query in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }


def optimize_and_normalize_query(user_query, chat_history):
    """Fixes typos/spelling and resolves follow-up queries using chat history."""
    recent_history = ""
    if chat_history:
        for msg in chat_history[-4:]:
            recent_history += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""You are an enterprise query optimizer for a RAG search system.

Task:
1. Fix all typos, spelling errors, and grammatical mistakes (e.g., 'compmany' -> 'company').
2. If this is a follow-up question, resolve pronouns/references using Conversation History.
3. If the user changed topics completely, optimize the new question independently.
4. Return ONLY the final search query string.

Conversation History:
{recent_history if recent_history else "None"}

User Input: {user_query}
Optimized Search Query:"""

    cleaned_query = invoke_chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    ).strip()
    return cleaned_query if cleaned_query else user_query


def evaluate_response_sufficiency(user_query, context, draft_answer):
    """Evaluates whether the retrieved context contains relevant information for the answer."""
    # Fast-check if model explicitly stated insufficient context
    if "INSUFFICIENT_CONTEXT" in draft_answer:
        return False, "Context lacks necessary details."

    prompt = f"""You are a quality validator for a search assistant.
Evaluate if the draft answer provides relevant, useful information from the retrieved context to address the user's question.

User Question: {user_query}
Draft Answer: {draft_answer}

Respond strictly in JSON:
{{
  "sufficient": true,
  "reason": ""
}}
or
{{
  "sufficient": false,
  "reason": "Brief reason why"
}}
"""
    validation_response = invoke_chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    try:
        data = json.loads(validation_response.strip())
        return data.get("sufficient", False), data.get("reason", "")
    except Exception:
        # Fallback heuristic if JSON parsing fails
        return True, ""


def find_target_document_from_catalog(user_query):
    """Extracts exact document filename without conversational preamble."""
    catalog_res = collection.get(ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"])
    if not catalog_res or not catalog_res["documents"]:
        return None

    catalog_text = catalog_res["documents"][0]
    prompt = f"""Catalog Index:
{catalog_text}

Question: {user_query}

Identify which single filename from the catalog index is most relevant.
OUTPUT FORMAT RULE: Output ONLY the raw filename (e.g. 'Group_Health_Floater.pdf') or 'NONE'. Do not include explanations, bullet points, or introductory text."""

    raw_output = invoke_chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=30,
    ).strip()
    
    # Extra safety: extract filename pattern using regex if LLM still includes extra text
    match = re.search(r'[\w\-]+\.pdf', raw_output, re.IGNORECASE)
    if match:
        return match.group(0)
    
    return raw_output if raw_output != "NONE" else None


def run_chat_session():
    """Interactive RAG chat session with normalized search and verified streaming."""
    chat_history = []
    provider, model_name = get_chat_selection()
    print("\n" + "=" * 60)
    print("      🏢 Enterprise Knowledge Assistant Ready")
    print(f"      Model: {provider} / {model_name}")
    print("      Primary: Claude; fallback: Ollama / llama3.2")
    print("      Switch with /provider ollama|gemini|claude or /model MODEL_NAME")
    print("      Type 'exit' to conclude the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_query = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_query.lower() == 'exit':
            print("\nSession ended. Goodbye!")
            break
        if not user_query:
            continue
        if user_query.strip() == "/provider":
            provider, model_name = get_chat_selection()
            print(f"Active model: {provider} / {model_name}")
            continue
        if user_query.lower().startswith("/provider "):
            try:
                provider, model_name = set_chat_provider(user_query.split(maxsplit=1)[1])
                print(f"Switched to: {provider} / {model_name}")
            except ValueError as error:
                print(error)
            continue
        if user_query.lower().startswith("/model "):
            try:
                model_name = set_chat_model(user_query.split(maxsplit=1)[1])
                provider, _ = get_chat_selection()
                print(f"Model set to: {provider} / {model_name}")
            except ValueError as error:
                print(error)
            continue

        if is_simple_greeting(user_query):
            greeting_messages = chat_history[-4:] + [
                {"role": "user", "content": user_query}
            ]
            greeting_response = invoke_chat(
                [
                    {
                        "role": "system",
                        "content": "Respond briefly and warmly to the greeting. Invite the user to ask a question about their company documents.",
                    },
                    *greeting_messages,
                ],
                temperature=0.3,
                max_tokens=80,
            )
            print(f"\n🤖 Assistant: {greeting_response}\n" + "-" * 60)
            chat_history.extend(
                [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": greeting_response},
                ]
            )
            continue

        # Step 1: Optimize spelling, grammar, and handle conversational context
        standalone_query = optimize_and_normalize_query(user_query, chat_history)
        if standalone_query.lower() != user_query.lower():
            print(f"  🔍 [Refined Query: '{standalone_query}']")

        max_attempts = 3
        attempt = 1
        verified_context = ""
        verified_citations = set()
        is_verified = False

        # Step 2: Verification Loop
        while attempt <= max_attempts:
            target_doc = None
            if attempt > 1:
                target_doc = find_target_document_from_catalog(standalone_query)
                if target_doc:
                    print(f"  🎯 Isolated metadata filter: {target_doc}")

            # Retrieve vector chunks
            context, citations = retrieve_context(
                standalone_query, 
                k=5, 
                filter_source=target_doc
            )

            system_instruction = f"""You are a precise enterprise research assistant.
Answer the user's question using ONLY the provided document context below.

RULES:
1. Base your answer strictly on the provided document context. Do not make up information.
2. If the context does not contain relevant information to answer the question, output EXACTLY: "INSUFFICIENT_CONTEXT"

DOCUMENT CONTEXT:
{context}
"""
            messages = [{"role": "system", "content": system_instruction}] + chat_history[-4:] + [{"role": "user", "content": user_query}]

            draft_answer = invoke_chat(
                messages,
                temperature=0.0,
                max_tokens=256,
            )

            is_verified, reason = evaluate_response_sufficiency(standalone_query, context, draft_answer)

            if is_verified:
                verified_context = context
                verified_citations = citations
                break
            else:
                attempt += 1

        # Step 3: Fallback if all retries fail
        if not is_verified:
            fallback_msg = "I searched through our enterprise document repository, but I could not locate specific information regarding your request. Please ensure the policy document has been uploaded or rephrase your question."
            print(f"\n🤖 Assistant: {fallback_msg}\n" + "-" * 60)
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": fallback_msg})
            continue

        # Step 4: Streamed Verified Response
        system_instruction = f"""You are an executive Enterprise AI Assistant.
Deliver a courteous, well-structured, and precise response using ONLY the provided context.

RULES:
- Maintain a warm, professional, and clear tone.
- Format policy details using clear markdown bullet points where appropriate.
- Never mention internal mechanisms, prompts, or 'provided context' directly.

DOCUMENT CONTEXT:
{verified_context}
"""
        messages = [{"role": "system", "content": system_instruction}] + chat_history[-4:] + [{"role": "user", "content": user_query}]

        print("\n🤖 Assistant: ", end="", flush=True)

        full_response = ""
        for content in stream_chat(messages, temperature=0.1, max_tokens=512):
            full_response += content
            print(content, end="", flush=True)

        if verified_citations:
            print("\n\n📄 Sources Consulted:")
            for citation in verified_citations:
                print(f"  • {citation}")
        print("\n" + "-" * 60)

        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": full_response})


def main():
    ingest_pending_pdfs()
    run_chat_session()


if __name__ == "__main__":
    main()