"""LangChain-backed chat model selection for Ollama, Gemini, and Claude."""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import CHAT_MODEL, CHAT_PROVIDER, FALLBACK_MODEL, FALLBACK_PROVIDER


_active_provider = CHAT_PROVIDER
_active_model = CHAT_MODEL
_DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-5",
}
_PROVIDER_API_KEYS = {
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


def resolve_chat_settings(provider=None, model=None):
    """Resolve and validate the selected chat provider and model name."""
    selected_provider = (provider or _active_provider).strip().lower()
    if selected_provider not in _DEFAULT_MODELS:
        raise ValueError("Chat provider must be 'ollama', 'gemini', or 'claude'.")

    selected_model = model or _active_model
    if not selected_model:
        selected_model = _DEFAULT_MODELS[selected_provider]
    return selected_provider, selected_model


def set_chat_provider(provider):
    """Select a chat provider for subsequent model requests in this process."""
    global _active_provider, _active_model
    selected_provider = provider.strip().lower()
    if selected_provider not in _DEFAULT_MODELS:
        raise ValueError("Chat provider must be 'ollama', 'gemini', or 'claude'.")
    api_key_name = _PROVIDER_API_KEYS.get(selected_provider)
    if api_key_name and not os.getenv(api_key_name):
        provider_name = "Gemini" if selected_provider == "gemini" else "Claude"
        raise ValueError(f"Set {api_key_name} in .env before selecting {provider_name}.")

    _active_provider = selected_provider
    if _active_model in _DEFAULT_MODELS.values():
        _active_model = _DEFAULT_MODELS[selected_provider]
    return _active_provider, _active_model


def set_chat_model(model):
    """Set the model identifier used by subsequent requests in this process."""
    global _active_model
    if not model.strip():
        raise ValueError("Model name cannot be empty.")
    _active_model = model.strip()
    return _active_model


def get_chat_selection():
    """Return the active provider and model for display in the chat prompt."""
    return resolve_chat_settings()


def _to_langchain_messages(messages):
    """Convert role/content dictionaries into LangChain message objects."""
    message_types = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    converted = []
    for message in messages:
        message_type = message_types.get(message["role"])
        if message_type is None:
            raise ValueError(f"Unsupported chat role: {message['role']}")
        converted.append(message_type(content=message["content"]))
    return converted


def _create_chat_model(temperature, max_tokens, provider=None, model=None):
    """Create a LangChain chat client for the selected or fallback provider."""
    provider, model_name = resolve_chat_settings(provider, model)
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("Gemini is selected but GOOGLE_API_KEY is not set in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Claude is selected but ANTHROPIC_API_KEY is not set in .env.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
        keep_alive="24h",
    )


def _content_to_text(content):
    """Normalize LangChain text or content blocks into plain response text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def invoke_chat(messages, temperature=0.0, max_tokens=256):
    """Invoke the primary model, falling back to Ollama if Claude fails."""
    provider, _ = resolve_chat_settings()
    try:
        model = _create_chat_model(temperature, max_tokens)
        response = model.invoke(_to_langchain_messages(messages))
        return _content_to_text(response.content)
    except Exception as primary_error:
        if provider != "claude" or FALLBACK_PROVIDER == provider:
            raise
        print(
            f"\n⚠️ Claude request failed ({primary_error}). "
            f"Falling back to {FALLBACK_PROVIDER} / {FALLBACK_MODEL}."
        )
        fallback_model = _create_chat_model(
            temperature,
            max_tokens,
            provider=FALLBACK_PROVIDER,
            model=FALLBACK_MODEL,
        )
        response = fallback_model.invoke(_to_langchain_messages(messages))
        return _content_to_text(response.content)


def stream_chat(messages, temperature=0.1, max_tokens=512):
    """Stream from Claude, using Ollama if Claude fails before producing text."""
    provider, _ = resolve_chat_settings()
    try:
        model = _create_chat_model(temperature, max_tokens)
        chunks = iter(model.stream(_to_langchain_messages(messages)))
        try:
            first_chunk = next(chunks)
        except StopIteration:
            return
    except Exception as primary_error:
        if provider != "claude" or FALLBACK_PROVIDER == provider:
            raise
        print(
            f"\n⚠️ Claude stream failed ({primary_error}). "
            f"Falling back to {FALLBACK_PROVIDER} / {FALLBACK_MODEL}."
        )
        fallback_model = _create_chat_model(
            temperature,
            max_tokens,
            provider=FALLBACK_PROVIDER,
            model=FALLBACK_MODEL,
        )
        chunks = fallback_model.stream(_to_langchain_messages(messages))
    else:
        first_text = _content_to_text(first_chunk.content)
        if first_text:
            yield first_text

    for chunk in chunks:
        text = _content_to_text(chunk.content)
        if text:
            yield text