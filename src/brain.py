"""Brain module — LLM clients, web search, conversation management."""

import json
import logging
import threading
from typing import Generator
from openai import OpenAI

from config import OPENAI_API_KEY, XAI_API_KEY, PERPLEXITY_API_KEY, HARDWARE
import personality

log = logging.getLogger("botijo.brain")

_client: OpenAI | None = None
_px_client: OpenAI | None = None
_history: list[dict] = []
_history_lock = threading.Lock()
_llm_config: dict = {}
_max_history: int = 20


def init(persona_name: str) -> None:
    """Load personality and initialize LLM clients."""
    global _client, _px_client, _llm_config, _max_history

    persona = personality.load(persona_name)
    _llm_config = personality.get_llm_config()
    _max_history = HARDWARE["behavior"]["history_max_messages"]

    if _llm_config["llm"] == "grok":
        _client = OpenAI(api_key=XAI_API_KEY, base_url=_llm_config["grok_base_url"])
    else:
        _client = OpenAI(api_key=OPENAI_API_KEY)

    if PERPLEXITY_API_KEY and persona.get("search_enabled"):
        _px_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    _history.clear()
    _history.append({"role": "system", "content": personality.get_system_prompt()})

    log.info("Brain initialized — llm: %s, model: %s, search: %s",
             _llm_config["llm"], _llm_config["model"],
             "enabled" if _px_client else "disabled")


def chat(user_text: str) -> str:
    """Send message to LLM, return complete response (blocking)."""
    return "".join(chat_stream(user_text))


def chat_stream(user_text: str) -> Generator[str, None, None]:
    """Stream text chunks from LLM. Handles function calling internally."""
    if not _client:
        yield "Error: brain not initialized"
        return

    with _history_lock:
        _history.append({"role": "user", "content": user_text})
        _trim_history()
        messages = list(_history)

    tools = personality.get_tools()
    kwargs = {
        "model": _llm_config["model"],
        "messages": messages,
        "max_completion_tokens": _llm_config["max_completion_tokens"],
        "stream": True,
    }
    # GPT-5 only supports temperature=1 (the default) — omit for those models
    model_name = _llm_config["model"].lower()
    if not model_name.startswith("gpt-5"):
        kwargs["temperature"] = _llm_config["temperature"]
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    # GPT-5 specific params
    if _llm_config.get("verbosity"):
        kwargs["verbosity"] = _llm_config["verbosity"]
    if _llm_config.get("reasoning_effort"):
        kwargs["reasoning_effort"] = _llm_config["reasoning_effort"]

    full_response = ""
    tool_calls_buffer = {}

    try:
        stream = _client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            if delta.content:
                full_response += delta.content
                yield delta.content

        # Resolve tool calls if any
        if tool_calls_buffer:
            _handle_tool_calls(tool_calls_buffer, messages)
            for chunk_text in _continue_after_tools(messages):
                full_response += chunk_text
                yield chunk_text

    except Exception as e:
        log.error("LLM error: %s", e)
        error_msg = "Error procesando tu petición, zarria humana."
        full_response = error_msg
        yield error_msg

    with _history_lock:
        _history.append({"role": "assistant", "content": full_response})


def search(query: str) -> str:
    """Web search via Perplexity."""
    if not _px_client:
        return "Search not available"
    try:
        response = _px_client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": query}],
            temperature=0.2,
            max_tokens=400,
            top_p=0.9,
        )
        return response.choices[0].message.content or "No results"
    except Exception as e:
        log.error("Search error: %s", e)
        return f"Search failed: {e}"


def note_interruption(spoken_text: str) -> None:
    """Record that the last response was interrupted."""
    with _history_lock:
        if _history and _history[-1]["role"] == "assistant":
            _history[-1]["content"] += "\n[INTERRUPTED — user cut off here]"
    log.debug("Interruption noted after: %s...", spoken_text[:50])


def reset_history() -> None:
    with _history_lock:
        system = _history[0] if _history else None
        _history.clear()
        if system:
            _history.append(system)


def cleanup():
    pass


def _trim_history():
    if len(_history) > _max_history + 1:
        system = _history[0]
        _history[:] = [system] + _history[-_max_history:]


def _handle_tool_calls(tool_calls_buffer: dict, messages: list) -> bool:
    """Execute tool calls and append results to messages."""
    if not tool_calls_buffer:
        return False

    # ONE assistant message with ALL tool_calls (OpenAI API requirement)
    all_tool_calls = []
    for idx in sorted(tool_calls_buffer.keys()):
        tc = tool_calls_buffer[idx]
        all_tool_calls.append({
            "id": tc["id"], "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]}
        })
    messages.append({"role": "assistant", "content": None, "tool_calls": all_tool_calls})

    # Execute each tool
    for idx in sorted(tool_calls_buffer.keys()):
        tc = tool_calls_buffer[idx]
        if tc["name"] == "web_search":
            try:
                args = json.loads(tc["arguments"])
                result = search(args.get("query", ""))
            except Exception as e:
                result = f"Tool error: {e}"
        else:
            result = f"Unknown tool: {tc['name']}"

        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })
    return True


def _continue_after_tools(messages: list) -> Generator[str, None, None]:
    kwargs = {
        "model": _llm_config["model"],
        "messages": messages,
        "max_completion_tokens": _llm_config["max_completion_tokens"],
        "stream": True,
    }
    model_name = _llm_config["model"].lower()
    if not model_name.startswith("gpt-5"):
        kwargs["temperature"] = _llm_config["temperature"]
    try:
        stream = _client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        log.error("Continue after tools error: %s", e)
        yield "Error continuando la respuesta."
