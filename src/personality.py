"""Personality management — system prompts, tools, phrases."""

import logging
from config import get_personality as _get_personality, get_phrases as _get_phrases

log = logging.getLogger("botijo.personality")

_current: dict | None = None


def load(name: str) -> dict:
    """Load personality config. Stores as current."""
    global _current
    _current = _get_personality(name)
    log.info("Personality loaded: %s (llm: %s)", _current["name"], _current["llm"])
    return _current


def get_current() -> dict:
    """Return the current personality config dict, or empty dict if none loaded."""
    return _current or {}


def get_system_prompt() -> str:
    if not _current:
        raise RuntimeError("No personality loaded")
    return _current["system_prompt"]


def get_tools() -> list | None:
    """Return tool definitions for function calling, or None if disabled."""
    if not _current or not _current.get("tools_enabled"):
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def get_greeting() -> str:
    if not _current:
        return "Botijo online."
    return _current.get("greeting", "Botijo online.")


def get_phrases() -> dict | None:
    if not _current:
        return None
    return _get_phrases(_current["name"])


def get_llm_config() -> dict:
    if not _current:
        raise RuntimeError("No personality loaded")
    return {
        "llm": _current.get("llm", "openai"),
        "model": _current.get("model", "gpt-5"),
        "temperature": _current.get("temperature", 0.7),
        "max_completion_tokens": _current.get("max_completion_tokens", 800),
        "verbosity": _current.get("verbosity"),
        "reasoning_effort": _current.get("reasoning_effort"),
        "grok_base_url": _current.get("grok_base_url"),
    }
