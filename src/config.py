"""Botijo configuration loader.

Loads .env (API keys) and config files (hardware, personalities).
No dependencies on other src/ modules.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Find project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

# Load .env from home directory (RPi) or project root
load_dotenv(Path.home() / ".env")
load_dotenv(_PROJECT_ROOT / ".env")

# Hardware config
def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

HARDWARE = _load_json(_CONFIG_DIR / "hardware.json")

def get_personality(name: str) -> dict:
    """Load a personality config by name. Injects current datetime into system_prompt."""
    path = _CONFIG_DIR / "personalities" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Personality not found: {name}")
    data = _load_json(path)
    if "system_prompt" in data:
        data["system_prompt"] = data["system_prompt"].format(
            datetime=datetime.now().strftime("%d/%m/%Y %H:%M")
        )
    return data

def get_phrases(name: str) -> dict | None:
    """Load phrase pools for autonomous modes. Returns None if no phrases file exists."""
    path = _CONFIG_DIR / "personalities" / f"{name}_phrases.json"
    if not path.exists():
        return None
    return _load_json(path)

# Convenience: API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
XAI_API_KEY = os.getenv("XAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
