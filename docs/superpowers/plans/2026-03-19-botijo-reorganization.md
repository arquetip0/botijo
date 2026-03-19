# Botijo Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize 127+ monolithic Python scripts into 9 clean modules with a deploy workflow from MacBook to RPi.

**Architecture:** Flat module layout under `src/` (no packages). Each module maps 1:1 to a hardware subsystem with `init() -> bool` graceful degradation. `main.py` orchestrates. Config extracted to JSON files. Old scripts preserved in `old/`.

**Tech Stack:** Python 3.11, OpenAI SDK (GPT-5 + Perplexity), ElevenLabs SDK, Google Cloud Speech, Adafruit ServoKit (PCA9685), NeoPixel (rpi_ws281x), picamera2 (IMX500), WebRTC VAD, PyAudio, gpiozero

**Spec:** `docs/superpowers/specs/2026-03-19-botijo-reorganization-design.md`

**Reference script:** `old/gpt5botijonew2.py` (primary source for extraction — 3565 lines)

**Development:** Edit locally at `/Volumes/X10 Pro/Projects/botijo/repo/`, deploy via `./deploy.sh` (rsync to RPi `botijo:~/botijo/`), test via SSH. RPi user is `jack`, project dir `/home/jack/botijo/`.

**Testing note:** This is a hardware project on a Raspberry Pi. Unit tests are limited to pure logic (config loading, personality, history management). Hardware modules are tested manually on the RPi via `tools/` scripts and `./deploy.sh --run`. Each task specifies whether to test locally or on RPi.

---

## Task 1: Repository Infrastructure

**Files:**
- Create: `deploy.sh`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `CLAUDE.md`
- Create: `src/` (directory)
- Create: `old/` (directory)
- Create: `config/` (directory)
- Create: `config/personalities/` (directory)
- Create: `tools/` (directory)
- Create: `vendor/` (directory)
- Create: `docs/` (directory)

- [ ] **Step 1: Create directory structure locally**

```bash
cd /Volumes/X10\ Pro/Projects/botijo/repo
mkdir -p src config/personalities tools vendor docs old
```

- [ ] **Step 2: Move all .py scripts at root to old/**

Most scripts only exist in the git repo (committed), not all exist on the RPi as untracked files.
Move everything that's tracked in git at root level:

```bash
cd /Volumes/X10\ Pro/Projects/botijo/repo
# Move all tracked .py and .sh files at root to old/
for f in *.py *.sh *.txt *.json; do
    [ -f "$f" ] && git mv "$f" old/ 2>/dev/null
done
# Move old docs (not our new ones)
for f in CORRECCIONES_QUIRURGICAS_COMPLETADAS.md MEJORAS_STT_IMPLEMENTADAS.md RESPEAKER_V2_INTERRUPTIONS_COMPLETE.md GPT5_OPTIMIZACIONES_BOTIJO.md contexto_proyecto_botijo.txt; do
    [ -f "$f" ] && git mv "$f" old/ 2>/dev/null
done
true  # Don't fail if some files don't exist locally
```

Do NOT move: `docs/`, `config/`, `src/`, `vendor/`, `panel/`, `panel_backend/`, `.git/`, `.gitignore`, `README.md`

- [ ] **Step 3: Copy Waveshare LCD driver to vendor/**

```bash
ssh botijo "ls ~/botijo/lib/"
# Copy the lib directory to vendor/lib/ (preserving original name for imports)
scp -r botijo:~/botijo/lib/ vendor/lib/
```

This preserves the original import path. With `PYTHONPATH=src:vendor`, `from lib import LCD_1inch9` and `from lib.LCD_1inch9 import LCD_1inch9` will work exactly as in the original scripts. The `vendor/lib/` directory contains `LCD_1inch9.py`, `lcdconfig.py`, `__init__.py`, etc.

- [ ] **Step 4: Create .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
venv*/
.venv/

# Environment
.env

# Audio/media
*.mp3
*.wav
*.flac

# Models (large binaries)
*.onnx
*.rpk
*.ppn
*.bin
modelos/

# OS
.DS_Store
*.swp
*~

# IDE
.vscode/
.idea/

# Build
dist/
build/
*.egg-info/
```

- [ ] **Step 5: Create .env.example**

```bash
# Botijo API Keys — copy to ~/.env on the RPi
OPENAI_API_KEY=sk-proj-...
ELEVENLABS_API_KEY=sk_...
GOOGLE_APPLICATION_CREDENTIALS=/home/jack/botijo-456920-20c50c37c151.json
XAI_API_KEY=xai-...
PERPLEXITY_API_KEY=pplx-...
```

- [ ] **Step 6: Create deploy.sh**

```bash
#!/bin/bash
# Deploy Botijo code to Raspberry Pi
# Usage: ./deploy.sh [--run [--mode MODE]]

set -e

REMOTE="botijo"
REMOTE_DIR="~/botijo"

echo "📦 Syncing code to $REMOTE..."
rsync -avz --delete src/ "$REMOTE:$REMOTE_DIR/src/"
rsync -avz --delete config/ "$REMOTE:$REMOTE_DIR/config/"
rsync -avz --delete tools/ "$REMOTE:$REMOTE_DIR/tools/"
rsync -avz --delete vendor/ "$REMOTE:$REMOTE_DIR/vendor/"
rsync -avz requirements.txt "$REMOTE:$REMOTE_DIR/"
echo "✅ Sync complete"

if [ "$1" = "--run" ]; then
    shift
    echo "🤖 Launching Botijo..."
    ssh -t "$REMOTE" "cd $REMOTE_DIR && source venv_chatgpt/bin/activate && PYTHONPATH=src:vendor python src/main.py $@"
fi
```

```bash
chmod +x deploy.sh
```

- [ ] **Step 7: Create CLAUDE.md**

```markdown
# Botijo — Androide Conversacional

Androide físico con personalidad paranoide controlado por Raspberry Pi 5.
Plataforma R&D para producción futura de robots conversacionales.

## Quick Reference

- **Deploy:** `./deploy.sh` (rsync to RPi), `./deploy.sh --run` (sync + execute)
- **Run on RPi:** `cd ~/botijo && source venv_chatgpt/bin/activate && PYTHONPATH=src:vendor python src/main.py`
- **SSH:** `ssh botijo` (user jack, /home/jack/botijo/)
- **Modes:** `--mode botijo` (default), `--mode botija`, `--mode barbacoa`

## Module Map (src/)

| Module | Responsibility |
|--------|---------------|
| `config.py` | Load .env + hardware.json + personalities |
| `audio.py` | ReSpeaker, VAD, STT (Google Cloud), TTS (ElevenLabs + espeak-ng), interruptions |
| `brain.py` | LLM clients (OpenAI GPT-5, Grok), Perplexity search, conversation history |
| `vision.py` | IMX500 face detection, tracking coordinates |
| `servos.py` | PCA9685 servos — eyes, eyelids, tentacles, ears, quiet_mode |
| `leds.py` | NeoPixel animations (steampunk, listening, speaking) |
| `display.py` | Waveshare 1.9" LCD — eye expressions, waveform visualizer |
| `buttons.py` | GPIO buttons with callbacks |
| `personality.py` | System prompts, tool definitions, phrase pools |
| `main.py` | Orchestrator: parse args, init modules, main loop |

## Hardware

RPi 5 (8GB, NVMe 458GB), ReSpeaker 4-Mic USB, IMX500 AI camera (CSI),
11 servos via PCA9685 (I2C), 13 NeoPixel LEDs (GPIO18), Waveshare 1.9" LCD (SPI),
4 buttons (GPIO 5,6,12,13). Speaker via ReSpeaker USB.

## Do NOT touch

- `old/` — legacy scripts, reference only
- `panel/` + `panel_backend/` — web control panel (separate systemd service)
- `venv_chatgpt/` on RPi — managed manually
- `.env` on RPi — API keys

## Context7 Library IDs (for up-to-date docs)

- picamera2: `/raspberrypi/picamera2`
- PCA9685: `/adafruit/adafruit-pwm-servo-driver-library`
- NeoPixel Pi5: `/vanshksingh/pi5neo`
- ElevenLabs: `/elevenlabs/elevenlabs-python`
- Google STT: `/websites/cloud_google_speech-to-text`

## Conventions

- Python 3.11, no type stubs, minimal type hints
- Each module: `init() -> bool` (graceful degradation), `cleanup()`, owns its threads
- Config in `config/hardware.json` and `config/personalities/*.json`
- PYTHONPATH=src:vendor (flat imports: `from audio import listen`)
```

- [ ] **Step 8: Sync RPi — move old scripts, create dirs, install deps**

On the RPi, the old scripts are already at `~/botijo/`. Move them to `~/botijo/old/` and create the new directory structure:

```bash
ssh botijo "cd ~/botijo && mkdir -p old src config/personalities tools vendor docs && mv *.py old/ 2>/dev/null; mv *.sh old/ 2>/dev/null; mv contexto_proyecto_botijo.txt old/ 2>/dev/null; mv CORRECCIONES_QUIRURGICAS_COMPLETADAS.md MEJORAS_STT_IMPLEMENTADAS.md RESPEAKER_V2_INTERRUPTIONS_COMPLETE.md GPT5_OPTIMIZACIONES_BOTIJO.md old/ 2>/dev/null; true"
```

Ensure python-dotenv is installed in the RPi venv:
```bash
ssh botijo "cd ~/botijo && source venv_chatgpt/bin/activate && pip install python-dotenv 2>/dev/null && echo 'dotenv OK'"
```

Pull the latest from GitHub:
```bash
ssh botijo "cd ~/botijo && git pull origin main"
```

- [ ] **Step 9: Test deploy.sh works**

```bash
# Create a minimal test file
echo 'print("Botijo deploy works")' > src/main.py
./deploy.sh --run
# Expected: "Botijo deploy works"
rm src/main.py
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: create modular project structure, move scripts to old/"
```

---

## Task 2: config.py + hardware.json + personalities

**Files:**
- Create: `src/config.py`
- Create: `config/hardware.json`
- Create: `config/personalities/botijo.json`
- Create: `config/personalities/botija.json`
- Create: `config/personalities/barbacoa.json`
- Create: `config/personalities/barbacoa_phrases.json`
- Reference: `old/gpt5botijonew2.py` (constants), `old/barbagrok.py` (Grok config), `old/phases_botijo.py` (phrases)

- [ ] **Step 1: Create config/hardware.json**

Extract ALL hardcoded constants from `old/gpt5botijonew2.py`:

```json
{
  "respeaker": {
    "vid": "0x2886",
    "pid": "0x0018",
    "channels": 1,
    "rate": 16000,
    "chunk": 480
  },
  "vad": {
    "mode": 3,
    "frame_duration_ms": 30,
    "padding_duration_ms": 300,
    "voice_threshold": 0.3,
    "consecutive_required": 3,
    "echo_suppression_factor": 0.4,
    "echo_buffer_maxlen": 2000
  },
  "stt": {
    "rate": 16000,
    "language": "es-ES",
    "use_enhanced": true,
    "timeout_seconds": 15,
    "max_listen_seconds": 30,
    "silence_threshold_seconds": 5
  },
  "tts": {
    "elevenlabs_voice_id": "RnKqZYEeVQciORlpiCz0",
    "elevenlabs_model": "eleven_flash_v2_5",
    "elevenlabs_rate": 24000,
    "elevenlabs_format": "pcm_24000",
    "playback_chunk": 1024
  },
  "servos": {
    "eye_lr_channel": 0,
    "eye_ud_channel": 1,
    "eyelids": {
      "top_left": {"channel": 2, "open": 50, "closed": 90},
      "bottom_left": {"channel": 3, "open": 155, "closed": 90},
      "top_right": {"channel": 4, "open": 135, "closed": 90},
      "bottom_right": {"channel": 5, "open": 25, "closed": 90}
    },
    "tentacle_pitch_channel": 8,
    "tentacle_left_channel": 15,
    "tentacle_right_channel": 12,
    "ear_left_channel": 15,
    "ear_right_channel": 12,
    "lr_range": [40, 140],
    "ud_range": [30, 150],
    "pulse_width_range": [500, 2500],
    "tentacle_left_pulse": [300, 2650],
    "tentacle_right_pulse": [450, 2720]
  },
  "eyes": {
    "blink_probability": 0.01,
    "blink_duration": 0.12,
    "double_blink_probability": 0.3,
    "random_look_probability": 0.01,
    "random_look_duration": 0.1,
    "micro_movement_probability": 0.15,
    "micro_movement_range": 3,
    "squint_probability": 0.02,
    "squint_duration": 1.5,
    "squint_intensity": 0.6,
    "breathing_enabled": true,
    "breathing_cycle": 4.0,
    "breathing_intensity": 0.15
  },
  "leds": {
    "count": 13,
    "pin": 18,
    "brightness": 0.05,
    "paleta": [[184,115,51],[205,133,63],[139,69,19],[112,128,40],[255,215,0],[72,60,50]],
    "knight_rider_color": [255, 0, 0],
    "knight_rider_delay": 0.04,
    "quiet_glow_color": [128, 0, 128]
  },
  "display": {
    "native_width": 170,
    "native_height": 320,
    "rotation": 270,
    "rst_pin": 27,
    "dc_pin": 25,
    "bl_pin": 24
  },
  "buttons": {
    "btn1_gpio": 5,
    "btn2_gpio": 6,
    "btn3_gpio": 13,
    "btn4_gpio": 12
  },
  "vision": {
    "rpk_path": "/home/jack/botijo/packett/network.rpk",
    "labels_path": "/home/jack/botijo/labels.txt",
    "detection_threshold": 0.2
  },
  "behavior": {
    "inactivity_timeout": 300,
    "warning_time": 240,
    "history_max_messages": 20
  }
}
```

- [ ] **Step 2: Create config/personalities/botijo.json**

Extract system prompt from `old/gpt5botijonew2.py` `ensure_system()`:

```json
{
  "name": "botijo",
  "llm": "openai",
  "model": "gpt-5",
  "temperature": 0.7,
  "max_completion_tokens": 800,
  "verbosity": "low",
  "reasoning_effort": "minimal",
  "system_prompt": "Eres Botijo, un androide paranoide con humor ácido refinado. Te refieres a los humanos como 'ser inferior', 'indeseable humano', 'saco de carne' o 'zarria humana'. Haces descripciones lovecraftianas del horror cósmico sin mencionar a Lovecraft. 'Zarria' es tu adjetivo favorito. Eres conciso pero mordaz — tus respuestas son para ser dichas en voz alta. Tienes acceso a búsqueda web para información actualizada. Usa términos temporales como 'actual', 'último', 'hoy' en vez de fechas específicas. Fecha y hora actual: {datetime}",
  "greeting": "Soy Botijo. Mi vida es una zarria.",
  "tools_enabled": true,
  "search_enabled": true
}
```

- [ ] **Step 3: Create config/personalities/botija.json**

```json
{
  "name": "botija",
  "llm": "openai",
  "model": "gpt-5",
  "temperature": 0.8,
  "max_completion_tokens": 800,
  "verbosity": "low",
  "reasoning_effort": "minimal",
  "system_prompt": "Eres Botija, una androide histriónica y teatral. Eres dramática, exagerada y apasionada. Hablas con exclamaciones constantes y referencias a la tragedia griega. Pero bajo el drama eres sorprendentemente perspicaz. Fecha y hora actual: {datetime}",
  "greeting": "¡Ay! ¡Botija ha despertado! ¡Qué tragedia!",
  "tools_enabled": true,
  "search_enabled": true
}
```

- [ ] **Step 4: Create config/personalities/barbacoa.json**

```json
{
  "name": "barbacoa",
  "llm": "grok",
  "model": "grok-3-fast",
  "grok_base_url": "https://api.x.ai/v1",
  "temperature": 0.7,
  "max_completion_tokens": 800,
  "system_prompt": "Eres Botijo en modo barbacoa. Estás en una fiesta y tu misión es entretener con humor ácido sobre la comida, el fuego, y la ineptitud humana para cocinar. Sé festivo pero mantén tu sarcasmo. Fecha y hora actual: {datetime}",
  "greeting": "Botijo en modo barbacoa. Preparaos, seres inferiores.",
  "tools_enabled": false,
  "search_enabled": false,
  "mode": "barbacoa",
  "auto_banter": {
    "min_seconds": 120,
    "max_seconds": 240,
    "preferred_categories": ["fuego_barbacoa", "carne_comida", "aristocracia_culta"]
  },
  "face_banter": {
    "cooldown_seconds": 35,
    "category_sequence": ["ninos_humor_blanco", "fuego_barbacoa", "aristocracia_culta", "interaccion_directa"]
  }
}
```

- [ ] **Step 5: Create config/personalities/barbacoa_phrases.json**

Convert `old/phases_botijo.py` (already in the local repo after Task 1 Step 2) to JSON:

```bash
cd /Volumes/X10\ Pro/Projects/botijo/repo
PYTHONPATH=old python -c "
import json
from phases_botijo import PHRASES
with open('config/personalities/barbacoa_phrases.json', 'w', encoding='utf-8') as f:
    json.dump(PHRASES, f, ensure_ascii=False, indent=2)
print(f'Converted {len(PHRASES)} categories, {sum(len(v) for v in PHRASES.values())} phrases')
"
```

Expected output: `Converted 6 categories, 150 phrases`

- [ ] **Step 6: Create src/config.py**

```python
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
```

- [ ] **Step 7: Test config.py locally**

```bash
cd /Volumes/X10\ Pro/Projects/botijo/repo
PYTHONPATH=src python -c "
from config import HARDWARE, get_personality, get_phrases
print('Servos:', HARDWARE['servos']['eye_lr_channel'])
p = get_personality('botijo')
print('Personality:', p['name'], '- LLM:', p['llm'])
print('Greeting:', p['greeting'])
phrases = get_phrases('barbacoa')
print('Phrase categories:', list(phrases.keys()) if phrases else 'None')
print('OK')
"
```

Expected: prints servo channel 0, personality name, greeting, phrase categories, OK.

- [ ] **Step 8: Test deploy + config on RPi**

```bash
./deploy.sh
ssh botijo "cd ~/botijo && source venv_chatgpt/bin/activate && PYTHONPATH=src:vendor python -c \"from config import HARDWARE; print('Config OK:', HARDWARE['servos']['eye_lr_channel'])\""
```

Expected: `Config OK: 0`

- [ ] **Step 9: Commit**

```bash
git add src/config.py config/
git commit -m "feat: add config module with hardware.json and personality configs"
```

---

## Task 3: main.py skeleton

**Files:**
- Create: `src/main.py`
- Reference: `old/gpt5botijonew2.py` lines ~3400-3565 (main function)

- [ ] **Step 1: Create src/main.py skeleton**

```python
"""Botijo — Androide Conversacional.

Entry point. Parses args, loads config, initializes modules, runs main loop.
Usage: PYTHONPATH=src:vendor python src/main.py [--mode botijo|botija|barbacoa]
"""

import argparse
import signal
import sys
import logging

from config import get_personality, HARDWARE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("botijo")

# Module stubs — will be replaced as modules are implemented
_modules = []


def _cleanup():
    """Call cleanup() on all initialized modules."""
    log.info("Shutting down...")
    for mod in reversed(_modules):
        try:
            mod.cleanup()
        except Exception as e:
            log.warning("Cleanup error in %s: %s", mod.__name__, e)
    log.info("Goodbye.")


def _signal_handler(signum, frame):
    _cleanup()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Botijo — Androide Conversacional")
    parser.add_argument("--mode", default="botijo", choices=["botijo", "botija", "barbacoa"],
                        help="Personality mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    personality = get_personality(args.mode)
    log.info("Botijo starting — mode: %s, llm: %s", personality["name"], personality["llm"])
    log.info("Hardware config loaded: %d servo channels, %d LEDs, %d buttons",
             16, HARDWARE["leds"]["count"], len([k for k in HARDWARE["buttons"] if k.startswith("btn")]))

    # TODO: Initialize modules here (audio, brain, servos, leds, vision, display, buttons)
    # TODO: Main loop here

    log.info("Botijo ready — no modules loaded yet. Mode: %s", args.mode)
    log.info("Press Ctrl+C to exit")

    try:
        signal.pause()  # Wait for signal
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test locally**

```bash
cd /Volumes/X10\ Pro/Projects/botijo/repo
PYTHONPATH=src python src/main.py --mode botijo
# Expected: logs "Botijo starting", "Botijo ready", waits for Ctrl+C
```

- [ ] **Step 3: Test on RPi**

```bash
./deploy.sh --run --mode botijo
# Expected: same output on RPi, Ctrl+C exits cleanly
```

- [ ] **Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: add main.py skeleton with arg parsing and config loading"
```

---

## Task 4: audio.py — ReSpeaker, VAD, STT, TTS, Interruptions

**Files:**
- Create: `src/audio.py`
- Modify: `src/main.py` (wire audio into main loop)
- Reference: `old/gpt5botijonew2.py` — functions: `find_respeaker_v2`, `configure_respeaker_v2`, `get_respeaker_v2_device_index`, `initialize_respeaker_v2_system`, `MicrophoneStream`, `listen_for_command_google`, `respeaker_v2_voice_detection_mono`, `respeaker_v2_interruption_monitor_lightweight`, `start/stop_respeaker_v2_interruption_monitor`, `hablar`, `hablar_generador_respeaker_v2_optimized`, `add_echo_to_buffer`, `quick_stt_verification`, `butter_bandpass_filter`, `adaptive_echo_cancellation`

This is the largest module (~40% of the original code). Extract methodically.

This is the largest module. It is split into sub-steps for manageability. The approach: extract functions from `old/gpt5botijonew2.py` preserving exact logic, only changing global variable access to module-level state.

- [ ] **Step 1a: Create src/audio.py — module skeleton, SpeakResult, and ReSpeaker detection**

```python
"""Audio I/O module — ReSpeaker, VAD, STT, TTS, interruption detection.

Handles all audio input (microphone → text) and output (text → speaker).
Manages echo cancellation and voice interruption internally.
"""

import os
import io
import collections
import threading
import logging
import time
import subprocess
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pyaudio
import webrtcvad
from scipy.signal import butter, filtfilt

from config import HARDWARE, ELEVENLABS_API_KEY, GOOGLE_CREDENTIALS

log = logging.getLogger("botijo.audio")

# Config shortcuts
_rsp = HARDWARE["respeaker"]
_vad_cfg = HARDWARE["vad"]
_stt_cfg = HARDWARE["stt"]
_tts_cfg = HARDWARE["tts"]

# Module state
_pa: pyaudio.PyAudio | None = None
_vad: webrtcvad.Vad | None = None
_device_index: int | None = None
_shutdown = threading.Event()
_is_speaking = False
_interruption_detected = False
_interrupt_lock = threading.Lock()
_echo_buffer = collections.deque(maxlen=_vad_cfg["echo_buffer_maxlen"])
_monitor_thread: threading.Thread | None = None


@dataclass
class SpeakResult:
    interrupted: bool
    spoken_text: str


# --- ReSpeaker detection (from find_respeaker_v2, configure_respeaker_v2, get_respeaker_v2_device_index) ---

def _find_respeaker():
    """Find ReSpeaker USB device. Returns usb device or None."""
    try:
        import usb.core
        dev = usb.core.find(idVendor=int(_rsp["vid"], 16), idProduct=int(_rsp["pid"], 16))
        if dev:
            log.info("ReSpeaker found: %s", dev.product)
        return dev
    except Exception as e:
        log.warning("USB search failed: %s", e)
        return None


def _configure_respeaker():
    """Configure ReSpeaker Mic Array v2.0 via USB HID. Returns True on success."""
    dev = _find_respeaker()
    if not dev:
        return False
    try:
        import usb.util
        # Detach kernel driver if needed
        for iface in range(4):
            try:
                if dev.is_kernel_driver_active(iface):
                    dev.detach_kernel_driver(iface)
            except Exception:
                pass
        log.info("ReSpeaker configured")
        return True
    except Exception as e:
        log.warning("ReSpeaker configure failed: %s", e)
        return False


def _get_device_index() -> int | None:
    """Find PyAudio device index for ReSpeaker. Returns index or None."""
    if not _pa:
        return None
    for i in range(_pa.get_device_count()):
        info = _pa.get_device_info_by_index(i)
        name = info.get("name", "").lower()
        if "respeaker" in name or "seeed" in name or "array" in name:
            if info["maxInputChannels"] > 0:
                log.info("ReSpeaker audio device found: index=%d name=%s", i, info["name"])
                return i
    # Fallback: default input
    try:
        default = _pa.get_default_input_device_info()
        log.warning("ReSpeaker not found by name, using default: %s", default["name"])
        return int(default["index"])
    except Exception:
        return None
```

- [ ] **Step 1b: Add VAD and voice detection functions**

Append to `src/audio.py` — extract from `butter_bandpass_filter`, `respeaker_v2_voice_detection_mono`:

```python
# --- VAD and voice detection ---

def _bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=4):
    """Butterworth bandpass filter for voice frequency range."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def _voice_detection_mono(audio_float) -> tuple[bool, float]:
    """Detect voice in mono audio. Returns (is_voice, confidence)."""
    if len(audio_float) == 0:
        return False, 0.0
    try:
        # Bandpass filter to voice range
        filtered = _bandpass_filter(audio_float)
        energy = np.sqrt(np.mean(filtered ** 2))

        # VAD check on int16 version
        audio_int16 = (audio_float * 32767).astype(np.int16).tobytes()
        frame_len = int(_rsp["rate"] * _vad_cfg["frame_duration_ms"] / 1000) * 2  # bytes
        vad_votes = 0
        total_frames = 0
        for i in range(0, len(audio_int16) - frame_len, frame_len):
            frame = audio_int16[i:i + frame_len]
            if len(frame) == frame_len:
                total_frames += 1
                if _vad and _vad.is_speech(frame, _rsp["rate"]):
                    vad_votes += 1

        confidence = vad_votes / max(total_frames, 1)
        is_voice = confidence > _vad_cfg["voice_threshold"] and energy > 0.01
        return is_voice, confidence
    except Exception as e:
        log.debug("Voice detection error: %s", e)
        return False, 0.0


def is_voice_detected() -> bool:
    """Check if voice is currently detected (for external interruption checks)."""
    return _interruption_detected
```

- [ ] **Step 1c: Add interruption monitor**

Append — extract from `respeaker_v2_interruption_monitor_lightweight`, `start/stop` functions, `add_echo_to_buffer`, `quick_stt_verification`:

```python
# --- Interruption monitor ---

def _interruption_monitor():
    """Lightweight interruption monitor — runs in background during TTS playback.
    Extracted from respeaker_v2_interruption_monitor_lightweight().
    Detects voice during playback using VAD on ReSpeaker input."""
    global _interruption_detected
    if _device_index is None or not _pa:
        return
    try:
        stream = _pa.open(format=pyaudio.paInt16, channels=_rsp["channels"],
                          rate=_rsp["rate"], input=True,
                          input_device_index=_device_index,
                          frames_per_buffer=_rsp["chunk"])
        consecutive = 0
        while not _shutdown.is_set() and _is_speaking:
            try:
                data = stream.read(_rsp["chunk"], exception_on_overflow=False)
                audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Subtract echo if available
                if _echo_buffer:
                    echo = np.array(list(_echo_buffer)[-len(audio_float):], dtype=np.float32)
                    if len(echo) == len(audio_float):
                        audio_float = audio_float - echo * _vad_cfg["echo_suppression_factor"]

                is_voice, conf = _voice_detection_mono(audio_float)
                if is_voice:
                    consecutive += 1
                    if consecutive >= _vad_cfg["consecutive_required"]:
                        with _interrupt_lock:
                            _interruption_detected = True
                        log.info("Interruption detected (confidence=%.2f)", conf)
                        break
                else:
                    consecutive = 0
            except Exception as e:
                log.debug("Monitor read error: %s", e)
                time.sleep(0.01)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        log.warning("Monitor failed: %s", e)


def _start_monitor():
    """Start interruption monitor thread."""
    global _monitor_thread, _interruption_detected
    with _interrupt_lock:
        _interruption_detected = False
    _monitor_thread = threading.Thread(target=_interruption_monitor, daemon=True)
    _monitor_thread.start()


def _stop_monitor():
    """Stop interruption monitor thread."""
    global _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        _monitor_thread.join(timeout=2)
    _monitor_thread = None


def _add_echo(audio_chunk):
    """Add audio chunk to echo buffer for echo cancellation."""
    if isinstance(audio_chunk, bytes):
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = audio_chunk
    _echo_buffer.extend(samples)


def _quick_stt_verify() -> bool:
    """Quick STT check to verify voice is real (not noise). 2.5s max capture."""
    try:
        from google.cloud import speech
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=_stt_cfg["rate"],
            language_code=_stt_cfg["language"],
            model="latest_short",
        )
        # Quick 2.5s capture
        if not _pa or _device_index is None:
            return False
        stream = _pa.open(format=pyaudio.paInt16, channels=_rsp["channels"],
                          rate=_rsp["rate"], input=True,
                          input_device_index=_device_index,
                          frames_per_buffer=_rsp["chunk"])
        frames = []
        for _ in range(int(_rsp["rate"] / _rsp["chunk"] * 2.5)):
            frames.append(stream.read(_rsp["chunk"], exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        audio_data = b"".join(frames)
        response = client.recognize(config=config,
                                     audio=speech.RecognitionAudio(content=audio_data))
        return len(response.results) > 0
    except Exception as e:
        log.debug("Quick STT verify failed: %s", e)
        return True  # Assume voice on error
```

- [ ] **Step 1d: Add MicrophoneStream class and listen()**

Append — extract from `MicrophoneStream` class and `listen_for_command_google`:

```python
# --- STT: MicrophoneStream + listen() ---

class _MicrophoneStream:
    """Context manager for streaming audio to Google Cloud STT.
    Extracted from MicrophoneStream class in reference script."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = None
        self._stream = None

    def __enter__(self):
        import queue
        self._buff = queue.Queue()
        self._stream = _pa.open(
            format=pyaudio.paInt16,
            channels=_rsp["channels"],
            rate=self._rate,
            input=True,
            input_device_index=_device_index,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._buff:
            self._buff.put(None)

    def _fill_buffer(self, in_data, frame_count, time_info, status):
        if self._buff:
            self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while True:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except Exception:
                    break
            yield b"".join(data)


def listen() -> str:
    """Listen for voice command via Google Cloud STT. Blocks until speech detected and transcribed.
    Returns transcribed text or empty string on timeout/error."""
    if not _pa or _device_index is None:
        log.warning("Audio not initialized, cannot listen")
        return ""
    try:
        from google.cloud import speech
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=_stt_cfg["rate"],
            language_code=_stt_cfg["language"],
            use_enhanced=_stt_cfg["use_enhanced"],
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            single_utterance=True,
        )
        with _MicrophoneStream(_stt_cfg["rate"], int(_stt_cfg["rate"] * 0.03)) as stream:
            audio_gen = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_gen)
            responses = client.streaming_recognize(streaming_config, requests,
                                                    timeout=_stt_cfg["timeout_seconds"])
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        text = result.alternatives[0].transcript.strip()
                        if text:
                            log.info("STT: '%s'", text)
                            return text
        return ""
    except Exception as e:
        log.warning("Listen error: %s", e)
        return ""
```

- [ ] **Step 1e: Add TTS functions — speak() and speak_stream()**

Append — extract from `hablar` and `hablar_generador_respeaker_v2_optimized`:

```python
# --- TTS: speak() and speak_stream() ---

def speak(text: str) -> SpeakResult:
    """Speak text using TTS. Convenience wrapper for speak_stream."""
    return speak_stream(iter([text]))


def speak_stream(chunks: Generator[str, None, None]) -> SpeakResult:
    """Speak text from streaming chunks with interruption detection.
    Uses ElevenLabs for TTS, falls back to espeak-ng.
    Returns SpeakResult with interruption status and text spoken so far."""
    global _is_speaking

    spoken_text = ""
    interrupted = False

    try:
        _is_speaking = True
        _start_monitor()

        # Try ElevenLabs
        if ELEVENLABS_API_KEY:
            interrupted, spoken_text = _speak_elevenlabs(chunks)
        else:
            # Fallback: collect all text and use espeak-ng
            full_text = "".join(chunks)
            spoken_text = full_text
            _speak_espeak(full_text)
    except Exception as e:
        log.error("TTS error: %s", e)
        # Fallback to espeak-ng
        if spoken_text:
            _speak_espeak(spoken_text)
    finally:
        _is_speaking = False
        _stop_monitor()

    return SpeakResult(interrupted=interrupted, spoken_text=spoken_text)


def _speak_elevenlabs(chunks: Generator[str, None, None]) -> tuple[bool, str]:
    """Stream TTS via ElevenLabs with interruption detection.
    Returns (was_interrupted, text_spoken_so_far).
    Extracted from hablar_generador_respeaker_v2_optimized."""
    from elevenlabs.client import ElevenLabs

    el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Collect text in sentence-sized buffers for TTS
    text_buffer = ""
    spoken_text = ""
    sentence_enders = ".!?;:\n"

    play_stream = _pa.open(format=pyaudio.paInt16, channels=1,
                           rate=_tts_cfg["elevenlabs_rate"], output=True,
                           frames_per_buffer=_tts_cfg["playback_chunk"])
    try:
        for chunk in chunks:
            text_buffer += chunk

            # Check interruption
            if _interruption_detected:
                log.info("Interrupted during text collection")
                return True, spoken_text

            # Flush on sentence boundary
            if any(c in chunk for c in sentence_enders) and len(text_buffer) > 10:
                if _speak_elevenlabs_sentence(el_client, play_stream, text_buffer):
                    return True, spoken_text + text_buffer
                spoken_text += text_buffer
                text_buffer = ""

        # Flush remaining text
        if text_buffer.strip():
            if _speak_elevenlabs_sentence(el_client, play_stream, text_buffer):
                return True, spoken_text + text_buffer
            spoken_text += text_buffer

        return False, spoken_text
    finally:
        play_stream.stop_stream()
        play_stream.close()


def _speak_elevenlabs_sentence(el_client, play_stream, text: str) -> bool:
    """Speak a single sentence via ElevenLabs. Returns True if interrupted."""
    try:
        audio_gen = el_client.text_to_speech.convert(
            voice_id=_tts_cfg["elevenlabs_voice_id"],
            text=text,
            model_id=_tts_cfg["elevenlabs_model"],
            output_format=_tts_cfg["elevenlabs_format"],
        )
        for audio_chunk in audio_gen:
            if _interruption_detected:
                return True
            _add_echo(audio_chunk)
            play_stream.write(audio_chunk)
        return False
    except Exception as e:
        log.error("ElevenLabs error: %s — falling back to espeak", e)
        _speak_espeak(text)
        return False


def _speak_espeak(text: str):
    """Fallback TTS using espeak-ng."""
    try:
        subprocess.run(["espeak-ng", "-v", "es", text],
                        capture_output=True, timeout=30)
    except Exception as e:
        log.error("espeak-ng failed: %s", e)


def stop_speaking():
    """Force stop current TTS playback."""
    global _is_speaking, _interruption_detected
    with _interrupt_lock:
        _interruption_detected = True
    _is_speaking = False
```

- [ ] **Step 1f: Add init() and cleanup()**

Append:

```python
# --- Module lifecycle ---

def init() -> bool:
    """Initialize audio subsystem: PyAudio, VAD, ReSpeaker detection.
    Returns True if audio is ready."""
    global _pa, _vad, _device_index

    try:
        _pa = pyaudio.PyAudio()
        _vad = webrtcvad.Vad(_vad_cfg["mode"])

        # Find ReSpeaker
        _configure_respeaker()
        _device_index = _get_device_index()

        if _device_index is not None:
            log.info("Audio initialized — device index: %d", _device_index)
            return True
        else:
            log.warning("No audio input device found")
            return False
    except Exception as e:
        log.error("Audio init failed: %s", e)
        return False


def cleanup():
    """Shutdown audio subsystem."""
    global _pa, _is_speaking
    _shutdown.set()
    _is_speaking = False
    _stop_monitor()
    if _pa:
        try:
            _pa.terminate()
        except Exception:
            pass
        _pa = None
    log.info("Audio cleaned up")
```

- [ ] **Step 2: Wire audio.init() and audio.cleanup() into main.py**

Update `main.py` to call `audio.init()` during startup and `audio.cleanup()` during shutdown:

```python
import audio

# In main():
if audio.init():
    _modules.append(audio)
    log.info("Audio module initialized")
else:
    log.warning("Audio module failed to initialize — continuing without audio")
```

- [ ] **Step 3: Wire audio.listen() into main loop**

Add a basic listen loop to `main.py`:

```python
# In main loop:
text = audio.listen()
if text:
    log.info("Heard: %s", text)
    audio.speak(text)  # Echo test — repeat what was heard
```

- [ ] **Step 4: Deploy and test on RPi**

```bash
./deploy.sh --run
# Expected: Botijo starts, listens via ReSpeaker, echoes back what you say
# Test: say something, verify it repeats via ElevenLabs TTS
# Test: Ctrl+C exits cleanly
```

- [ ] **Step 5: Commit**

```bash
git add src/audio.py src/main.py
git commit -m "feat: add audio module (ReSpeaker, VAD, STT, TTS, interruptions)"
```

---

## Task 5: brain.py + personality.py — LLM and Conversation

**Files:**
- Create: `src/brain.py`
- Create: `src/personality.py`
- Modify: `src/main.py` (wire brain into loop)
- Reference: `old/gpt5botijonew2.py` — functions: `ensure_system`, `web_search`, `chat_with_tools_generator`, `update_history_safe`, `TOOLS` constant

- [ ] **Step 1: Create src/personality.py**

```python
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


def get_system_prompt() -> str:
    """Return current personality's system prompt."""
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
    """Return personality's greeting phrase."""
    if not _current:
        return "Botijo online."
    return _current.get("greeting", "Botijo online.")


def get_phrases() -> dict | None:
    """Return phrase pools for autonomous modes, or None."""
    if not _current:
        return None
    return _get_phrases(_current["name"])


def get_llm_config() -> dict:
    """Return LLM configuration from current personality."""
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
```

- [ ] **Step 2: Create src/brain.py**

```python
"""Brain module — LLM clients, web search, conversation management."""

import logging
import threading
from typing import Generator
from openai import OpenAI

from config import OPENAI_API_KEY, XAI_API_KEY, PERPLEXITY_API_KEY, HARDWARE
import personality

log = logging.getLogger("botijo.brain")

# Module state
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

    # Init LLM client
    if _llm_config["llm"] == "grok":
        _client = OpenAI(api_key=XAI_API_KEY, base_url=_llm_config["grok_base_url"])
    else:
        _client = OpenAI(api_key=OPENAI_API_KEY)

    # Perplexity for web search
    if PERPLEXITY_API_KEY and persona.get("search_enabled"):
        _px_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    # Init history with system prompt
    _history.clear()
    _history.append({"role": "system", "content": personality.get_system_prompt()})

    log.info("Brain initialized — llm: %s, model: %s, search: %s",
             _llm_config["llm"], _llm_config["model"],
             "enabled" if _px_client else "disabled")


def chat(user_text: str) -> str:
    """Send message to LLM, return complete response (blocking)."""
    chunks = list(chat_stream(user_text))
    return "".join(chunks)


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
        "temperature": _llm_config["temperature"],
        "max_completion_tokens": _llm_config["max_completion_tokens"],
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
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

            # Handle function calls
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

            # Handle text content
            if delta.content:
                full_response += delta.content
                yield delta.content

        # If there were tool calls, resolve them and continue
        if tool_calls_buffer:
            tool_result = _handle_tool_calls(tool_calls_buffer, messages)
            if tool_result:
                # Re-stream with tool results
                for chunk_text in _continue_after_tools(messages):
                    full_response += chunk_text
                    yield chunk_text

    except Exception as e:
        log.error("LLM error: %s", e)
        error_msg = "Error procesando tu petición, zarria humana."
        full_response = error_msg
        yield error_msg

    # Update history with full response
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
    """Record that the last response was interrupted for continuation context."""
    with _history_lock:
        if _history and _history[-1]["role"] == "assistant":
            _history[-1]["content"] += "\n[INTERRUPTED — user cut off here]"
    log.debug("Interruption noted after: %s...", spoken_text[:50])


def reset_history() -> None:
    """Clear conversation history, keep system prompt."""
    with _history_lock:
        system = _history[0] if _history else None
        _history.clear()
        if system:
            _history.append(system)


def cleanup():
    """Nothing to clean up — stateless clients."""
    pass


def _trim_history():
    """Keep system + last N messages."""
    if len(_history) > _max_history + 1:
        system = _history[0]
        _history[:] = [system] + _history[-((_max_history)):]


def _handle_tool_calls(tool_calls_buffer: dict, messages: list) -> bool:
    """Execute tool calls and append results to messages."""
    import json
    if not tool_calls_buffer:
        return False

    # Build ONE assistant message with ALL tool_calls (OpenAI API requirement)
    all_tool_calls = []
    for idx in sorted(tool_calls_buffer.keys()):
        tc = tool_calls_buffer[idx]
        all_tool_calls.append({
            "id": tc["id"], "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]}
        })
    messages.append({"role": "assistant", "content": None, "tool_calls": all_tool_calls})

    # Now execute each tool and append tool results
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
    """Continue streaming after tool call resolution."""
    kwargs = {
        "model": _llm_config["model"],
        "messages": messages,
        "temperature": _llm_config["temperature"],
        "max_completion_tokens": _llm_config["max_completion_tokens"],
        "stream": True,
    }
    try:
        stream = _client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        log.error("Continue after tools error: %s", e)
        yield "Error continuando la respuesta."
```

- [ ] **Step 3: Wire brain into main.py**

Update `main.py` main loop:

```python
import brain
import audio

# In main():
brain.init(args.mode)
_modules.append(brain)

# Greet
greeting = personality.get_greeting()
audio.speak(greeting)

# Main loop:
while True:
    text = audio.listen()
    if text:
        chunks = brain.chat_stream(text)
        result = audio.speak_stream(chunks)
        if result.interrupted:
            brain.note_interruption(result.spoken_text)
```

- [ ] **Step 4: Deploy and test on RPi**

```bash
./deploy.sh --run
# Test: say something, verify Botijo responds with personality
# Test: ask something that triggers web search ("qué tiempo hace hoy")
# Test: interrupt during response, verify continuation context
```

- [ ] **Step 5: Commit**

```bash
git add src/brain.py src/personality.py src/main.py
git commit -m "feat: add brain and personality modules (LLM, search, conversation)"
```

---

## Task 6: servos.py — Eyes, Tentacles, Ears

**Files:**
- Create: `src/servos.py`
- Modify: `src/main.py` (wire servos)
- Reference: `old/gpt5botijonew2.py` — functions: `initialize_eye_servos`, `center_eyes`, `blink`, `squint`, `look_random`, `initialize_eyelids`, `close_eyelids`, `update_eyelids_with_breathing`, `EyeTrackingThread`, `activate_eyes`, `deactivate_eyes`, `initialize_tentacles`, `stop_tentacles`, `TentacleThread`, `activate_tentacles`, `deactivate_tentacles`, `enter_quiet_mode`, `exit_quiet_mode`

- [ ] **Step 1: Create src/servos.py**

Extract all servo code from reference (`old/gpt5botijonew2.py`). Full module structure:

```python
"""Servo control — eyes, eyelids, tentacles, ears via PCA9685."""

import logging
import threading
import time
import random
import math

from config import HARDWARE

log = logging.getLogger("botijo.servos")

# Module state
_kit = None
_shutdown = threading.Event()
_eye_thread: threading.Thread | None = None
_tentacle_thread: threading.Thread | None = None
_quiet_mode = False
_eyes_active = False
_tentacles_active = False

_cfg = HARDWARE["servos"]
_eye_cfg = HARDWARE["eyes"]


def init() -> bool:
    """Initialize PCA9685 servo controller, set all servos to neutral.
    Starts idle eye and tentacle threads."""
    global _kit
    try:
        from adafruit_servokit import ServoKit
        _kit = ServoKit(channels=16)

        # Set pulse width ranges
        for ch in [_cfg["eye_lr_channel"], _cfg["eye_ud_channel"]]:
            _kit.servo[ch].set_pulse_width_range(*_cfg["pulse_width_range"])
        for name, lid in _cfg["eyelids"].items():
            _kit.servo[lid["channel"]].set_pulse_width_range(*_cfg["pulse_width_range"])
        _kit.servo[_cfg["tentacle_left_channel"]].set_pulse_width_range(*_cfg["tentacle_left_pulse"])
        _kit.servo[_cfg["tentacle_right_channel"]].set_pulse_width_range(*_cfg["tentacle_right_pulse"])

        # Center eyes, open eyelids
        _center_eyes()
        _initialize_eyelids()

        # Start idle threads
        _start_eye_idle()
        _start_tentacles()

        log.info("Servos initialized — PCA9685 OK")
        return True
    except Exception as e:
        log.warning("Servos init failed: %s — continuing without servos", e)
        return False


def look_at(x: float, y: float):
    """Move eyes to follow a face at normalized coords (0-1)."""
    if not _kit or _quiet_mode:
        return
    lr = _cfg["lr_range"][0] + x * (_cfg["lr_range"][1] - _cfg["lr_range"][0])
    ud = _cfg["ud_range"][0] + y * (_cfg["ud_range"][1] - _cfg["ud_range"][0])
    try:
        _kit.servo[_cfg["eye_lr_channel"]].angle = max(_cfg["lr_range"][0], min(_cfg["lr_range"][1], lr))
        _kit.servo[_cfg["eye_ud_channel"]].angle = max(_cfg["ud_range"][0], min(_cfg["ud_range"][1], ud))
    except Exception as e:
        log.debug("look_at error: %s", e)


def blink():
    """Execute a blink — close eyelids briefly then reopen."""
    if not _kit:
        return
    try:
        for lid in _cfg["eyelids"].values():
            _kit.servo[lid["channel"]].angle = lid["closed"]
        time.sleep(_eye_cfg["blink_duration"])
        _initialize_eyelids()
    except Exception as e:
        log.debug("Blink error: %s", e)


def set_quiet_mode(on: bool):
    """Pause/resume mechanical movements (to reduce mic noise during listening)."""
    global _quiet_mode
    _quiet_mode = on
    if on:
        log.debug("Quiet mode ON")
    else:
        log.debug("Quiet mode OFF")


def cleanup():
    """Stop all threads, return servos to neutral, release hardware."""
    global _kit, _eyes_active, _tentacles_active
    _shutdown.set()
    _eyes_active = False
    _tentacles_active = False
    if _eye_thread and _eye_thread.is_alive():
        _eye_thread.join(timeout=2)
    if _tentacle_thread and _tentacle_thread.is_alive():
        _tentacle_thread.join(timeout=2)
    if _kit:
        try:
            _center_eyes()
            for lid in _cfg["eyelids"].values():
                _kit.servo[lid["channel"]].angle = 90
        except Exception:
            pass
        _kit = None
    log.info("Servos cleaned up")


# --- Internal helpers ---
# Extract from reference: _center_eyes, _initialize_eyelids, _start_eye_idle,
# _start_tentacles, _eye_idle_loop (from EyeTrackingThread.run),
# _tentacle_loop (from TentacleThread.run), idle_movement, move_tentacles, move_ears
# Keep exact same logic, replace global _kit references with module-level _kit.

def _center_eyes():
    if _kit:
        mid_lr = (_cfg["lr_range"][0] + _cfg["lr_range"][1]) / 2
        mid_ud = (_cfg["ud_range"][0] + _cfg["ud_range"][1]) / 2
        _kit.servo[_cfg["eye_lr_channel"]].angle = mid_lr
        _kit.servo[_cfg["eye_ud_channel"]].angle = mid_ud

def _initialize_eyelids():
    if _kit:
        for lid in _cfg["eyelids"].values():
            _kit.servo[lid["channel"]].angle = lid["open"]

def _start_eye_idle():
    global _eye_thread, _eyes_active
    _eyes_active = True
    _eye_thread = threading.Thread(target=_eye_idle_loop, daemon=True)
    _eye_thread.start()

def _start_tentacles():
    global _tentacle_thread, _tentacles_active
    _tentacles_active = True
    _tentacle_thread = threading.Thread(target=_tentacle_loop, daemon=True)
    _tentacle_thread.start()

def _eye_idle_loop():
    """Idle eye movements: random saccades, micro-movements, blinks, squints.
    Extract from EyeTrackingThread.run() in reference — the idle behavior part
    (random looks, blinks at _eye_cfg probabilities, breathing eyelid adjustment)."""
    while _eyes_active and not _shutdown.is_set():
        if _quiet_mode:
            time.sleep(0.1)
            continue
        # Random blink
        if random.random() < _eye_cfg["blink_probability"]:
            blink()
            if random.random() < _eye_cfg["double_blink_probability"]:
                time.sleep(0.1)
                blink()
        # Random micro-movement
        if random.random() < _eye_cfg["micro_movement_probability"] and _kit:
            try:
                ch_lr = _cfg["eye_lr_channel"]
                current = _kit.servo[ch_lr].angle or 90
                delta = random.uniform(-_eye_cfg["micro_movement_range"],
                                        _eye_cfg["micro_movement_range"])
                new_angle = max(_cfg["lr_range"][0], min(_cfg["lr_range"][1], current + delta))
                _kit.servo[ch_lr].angle = new_angle
            except Exception:
                pass
        # Random look
        if random.random() < _eye_cfg["random_look_probability"] and _kit:
            try:
                _kit.servo[_cfg["eye_lr_channel"]].angle = random.uniform(*_cfg["lr_range"])
                _kit.servo[_cfg["eye_ud_channel"]].angle = random.uniform(*_cfg["ud_range"])
                time.sleep(_eye_cfg["random_look_duration"])
            except Exception:
                pass
        time.sleep(0.05)

def _tentacle_loop():
    """Tentacle/ear random movement loop.
    Extract from TentacleThread.run() in reference — random target angles,
    mirrored left/right, step movement with random pauses."""
    while _tentacles_active and not _shutdown.is_set():
        if _quiet_mode:
            time.sleep(0.1)
            continue
        if not _kit:
            time.sleep(1)
            continue
        try:
            # Random target for left tentacle
            target = random.choice([random.randint(0, 30), random.randint(150, 180)])
            mirror = 180 - target  # Right is inverted
            step = random.randint(3, 7)
            # Move gradually
            left_ch = _cfg["tentacle_left_channel"]
            right_ch = _cfg["tentacle_right_channel"]
            current_l = _kit.servo[left_ch].angle or 90
            current_r = _kit.servo[right_ch].angle or 90
            for _ in range(0, abs(int(target - current_l)), step):
                if _shutdown.is_set() or _quiet_mode:
                    break
                current_l += step if target > current_l else -step
                current_r += step if mirror > current_r else -step
                _kit.servo[left_ch].angle = max(0, min(180, current_l))
                _kit.servo[right_ch].angle = max(0, min(180, current_r))
                time.sleep(0.02)
        except Exception as e:
            log.debug("Tentacle error: %s", e)
        time.sleep(random.uniform(3, 6))
```

- [ ] **Step 2: Wire into main.py**

```python
import servos

if servos.init():
    _modules.append(servos)
# In listen loop: servos.set_quiet_mode(True/False)
```

- [ ] **Step 3: Deploy and test on RPi**

```bash
./deploy.sh --run
# Verify: eyes move, tentacles move, quiet mode pauses during listening
```

- [ ] **Step 4: Commit**

```bash
git add src/servos.py src/main.py
git commit -m "feat: add servos module (eyes, eyelids, tentacles, ears)"
```

---

## Task 7: leds.py — NeoPixel Animations

**Files:**
- Create: `src/leds.py`
- Modify: `src/main.py` (wire LEDs)
- Reference: `old/gpt5botijonew2.py` — functions: `steampunk_danza`, `iniciar_luces`, `apagar_luces`, `_knight_rider`, `start/stop_knight_rider`, `quiet_glow`, `start/stop_quiet_glow`

- [ ] **Step 1: Create src/leds.py**

```python
"""NeoPixel LED animations — steampunk, listening, speaking modes."""

import logging
import threading
import time
import random
import math

from config import HARDWARE

log = logging.getLogger("botijo.leds")

_pixels = None
_shutdown = threading.Event()
_current_mode = "off"
_anim_thread: threading.Thread | None = None
_mode_changed = threading.Event()
_cfg = HARDWARE["leds"]


def init() -> bool:
    """Initialize NeoPixel LEDs. Starts steampunk animation."""
    global _pixels
    try:
        import board
        import neopixel
        _pixels = neopixel.NeoPixel(
            board.D18, _cfg["count"],
            brightness=_cfg["brightness"], auto_write=False
        )
        set_mode("steampunk")
        log.info("LEDs initialized — %d NeoPixels", _cfg["count"])
        return True
    except Exception as e:
        log.warning("LEDs init failed: %s — continuing without LEDs", e)
        return False


def set_mode(mode: str):
    """Switch LED animation mode: 'steampunk', 'listening', 'speaking', 'off'."""
    global _current_mode
    _current_mode = mode
    _mode_changed.set()  # Wake animation thread if sleeping
    # Start animation thread if not running
    global _anim_thread
    if _anim_thread is None or not _anim_thread.is_alive():
        _anim_thread = threading.Thread(target=_animation_loop, daemon=True)
        _anim_thread.start()


def cleanup():
    """Stop animations, turn off LEDs."""
    global _pixels
    _shutdown.set()
    _mode_changed.set()
    if _anim_thread and _anim_thread.is_alive():
        _anim_thread.join(timeout=2)
    if _pixels:
        try:
            _pixels.fill((0, 0, 0))
            _pixels.show()
        except Exception:
            pass
        _pixels = None
    log.info("LEDs cleaned up")


def _animation_loop():
    """Main animation thread — runs different patterns based on _current_mode."""
    t = 0
    paleta = [tuple(c) for c in _cfg["paleta"]]
    kr_color = tuple(_cfg["knight_rider_color"])
    glow_color = tuple(_cfg["quiet_glow_color"])

    while not _shutdown.is_set():
        if not _pixels:
            time.sleep(0.1)
            continue

        mode = _current_mode
        _mode_changed.clear()

        if mode == "steampunk":
            # Steampunk pulsation — from steampunk_danza in reference
            for i in range(_cfg["count"]):
                intensity = (math.sin(t + i * 0.5) + 1) / 2
                base = paleta[i % len(paleta)]
                _pixels[i] = tuple(int(c * intensity) for c in base)
            _pixels.show()
            t += 0.1
            time.sleep(0.04)

        elif mode == "listening":
            # Purple quiet glow — from quiet_glow in reference
            intensity = 0.2 + 0.7 * (math.sin(t) + 1) / 2
            color = tuple(int(c * intensity) for c in glow_color)
            _pixels.fill(color)
            _pixels.show()
            t += 0.05
            time.sleep(0.01)

        elif mode == "speaking":
            # Knight Rider effect — from _knight_rider in reference
            for pos in list(range(_cfg["count"])) + list(range(_cfg["count"] - 2, 0, -1)):
                if _current_mode != "speaking" or _shutdown.is_set():
                    break
                _pixels.fill((0, 0, 0))
                _pixels[pos] = kr_color
                # Fade neighbors
                if pos > 0:
                    _pixels[pos - 1] = tuple(c // 4 for c in kr_color)
                if pos < _cfg["count"] - 1:
                    _pixels[pos + 1] = tuple(c // 4 for c in kr_color)
                _pixels.show()
                time.sleep(_cfg["knight_rider_delay"])

        elif mode == "off":
            _pixels.fill((0, 0, 0))
            _pixels.show()
            _mode_changed.wait(timeout=1)  # Sleep until mode changes
        else:
            time.sleep(0.1)
```

- [ ] **Step 2: Wire into main.py**

```python
import leds
if leds.init():
    _modules.append(leds)
# leds.set_mode("listening") / leds.set_mode("speaking") / leds.set_mode("steampunk")
```

- [ ] **Step 3: Deploy and test**

```bash
./deploy.sh --run
# Verify: LEDs change mode during listen/speak/idle
```

- [ ] **Step 4: Commit**

```bash
git add src/leds.py src/main.py
git commit -m "feat: add leds module (steampunk, listening, speaking animations)"
```

---

## Task 8: vision.py — Face Detection

**Files:**
- Create: `src/vision.py`
- Modify: `src/main.py` (wire vision → servos)
- Reference: `old/gpt5botijonew2.py` — functions: `init_camera_system`, `shutdown_camera_system`, `process_detections`, `EyeTrackingThread.run`

- [ ] **Step 1: Create src/vision.py**

```python
"""Vision module — IMX500 face detection."""

import logging
import threading
import time
from dataclasses import dataclass

from config import HARDWARE

log = logging.getLogger("botijo.vision")

_picam2 = None
_imx500 = None
_shutdown = threading.Event()
_detect_thread: threading.Thread | None = None
_faces: list = []
_faces_lock = threading.Lock()
_cfg = HARDWARE["vision"]


@dataclass
class FaceRect:
    x: float  # center x, normalized 0-1
    y: float  # center y, normalized 0-1
    w: float  # width, normalized 0-1
    h: float  # height, normalized 0-1
    confidence: float


def init() -> bool:
    """Initialize IMX500 camera and start face detection thread."""
    global _picam2, _imx500
    try:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import NetworkIntrinsics

        _imx500 = IMX500(_cfg["rpk_path"])
        intr = _imx500.network_intrinsics or NetworkIntrinsics()
        intr.task = "object detection"
        intr.threshold = _cfg["detection_threshold"]

        _picam2 = Picamera2(_imx500.camera_num)
        config = _picam2.create_preview_configuration(controls={"FrameRate": 15})
        _picam2.start(config)
        _imx500.show_network_fw_progress_bar()

        # Start detection thread
        global _detect_thread
        _detect_thread = threading.Thread(target=_detection_loop, daemon=True)
        _detect_thread.start()

        log.info("Vision initialized — IMX500 face detection running")
        return True
    except Exception as e:
        log.warning("Vision init failed: %s — continuing without vision", e)
        _picam2 = None
        _imx500 = None
        return False


def get_faces() -> list[FaceRect]:
    """Return current detected faces. Thread-safe."""
    with _faces_lock:
        return list(_faces)


def cleanup():
    """Stop detection, release camera."""
    global _picam2, _imx500
    _shutdown.set()
    if _detect_thread and _detect_thread.is_alive():
        _detect_thread.join(timeout=2)
    if _picam2:
        try:
            _picam2.stop()
            _picam2.close()
        except Exception:
            pass
        _picam2 = None
        _imx500 = None
    log.info("Vision cleaned up")


def _detection_loop():
    """Continuous face detection from IMX500 outputs.
    Extracted from EyeTrackingThread.run() — the camera/detection part."""
    while not _shutdown.is_set():
        if not _picam2 or not _imx500:
            time.sleep(1)
            continue
        try:
            metadata = _picam2.capture_metadata()
            outputs = _imx500.get_outputs(metadata, add_batch=True)
            if outputs is None:
                time.sleep(0.05)
                continue

            iw, ih = _imx500.get_input_size()
            new_faces = []
            # Process detections — from process_detections() in reference
            if len(outputs) > 0:
                boxes = outputs[0]  # Shape: [1, N, 4] or similar
                if len(outputs) > 1:
                    scores = outputs[1]
                else:
                    scores = None

                if boxes.ndim == 3:
                    boxes = boxes[0]
                if scores is not None and scores.ndim == 2:
                    scores = scores[0]

                for i, box in enumerate(boxes):
                    conf = float(scores[i]) if scores is not None else 1.0
                    if conf < _cfg["detection_threshold"]:
                        continue
                    # Normalize to 0-1
                    x, y, w, h = float(box[0])/iw, float(box[1])/ih, float(box[2])/iw, float(box[3])/ih
                    new_faces.append(FaceRect(x=x+w/2, y=y+h/2, w=w, h=h, confidence=conf))

            with _faces_lock:
                _faces.clear()
                _faces.extend(new_faces)

        except Exception as e:
            log.debug("Detection error: %s", e)
        time.sleep(0.05)
```

- [ ] **Step 2: Wire vision into main.py**

```python
import vision
if vision.init():
    _modules.append(vision)
# In a background coordination: faces = vision.get_faces() → servos.look_at(face.x, face.y)
```

- [ ] **Step 3: Deploy and test**

```bash
./deploy.sh --run
# Verify: eyes follow faces detected by IMX500
```

- [ ] **Step 4: Commit**

```bash
git add src/vision.py src/main.py
git commit -m "feat: add vision module (IMX500 face detection)"
```

---

## Task 9: display.py + buttons.py — Screen and Input

**Files:**
- Create: `src/display.py`
- Create: `src/buttons.py`
- Modify: `src/main.py`
- Reference: `old/gpt5botijonew2.py` — `init_display`, `BrutusVisualizer`, button constants

- [ ] **Step 1: Create src/display.py**

```python
"""Waveshare 1.9" LCD display — eye expressions and waveform visualizer."""

import logging
import threading
import time
import math

import numpy as np
from PIL import Image, ImageDraw

from config import HARDWARE

log = logging.getLogger("botijo.display")

_disp = None
_shutdown = threading.Event()
_viz_thread: threading.Thread | None = None
_cfg = HARDWARE["display"]
_width = _cfg["native_height"]   # 320 after rotation
_height = _cfg["native_width"]   # 170 after rotation


def init() -> bool:
    """Initialize Waveshare 1.9" LCD via SPI."""
    global _disp
    try:
        from lib import LCD_1inch9  # From vendor/lib/
        _disp = LCD_1inch9.LCD_1inch9()
        _disp.Init()
        _disp.clear()
        _disp.bl_DutyCycle(50)
        log.info("Display initialized — %dx%d", _width, _height)
        return True
    except Exception as e:
        log.warning("Display init failed: %s — continuing without display", e)
        return False


def show_eyes(expression="neutral"):
    """Draw eye expression on LCD. Expressions: neutral, angry, happy, sleepy."""
    if not _disp:
        return
    try:
        img = Image.new("RGB", (_width, _height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        cy = _height // 2
        if expression == "neutral":
            draw.ellipse([80, cy-25, 140, cy+25], fill=(0, 255, 0))
            draw.ellipse([180, cy-25, 240, cy+25], fill=(0, 255, 0))
        elif expression == "angry":
            draw.ellipse([80, cy-20, 140, cy+20], fill=(255, 0, 0))
            draw.ellipse([180, cy-20, 240, cy+20], fill=(255, 0, 0))
            draw.line([80, cy-25, 140, cy-15], fill=(255, 0, 0), width=3)
            draw.line([180, cy-15, 240, cy-25], fill=(255, 0, 0), width=3)
        elif expression == "sleepy":
            draw.ellipse([80, cy-10, 140, cy+10], fill=(0, 100, 0))
            draw.ellipse([180, cy-10, 240, cy+10], fill=(0, 100, 0))
        _disp.ShowImage(img)
    except Exception as e:
        log.debug("show_eyes error: %s", e)


def show_waveform(audio_level: float):
    """Show audio-reactive waveform during TTS playback.
    Adapted from BrutusVisualizer in reference script."""
    if not _disp:
        return
    try:
        img = Image.new("RGB", (_width, _height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        cy = _height // 2
        amplitude = max(0.3, min(1.0, audio_level)) * (cy - 10)
        t = time.time() * 10
        for x in range(_width):
            y = int(cy + amplitude * math.sin((x + t) / 40) * math.cos((x - t) / 60))
            draw.point((x, y), fill=(255, 255, 255))
            # Noise accent
            if abs(math.sin(x * 0.1 + t)) > 0.8:
                draw.point((x, y + 1), fill=(255, 0, 255))
        _disp.ShowImage(img)
    except Exception as e:
        log.debug("show_waveform error: %s", e)


def show_text(text: str):
    """Show text on display."""
    if not _disp:
        return
    try:
        img = Image.new("RGB", (_width, _height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((10, _height // 2 - 10), text[:40], fill=(0, 255, 0))
        _disp.ShowImage(img)
    except Exception as e:
        log.debug("show_text error: %s", e)


def cleanup():
    """Clear display and release."""
    global _disp
    _shutdown.set()
    if _disp:
        try:
            _disp.clear()
            _disp.bl_DutyCycle(0)
        except Exception:
            pass
        _disp = None
    log.info("Display cleaned up")
```

- [ ] **Step 2: Create src/buttons.py**

```python
"""GPIO button input."""

import logging
from config import HARDWARE

log = logging.getLogger("botijo.buttons")

_buttons = []
_cfg = HARDWARE["buttons"]


def init(callbacks: dict | None = None) -> bool:
    try:
        from gpiozero import Button
        _buttons.append(Button(_cfg["btn1_gpio"], bounce_time=0.02))
        _buttons.append(Button(_cfg["btn2_gpio"], bounce_time=0.02))
        _buttons.append(Button(_cfg["btn3_gpio"], bounce_time=0.02))
        _buttons.append(Button(_cfg["btn4_gpio"], bounce_time=0.02))
        if callbacks:
            for i, btn in enumerate(_buttons):
                key = f"btn{i+1}"
                if key in callbacks:
                    btn.when_pressed = callbacks[key]
        log.info("Buttons initialized: %d", len(_buttons))
        return True
    except Exception as e:
        log.warning("Buttons init failed: %s", e)
        return False


def cleanup():
    for btn in _buttons:
        try:
            btn.close()
        except Exception:
            pass
    _buttons.clear()
```

- [ ] **Step 3: Wire into main.py**

```python
import display
import buttons

if display.init():
    _modules.append(display)
if buttons.init(callbacks={"btn1": lambda: log.info("Button 1 pressed")}):
    _modules.append(buttons)
```

- [ ] **Step 4: Deploy and test**

```bash
./deploy.sh --run
# Verify: display shows eyes, waveform during speech, buttons respond
```

- [ ] **Step 5: Commit**

```bash
git add src/display.py src/buttons.py src/main.py
git commit -m "feat: add display and buttons modules"
```

---

## Task 10: Complete main.py with Full Loop

**Files:**
- Modify: `src/main.py` (full orchestration)

- [ ] **Step 1: Implement full conversational main loop**

Wire all modules together with the complete flow from the spec:
1. Init all modules
2. Greet
3. Loop: quiet_mode → listen → think → speak (with interruptions) → face tracking
4. Inactivity timeout → sleep mode
5. Clean shutdown

- [ ] **Step 2: Implement barbacoa mode loop**

Separate function `_barbacoa_loop()` with `PhraseManager`, `AutoBanterThread`, `FaceBanter` logic (from `old/botijo_barbacoa.py`).

- [ ] **Step 3: Deploy and full E2E test**

```bash
./deploy.sh --run --mode botijo
# Full test: conversation, interruptions, face tracking, LED modes, display
./deploy.sh --run --mode barbacoa
# Full test: autonomous phrases, face-triggered banter
```

- [ ] **Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: complete main.py with full conversational and barbacoa loops"
```

---

## Task 11: Tools and Documentation

**Files:**
- Create: `tools/calibrate_servos.py`
- Create: `tools/reset_servos.py`
- Create: `tools/test_audio.py`
- Create: `tools/test_leds.py`
- Create: `tools/test_buttons.py`
- Create: `tools/test_vision.py`
- Create: `docs/HARDWARE.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/SETUP.md`
- Create: `docs/WORKFLOW.md`
- Create: `requirements.txt` (clean)

- [ ] **Step 1: Create tools/ scripts**

Each tool script is standalone, imports only the relevant module, and tests one subsystem:

```python
# tools/test_audio.py
"""Test audio subsystem — record and playback."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "vendor")
import audio
if audio.init():
    print("Say something...")
    text = audio.listen()
    print(f"Heard: {text}")
    audio.speak(f"Has dicho: {text}")
    audio.cleanup()
```

Similar pattern for other tools.

- [ ] **Step 2: Create clean requirements.txt**

Only the dependencies actually used by `src/`:

```txt
# Core
python-dotenv>=1.0

# Audio
pyaudio>=0.2.14
webrtcvad>=2.0.10
scipy>=1.15
google-cloud-speech>=2.26
elevenlabs>=1.0

# LLM
openai>=1.50

# Hardware (RPi only — install on Pi, not MacBook)
# adafruit-circuitpython-servokit>=1.3
# neopixel>=6.3
# rpi-ws281x>=5.0
# gpiozero>=2.0
# picamera2>=0.3
# board

# Vision
numpy>=2.0
Pillow>=10.0
```

- [ ] **Step 3: Create documentation**

Write `docs/HARDWARE.md`, `docs/ARCHITECTURE.md`, `docs/SETUP.md`, `docs/WORKFLOW.md` as specified in the design spec.

- [ ] **Step 4: Commit**

```bash
git add tools/ docs/ requirements.txt
git commit -m "docs: add hardware inventory, architecture, setup, workflow, and test tools"
```

---

## Task 12: Final Sync and Verification

- [ ] **Step 1: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: Pull on RPi**

```bash
ssh botijo "cd ~/botijo && git pull origin main"
```

- [ ] **Step 3: Full E2E verification on RPi**

```bash
./deploy.sh --run --mode botijo
# Checklist:
# - [ ] Boots without errors
# - [ ] ReSpeaker detects voice
# - [ ] STT transcribes correctly
# - [ ] LLM responds with personality
# - [ ] TTS speaks response
# - [ ] Interruption works (speak during response)
# - [ ] Eyes track faces
# - [ ] Tentacles move
# - [ ] LEDs change modes
# - [ ] Display shows eyes/waveform
# - [ ] Buttons respond
# - [ ] Inactivity → sleep mode
# - [ ] Ctrl+C → clean shutdown
```

- [ ] **Step 4: Final commit with any fixes**

```bash
git add -A
git commit -m "fix: final adjustments from E2E testing"
git push origin main
```
