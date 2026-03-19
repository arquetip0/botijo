# Botijo Reorganization — Design Spec

**Date:** 2026-03-19
**Author:** Jacobo + Claude Code
**Status:** Reviewed (spec review v1 — all critical issues resolved)

---

## 1. Context

Botijo is a physical android robot controlled by a Raspberry Pi 5. It listens, thinks, and responds with acid humor while moving animatronic eyes, tentacles, and LEDs. The project has grown organically over ~1 year (April 2025 — present) resulting in 127+ Python scripts, many of them iterative copies of the same functionality with no modular structure.

### Current State
- **RPi:** Raspberry Pi 5, 8GB RAM, 458GB NVMe, Debian Bookworm
- **Repo:** github.com/arquetip0/botijo (public), 19 commits on `main`
- **Code:** 127+ .py files at root level, no package structure, no imports between scripts
- **Scripts are monolithic:** each major script (60K-142K bytes) contains ALL subsystem code embedded (audio, servos, LEDs, vision, LLM, etc.)
- **No Docker, no CI:** everything runs directly in a virtualenv (`venv_chatgpt`)
- **Panel:** Simple web frontend + Flask backend, running as systemd service
- **662 untracked files** on the RPi (scripts, binaries, models, downloaded libs)

### Goal
Transform Botijo into a **modular conversational android platform** — a perpetual R&D prototype for a future production line of conversational androids and robots.

- A) **Core:** Best-in-class conversational android (personality, voice, interruptions, memory, multimodal)
- B) **Platform:** Generic modular framework where different personalities, capabilities, and hardware can be swapped via config
- C) **Experimental:** Test integrations like LiveKit/Néstor, gallery reception, etc. — these inform the platform but are not the primary purpose. A dedicated gallery android will be built separately.

### Constraints
- Keep all original scripts in `old/` — nothing gets deleted
- Don't change LLM providers now (keep OpenAI GPT-5, Grok, Perplexity as-is)
- Hardware is fully connected and working (except Picovoice and Piper, unused)
- Panel (web + Flask backend) is preserved, will be improved later (circular display)
- Development happens on MacBook Air M4, testing on RPi via rsync

---

## 2. Repository Structure

```
botijo/
├── src/
│   ├── main.py              # Entry point: parse args, load config, launch mode
│   ├── audio.py             # ReSpeaker init, VAD (WebRTC), echo suppression,
│   │                        #   STT (Google Cloud + Vosk fallback),
│   │                        #   TTS (ElevenLabs + espeak-ng fallback),
│   │                        #   voice interruption detection
│   ├── brain.py             # LLM clients (OpenAI, Grok), web search (Perplexity),
│   │                        #   conversation history/context management
│   ├── vision.py            # IMX500 init, face detection, tracking coordinates
│   ├── servos.py            # PCA9685 init, eyes (LR/UD), eyelids, tentacles, ears,
│   │                        #   idle movements (saccades, blink), quiet_mode
│   ├── leds.py              # NeoPixel init, animations (steampunk, knight, listening, speaking)
│   ├── display.py           # Waveshare 1.9" LCD driver, eye expressions on screen
│   ├── buttons.py           # GPIO buttons with gpiozero, callbacks per mode
│   ├── personality.py       # System prompts per personality (botijo, botija, barbacoa...),
│   │                        #   tool definitions for function calling
│   └── config.py            # Load .env + config.json, hardware constants
│                            #   (pins, channels, rates, thresholds)
├── vendor/
│   └── lib/                 # Waveshare LCD drivers (vendored from RPi, preserves original import path)
├── panel/                   # Web panel (HTML/JS/CSS) — keep as-is
├── panel_backend/           # Flask backend — keep as-is
├── tools/                   # Standalone utility scripts
│   ├── calibrate_servos.py
│   ├── reset_servos.py
│   ├── test_audio.py
│   ├── test_leds.py
│   ├── test_buttons.py
│   └── test_vision.py
├── old/                     # ALL original scripts, untouched (127+ files)
├── config/
│   ├── personalities/       # Personality JSONs (botijo.json, botija.json, barbacoa.json)
│   └── hardware.json        # Servo channels (0-15), LED pin/count, display pins (RST/DC/BL),
│                            #   button GPIOs, ReSpeaker config (rate/channels/chunk),
│                            #   VAD thresholds, echo suppression factor, inactivity timeout
├── docs/
│   ├── HARDWARE.md          # Real inventory of currently connected hardware
│   ├── ARCHITECTURE.md      # System diagram and data flow
│   ├── SETUP.md             # How to set up a fresh RPi from scratch
│   └── WORKFLOW.md          # How to develop from MacBook with Claude Code
├── deploy.sh                # rsync src/ + config/ to RPi, optionally restart
├── requirements.txt         # Clean dependencies (only what's needed)
├── .env.example             # API keys template
├── CLAUDE.md                # Context for Claude Code sessions
└── .gitignore               # Clean, no junk
```

### Design Principles
- Each module in `src/` has an `init()` with graceful degradation (if hardware missing, log warning and continue)
- `main.py` is the ONLY entry point: `python src/main.py` or `python src/main.py --mode barbacoa`
- Old scripts moved untouched to `old/` — always available as reference
- `config/hardware.json` extracts hardcoded constants (servo channels, VAD thresholds, etc.) to a centralized, editable location

---

## 3. Module Interfaces

### `config.py` — Foundation
```python
# Loads .env (API keys) + config/hardware.json (pins, channels, thresholds)
# Exposes: CONFIG global dict, get_personality(name) -> dict
# No dependencies on other modules
```

### `audio.py` — Sound I/O
```python
init() -> bool                              # Detect ReSpeaker, open stream, init VAD
listen() -> str                             # VAD + STT → text (blocks until silence)
speak_stream(chunks: Generator[str, None, None]) -> SpeakResult
                                            # TTS from streaming text chunks → play audio
                                            # Returns SpeakResult(interrupted: bool, spoken_text: str)
                                            # Internally manages echo cancellation buffer
                                            # and interruption monitor (start/stop around playback)
speak(text: str) -> SpeakResult             # Convenience wrapper: speak_stream(iter([text]))
                                            # Fallback chain: ElevenLabs → espeak-ng (automatic)
is_voice_detected() -> bool                 # For external interruption checks
stop_speaking() -> None                     # Cut playback immediately
cleanup()                                   # Close streams
```
Note: `speak_stream` is the primary interface — it consumes text chunks from `brain.chat_stream()` so TTS begins before the full LLM response is ready (low latency). Echo cancellation and interruption monitoring run internally during playback.

### `brain.py` — Think and Search
```python
init(persona_name: str) -> None
                                            # Load personality (which determines LLM), init history
                                            # LLM provider comes from personality config (e.g., barbacoa → grok)
chat(user_text: str) -> str                 # Send to LLM, return complete response (blocking)
chat_stream(user_text: str) -> Generator[str, None, None]
                                            # Stream text chunks from LLM (for speak_stream)
                                            # Handles function calling (web_search tool) internally
search(query: str) -> str                   # Perplexity web search (also called by function calling)
reset_history() -> None                     # Clear conversational context
```
Note: `chat_stream` is the primary interface — yields text chunks that feed directly into `audio.speak_stream()`. Function calls (e.g., web_search via Perplexity) are resolved internally before yielding the response text. The `llm` parameter is set from the personality config (e.g., barbacoa uses Grok, default uses GPT-5).

### `vision.py` — See Faces
```python
init() -> bool                              # Initialize IMX500 + picamera2
get_faces() -> list[FaceRect]               # Return detected bounding boxes
cleanup()
```

### `servos.py` — Move Things
```python
init() -> bool                              # PCA9685 via I2C
look_at(x, y) -> None                       # Eyes follow coordinate
blink() -> None                             # Blink
idle_movement() -> None                     # Random saccades
move_tentacles(pattern="random") -> None
move_ears(pattern="alert") -> None
set_quiet_mode(on: bool) -> None            # Pause mechanics during listening
cleanup()                                   # Servos to neutral position
```

### `leds.py` — Light Up
```python
init() -> bool                              # NeoPixel on GPIO18
set_mode(mode: str) -> None                 # "steampunk", "listening", "speaking", "idle", "off"
cleanup()
```

### `display.py` — Screen
```python
init() -> bool                              # Waveshare SPI (uses vendor/LCD_1inch9)
show_eyes(expression="neutral") -> None     # Draw eyes on LCD
show_waveform(audio_level: float) -> None   # Audio-reactive visualizer during TTS playback
show_text(text: str) -> None
cleanup()
```
Note: Display pins (RST=27, DC=25, BL=24) and SPI config come from `config/hardware.json`.

### `buttons.py` — Physical Buttons
```python
init(callbacks: dict) -> None               # GPIO 5,6,12,13 with callbacks
cleanup()
```

### `personality.py` — Who Is Botijo
```python
load(name: str) -> dict                     # Load config/personalities/botijo.json
get_system_prompt(name: str) -> str         # Formatted system prompt
get_tools(name: str) -> list                # Tool definitions for function calling
get_phrases(name: str) -> dict | None       # Phrase categories for autonomous modes (barbacoa)
                                            # Returns None for conversational-only personalities
```
Note: Barbacoa phrase pools (fuego, carne, surrealismo, etc.) live in `config/personalities/barbacoa_phrases.json`, loaded by `get_phrases()`. The barbacoa main loop in `main.py` uses these for autonomous banter.

### `main.py` — The Conductor
```python
# 1. Parse args (--mode botijo|botija|barbacoa)
# 2. Load config
# 3. init() each module (graceful if any fails)
#    - Each module manages its own internal threads (e.g., servos.idle runs
#      its own daemon thread, leds run their own animation thread)
#    - main.py does NOT spawn threads for modules — only calls start/stop
# 4. Main loop:
#    - servos.set_quiet_mode(True)         # pause mechanics while listening
#    - leds.set_mode("listening")
#    - text = audio.listen()
#    - servos.set_quiet_mode(False)        # resume mechanics
#    - leds.set_mode("speaking")
#    - chunks = brain.chat_stream(text)    # streaming LLM response
#    - result = audio.speak_stream(chunks) # TTS with interruption detection
#    - if result.interrupted:
#        brain.note_interruption(result.spoken_text)  # for continuation context
#    - Inactivity timeout: 300s idle → sleep mode (deactivate servos/leds),
#      240s warning. Voice reactivates from sleep.
# 5. vision.get_faces() → servos.look_at() runs as background task via vision module
# 6. Signal handler for clean cleanup (SIGINT, SIGTERM) — calls cleanup() on all modules
#
# Barbacoa mode: uses a different main loop with autonomous phrase emission
# (PhraseManager with timers, face-triggered banter). This is a distinct behavioral
# mode, not just a personality swap. Implemented as a separate loop function
# selected by --mode barbacoa.
```

### Thread Management Convention
Each module owns its threads. `main.py` never creates threads for module work.
- `servos.init()` starts idle movement daemon thread internally
- `leds.init()` starts animation daemon thread internally
- `vision.init()` starts face detection loop thread internally
- `audio.speak_stream()` starts/stops echo monitor internally
- All daemon threads check a module-level `_shutdown` flag
- `cleanup()` sets `_shutdown`, joins threads with timeout(2s), releases hardware

### Python Import Path
`src/` is a flat module directory (no `__init__.py`). The entry point runs as:
```bash
cd ~/botijo && PYTHONPATH=src:vendor python src/main.py
```
This allows `from audio import listen` and `from lib import LCD_1inch9` to work (same import as original scripts). The `deploy.sh --run` command sets this automatically.

---

## 4. Development Workflow

### MacBook → RPi

```
MacBook (edit)                      RPi (execute)
─────────────────                   ──────────────
/Volumes/X10 Pro/Projects/          /home/jack/botijo/
  botijo/repo/                        src/
    src/                              config/
    config/                           tools/
    tools/                            old/
```

### `deploy.sh`

```bash
#!/bin/bash
# Sync local code → RPi (each directory to its correct destination)
rsync -avz --delete src/ botijo:~/botijo/src/
rsync -avz --delete config/ botijo:~/botijo/config/
rsync -avz --delete tools/ botijo:~/botijo/tools/
rsync -avz --delete vendor/ botijo:~/botijo/vendor/
rsync -avz requirements.txt botijo:~/botijo/

# If --run passed, execute main.py on RPi
if [ "$1" = "--run" ]; then
    ssh botijo "cd ~/botijo && source venv_chatgpt/bin/activate && PYTHONPATH=src:vendor python src/main.py ${@:2}"
fi
```

**Usage:**
- `./deploy.sh` — sync only
- `./deploy.sh --run` — sync and execute
- `./deploy.sh --run --mode barbacoa` — sync and execute in barbecue mode

### What does NOT sync (lives only on RPi)
- `venv_chatgpt/` — virtualenv stays on RPi
- `.env` — API keys live only there
- `old/` — legacy scripts already there, don't travel
- Models (vosk, piper, onnx) — too large, already on RPi

### CLAUDE.md
Contains: what Botijo is, how to deploy, how to run, module map, hardware summary, Context7 library IDs, things NOT to touch.

---

## 5. Dependency Verification Protocol

Context7 provides up-to-date documentation for all key libraries:

| Library | Context7 ID | Snippets | Reputation |
|---|---|---|---|
| picamera2 (IMX500) | `/raspberrypi/picamera2` | 49 | High |
| PCA9685 servos | `/adafruit/adafruit-pwm-servo-driver-library` | 14 | High |
| NeoPixel (Pi5) | `/vanshksingh/pi5neo` | 39 | High |
| ElevenLabs TTS | `/elevenlabs/elevenlabs-python` | 629 | High |
| Google Cloud STT | `/websites/cloud_google_speech-to-text` | 9129 | High |

### Known Issue: NeoPixel on RPi 5
Current code uses `rpi_ws281x` which has known issues on RPi 5 (DMA access). `Pi5Neo` uses SPI instead. Must investigate during Fase 4 implementation.

### Protocol
1. **When refactoring each module:** consult Context7 for the module's primary library, check for breaking changes or deprecated APIs
2. **In CLAUDE.md:** list of Context7 IDs so any future Claude Code session can query up-to-date docs
3. **In `requirements.txt`:** pinned versions with verification date comment

---

## 6. Documentation

### `docs/HARDWARE.md`
Real inventory of what's connected now:

| Subsystem | Component | Connection | Status |
|---|---|---|---|
| Compute | Raspberry Pi 5, 8GB, NVMe 458GB | — | Active |
| Audio In | ReSpeaker Mic Array v2.0 (4 mics) | USB (2886:0018) | Active |
| Audio Out | Speaker via ReSpeaker (playback device 0) | USB | Active |
| Camera | RPi Camera + IMX500 AI | CSI | Active |
| Servos | 11x MG90S via PCA9685 | I2C (i2c-1) | Active |
| LEDs | 13x NeoPixel WS2812 | GPIO 18 | Active |
| Display | Waveshare 1.9" LCD (320×170) | SPI (spidev0.0) | Active |
| Buttons | 4x GPIO (5, 6, 12, 13) | GPIO | Active |
| Wake word | Picovoice Porcupine | Software | Not in use |
| TTS local | Piper | Software | Not in use |

Plus: servo channel map (0-15), button pinout, calibration notes.

### `docs/ARCHITECTURE.md`
System diagram, conversational flow, interruption flow, graceful degradation.

### `docs/SETUP.md`
How to flash RPi, install system deps, create venv, configure .env, test each subsystem.

### `docs/WORKFLOW.md`
MacBook prerequisites, edit → deploy → test cycle, Claude Code usage, debugging tips.

---

## 7. Migration Plan

### Phase 0: Infrastructure (no Botijo code touched)
- Sync RPi with GitHub (`git pull` for missing commit)
- Create folder structure (`src/`, `old/`, `config/`, `docs/`, `tools/`)
- Move all 127+ scripts to `old/`
- Create `deploy.sh`, `.env.example`, `.gitignore`
- Create `CLAUDE.md`
- **Result:** Clean repo, deploy works, originals safe in `old/`

### Phase 1: `config.py` + `main.py` skeleton
- Extract hardcoded constants from `gpt5botijonew2.py` to `config/hardware.json` and `config/personalities/`
- `config.py` loads everything
- `main.py` skeleton that only initializes config and logs
- **Result:** `./deploy.sh --run` boots and says "Botijo ready, no modules loaded"

### Phase 2: `audio.py` — the most critical module
- Extract ReSpeaker, VAD, STT, TTS, interruption logic from `gpt5botijonew2.py`
- Largest and most complex module (~40% of code)
- Context7 check: Google Cloud STT streaming API, ElevenLabs SDK
- **Result:** `main.py` listens and speaks

### Phase 3: `brain.py` + `personality.py`
- Extract LLM clients (OpenAI, Grok) and Perplexity
- System prompts to configurable JSON
- Conversation history with sliding window
- **Result:** Botijo converses (audio + brain)

### Phase 4: `servos.py` + `leds.py`
- Extract PCA9685, eye/tentacle/ear movements
- Extract NeoPixel animations
- Context7 check: PCA9685 API, Pi5Neo vs rpi_ws281x
- **Result:** Botijo converses and moves

### Phase 5: `vision.py` + `display.py` + `buttons.py`
- Extract IMX500 face detection + servo tracking
- Extract LCD Waveshare driver
- Extract GPIO buttons
- Context7 check: picamera2 IMX500 API
- **Result:** Complete Botijo, functional equivalent of best current script

### Phase 6: Documentation and tools
- Write `docs/HARDWARE.md`, `ARCHITECTURE.md`, `SETUP.md`, `WORKFLOW.md`
- Create scripts in `tools/` (test_audio, test_leds, calibrate_servos, etc.)
- **Result:** Documented and maintainable project

### Phase 7 (future): Integrations
- Connect to LiveKit/Néstor
- Improved web panel (circular display)
- New modes/personalities

Each phase leaves Botijo functional. If Phase 3 breaks, Phase 2 still works. Scripts in `old/` always available as reference and emergency fallback.
