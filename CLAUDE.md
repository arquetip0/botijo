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
| `audio.py` | ReSpeaker, VAD, STT (Google Cloud), TTS (ElevenLabs + espeak-ng), interruptions, ALSA suppression, shared PyAudio |
| `brain.py` | LLM clients (OpenAI GPT-5, Grok), Perplexity search, conversation history (with tool call persistence) |
| `vision.py` | IMX500 face detection, tracking coordinates |
| `servos.py` | PCA9685 servos — eyes, eyelids, tentacles, ears, quiet_mode (thread-safe Event) |
| `leds.py` | NeoPixel animations (steampunk, listening, speaking), clean shutdown via _shutdown Event |
| `display.py` | Waveshare 1.9" LCD — eye expressions, waveform visualizer (polyline, lock-protected thread) |
| `buttons.py` | GPIO buttons with callbacks |
| `personality.py` | System prompts, tool definitions, phrase pools, `get_current()` accessor |
| `main.py` | Orchestrator: parse args, init modules, main loop, thread-safe PhraseManager, button↔loop sleep sync |
| `main_livekit.py` | LiveKit variant of main.py — identical for now, will diverge to use remote STT/TTS via LiveKit |

## Launchers (repo root)

| File | Purpose |
|------|---------|
| `run_modular.py` | Launches `src/main.py` (panel btn7) |
| `run_livekit.py` | Launches `src/main_livekit.py` (panel btn6) |

## Panel Web (panel/ + panel_backend/)

Steampunk control panel served in Chromium kiosk mode on the RPi touchscreen.

- **`panel/`** — Static frontend (HTML/CSS/JS + button/background images). Served as `file://` in Chromium.
- **`panel_backend/`** — Flask backend (`app.py`, port 5000). Launches scripts, streams stdout, handles Ctrl-C. Runs as systemd service `panel-backend` (`WorkingDirectory=/home/jack/panel_backend`).
- **Chromium kiosk** — Auto-starts on boot via LXDE autostart, opens `file:///home/jack/panel/index.html`.
- **8 buttons:** btn1-5 = legacy scripts, btn6 = `run_livekit.py`, btn7 = `run_modular.py`, btn8 = close Chromium.
- **Deploy:** `deploy.sh` rsyncs `panel/` → `~/panel/` and `panel_backend/app.py` → `~/panel_backend/app.py` (outside `~/botijo/`), then restarts `panel-backend` service.

## Hardware

RPi 5 (8GB, NVMe 458GB), ReSpeaker 4-Mic USB, IMX500 AI camera (CSI),
11 servos via PCA9685 (I2C), 13 NeoPixel LEDs (GPIO18), Waveshare 1.9" LCD (SPI),
4 buttons (GPIO 5,6,12,13). Speaker via ReSpeaker USB.

## Do NOT touch

- `old/` — legacy scripts, reference only
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
