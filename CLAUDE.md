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
- `panel/` + `panel_backend/` — web control panel (separate systemd service, RPi only)
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
