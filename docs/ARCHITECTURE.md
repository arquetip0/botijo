# Architecture

## System Diagram

```
                     ┌─────────────────────────────────────────┐
                     │            Raspberry Pi 5                │
                     │                                          │
  ┌──────────┐       │  ┌──────────────────────────────────┐   │
  │ReSpeaker │──USB──┤  │            main.py               │   │
  │4-mic USB │       │  │         (orchestrator)           │   │
  └──────────┘       │  └───┬────────┬──────┬──────┬───────┘   │
                     │      │        │      │      │            │
  ┌──────────┐  CSI  │  ┌───┴──┐ ┌──┴──┐ ┌─┴──┐ ┌─┴────┐      │
  │ IMX500   │───────┤  │audio │ │brain│ │servos│ │ leds │      │
  │ Camera   │       │  └──────┘ └──┬──┘ └─────┘ └──────┘      │
  └──────────┘       │              │                           │
                     │  ┌───────┐   │     ┌────────┐            │
  ┌──────────┐  I2C  │  │vision │   │     │display │            │
  │ PCA9685  │───────┤  └───────┘   │     └────────┘            │
  │ 11 servos│       │              │                           │
  └──────────┘       │  ┌─────────┐ │     ┌─────────┐          │
                     │  │ buttons │ │     │personality│         │
  ┌──────────┐ GPIO18│  └─────────┘ │     └─────────┘          │
  │NeoPixels │───────┤              │                           │
  │  13 LEDs │       │  ┌───────────┴──────────────┐           │
  └──────────┘       │  │    config.py             │            │
                     │  │  hardware.json + .env    │            │
  ┌──────────┐  SPI  │  └──────────────────────────┘           │
  │Waveshare │───────┤                                          │
  │1.9" LCD  │       └─────────────────────────────────────────┘
  └──────────┘
                              │ Internet
                     ┌────────┴────────┐
                     │   OpenAI API    │
                     │ ElevenLabs API  │
                     │ Google STT API  │
                     └─────────────────┘
```

## Conversational Flow

```
IDLE (steampunk LEDs)
    │
    │ face detected  ───── optional gaze tracking via look_at()
    │
    ▼
LISTENING (purple LEDs, quiet_mode ON)
    │
    │ audio.listen() → Google STT streaming → text
    │
    ▼
THINKING (eyes animate)
    │
    │ brain.chat() → OpenAI GPT / Grok → stream chunks
    │
    ▼
SPEAKING (red Knight-Rider LEDs)
    │
    │ audio.speak_stream() → ElevenLabs PCM → ReSpeaker output
    │
    │ interruption detected? ──yes──► stop TTS ──► back to LISTENING
    │
    ▼
IDLE (steampunk LEDs, inactivity timer reset)
    │
    │ no activity for 300s ──► SLEEP mode (LEDs off, servos quiet)
```

## Interruption Flow

The interruption monitor runs in a daemon thread during TTS playback:

```
TTS playing
    │
    ├── monitor thread: reads ReSpeaker hardware VAD every 30ms
    │       (SPEECHDETECTED + VOICEACTIVITY registers via tuning API)
    │
    │   N consecutive voice frames detected?
    │       yes ──► set _interruption_detected = True
    │
    ├── speak_stream() checks flag per chunk → stops early
    │
    └── speak() returns SpeakResult(interrupted=True)
            │
            └── main loop: skips remaining TTS, goes to LISTENING
```

## Module Responsibilities

| Module | Init | Threads | Cleanup |
|--------|------|---------|---------|
| `config.py` | loads .env + hardware.json + personality | none | none |
| `audio.py` | ReSpeaker + VAD + STT + TTS clients | monitor thread (during TTS) | releases PyAudio + clients |
| `brain.py` | OpenAI + Grok clients | none | none |
| `personality.py` | loads personality JSON | none | none |
| `vision.py` | IMX500 + Picamera2 + AI firmware | detection thread | stops camera |
| `servos.py` | PCA9685 + eye/tentacle setup | eye-idle + tentacle threads | centers + releases |
| `leds.py` | NeoPixel strip | animation thread | turns off strip |
| `display.py` | SPI LCD + fonts | none | backlight off |
| `buttons.py` | gpiozero Button x4 | internal gpiozero threads | closes buttons |

## Graceful Degradation

Every module follows the same pattern:

- Hardware imports are wrapped in `try/except ImportError`
- `init()` returns `False` if hardware not found (never raises)
- All public functions are no-ops when hardware is absent
- `main.py` logs warnings but continues without the missing subsystem

This allows development on a MacBook (no GPIO, no camera, no servos, no LEDs).
Only `audio.py` can partially work on macOS with a regular microphone.
