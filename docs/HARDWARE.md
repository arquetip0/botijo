# Hardware Inventory

## Component Summary

| Subsystem | Component | Connection | Status |
|-----------|-----------|------------|--------|
| Compute | Raspberry Pi 5, 8GB RAM, NVMe 458GB | — | Active |
| Audio In | ReSpeaker Mic Array v2.0 (4 mics) | USB (VID:0x2886 PID:0x0018) | Active |
| Audio Out | Speaker via ReSpeaker playback channel | USB | Active |
| Camera | RPi Camera Module 3 + IMX500 AI | CSI | Active |
| Servos | 11x MG90S via PCA9685 PWM driver | I2C (i2c-1, addr 0x40) | Active |
| LEDs | 13x NeoPixel WS2812B | GPIO 18 | Active |
| Display | Waveshare 1.9" LCD ST7789 (320x170) | SPI (spidev0.0) | Active |
| Buttons | 4x tactile (active HIGH) | GPIO 5, 6, 12, 13 | Active |

## Servo Channel Map

PCA9685 I2C PWM driver. Default pulse range: 500–2500µs unless noted.

| Channel | Servo | Role | Range | Neutral |
|---------|-------|------|-------|---------|
| 0 | MG90S | Eye left/right | 40°–140° | 90° |
| 1 | MG90S | Eye up/down | 30°–150° | 90° |
| 2 | MG90S | Eyelid top-left | 50° (open) – 90° (closed) | 50° |
| 3 | MG90S | Eyelid bottom-left | 155° (open) – 90° (closed) | 155° |
| 4 | MG90S | Eyelid top-right | 135° (open) – 90° (closed) | 135° |
| 5 | MG90S | Eyelid bottom-right | 25° (open) – 90° (closed) | 25° |
| 8 | MG90S | Tentacle pitch | 0°–180° | 90° |
| 12 | MG90S | Tentacle right / ear right | 0°–180°, pulse 450–2720µs | 90° |
| 15 | MG90S | Tentacle left / ear left | 0°–180°, pulse 300–2650µs | 90° |

Channels 6, 7, 9, 10, 11, 13, 14 are unused.

### Eyelid Positions

All four eyelids close to 90°. Open positions differ per servo orientation:

| Eyelid | Open | Closed |
|--------|------|--------|
| Top-left (ch 2) | 50° | 90° |
| Bottom-left (ch 3) | 155° | 90° |
| Top-right (ch 4) | 135° | 90° |
| Bottom-right (ch 5) | 25° | 90° |

### Tentacle Pulse Ranges

Custom pulse widths to match the physical range of motion:

| Tentacle | Min pulse | Max pulse |
|----------|-----------|-----------|
| Left (ch 15) | 300µs | 2650µs |
| Right (ch 12) | 450µs | 2720µs |

## LED Strip

13x WS2812B NeoPixels on GPIO 18.  Controlled via the `neopixel` library.
Brightness default: 5% (prevents heat buildup, sufficient for indoor use).

Modes: `steampunk` (copper/brass pulsation), `listening` (purple glow), `speaking` (red Knight-Rider sweep), `off`.

## Display

Waveshare 1.9" ST7789 LCD.  Native resolution 170×320, mounted rotated 270° → effective 320×170 landscape.

SPI pins: RST=GPIO27, DC=GPIO25, BL (backlight)=GPIO24.

## Button Pinout

Buttons are active HIGH (generate HIGH signal when pressed). GPIO factory: lgpio (required for RPi 5).

| Button | GPIO | Physical location |
|--------|------|-------------------|
| btn1 | 5 | Top-left |
| btn2 | 6 | Top-right |
| btn3 | 13 | Bottom-left |
| btn4 | 12 | Bottom-right |

## Audio

**Input:** ReSpeaker Mic Array v2.0, 4 microphones in cross pattern. Detected via USB VID/PID. PyAudio device name matches "respeaker" or "seeed" or "mic array". Sample rate: 16 kHz, mono, chunk 480 frames (30ms frames for WebRTC VAD).

**Output:** Playback via the ReSpeaker USB audio interface. ElevenLabs PCM stream at 24 kHz. Fallback: `espeak-ng -v es`.

**VAD:** WebRTC VAD mode 3 (most aggressive), 30ms frames, 300ms padding, voice threshold 0.3.

## Camera

IMX500 on-chip AI. Neural network `.rpk` at `/home/jack/botijo/packett/network.rpk`. Labels at `/home/jack/botijo/labels.txt`. Detection threshold: 0.2 confidence. Target: ~20 FPS metadata capture.
