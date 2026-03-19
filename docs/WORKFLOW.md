# Development Workflow

## MacBook Prerequisites

```bash
# Python 3.11+
brew install python@3.11

# Audio libs (for partial testing on macOS)
brew install portaudio

# SSH config — add to ~/.ssh/config:
# Host botijo
#     HostName <pi-ip>
#     User jack
#     IdentityFile ~/.ssh/id_ed25519

# Verify SSH works:
ssh botijo hostname   # should print: botijo
```

## Edit → Deploy → Test Cycle

### 1. Edit on MacBook

Work in the repo at `/Volumes/X10 Pro/Projects/botijo/repo/`.

Most modules degrade gracefully on macOS — you can run `src/config.py` directly,
test brain.py logic, etc. Hardware modules (servos, leds, buttons, vision) will
log warnings and no-op without RPi hardware.

### 2. Deploy to RPi

```bash
# Sync source files (fast, rsync)
./deploy.sh

# Sync and start the process immediately
./deploy.sh --run
```

The deploy script rsyncs `src/`, `config/`, `vendor/`, and `tools/` to `/home/jack/botijo/`.
It does NOT sync `.env` or `secrets/` — manage those separately.

### 3. Test on RPi

```bash
# SSH in
ssh botijo

# Quick hardware checks
cd ~/botijo && source venv_chatgpt/bin/activate && export PYTHONPATH=src:vendor

python tools/reset_servos.py      # verify servos reach neutral
python tools/test_leds.py         # verify all 4 LED modes
python tools/test_buttons.py      # press each button, check output
python tools/test_audio.py        # test mic + TTS
python tools/test_vision.py       # test face detection

# Run full system
python src/main.py --mode botijo
```

## Claude Code Usage Tips

### Starting a session

Open Claude Code in the repo root. The project `CLAUDE.md` is loaded automatically.
For hardware-specific context, also mention: `config/hardware.json`.

### Useful prompts

- "Show me how audio.listen() works with the VAD"
- "Add a new LED mode called 'thinking' with a blue pulse"
- "The eye LR servo is drifting — check the pulse range config"
- "Add a button callback for btn3 in main.py"

### File structure to know

```
src/           — all Python modules (flat, no packages)
config/        — hardware.json + personalities/*.json
tools/         — standalone test scripts
vendor/        — vendored dependencies (if any)
docs/          — this documentation
old/           — legacy reference scripts (do not modify)
```

## Debugging via SSH

### View live logs

```bash
# If running in screen/tmux
ssh botijo
screen -r   # or tmux attach

# Or redirect to file and tail:
PYTHONPATH=src:vendor python src/main.py 2>&1 | tee /tmp/botijo.log
tail -f /tmp/botijo.log
```

### Check hardware

```bash
# I2C — PCA9685 servo driver
i2cdetect -y 1          # expect: 0x40

# SPI — LCD display
ls /dev/spidev0.0       # should exist

# USB — ReSpeaker
lsusb | grep 2886       # VID:2886 PID:0018

# GPIO — buttons
gpio readall            # or pinout

# Audio devices
aplay -l                # playback
arecord -l              # capture
```

### Common issues

| Symptom | Check |
|---------|-------|
| Servos don't move | `i2cdetect -y 1` shows 0x40? `adafruit_servokit` installed? |
| LEDs off | GPIO18 free? `rpi-ws281x` installed? Run as root or gpio group? |
| No audio input | ReSpeaker USB plugged in? `arecord -l` shows it? |
| TTS falls back to espeak | `ELEVENLABS_API_KEY` in `.env`? Internet access? |
| STT returns empty | `GOOGLE_APPLICATION_CREDENTIALS` set? GCP project active? |
| Vision stub mode | `picamera2` installed? Camera cable seated? CSI enabled? |
| Buttons don't respond | `gpiozero` + `lgpio` installed? `GPIOZERO_PIN_FACTORY=lgpio`? |

### Restart cleanly

```bash
# Kill any running instance
pkill -f "python src/main.py"

# Reset servos to safe position before restarting
PYTHONPATH=src:vendor python tools/reset_servos.py

# Restart
PYTHONPATH=src:vendor python src/main.py --mode botijo
```
