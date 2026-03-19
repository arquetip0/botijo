# Setup Guide

## 1. Flash OS

Use Raspberry Pi Imager to flash **Raspberry Pi OS Bookworm 64-bit** to the NVMe.

Settings:
- Hostname: `botijo`
- User: `jack`
- SSH: enabled (public key auth)
- Locale: `es_ES.UTF-8`

## 2. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y

# Audio
sudo apt install -y portaudio19-dev espeak-ng libespeak-ng1 alsa-utils pulseaudio

# GPIO / I2C / SPI
sudo apt install -y python3-lgpio i2c-tools

# Enable I2C and SPI via raspi-config (or /boot/firmware/config.txt)
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# Verify PCA9685 on I2C bus 1
i2cdetect -y 1   # should show 0x40

# NeoPixel (rpi-ws281x needs this)
sudo apt install -y python3-rpi.gpio
```

## 3. Python Environment

```bash
cd ~
python3 -m venv venv_chatgpt
source venv_chatgpt/bin/activate

# Install from requirements.txt
pip install -r ~/botijo/requirements.txt

# RPi-only packages (uncomment in requirements.txt first, or install directly)
pip install adafruit-circuitpython-servokit rpi-ws281x gpiozero picamera2 pyusb
pip install board neopixel
```

Note: `picamera2` is installed system-wide on Raspberry Pi OS — if pip fails, use the system package:
```bash
sudo apt install -y python3-picamera2
```

## 4. Google Cloud Credentials

```bash
mkdir -p ~/botijo/secrets
# Copy credentials JSON from your GCP project
scp credentials.json jack@botijo:~/botijo/secrets/

# Set in .env (see step 5)
```

## 5. Environment Variables

Create `~/botijo/.env`:

```bash
# LLM
OPENAI_API_KEY=sk-...
XAI_API_KEY=xai-...       # optional (Grok)

# TTS
ELEVENLABS_API_KEY=...

# STT
GOOGLE_APPLICATION_CREDENTIALS=/home/jack/botijo/secrets/credentials.json

# Personality (optional, default: botijo)
BOTIJO_MODE=botijo
```

Permissions:
```bash
chmod 600 ~/botijo/.env ~/botijo/secrets/credentials.json
```

## 6. IMX500 AI Model

The camera neural network must be on the Pi at the configured path:

```bash
# Default path from hardware.json:
# rpk_path: /home/jack/botijo/packett/network.rpk
# labels_path: /home/jack/botijo/labels.txt

mkdir -p ~/botijo/packett
# Copy your face detection .rpk file and labels.txt
```

## 7. Test Each Subsystem

Run each tool independently to verify hardware before running the full system:

```bash
cd ~/botijo
source venv_chatgpt/bin/activate
export PYTHONPATH=src:vendor

# Audio
python tools/test_audio.py

# LEDs (requires root or gpio group membership)
python tools/test_leds.py

# Buttons
python tools/test_buttons.py

# Vision
python tools/test_vision.py

# Servos — verify positions before full run
python tools/reset_servos.py
python tools/calibrate_servos.py
```

If a tool reports "stub mode" or returns False from init(), check:
- Physical connection (USB/I2C/SPI/GPIO cable seated)
- System packages installed
- Required entries in `.env`

## 8. Run the Full System

```bash
cd ~/botijo
source venv_chatgpt/bin/activate
PYTHONPATH=src:vendor python src/main.py --mode botijo
```

Other modes: `--mode botija`, `--mode barbacoa`

## 9. Deploy from MacBook

```bash
# From the repo root on your MacBook:
./deploy.sh          # rsync to RPi only
./deploy.sh --run    # rsync + start the process
```

SSH alias required: `ssh botijo` must resolve to `jack@<pi-ip>`. Add to `~/.ssh/config`:
```
Host botijo
    HostName <pi-ip-or-hostname>
    User jack
    IdentityFile ~/.ssh/id_ed25519
```
