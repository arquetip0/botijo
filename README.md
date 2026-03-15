# Botijo

Botijo es un robot conversacional físico con personalidad de androide paranoide. Escucha, piensa y responde con humor ácido y referencias lovecraftianas, mientras mueve ojos mecánicos, tentáculos y LEDs.

## Arquitectura General

```
Micrófono (ReSpeaker 4-Mic) → VAD + STT → LLM → TTS → Altavoz
                                  ↕              ↕
                              Wake Word      Búsqueda Web
                             (Picovoice)    (Perplexity)

        Cámara (IMX500) → Detección de caras → Seguimiento ocular (Servos)

        Botones GPIO → Control de modos / reset

        LEDs NeoPixel → Animaciones steampunk
```

### Flujo conversacional

1. **Escucha**: Captura de audio continua con VAD (WebRTC modo 3) para detectar voz
2. **Reconocimiento**: Google Cloud STT (online) o Vosk (offline) convierte voz a texto
3. **Procesamiento**: El texto se envía al LLM con el system prompt de personalidad
4. **Búsqueda** (opcional): Consulta Perplexity para información en tiempo real
5. **Respuesta**: TTS genera audio (ElevenLabs, Piper o espeak-ng) que se reproduce
6. **Interrupción**: El usuario puede interrumpir en cualquier momento; el sistema detecta nueva voz y corta la reproducción

## Hardware

### Computadora

| Componente | Detalle |
|---|---|
| Placa | Raspberry Pi |
| SO | Raspberry Pi OS (Linux) |
| Python | 3.11 (venv) |

### Audio

| Componente | Detalle |
|---|---|
| Micrófono | ReSpeaker Mic Array v2.0 (4 micrófonos, USB) |
| Características | Beamforming, cancelación de eco adaptativa (LMS), VAD |
| Sample rate | 16000 Hz |
| Altavoz | Altavoz externo (salida de audio estándar) |

### Visión

| Componente | Detalle |
|---|---|
| Cámara | Raspberry Pi Camera con acelerador AI IMX500 |
| Modelo de detección | Red neuronal en `/packett/network.rpk` |
| Capacidad | Detección de caras para seguimiento ocular |
| Resolución | 640×480 a 1920×1080 (configurable) |
| Display | Waveshare 1.9" LCD (320×170) para visualización |

### Servos (11 en total, controlados por PCA9685 vía I2C)

| Canal | Función | Rango |
|---|---|---|
| 0 | Ojo izquierda-derecha | 40°–140° |
| 1 | Ojo arriba-abajo | 30°–150° |
| 2 | Párpado superior izquierdo | 500–2500 µs |
| 3 | Párpado inferior izquierdo | 500–2500 µs |
| 4 | Párpado superior derecho | 500–2500 µs |
| 5 | Párpado inferior derecho | 500–2500 µs |
| 8 | Tentáculo pitch | 310–2500 µs |
| 9 | Tentáculo rotación izquierda | 420–2700 µs |
| 10 | Tentáculo rotación derecha | 500–2700 µs |
| 12 | Oreja derecha | 0°–180° |
| 15 | Oreja izquierda | 0°–180° |

### LEDs

| Componente | Detalle |
|---|---|
| Tipo | NeoPixel RGB (WS2812) |
| Cantidad | 13 LEDs |
| Pin | GPIO 18 |
| Brillo por defecto | 5% |
| Animaciones | Pulsación steampunk, Knight Rider, patrones personalizados |

### Botones

| Botón | GPIO | Pin físico |
|---|---|---|
| BTN1 | 5 | 29 |
| BTN2 | 6 | 31 |
| BTN3 | 13 | 33 |
| BTN4 | 12 | 32 |

Debounce: 20ms. Activos en HIGH. Controlados con `gpiozero`.

### Protocolos de comunicación

- **I2C**: Controladora de servos PCA9685
- **PWM**: Señales de servo (16-bit, 1600 Hz)
- **USB**: ReSpeaker Mic Array
- **GPIO**: Botones y LEDs NeoPixel
- **CSI**: Cámara Raspberry Pi

## Software

### APIs de LLM

| Proveedor | Modelos | Uso |
|---|---|---|
| OpenAI | GPT-4, GPT-5 | Motor conversacional principal |
| xAI (Grok) | grok-3-fast | Motor conversacional alternativo |
| Perplexity | — | Búsqueda web en tiempo real |

### Text-to-Speech (TTS)

| Motor | Tipo | Detalle |
|---|---|---|
| ElevenLabs | Cloud | Voz principal (eleven_multilingual_v2) |
| Piper | Local | Voz `es_MX-laura-high.onnx` (sin internet) |
| espeak-ng | Local | Fallback básico |

### Speech-to-Text (STT)

| Motor | Tipo | Detalle |
|---|---|---|
| Google Cloud STT | Cloud | Reconocimiento principal (streaming) |
| Vosk | Local | Modelo `vosk-model-small-es-0.42` (offline) |
| Picovoice Porcupine | Local | Wake word "botijo" / "amanece" |

### Personalidad

Botijo es un androide paranoide con humor ácido refinado. Se refiere a los humanos como "ser inferior", "saco de carne" o "zarria humana". Usa referencias lovecraftianas y mantiene un tono condescendiente pero informativo. Existe también una variante "Botija" con personalidad histriónica y teatral.

### Puntos de entrada principales

| Archivo | Descripción |
|---|---|
| `menu.py` | GUI Tkinter para pantalla táctil (WiFi, teclado, lanzar botijo, apagar) |
| `integrator5.py` | Versión siempre escuchando (sin wake word) |
| `integratorpico3.py` | Versión con wake word (Picovoice) + ojos |
| `gpt5botijonew2.py` | GPT-5 con features avanzadas + búsqueda web |
| `grokbotijo3.py` | Versión Grok/xAI con Perplexity |
| `ritmo8_elevenlabs.py` | Visión + ElevenLabs + display Waveshare |
| `botijo_barbacoa.py` | Modo barbacoa con frases automáticas por categorías |

### Scripts de lanzamiento

| Script | Función |
|---|---|
| `launch_botijo.sh` | Lanza la versión principal |
| `launch_botijo_interruptions.sh` | Lanza con sistema de interrupciones |
| `launch_integrator.sh` | Lanza integrator |
| `launch_menu.sh` | Lanza el menú GUI |
| `activate_venv.sh` | Activa el entorno virtual |

### Características destacadas

- **Interrupciones**: El usuario puede hablar durante la respuesta para cortarla
- **Supresión de eco**: Factor 0.4 para evitar que el bot se escuche a sí mismo
- **Watchdog de STT**: Detecta audio atascado y reinicia el streaming (~4 min límite)
- **Memoria conversacional**: Ventana de contexto de ~8-10 turnos
- **Modo barbacoa**: Frases automáticas en 6 categorías (fuego, carne, surrealismo, etc.) con detección de caras
- **Degradación elegante**: Si el hardware no está disponible, continúa sin él; fallback de TTS automático

## Dependencias principales

- `openai`, `google-cloud-speech`, `elevenlabs` — APIs cloud
- `vosk`, `pvporcupine` — Reconocimiento offline y wake word
- `adafruit-circuitpython-servokit` — Control de servos PCA9685
- `neopixel`, `board` — LEDs NeoPixel
- `gpiozero` — Botones GPIO
- `picamera2` — Cámara Raspberry Pi
- `pyaudio`, `sounddevice` — Captura de audio
- `webrtcvad` — Detección de actividad vocal
- `piper` (binario local) — TTS offline

## Rutas relevantes (en la Raspberry Pi)

```
/home/jack/botijo/              — Directorio del proyecto
/home/jack/.env                 — Variables de entorno (API keys)
/home/jack/piper/               — Motor TTS Piper
/home/jack/picovoice/           — Modelos de wake word
/home/jack/botijo/packett/      — Modelo de visión IMX500
/venv_chatgpt/                  — Entorno virtual Python
```
