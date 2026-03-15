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

## Clasificación de Scripts

El repositorio contiene muchas iteraciones de los mismos scripts. Esta sección documenta cuáles son los activos y cuáles son legacy.

### Scripts Fundamentales (Activos)

| Script | Familia | Por qué es fundamental |
|--------|---------|----------------------|
| `searchbotijoperplex.py` | Search | Lanzado por `launch_botijo.sh` (launcher principal) |
| `searchbotijoperplex10.py` | Search | Última iteración con Perplexity (la más completa) |
| `gpt5botijonew2.py` | GPT5 | Versión más completa: ReSpeaker v2.0, WebRTC VAD, interrupciones |
| `gpt5botijonew.py` | GPT5 | Lanzado por `launch_botijo_interruptions.sh` |
| `grokbotijo3.py` | Grok | Versión más avanzada con Grok/xAI + Perplexity |
| `grokbotijo3pija.py` | Grok | Variante "Botija" (personalidad femenina/histriónica) |
| `integrator3.py` | Integrator | Lanzado por `launch_integrator.sh` |
| `integrator5.py` | Integrator | Versión más reciente (sin wake word, siempre escuchando) |
| `streambotijo2vad.py` | Stream | Última versión streaming con VAD |
| `menu.py` | Sistema | Menú GUI Tkinter, lanzado por `launch_menu.sh` |
| `ritmo8_elevenlabs.py` | Ritmo | Visión + ElevenLabs + display Waveshare |
| `barbagrok.py` | Barbacoa | Grok + modo barbacoa |

Los scripts fundamentales son **autocontenidos**: llevan el código de LEDs, servos, eye tracking, etc. embebido directamente, no importado de módulos compartidos. La única dependencia local compartida es `lib/LCD_1inch9` (driver Waveshare).

### Módulos Locales (importados por otros scripts)

| Script | Importado por | Función |
|--------|---------------|---------|
| `botijo_barbacoa.py` | `barbagrok.py` | Setup del modo barbacoa |
| `phases_botijo.py` | `botijo_barbacoa.py` | Frases para modo barbacoa (6 categorías) |
| `spirob.py` | `integrapruebas.py`, `pruebastentacle.py` | Movimiento espiral para tentáculos |

### Scripts Standalone de Utilidad

Se ejecutan directamente para probar o controlar hardware:

| Script | Función |
|--------|---------|
| `servo8.py` / `servo9.py` / `servo10.py` | Calibración de servos individuales |
| `tentaclederecho.py` / `tentacleizquierdo.py` | Control tentáculo derecho/izquierdo |
| `tentaclerandom.py` / `tentaclerandom2.py` | Movimientos aleatorios tentáculos |
| `tentaclereset.py` / `ojosreset.py` | Reset a posición neutral |
| `leds.py` | Demo LEDs NeoPixel steampunk |
| `ojopipa3.py` | Eye tracking con saccades y parpadeo |
| `eyetracker4.py` | Eye tracker standalone |
| `face_eye_calibration.py` | Calibración ojos/cara |
| `brutus.py` | Visualizador de ondas retro |
| `stuff.py` | Diagnóstico IMX500 |
| `smooth.py` | Demo suavizado de servos |
| `debug_audio.py` | Debug de audio |
| `test_botones.py` | Test de botones GPIO |

### Legacy / Supersedidos

**Integrators:** `integrator1.py`, `integrator2.py`, `integrator4.py`, `integratorpico.py`, `integratorpico2.py`, `integratorpico3.py`, `integratorpico3_fixed.py` (vacío), `integratorsalvacion.py`, `integratorsave.py`, `integrapruebas.py`

**GPT5:** `gpt5botijo.py`, `gpt5botijo2.py`, `gpt5botijo3.py`

**Grok:** `grokbotijo.py`, `grokchat.py`, `grokchatfucional.py`, `grok3.py`, `grokcopilot.py`

**Search:** `searchbotijoperplex2.py` a `searchbotijoperplex9.py`, `searchbotijoperplex67.py`, `searchbotijoperplexbackup.py`, `searchbotijo.py` (DuckDuckGo)

**Stream:** `streambotijo.py`, `streambotijo2.py`, `streambotijointerim.py`, `streambotijogrok.py`

**Ritmo:** `ritmo6.py`, `ritmo7.py`, `ritmo8.py`, `ritmo9.py`

**Eye/Vision:** `eye.py`, `eye2.py`, `eyeraro.py`, `eyetracker.py`, `eyetracker2.py`, `eyetracker3.py`, `ojopipa.py`, `ojopipa2.py`

**Experimentos:** `o3.py`, `o3google.py`, `o3google2.py`, `avanzado.py`, `sinmiedo.py`, `deepseekloco.py`, `kivi.py`, `dcijdcncdijn.py`, `belcebug3.py`, `judasx.py`, `judasxtremo.py`, `gog.py`, `satan.py`, `suso.py`, `prus.py`, `pub.py`, `raruno.py`, `stuffpre.py`, `xmp.py`, `psonido.py`, `purru.py`, `knight.py`, `servoloco.py`, `servitor.py`, `fluid.py`

**Tests:** `test_gpt5.py`, `test_grok.py`, `test_grok_basic.py`, `test_grok_streaming.py`, `test_grok_final.py`, `test_grok_diagnostics.py`, `test_interruptions.py`, `test_respeaker.py`, `test_respeaker_interruptions.py`, `test_respeaker_levels.py`, `test_simple_interruption.py`, `test_final_interruption.py`, `test_amanece.py`, `test_cam_drm.py`, `test_voice_detection.py`, `test_two_keywords.py`, `testenv.py`, `testconvertereleven.py`, `validate_integratorpico3.py`, `test_audio_real.py` (vacío), `test_optimized_flow.py` (vacío)
