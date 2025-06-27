import board
import neopixel
import time
import math
import random

LED_COUNT = 13
pixels = neopixel.NeoPixel(board.D18, LED_COUNT, brightness=0.01, auto_write=False)

PALETA = [
    (184, 115, 51),   # Cobre
    (205, 133, 63),   # Latón
    (139, 69, 19),    # Óxido
    (112, 128, 40),   # Verdín
    (255, 215, 0),    # Oro viejo
    (72, 60, 50)      # Metal sucio
]

def pulso_oxidado(t, i):
    intensidad = (math.sin(t + i * 0.5) + 1) / 2
    base_color = PALETA[i % len(PALETA)]
    return tuple(int(c * intensidad) for c in base_color)

def steampunk_danza(delay=0.04):
    t = 0
    glitch_timer = 0
    try:
        while True:
            for i in range(LED_COUNT):
                if glitch_timer > 0 and i == random.randint(0, LED_COUNT - 1):
                    pixels[i] = random.choice([(0, 255, 180), (255, 255, 255)])
                else:
                    pixels[i] = pulso_oxidado(t, i)
            pixels.show()
            time.sleep(delay)
            t += 0.1
            glitch_timer = glitch_timer - 1 if glitch_timer > 0 else random.randint(0, 20)
    except KeyboardInterrupt:
        print("⛔ Apagando LEDs al salir…")
    finally:
        pixels.fill((0, 0, 0))
        pixels.show()

steampunk_danza()
