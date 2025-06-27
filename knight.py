import board
import neopixel
import time

LED_COUNT = 13  # Cambia si tienes más o menos
pixels = neopixel.NeoPixel(board.D18, LED_COUNT, brightness=0.05, auto_write=False)

def coche_fantastico(delay=0.05, tail=2):
    while True:
        # Hacia la derecha
        for i in range(LED_COUNT):
            pixels.fill((0, 0, 0))  # Apaga todos
            for j in range(tail + 1):
                if i - j >= 0:
                    brightness = int(255 * (1 - j / (tail + 1)))
                    pixels[i - j] = (brightness, 0, 0)
            pixels.show()
            time.sleep(delay)
        # Hacia la izquierda
        for i in reversed(range(LED_COUNT)):
            pixels.fill((0, 0, 0))
            for j in range(tail + 1):
                if i + j < LED_COUNT:
                    brightness = int(255 * (1 - j / (tail + 1)))
                    pixels[i + j] = (brightness, 0, 0)
            pixels.show()
            time.sleep(delay)

coche_fantastico()
