import pygame
import subprocess
from PIL import Image
import time

# Captura de la cámara sin rotar
subprocess.run([
    "libcamera-still",
    "-o", "/tmp/capture.jpg",
    "--width", "1080", "--height", "1920",
    "-n"  # sin preview
])

# Cargar y rotar con PIL
img = Image.open("/tmp/capture.jpg")
img = img.rotate(270, expand=True)

# Iniciar pygame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen.fill((0, 0, 0))

# Redimensionar manteniendo proporciones
img.thumbnail(screen.get_size(), Image.ANTIALIAS)
mode = img.mode
size = img.size
data = img.tobytes()

# Convertir a Surface y mostrar
image_surface = pygame.image.fromstring(data, size, mode)
x = (screen.get_width() - size[0]) // 2
y = (screen.get_height() - size[1]) // 2
screen.blit(image_surface, (x, y))
pygame.display.flip()

# Mostrar durante 3 segundos
time.sleep(3)
pygame.quit()
