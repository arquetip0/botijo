
import pygame
import sys
from PIL import Image

# Ruta de la imagen
IMAGE_PATH = "hello.jpeg"  # cámbiala según necesites
ROTATE_DEGREES = 270        # pon 0 si no quieres rotar

# Inicializar pygame
pygame.init()

# Detectar resolución de pantalla
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# Cargar imagen con PIL para rotarla sin deformar
img = Image.open(IMAGE_PATH)

if ROTATE_DEGREES != 0:
    img = img.rotate(ROTATE_DEGREES, expand=True)

# Redimensionar manteniendo proporción
img.thumbnail((screen_width, screen_height), Image.ANTIALIAS)
mode = img.mode
size = img.size
data = img.tobytes()

# Convertir a superficie de pygame
image_surface = pygame.image.fromstring(data, size, mode)

# Centrar la imagen en la pantalla
x = (screen_width - size[0]) // 2
y = (screen_height - size[1]) // 2

# Pintar la imagen
screen.fill((0, 0, 0))  # fondo negro
screen.blit(image_surface, (x, y))
pygame.display.flip()

# Esperar hasta que se pulse una tecla o se cierre la ventana
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            pygame.quit()
            sys.exit()
