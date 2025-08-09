import cv2
import pygame
import numpy as np

# Inicializar pygame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("BotijoCam")

# Captura desde la cámara IMX500 (probablemente index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotar 270 grados (clockwise 90 tres veces)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Convertir BGR a RGB para pygame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir a Surface
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame))  # rot90 porque Pygame usa diferente orientación

    # Escalar y mostrar
    frame_surface = pygame.transform.scale(frame_surface, screen.get_size())
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    # Salir con tecla ESC
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

cap.release()
pygame.quit()
