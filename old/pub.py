#!/usr/bin/env python3
# filepath: /home/jack/botijo/pub_fixed.py
# Calibración con lógica INVERTIDA

from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)
kit.servo[1].set_pulse_width_range(500, 2500)

# ✅ CLAVE: Configurar el rango de ángulos para permitir negativos
kit.servo[1].actuation_range = 270  # Permite rango más amplio
kit.servo[1].angle = None  # Reset del servo

print("🔧 Calibración con lógica INVERTIDA")
print("Si 0° está bien, probamos valores NEGATIVOS para bajar más")

def set_inverted_angle(servo, desired_angle):
    """
    Función para manejar la lógica invertida del servo.
    Ángulo positivo = bajar, ángulo negativo = subir
    """
    try:
        # Como el servo está al revés, invertir el signo
        actual_angle = -desired_angle
        servo.angle = actual_angle
        print(f"  Ángulo deseado: {desired_angle}° → Aplicando: {actual_angle}°")
    except ValueError:
        # Para ángulos fuera de rango, usar ancho de pulso
        pulse_width = 1500 + (actual_angle * 1000 / 90)
        pulse_width = max(300, min(2700, pulse_width))
        print(f"  Usando ancho de pulso: {pulse_width}μs")
        servo.set_pulse_width_range(300, 2700)
        equivalent_angle = ((pulse_width - 500) / 2000) * 180
        servo.angle = equivalent_angle

try:
    servo = kit.servo[1]
    
    # Ángulos de prueba: positivos deberían bajar, negativos subir
    angles_to_test = [0, 5, 10, 15, 20, 25, 30, -5, -10, -15]
    
    for angle in angles_to_test:
        print(f"Probando ángulo deseado: {angle}° (para {'bajar' if angle >= 0 else 'subir'})")
        set_inverted_angle(servo, angle)
        time.sleep(4)
        
except KeyboardInterrupt:
    try:
        current_angle = int(input("¿Qué ángulo funcionó mejor? "))
        print(f"✅ Posición elegida: {current_angle}°")
        
        # Aplicar el ángulo elegido con lógica invertida
        set_inverted_angle(servo, current_angle)
        
        # Calcular rango invertido
        min_angle = current_angle - 15
        max_angle = current_angle + 15
        print(f"💡 Rango sugerido: ({min_angle}, {max_angle})")
        print(f"💡 En servitor.py usa estos valores INVERTIDOS:")
        print(f"    'UD': ({-max_angle}, {-min_angle})")
        
    except ValueError as e:
        print(f"Error aplicando ángulo: {e}")