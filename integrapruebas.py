import spirob, time

# Parte del neutro (0 / 90 / 90)
spirob.neutral(1.0)

# 1· Baja un poco el tentáculo
spirob.move_to_bend(270, 0.25, 0.4)
time.sleep(0.5)

# 2· Desde ahí, gira suavemente a la izquierda
spirob.move_to_bend(270, 2, 0.4)
time.sleep(0.5)

# 3· Vuelve al centro (bajar + nada de giro)
spirob.move_to_bend(330, 0.25, 0.4)
time.sleep(0.5)

# 4· Sube de nuevo al neutro
spirob.neutral(0.8)