import time
import random
import threading
from evdev import InputDevice, ecodes, list_devices
from adafruit_servokit import ServoKit

class AnimatronicEyes:
    """
    Control de ojos animatrónicos con servos conectados a una Raspberry Pi 5 vía PCA9685.
    Compatible con control manual por joystick y movimiento automático aleatorio.
    """
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.SERVO_CHANNELS = {
            "LR": 0, "UD": 1, "TL": 2, "BL": 3, "TR": 4, "BR": 5
        }
        self.servo_limits = {
            "LR": (40, 140), "UD": (40, 140),
            "TL": (90, 10), "BL": (90, 145),
            "TR": (90, 180), "BR": (90, 35)
        }
        self.MAX_SPEED = 5
        self.DEAD_ZONE = 0.05

        # Variables de estado
        self.auto_mode = True
        self.lid_left_trigger = 0.0  # L2
        self.lid_right_trigger = 0.0  # R2
        self.controller_connected = False
        self.exit_flag = False
        self.joy_left_x = 0.0
        self.joy_left_y = 0.0
        self.current_lr = 90
        self.current_ud = 110
        self.current_blink = 0.0
        
        # ✅ Variables para tracking de cambios de ángulo
        self.previous_angles = {
            "LR": 90, "UD": 110, "TL": 10, "BL": 160, "TR": 180, "BR": 20
        }

        self._initialize_servos()

    def _approach(self, current, target):
        if abs(target - current) <= self.MAX_SPEED:
            return target
        return current + self.MAX_SPEED if target > current else current - self.MAX_SPEED

    def _map_joy(self, axis, value):
        a, b = self.servo_limits[axis]
        return a + (value + 1) / 2 * (b - a)

    def _initialize_servos(self):
        for i in range(6):
            self.kit.servo[i].set_pulse_width_range(500, 2500)
        for channel, angle in [("LR", 90), ("UD", 110), ("TL", 10), ("BL", 160), ("TR", 180), ("BR", 20)]:
            self.kit.servo[self.SERVO_CHANNELS[channel]].angle = angle
            print(f"🔧 [INIT] Servo {channel} inicializado → {angle}°")

    def normalize_axis(self, value, min_val=-32768, max_val=32767):
        norm = (value - min_val) / (max_val - min_val)
        return 2.0 * norm - 1.0

    def start_controller_thread(self):
        threading.Thread(target=self._controller_loop, daemon=True).start()

    def _controller_loop(self):
        gamepad = None
        while not self.exit_flag:
            if not self.controller_connected:
                devices = [InputDevice(path) for path in list_devices()]
                for dev in devices:
                    if dev.name and ("xbox" in dev.name.lower() or "controller" in dev.name.lower()):
                        print(f"[INFO] Mando detectado: {dev.name} en {dev.path}")
                        gamepad = InputDevice(dev.path)
                        self.controller_connected = True
                        self.auto_mode = False
                        break
            else:
                try:
                    if gamepad is not None:
                        for event in gamepad.read_loop():
                            if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_EAST and event.value == 1:
                                self.auto_mode = not self.auto_mode
                                print(f"[INFO] Modo {'AUTOMÁTICO' if self.auto_mode else 'MANUAL'}")
                            elif event.type == ecodes.EV_ABS:
                                if event.code == ecodes.ABS_X:
                                    self.joy_left_x = self.normalize_axis(event.value)
                                    if abs(self.joy_left_x) < self.DEAD_ZONE:
                                        self.joy_left_x = 0.0
                                elif event.code == ecodes.ABS_Y:
                                    self.joy_left_y = self.normalize_axis(event.value)
                                    if abs(self.joy_left_y) < self.DEAD_ZONE:
                                        self.joy_left_y = 0.0
                                elif event.code == ecodes.ABS_Z:  # L2
                                    self.lid_left_trigger = max(0.0, min(1.0, event.value / 255.0))
                                elif event.code == ecodes.ABS_RZ:  # R2
                                    self.lid_right_trigger = max(0.0, min(1.0, event.value / 255.0))
                except OSError:
                    print("[WARN] Mando desconectado")
                    self.controller_connected = False
                    gamepad = None
            time.sleep(0.5)

    def update(self):
        if self.auto_mode:
            target_lr = random.uniform(*self.servo_limits["LR"])
            target_ud = random.uniform(*self.servo_limits["UD"])
            target_blink = random.choice([0.0, 1.0])
            time.sleep(random.uniform(0.1, 0.3))
        else:
            target_lr = self._map_joy("LR", self.joy_left_x)
            target_ud = self._map_joy("UD", self.joy_left_y)
            target_blink = self.current_blink  # Se mantiene estable

        self.current_lr = self._approach(self.current_lr, target_lr)
        self.current_ud = self._approach(self.current_ud, target_ud)
        self.current_blink = target_blink

        self._apply_servo_positions()

    def run_loop(self):
        try:
            while True:
                self.update()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[INFO] Salida por teclado")
        finally:
            self.shutdown()

    def _apply_servo_positions(self):
        # ✅ Aplicar y reportar cambios en LR y UD
        self._set_servo_with_report("LR", self.current_lr)
        self._set_servo_with_report("UD", self.current_ud)

        if self.auto_mode:
            for lid in ["TL", "BL", "TR", "BR"]:
                a = self.servo_limits[lid][1] + self.current_blink * (self.servo_limits[lid][0] - self.servo_limits[lid][1])
                a = max(0, min(180, a))
                self._set_servo_with_report(lid, a)
        else:
            for lid in ["TL", "BL"]:
                a = self.servo_limits[lid][1] + self.lid_left_trigger * (self.servo_limits[lid][0] - self.servo_limits[lid][1])
                a = max(0, min(180, a))
                self._set_servo_with_report(lid, a)
            for lid in ["TR", "BR"]:
                a = self.servo_limits[lid][1] + self.lid_right_trigger * (self.servo_limits[lid][0] - self.servo_limits[lid][1])
                a = max(0, min(180, a))
                self._set_servo_with_report(lid, a)

    def _set_servo_with_report(self, channel, angle):
        """
        ✅ Aplica el ángulo al servo y reporta cambios en consola
        """
        # Redondear para evitar spam de cambios microscópicos
        rounded_angle = round(angle, 1)
        
        # Solo reportar si hay un cambio significativo (>= 1 grado)
        if abs(rounded_angle - self.previous_angles.get(channel, 0)) >= 1.0:
            # Aplicar el ángulo al servo
            self.kit.servo[self.SERVO_CHANNELS[channel]].angle = rounded_angle
            
            # Reportar el cambio
            servo_names = {
                "LR": "Izquierda-Derecha", 
                "UD": "Arriba-Abajo",
                "TL": "Párpado Superior Izq", 
                "BL": "Párpado Inferior Izq",
                "TR": "Párpado Superior Der", 
                "BR": "Párpado Inferior Der"
            }
            
            print(f"🎯 Servo {channel} ({servo_names.get(channel, channel)}) → {rounded_angle}°")
            
            # Actualizar el ángulo previo
            self.previous_angles[channel] = rounded_angle
        else:
            # Aplicar el ángulo sin reportar
            self.kit.servo[self.SERVO_CHANNELS[channel]].angle = rounded_angle

    def shutdown(self):
        self.exit_flag = True
        for c in self.SERVO_CHANNELS.values():
            self.kit.servo[c].angle = None
        print("[INFO] Servos desactivados")

# Función externa para control manual desde otros scripts
kit = ServoKit(channels=16)

SERVO_CHANNELS = {
    "LR": 0, "UD": 1
}

def set_angle(eje, valor):
    """✅ Función externa con reporte de cambios"""
    if eje in SERVO_CHANNELS:
        kit.servo[SERVO_CHANNELS[eje]].angle = valor
        servo_names = {"LR": "Izquierda-Derecha", "UD": "Arriba-Abajo"}
        print(f"🎯 [SERVO] {eje} ({servo_names.get(eje, eje)}) → {valor}°")
    else:
        raise ValueError(f"Eje desconocido: {eje}")


if __name__ == "__main__":
    eyes = AnimatronicEyes()
    eyes.start_controller_thread()
    eyes.run_loop()