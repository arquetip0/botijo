#!/usr/bin/env python3
# botijo_menu.py – Menú táctil fullscreen con teclado virtual (Kivy)

from kivy.config import Config
Config.set("graphics", "fullscreen", "auto")       # pantalla completa
Config.set("input", "mouse", "mouse,disable_multitouch")

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.metrics import dp
import subprocess, psutil, os

# Rutas de tu proyecto
VENV_PY = "/home/jack/botijo/venv_chatgpt/bin/python"
BOTIJO_SCRIPT = "/home/jack/botijo/integrator3.py"

# ------------------ KV MARKUP ------------------
KV = """
<MenuScreen>:
    BoxLayout:
        orientation: 'vertical'
        spacing: dp(20)
        padding: dp(60)
        Button:
            text: '🔌  Conectar Wi-Fi'
            font_size: '32sp'
            on_release: root.open_wifi_popup()
        Button:
            text: '⌨️  Teclado virtual'
            font_size: '32sp'
            on_release: root.show_keyboard_demo()
        Button:
            text: '🤖  Lanzar Botijo'
            font_size: '32sp'
            on_release: root.launch_botijo()
        Button:
            text: '⏏️  Salir del menú'
            font_size: '32sp'
            on_release: app.stop()
        Button:
            text: '⏻  Apagar'
            font_size: '32sp'
            on_release: root.shutdown_pi()
"""

Builder.load_string(KV)

# ------------------ CLASES ------------------
class KeyboardPopup(Popup):
    """Teclado virtual simple (4 filas). entry = TextInput destino"""

    def __init__(self, entry, **kw):
        super().__init__(**kw)
        self.title = "Teclado"
        self.size_hint = (0.9, 0.55)
        layout = BoxLayout(orientation="vertical", spacing=5, padding=5)
        self.entry = entry
        layout.add_widget(entry)
        grid = GridLayout(cols=10, rows=4, spacing=3, size_hint_y=0.6)
        layout.add_widget(grid)
        self.content = layout

        keys = [
            "1 2 3 4 5 6 7 8 9 0".split(),
            "q w e r t y u i o p".split(),
            "a s d f g h j k l ñ".split(),
            "z x c v b n m ⌫ OK".split()
        ]
        for row in keys:
            for ch in row:
                btn = Button(text=ch, font_size='24sp')
                btn.bind(on_release=self.on_key)
                grid.add_widget(btn)

    def on_key(self, btn):
        char = btn.text
        if char == "⌫":
            self.entry.text = self.entry.text[:-1]
        elif char == "OK":
            self.dismiss()
        else:
            self.entry.text += char


class MenuScreen(Screen):

    # ---------- Botones principales ----------
    def launch_botijo(self):
        if any("integrator3.py" in " ".join(p.cmdline())
               for p in psutil.process_iter(attrs=["cmdline"])):
            return
        subprocess.Popen([VENV_PY, BOTIJO_SCRIPT])

    def shutdown_pi(self):
        os.system("sudo shutdown -h now")

    # ---------- Teclado demostración ----------
    def show_keyboard_demo(self):
        entry = TextInput(font_size='28sp', multiline=False, hint_text="Escribe…")
        KeyboardPopup(entry).open()

    # ---------- Wi-Fi ----------
    def open_wifi_popup(self):
        nets = subprocess.check_output(
            "nmcli -t -f SSID dev wifi list | sort -u",
            shell=True, text=True).strip().splitlines() or ['(sin redes)']

        popup = Popup(title="Wi-Fi", size_hint=(0.9, 0.7))
        box = BoxLayout(orientation='vertical', spacing=10, padding=10)
        popup.add_widget(box)

        from kivy.uix.spinner import Spinner
        ssid_spin = Spinner(text="Elige red", values=nets,
                            size_hint_y=None, height=dp(48), font_size='24sp')
        pwd_in = TextInput(password=True, hint_text="Contraseña",
                           font_size='28sp', multiline=False)
        ok_btn = Button(text="Conectar", size_hint_y=None,
                        height=dp(48), font_size='24sp')

        # El teclado aparece al tocar el campo contraseña
        pwd_in.bind(focus=lambda w, v: KeyboardPopup(pwd_in).open() if v else None)

        def conectar(*a):
            ssid = ssid_spin.text
            pwd = pwd_in.text
            if ssid == "Elige red":
                return
            cmd = ["nmcli", "dev", "wifi", "connect", ssid]
            if pwd:
                cmd += ["password", pwd]
            subprocess.Popen(cmd)
            popup.dismiss()

        ok_btn.bind(on_release=conectar)

        for w in (ssid_spin, pwd_in, ok_btn):
            box.add_widget(w)
        popup.open()


class BotijoApp(App):
    def build(self):
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(MenuScreen(name="menu"))
        return sm


# ------------------ MAIN ------------------
if __name__ == "__main__":
    BotijoApp().run()
