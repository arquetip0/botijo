#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Menú táctil para Botijo en Raspberry Pi  — Botita Edition 😼
"""

# ──────────────────── IMPORTS ────────────────────
import pathlib
import subprocess
import os
import signal
import sys
import threading
import psutil                    # sudo apt install python3-psutil

import tkinter as tk
from tkinter import ttk, messagebox

# ─────────────── CONFIGURACIÓN GENERAL ───────────────
BASE_DIR      = pathlib.Path(__file__).resolve().parent
VENV_DIR      = BASE_DIR / "venv_chatgpt"
VENV_PYTHON   = VENV_DIR / "bin" / "python"
SCRIPT_PATH   = BASE_DIR / "integrator3.py"

EMOJI_FONT    = ("Arial", 28)
BG_COLOR      = "black"
FULLSCREEN    = True            # Cámbialo a False para ventana normal
TERMINAL_FG   = "lime"          # Color texto consola
TERMINAL_BG   = "black"

# ─────────────── FUNCIONES AUXILIARES ───────────────
def _botijo_process():
    """Devuelve el proceso de integrator3.py si está en ejecución."""
    for p in psutil.process_iter(attrs=["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if str(SCRIPT_PATH) in cmd:
            return p
    return None


def _spawn_integrator_with_venv():
    """
    Lanza integrator3.py usando el Python del sistema.
    Asume que launch_menu.sh ha configurado PYTHONPATH y las variables de .env.
    """
    # El entorno (PYTHONPATH, variables de .env) debería haber sido
    # establecido por launch_menu.sh y heredado por este script menu.py.
    # Simplemente pasamos una copia del entorno actual de menu.py al subprocess.
    
    env_for_subprocess = os.environ.copy()

    # DEBUG: Imprime el PYTHONPATH que integrator3.py recibirá (se verá en la consola de Tkinter)
    # print(f"[DEBUG menu.py/_spawn] PYTHONPATH para integrator3: {env_for_subprocess.get('PYTHONPATH')}")
    # print(f"[DEBUG menu.py/_spawn] OPENAI_API_KEY para integrator3: {env_for_subprocess.get('OPENAI_API_KEY')}")


    # Llama a integrator3.py usando el Python del sistema explícitamente.
    # cwd asegura que integrator3.py se ejecute desde su propio directorio.
    return subprocess.Popen(
        ["/usr/bin/python3", str(SCRIPT_PATH)],  # Usa el Python del sistema
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Combina stderr con stdout para verlo en la consola Tkinter
        text=True,
        cwd=str(BASE_DIR),        # Establece el directorio de trabajo para integrator3.py
        env=env_for_subprocess    # Pasa el entorno heredado y configurado
    )

# ─────────────── CALLBACKS DE BOTONES ───────────────
def lanzar_botijo():
    """Mata instancia previa, abre consola fullscreen y arranca integrator."""
    if (p := _botijo_process()):
        try:
            p.terminate(); p.wait(timeout=3)
        except psutil.TimeoutExpired:
            p.kill()

    # Ventana consola fullscreen
    consola = tk.Toplevel(root)
    if FULLSCREEN:
        consola.attributes("-fullscreen", True)
    consola.configure(bg=TERMINAL_BG)  # ✅ FIX: Faltaba cerrar paréntesis
    text_area = tk.Text(
        consola, bg=TERMINAL_BG, fg=TERMINAL_FG,
        insertbackground="white", wrap="none"
    )
    text_area.pack(expand=True, fill="both", padx=12, pady=12)

    # Botón para cerrar consola
    close_btn = tk.Button(
        consola, text="✕ Cerrar", 
        command=lambda: _cerrar_consola(proc, consola),
        bg="red", fg="white", font=("Arial", 12)
    )
    close_btn.pack(side="bottom", pady=5)

    # Lanza integrator3.py con venv
    try:
        proc = _spawn_integrator_with_venv()
    except FileNotFoundError:
        messagebox.showerror(
            "Error",
            "No encuentro el intérprete o el script.\n"
            "Revisa las rutas en la cabecera."
        )
        consola.destroy()
        return

    # Hilo para volcar stdout/err en tiempo real
    def _leer_salida():
        try:
            for linea in iter(proc.stdout.readline, ""):
                if linea:  # ✅ FIX: Verificar que la línea no esté vacía
                    consola.after(0, lambda l=linea: [
                        text_area.insert("end", l),
                        text_area.see("end")
                    ])
        except Exception as e:
            print(f"Error leyendo salida: {e}")
        finally:
            if proc.stdout:
                proc.stdout.close()

    threading.Thread(target=_leer_salida, daemon=True).start()

    # ✅ FIX: Escape para cerrar consola
    consola.bind("<Escape>", lambda e: _cerrar_consola(proc, consola))


def _cerrar_consola(proc, win):
    """Mata el proceso de integrator (si sigue vivo) y cierra la ventana."""
    if proc and proc.poll() is None:
        try:
            proc.terminate(); proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    win.destroy()


def conectar_wifi():
    subprocess.Popen(["nm-connection-editor"])


def lanzar_teclado():
    subprocess.Popen(["onboard"])


def apagar_sistema():
    if os.geteuid() != 0:
        messagebox.showwarning(
            "Permisos",
            "Necesitas privilegios para apagar.\n"
            "Ejecuta el menú con sudo o configura polkit."
        )
        return
    subprocess.Popen(["systemctl", "poweroff"])


def salir_menu():
    root.destroy()


# ───────────────── GUI PRINCIPAL ─────────────────
if __name__ == "__main__":  # ✅ FIX: Proteger el código principal
    root = tk.Tk()
    root.title("Menú de Botijo")
    if FULLSCREEN:
        try:
            root.attributes("-fullscreen", True)  # X11
        except tk.TclError:
            root.state("zoomed")                  # Wayland fallback
    root.configure(bg=BG_COLOR)

    style = ttk.Style()
    style.configure("Boton.TButton", font=EMOJI_FONT, padding=20)

    frame = ttk.Frame(root)  # ✅ FIX: Quité el style incorrecto
    frame.pack(expand=True, padx=40, pady=40)

    BOTONES = (
        ("🔌 Conectar a WiFi", conectar_wifi),
        ("⌨️  Teclado táctil",   lanzar_teclado),
        ("🤖 Lanzar Botijo",     lanzar_botijo),
        ("⏻  Apagar",           apagar_sistema),
        ("⏏️  Salir del menú",   salir_menu),
    )

    for texto, comando in BOTONES:
        ttk.Button(
            frame, text=texto, command=comando,
            style="Boton.TButton"
        ).pack(fill="x", pady=15)

    # ESC cierra el menú
    root.bind("<Escape>", lambda e: salir_menu())

    root.mainloop()
