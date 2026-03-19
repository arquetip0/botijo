#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend del panel (Flask) para lanzar scripts de Botijo con audio correcto
desde la sesión del usuario `jack` y usando el intérprete del venv.

Endpoints:
- GET  /run_script?name=<script.py>   → lanza script y stream de stdout
- POST /interrupt                     → envía Ctrl-C al proceso activo
- POST /close_chromium               → cierra Chromium kiosko
- GET  /health                        → ok de salud
"""
import os
import sys
import signal
import subprocess
import shlex
import time
from pathlib import Path
from flask import Flask, request, Response, jsonify, stream_with_context, make_response

app = Flask(__name__)

# ───────────────────────────────────────────────────────────────
#  Entorno de audio correcto (sesión usuario jack + venv)
# ───────────────────────────────────────────────────────────────
JACK_UID = 1000
JACK_HOME = "/home/jack"
VENV_DEFAULT = f"{JACK_HOME}/botijo/venv_chatgpt"

# Intérprete de Python para scripts:
# 1) si el backend corre dentro de un venv, úsalo
# 2) si no, usa el venv por defecto del proyecto
if os.environ.get("VIRTUAL_ENV"):
    PYTHON_BIN = Path(os.environ["VIRTUAL_ENV"]).joinpath("bin/python").as_posix()
else:
    PYTHON_BIN = f"{VENV_DEFAULT}/bin/python"

# Directorio de trabajo del proyecto
BOTIJO_CWD = f"{JACK_HOME}/botijo"

# Proceso actual
PROC = None

def _log(msg: str):
    print(f"[PANEL] {msg}", flush=True)

def make_env_for_user():
    """Entorno con variables de sesión de usuario (Pulse/DBus) y venv al frente del PATH."""
    env = os.environ.copy()
    env["HOME"] = JACK_HOME
    env["USER"] = "jack"
    env["LOGNAME"] = "jack"
    env["XDG_RUNTIME_DIR"] = f"/run/user/{JACK_UID}"
    env["PULSE_SERVER"] = f"unix:/run/user/{JACK_UID}/pulse/native"
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{JACK_UID}/bus")
    venv_bin = Path(PYTHON_BIN).parent.as_posix()
    env["PATH"] = f"{venv_bin}:{env.get('PATH','')}"
    return env

# ───────────────────────────────────────────────────────────────
#  Utilidades
# ───────────────────────────────────────────────────────────────
def _validate_script_name(name: str) -> Path:
    """Restringe a scripts .py dentro de /home/jack/botijo y devuelve Path seguro."""
    if not name or ".." in name or name.startswith("."):
        raise ValueError("Nombre de script inválido")
    if not name.endswith(".py"):
        raise ValueError("Solo se permiten scripts .py")
    p = Path(BOTIJO_CWD).joinpath(name).resolve()
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}")
    if not str(p).startswith(str(Path(BOTIJO_CWD).resolve())):
        raise ValueError("Ruta fuera del directorio permitido")
    return p

def _stream_lines(proc: subprocess.Popen):
    try:
        for line in iter(proc.stdout.readline, ''):
            yield line
        rem = proc.stdout.read()
        if rem:
            yield rem
    finally:
        code = proc.poll()
        _log(f"Proceso finalizado con código {code}")

# ───────────────────────────────────────────────────────────────
#  CORS básico (file:// → http://localhost:5000)
# ───────────────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "python": sys.executable,
        "venv": os.environ.get("VIRTUAL_ENV", "<none>"),
        "cwd": BOTIJO_CWD,
    })

@app.route('/run_script')
def run_script():
    global PROC
    name = request.args.get('name', '').strip()
    try:
        script_path = _validate_script_name(name)
    except Exception as e:
        return make_response(str(e), 400)

    if PROC and PROC.poll() is None:
        return make_response("Ya hay un proceso en ejecución", 409)

    cmd = [PYTHON_BIN, script_path.as_posix()]
    env = make_env_for_user()

    _log(f"Lanzando: {' '.join(shlex.quote(c) for c in cmd)}")
    _log(f"Usando intérprete: {PYTHON_BIN}")
    _log(f"XDG_RUNTIME_DIR={env.get('XDG_RUNTIME_DIR')}  PULSE_SERVER={env.get('PULSE_SERVER')}")

    PROC = subprocess.Popen(
        cmd,
        cwd=BOTIJO_CWD,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,  # grupo propio → Ctrl-C real
    )

    return Response(stream_with_context(_stream_lines(PROC)), mimetype='text/plain; charset=utf-8')

@app.route('/interrupt', methods=['POST'])
def interrupt():
    global PROC
    if not (PROC and PROC.poll() is None):
        return make_response("No hay proceso activo", 409)
    try:
        pgid = os.getpgid(PROC.pid)

        # 1) Intento limpio: SIGINT
        os.killpg(pgid, signal.SIGINT)
        for _ in range(20):  # ~2s
            if PROC.poll() is not None:
                code = PROC.returncode
                PROC = None
                return f"SIGINT enviado (proceso terminó con código {code})", 200
            time.sleep(0.1)

        # 2) Más firme: SIGTERM
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(20):  # ~2s
            if PROC.poll() is not None:
                code = PROC.returncode
                PROC = None
                return f"SIGTERM enviado (proceso terminó con código {code})", 200
            time.sleep(0.1)

        # 3) Último recurso: SIGKILL
        os.killpg(pgid, signal.SIGKILL)
        try:
            code = PROC.wait(timeout=1)
        except Exception:
            code = PROC.returncode
        PROC = None
        return f"SIGKILL enviado (proceso forzado, código {code})", 200

    except Exception as e:
        return make_response(f"No se pudo interrumpir: {e}", 500)
@app.route('/close_chromium', methods=['POST'])
def close_chromium():
    try:
        subprocess.call(["pkill", "-f", "chromium"])
        return "Chromium cerrado", 200
    except Exception as e:
        return make_response(str(e), 500)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)