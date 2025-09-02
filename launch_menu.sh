#!/bin/bash
# filepath: /home/jack/botijo/launch_menu.sh

# Establece el directorio base para evitar problemas con rutas relativas
export BOTIJO_BASE_DIR="/home/jack/botijo"
cd "$BOTIJO_BASE_DIR" || exit 1 # Salir si el directorio no existe

# Rutas importantes
VENV_SITE_PACKAGES="$BOTIJO_BASE_DIR/venv_chatgpt/lib/python3.11/site-packages"
SYSTEM_PYTHON_EXEC="/usr/bin/python3"
MENU_SCRIPT="$BOTIJO_BASE_DIR/menu.py"

# Configura PYTHONPATH para incluir los paquetes del venv
# Esto permite que el Python del sistema (/usr/bin/python3) encuentre los paquetes del venv.
if [ -n "$PYTHONPATH" ]; then
  export PYTHONPATH="$VENV_SITE_PACKAGES:$PYTHONPATH"
else
  export PYTHONPATH="$VENV_SITE_PACKAGES"
fi

# Carga las variables de entorno desde el archivo .env
# Estas variables estarán disponibles para menu.py y los scripts que lance.
if [ -f ".env" ]; then
  echo "[launch_menu.sh] Cargando variables desde .env..."
  set -a # Exporta automáticamente todas las variables que se definan o modifiquen
  source ".env"
  set +a
else
  echo "[launch_menu.sh] ADVERTENCIA: No se encontró el archivo .env en $BOTIJO_BASE_DIR"
fi

# Mensajes de depuración (se verán en los logs de xinit o en la consola si xinit lo permite)
echo "[launch_menu.sh] Ejecutando como usuario: $(whoami)"
echo "[launch_menu.sh] Usando Python: $SYSTEM_PYTHON_EXEC"
echo "[launch_menu.sh] Script del menú: $MENU_SCRIPT"
echo "[launch_menu.sh] PYTHONPATH efectivo: $PYTHONPATH"
echo "[launch_menu.sh] Variables de .env cargadas (ej. OPENAI_API_KEY): $OPENAI_API_KEY" # Verifica una variable clave

# Ejecuta menu.py usando el Python del sistema
# El entorno configurado arriba (PYTHONPATH, variables de .env) será heredado.
exec "$SYSTEM_PYTHON_EXEC" "$MENU_SCRIPT"