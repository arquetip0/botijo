#!/bin/bash
# launch_botijo.sh  –– envuelve el script para asegurarse de que
# se ejecuta dentro del virtual‑env correcto.

cd /home/jack/botijo                # 1) Nos situamos en la carpeta del proyecto
source venv_chatgpt/bin/activate    # 2) Activamos el virtual‑env
python3 searchbotijoperplex.py "$@" # 3) Ejecutamos el script y pasamos cualquier parámetro
